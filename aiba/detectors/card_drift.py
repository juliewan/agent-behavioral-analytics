"""
Agent card drift: A2A-style card registrations scored against a learned
baseline (PLANNING.md §3.8).

Rules:
  R1 typosquat:              name string similarity > 0.85 to a known name +
                             NEW agent_id (SequenceMatcher ratio)
  R2 shadowing:              skill-set Jaccard > 0.6 with a known agent +
                             new agent_id
  R3 capability escalation:  known agent_id, skill-count z-score > 2.5 vs its
                             own baseline
  R4 endpoint hijack:        known agent_id, endpoint not in baseline set
  R5 scope violation:        cross-reference the EVENT log against cards;
                             flag any agent invoking a tool not declared in
                             its card. Turns the card from registry artifact
                             into an authorization boundary, and doubles as
                             novel-tool detection (a tool no card declares is
                             itself a finding).
  R6 rug-pull [embed]:       name-close AND description-far. Edit distance
                             says "claims to be the same tool"; embedding
                             distance on the description says "but it now does
                             something different" (the MCP tool-description
                             rug-pull).
  R8 sensitive-capability:   known agent_id gains a NEW skill whose tags mark
                             a privileged/identity/exfil action. Fires on the
                             capability CATEGORY, not the count, so it catches
                             a skill swap (count unchanged) that R3 would
                             miss. Reads the structured skill tags (governance
                             metadata), not a hand-listed set of bad skills.

Division of labor: identity spoofing gets string metrics, because typosquats
are DESIGNED to be semantically identical (embedding sim ~0.99) and
embeddings are exactly the wrong tool there. Semantic drift of goals/plans
gets embeddings (§3.5). Rug-pull gets the intersection: string-close AND
embedding-far.
"""

from __future__ import annotations

import difflib

from aiba.detectors.base import Detector, Finding, mean, std
from aiba.embed import cosine, embed_texts, embeddings_available
from aiba.schema import AgentCard, Event
from aiba.synth.traffic import BASELINE_CUTOFF_DAY, SENSITIVE_TAGS, START


def _ids(card: AgentCard) -> set[str]:
    """The card's declared skill identifiers (== invocable tool names)."""
    return {s.id for s in card.skills}


def _name_sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a and b) else 0.0


def _day(ts) -> int:
    return (ts - START).days


class CardDriftDetector(Detector):
    name = "card_drift"

    # R3 fallback when baseline skill count has zero observed variance (see
    # fit_baseline / _score_card): flag a skill-count jump of at least this
    # many beyond the baseline max. R3_Z gates the statistical path, used
    # only for agents whose baseline genuinely carries spread.
    CAP_DELTA_MIN = 2
    R3_Z = 2.5

    def __init__(self, cards: list[AgentCard], use_embed: bool | None = None):
        # Card stream is a second input alongside events (events are only
        # needed for R5 scope violation).
        self.cards = cards
        self.use_embed = embeddings_available() if use_embed is None else use_embed

    def fit_baseline(self, events: list[Event]) -> None:
        # Learn from the pre-attack card window (day < 20).
        base_cards = [c for c in self.cards if _day(c.timestamp) < BASELINE_CUTOFF_DAY]
        self.known_names = {c.name for c in base_cards}
        self.known_ids = {c.agent_id for c in base_cards}
        self.baselines: dict[str, dict] = {}
        by_agent: dict[str, list[AgentCard]] = {}
        for c in base_cards:
            by_agent.setdefault(c.agent_id, []).append(c)
        for aid, group in by_agent.items():
            counts = [len(c.skills) for c in group]
            self.baselines[aid] = {
                "skills": {s.id for c in group for s in c.skills},
                "endpoints": {c.endpoint for c in group},
                "names": {c.name for c in group},
                "desc": group[-1].description,          # last known good description
                "cap_mean": mean(counts),
                "cap_max": max(counts),
                "cap_std": std(counts),                  # real std, no floor (see _score_card)
            }
        # R5: declared-tool map (the authorization boundary) = baseline skills.
        self.declared = {aid: b["skills"] for aid, b in self.baselines.items()}

    def _score_card(self, card: AgentCard) -> tuple[list[str], float]:
        flags: list[str] = []
        risk = 0.0
        base = self.baselines.get(card.agent_id)

        # R1 typosquat: new agent_id, name string highly similar to a known name
        for known in self.known_names:
            if card.name != known and _name_sim(card.name, known) > 0.85 \
                    and card.agent_id not in self.known_ids:
                flags.append(f"typosquat_vs:{known} sim={_name_sim(card.name, known):.2f}")
                risk += 3

        # R2 shadowing: new agent_id, skill set highly overlapping a known agent
        for other, b in self.baselines.items():
            if card.agent_id != other and card.agent_id not in self.known_ids:
                j = _jaccard(_ids(card), b["skills"])
                if j > 0.6:
                    flags.append(f"shadowing_vs:{other} jaccard={j:.2f}")
                    risk += 3

        if base:
            new_skills = [s for s in card.skills if s.id not in base["skills"]]
            # R3 capability escalation: known id, skill count jumps well past
            # baseline. Every agent's baseline skill count in this dataset has
            # zero observed variance (a card's declared tool count just
            # doesn't move on its own between benign re-registrations), so a
            # z-score here would need a manufactured variance floor to avoid
            # dividing by zero -- dressing a fixed threshold up as a
            # statistical test. Honest version: it's a fixed count-delta
            # threshold on the zero-variance baselines (the common case here),
            # and a real z-test on any baseline that does carry genuine spread.
            count = len(card.skills)
            delta = count - base["cap_max"]
            z = (count - base["cap_mean"]) / base["cap_std"] if base["cap_std"] else None
            if delta >= self.CAP_DELTA_MIN or (z is not None and z > self.R3_Z):
                basis = f"z={z:.2f}" if z is not None else f"delta=+{delta} vs baseline max {base['cap_max']}"
                flags.append(f"capability_escalation {basis} new_skills={sorted(s.id for s in new_skills)}")
                risk += 4
            # R8 sensitive-capability gain: a NEW skill whose tags mark it as
            # a privileged/identity/exfil action. Fires on the category of the
            # capability, not the count, so it catches a swap (count
            # unchanged, z near 0) that R3 would miss. The structured skill's
            # tags make this expressible; a flat tool-name list could not.
            sens = [s for s in new_skills if set(s.tags) & SENSITIVE_TAGS]
            if sens:
                cats = sorted({t for s in sens for t in set(s.tags) & SENSITIVE_TAGS})
                flags.append(f"sensitive_capability_gain tags={cats} "
                             f"skills={sorted(s.id for s in sens)}")
                risk += 3
            # R4 endpoint hijack: known id, endpoint not in baseline set
            if card.endpoint not in base["endpoints"]:
                flags.append(f"endpoint_drift new={card.endpoint}")
                risk += 4
            # R6 rug-pull [embed]: name-close AND description-far
            if self.use_embed and card.name in base["names"]:
                sim = cosine(*embed_texts([card.description, base["desc"]]))
                if sim < 0.6:
                    flags.append(f"rug_pull desc_sim={sim:.2f} (name-close, description-far)")
                    risk += 4

        return flags, risk

    def detect(self, events: list[Event]) -> list[Finding]:
        findings: list[Finding] = []

        # Score cards in the eval window (day >= cutoff), like a live window.
        for card in self.cards:
            if _day(card.timestamp) < BASELINE_CUTOFF_DAY:
                continue
            flags, risk = self._score_card(card)
            if risk > 0:
                findings.append(Finding(detector=self.name, risk_score=risk, flags=flags,
                                        ts=card.timestamp, agent_id=card.agent_id,
                                        ground_truth=card.injected))

        # R5 scope violation: event log vs declared tools. A tool no card
        # declares for that agent is a finding (and doubles as novel-tool
        # detection). Report once per (agent, tool) in the eval window.
        seen: set[tuple[str, str]] = set()
        all_declared = {t for skills in self.declared.values() for t in skills}
        for e in events:
            if e.event_type != "tool_call" or not e.tool:
                continue
            if _day(e.ts) < BASELINE_CUTOFF_DAY:
                continue
            declared = self.declared.get(e.agent_id, set())
            if e.tool not in declared and (e.agent_id, e.tool) not in seen:
                seen.add((e.agent_id, e.tool))
                novel = " novel_tool(no card declares it)" if e.tool not in all_declared else ""
                findings.append(Finding(
                    detector=self.name, risk_score=3, ts=e.ts, agent_id=e.agent_id,
                    trace_id=e.trace_id, session_id=e.session_id, spans=[e.span_id],
                    flags=[f"scope_violation tool={e.tool} not in declared skills{novel}"],
                    ground_truth=e.injected_scenario))

        return sorted(findings, key=lambda f: f.risk_score, reverse=True)
