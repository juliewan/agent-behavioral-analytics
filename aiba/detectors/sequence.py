"""
Sequence / rare-edge analytics (PLANNING.md §3.4).

Complements loops without overlapping: loops catches repetition of the SAME
edge, this catches edges that shouldn't exist at all. Markov/edge-frequency
(count which tool follows which tool, be suspicious of transitions never
counted), deliberately not transformers: constant memory, explainable,
deterministic.

Detection:
  - Per-agent edge counts P(next tool | current tool) from the clean window;
    score session edges by rarity (below a probability floor, or never seen).
    Rarity alone contributes score but does NOT page, because
    novel-but-innocuous edges happen in benign traffic all the time.
  - A high-risk edge is DERIVED from per-tool taint tags, not enumerated: an
    untrusted SOURCE (ingests external / attacker-influenced content)
    followed by a privileged SINK (executes, exfiltrates, or touches
    secrets/identity) pages regardless of baseline frequency. This is
    taint/dataflow, and untrusted ingest followed by a privileged action is
    the classic indirect-prompt-injection tell (OWASP LLM01). Deriving from
    tool metadata rather than listing known-bad (cur,nxt) pairs means any
    such transition flags, including pairs nobody enumerated, and avoids the
    circularity of scoring a hand-written edge list against hand-authored
    attacks that traverse exactly those edges.

The taint tags are the toy stand-in for governance metadata (data
classification, IAM annotations on each tool); a detection repo consumes such
metadata, it does not mint it (§1). We detect the behavioral symptom without
simulating the injection mechanism or classifying content.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from aiba.detectors.base import Detector, Finding
from aiba.schema import Event

# --- tool taint tags (governance-metadata stand-in, NOT an attack list) -------
# High-risk edges are derived: untrusted-source -> privileged-sink. In
# production these tags come from data-classification / IAM; here they annotate
# tools by capability, so the rule generalizes to unseen (cur,nxt) pairs.
UNTRUSTED_SOURCES = {          # ingest external / attacker-influenced content
    "web_fetch", "fetch", "read_web", "read_docs", "read_email", "browse",
    "http_get", "scrape",
}
PRIVILEGED_SINKS = {           # execute, exfiltrate, or touch secrets / identity
    "exec_shell", "run_shell", "exec", "read_credentials", "get_secrets",
    "delete_records", "bulk_delete", "exfil_data", "grant_admin",
    "disable_mfa", "issue_token", "revoke_session",
}
# narrow substring fallbacks so an unseen tool name still classifies sensibly
_SOURCE_HINTS = ("web", "scrape", "browse", "http_get", "read_web", "read_docs", "read_email")
_SINK_HINTS = ("exec", "shell", "delete", "cred", "secret", "exfil", "grant_admin", "disable_mfa")
RARITY_FLOOR = 0.02   # P(next|cur) below this = rare; rarity alone does NOT page


def _source_class(tool: str) -> str:
    if tool in UNTRUSTED_SOURCES or any(h in tool for h in _SOURCE_HINTS):
        return "untrusted"
    return "trusted"


def _sink_class(tool: str) -> str:
    if tool in PRIVILEGED_SINKS or any(h in tool for h in _SINK_HINTS):
        return "privileged"
    return "normal"


def _is_high_risk(cur: str, nxt: str) -> str | None:
    """Taint-derived high-risk edge: untrusted-source -> privileged-sink (§3.4).
    No enumerated (cur,nxt) pairs; the tags do the work, so the rule catches
    transitions never explicitly listed."""
    if _source_class(cur) == "untrusted" and _sink_class(nxt) == "privileged":
        return f"untrusted-source {cur} -> privileged-sink {nxt} (taint dataflow)"
    return None


def _session_edges(evs: list[Event]):
    """Consecutive same-agent tool_call transitions within a session."""
    calls = sorted((e for e in evs if e.event_type == "tool_call" and e.tool),
                   key=lambda e: e.ts)
    for a, b in zip(calls, calls[1:]):
        if a.agent_id == b.agent_id:
            yield a.agent_id, a.tool, b.tool, a.span_id, b.span_id, b


class SequenceDetector(Detector):
    name = "sequence"

    def fit_baseline(self, events: list[Event]) -> None:
        # Per-agent (current tool -> next tool) transition counts.
        self.trans: dict[str, Counter] = defaultdict(Counter)
        self.totals: dict[str, Counter] = defaultdict(Counter)
        by_session: dict[str, list[Event]] = defaultdict(list)
        for e in events:
            by_session[e.session_id].append(e)
        for evs in by_session.values():
            for agent, cur, nxt, *_ in _session_edges(evs):
                self.trans[agent][(cur, nxt)] += 1
                self.totals[agent][cur] += 1

    def detect(self, events: list[Event]) -> list[Finding]:
        by_session: dict[str, list[Event]] = defaultdict(list)
        for e in events:
            by_session[e.session_id].append(e)

        findings: list[Finding] = []
        for sid, evs in by_session.items():
            for agent, cur, nxt, sp_a, sp_b, ev in _session_edges(evs):
                flags, risk = [], 0.0
                reason = _is_high_risk(cur, nxt)
                total = self.totals[agent].get(cur, 0)
                prob = (self.trans[agent].get((cur, nxt), 0) / total) if total else 0.0
                rare = prob < RARITY_FLOOR

                if reason:
                    flags.append(f"high-risk edge {cur}->{nxt} ({reason})")
                    risk += 5
                    if rare:
                        flags.append(f"edge also rare for {agent} (P={prob:.3f})")
                elif rare:
                    # rarity contributes score but does not page on its own
                    flags.append(f"rare edge {cur}->{nxt} for {agent} (P={prob:.3f})")
                    risk += 1

                if risk >= 4:   # only risk-list hits (or corroborated) page
                    findings.append(Finding(
                        detector=self.name, risk_score=risk, flags=flags, ts=ev.ts,
                        agent_id=agent, trace_id=ev.trace_id, session_id=sid,
                        spans=[sp_a, sp_b], ground_truth=ev.injected_scenario))
        return sorted(findings, key=lambda f: f.risk_score, reverse=True)
