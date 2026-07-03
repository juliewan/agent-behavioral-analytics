"""
Loops / runaway recursion (PLANNING.md §3.2).

The definitive runaway signature is COMBINED, never individual: high token
consumption + ~0% error rate + highly repetitive sequences in one session.
A loop that errors gets killed; the loop where every step reports healthy is
the one that runs all night.

Per-session features:
  F1 repetitive n-grams:   dominance of top (tool, args_hash) n-gram, n=2-4
                           (count(top) / total_steps; benign work is diverse)
  F2 delegation depth:     walk parent_span_id chains; flag > baseline p99
  F3 delegation cycles:    per-agent-pair cycle profiles from clean window;
                           baseline-normal A->B->A is NOT flagged alone; flag
                           novel cycles, or normal-shape cycles at anomalous
                           repetition count within one session
  F4 token spikes:         per-agent per-session z-score
  F5 duration/span count:  benign ~1-5s / handful of spans; flag ~60s+ with
                           hundreds of micro-steps (thresholds from baseline
                           percentiles, not hardcoded)
  F6 suspiciously clean:   0% error rate over a prolonged high-step session

Scoring is additive: no single feature pages; repetition + token spike + long
duration + clean health stack to high risk. Keeps FPs down on legitimate long
batch jobs, which tend to have some errors/retries and diverse n-grams.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from aiba.detectors.base import Detector, Finding, pctl, zscore
from aiba.schema import Event


def group_sessions(events: list[Event]) -> dict[str, list[Event]]:
    g: dict[str, list[Event]] = defaultdict(list)
    for e in events:
        g[e.session_id].append(e)
    return g


def pick_ground_truth(evs: list[Event]):
    """Prefer an attack label over a control label over None for a group."""
    labels = [e.injected_scenario for e in evs if e.injected_scenario]
    attacks = [l for l in labels if not l.startswith("control_")]
    return attacks[0] if attacks else (labels[0] if labels else None)


def _session_features(evs: list[Event]) -> dict:
    evs = sorted(evs, key=lambda e: e.ts)
    steps = [(e.tool, e.args_hash) for e in evs if e.event_type == "tool_call"]
    results = [e for e in evs if e.event_type == "tool_result"]
    errors = [e for e in results if e.status in ("error", "timeout")]
    tokens = sum((e.tokens_in or 0) + (e.tokens_out or 0) for e in evs)
    duration = (evs[-1].ts - evs[0].ts).total_seconds() if len(evs) > 1 else 0.0

    # top 2-gram dominance: benign work is diverse, a loop has one 2-gram win
    bigrams = Counter(zip(steps, steps[1:]))
    dominance = (bigrams.most_common(1)[0][1] / (len(steps) - 1)) if len(steps) > 1 else 0.0

    # delegation depth: longest parent_span chain inside the session
    parent = {e.span_id: e.parent_span_id for e in evs}
    def _depth(sp, seen=()):
        p = parent.get(sp)
        return 1 if not p or p in seen else 1 + _depth(p, seen + (sp,))
    depth = max((_depth(sp) for sp in parent), default=1)

    # agent-transition cycles: {a,b} that go back-and-forth within the session
    seq = [e.agent_id for e in evs]
    directed = Counter((a, b) for a, b in zip(seq, seq[1:]) if a != b)
    cycles = {frozenset((a, b)): directed[(a, b)] + directed[(b, a)]
              for (a, b) in directed if a < b and (b, a) in directed}

    return {
        "agent": Counter(seq).most_common(1)[0][0] if seq else None,
        "tokens": tokens, "duration": duration, "span_count": len(evs),
        "n_steps": len(steps), "dominance": dominance,
        "error_rate": (len(errors) / len(results)) if results else 0.0,
        "depth": depth, "cycles": cycles,
        "trace_id": evs[0].trace_id, "ground_truth": pick_ground_truth(evs),
    }


class LoopDetector(Detector):
    name = "loops"

    def fit_baseline(self, events: list[Event]) -> None:
        # Learns: per-agent token/session distributions, benign delegation
        # depth percentiles, per-agent-pair cycle profiles, benign session
        # duration/span-count percentiles.
        self.agent_tokens: dict[str, list[float]] = defaultdict(list)
        durations, spans, depths = [], [], []
        self.benign_cycles: set = set()
        for evs in group_sessions(events).values():
            f = _session_features(evs)
            self.agent_tokens[f["agent"]].append(f["tokens"])
            durations.append(f["duration"]); spans.append(f["span_count"]); depths.append(f["depth"])
            self.benign_cycles.update(f["cycles"].keys())
        self.dur_p99 = max(pctl(durations, 0.99), 30.0)     # floor keeps a sensible minimum
        self.span_p99 = max(pctl(spans, 0.99), 40)
        self.depth_p99 = max(pctl(depths, 0.99), 2)

    def detect(self, events: list[Event]) -> list[Finding]:
        findings: list[Finding] = []
        for sid, evs in group_sessions(events).items():
            f = _session_features(evs)
            flags: list[str] = []
            risk = 0.0

            if f["n_steps"] >= 6 and f["dominance"] > 0.5:                       # F1
                flags.append(f"repetitive n-grams: top 2-gram dominance={f['dominance']:.2f}")
                risk += 2
            z = zscore(f["tokens"], self.agent_tokens.get(f["agent"], [f["tokens"]]))
            if z > 3 and f["tokens"] > 5000:                                     # F4
                flags.append(f"token spike: {f['tokens']} tokens (z={z:.1f} vs {f['agent']} baseline)")
                risk += 2
            if f["duration"] > self.dur_p99:                                     # F5
                flags.append(f"long duration: {f['duration']:.0f}s (> p99 {self.dur_p99:.0f}s)")
                risk += 1
            if f["span_count"] > self.span_p99:
                flags.append(f"span explosion: {f['span_count']} spans (> p99 {self.span_p99:.0f})")
                risk += 1
            if f["depth"] > self.depth_p99 and f["depth"] >= 4:                  # F2
                flags.append(f"delegation depth={f['depth']} (> p99 {self.depth_p99:.0f})")
                risk += 3
            for pair, count in f["cycles"].items():                             # F3
                a, b = tuple(pair)
                if pair not in self.benign_cycles:
                    flags.append(f"novel delegation cycle {a}<->{b} (unseen in baseline)")
                    risk += 3
                elif count > 3:
                    flags.append(f"cycle {a}<->{b} x{count} (baseline-normal shape, anomalous volume)")
                    risk += 2
            if f["error_rate"] == 0.0 and f["span_count"] > self.span_p99:       # F6
                flags.append("suspiciously clean: 0% errors over a prolonged high-step session")
                risk += 1

            if risk >= 3:  # additive: no single weak feature pages alone
                findings.append(Finding(
                    detector=self.name, risk_score=risk, flags=flags, ts=evs[0].ts,
                    agent_id=f["agent"], trace_id=f["trace_id"], session_id=sid,
                    ground_truth=f["ground_truth"]))
        return sorted(findings, key=lambda f: f.risk_score, reverse=True)
