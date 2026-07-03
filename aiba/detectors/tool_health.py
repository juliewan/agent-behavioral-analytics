"""
Tool health: latency and failure-rate baselines (PLANNING.md §3.6).

Per-tool baselines come from the clean window: latency distribution and
failure rate. The security framing is that a latency or failure spike can be
the symptom of an endpoint hijack, which makes this detector the bridge
between card_drift's endpoint-change flag and the downstream hallucination
and loop findings in the correlated incident (§4.1: the hijacked endpoint is
slow and flaky, so this fires second in the chain).

Signal weighting: the latency spike is the standalone-worthy trigger. The
failure-rate spike is corroboration only. It carries risk below the emit
floor, so on its own it cannot page. It earns its place by stacking on the
latency spike: slow AND flaky is the hijack signature. Same additive-risk
corroboration model as base.py, applied within one detector.

Failure rate is a proportion (k failures of n calls), not a continuous
measurement, and most (session, tool) groups in the eval window are tiny:
n=1 or n=2 for the large majority. A raw rate takes a coin flip at face
value -- 1 failure of 2 calls reads as a 50% rate, which was every one of
this detector's false positives before the fix. The fix is not a bigger
minimum-sample floor (that just trades false positives for false negatives
at whatever n you pick); it's using the right instrument for a small-sample
proportion in the first place: a Wilson score lower bound (`base.wilson_lower`)
instead of the naive rate. Wilson shrinks toward 0 as n shrinks and as the
observed rate sits closer to the edges (0% or 100%), so "1-of-2 failed" comes
out around a 12% lower bound -- below almost every tool's baseline rate,
so it correctly does not fire -- while a genuinely sustained flakiness (e.g.
3-of-6, the hijack-chain shape) clears it easily. No hand-picked sample-size
cliff required.
"""

from __future__ import annotations

from collections import defaultdict

from aiba.detectors.base import Detector, Finding, pctl, wilson_lower
from aiba.detectors.loops import group_sessions, pick_ground_truth
from aiba.schema import Event


class ToolHealthDetector(Detector):
    name = "tool_health"

    # One-sided confidence level for the failure-rate Wilson bound: 1.645 ~=
    # 95% one-sided. Higher = more conservative (fewer, more certain flags).
    FAIL_Z = 1.645
    # The Wilson lower bound must clear the tool's own baseline rate by this
    # much before the failure signal contributes any risk.
    FAIL_MARGIN = 0.10
    # Latency spike = this many times the tool's p99 baseline. Robust to
    # heavy-tailed benign latency where a z-score is not (see detect()).
    LAT_SPIKE_MULT = 4.0

    def fit_baseline(self, events: list[Event]) -> None:
        # Per-tool latency distribution (mean/p95) and failure rate.
        lat: dict[str, list[float]] = defaultdict(list)
        n_ok: dict[str, int] = defaultdict(int)
        n_fail: dict[str, int] = defaultdict(int)
        for e in events:
            if e.event_type != "tool_result" or not e.tool:
                continue
            if e.latency_ms is not None:
                lat[e.tool].append(e.latency_ms)
            if e.status in ("error", "timeout"):
                n_fail[e.tool] += 1
            else:
                n_ok[e.tool] += 1
        self.lat = lat
        self.fail_rate = {t: n_fail[t] / (n_fail[t] + n_ok[t])
                          for t in set(n_ok) | set(n_fail)}

    def detect(self, events: list[Event]) -> list[Finding]:
        # Aggregate the eval window per (tool, session) so a hijacked endpoint's
        # latency + failure spike surfaces as one finding, not one per span.
        findings: list[Finding] = []
        for sid, evs in group_sessions(events).items():
            per_tool: dict[str, list[Event]] = defaultdict(list)
            for e in evs:
                if e.event_type == "tool_result" and e.tool:
                    per_tool[e.tool].append(e)
            for tool, results in per_tool.items():
                base_lat = self.lat.get(tool, [])
                if len(base_lat) < 3:
                    continue
                lats = [r.latency_ms for r in results if r.latency_ms is not None]
                fails = sum(1 for r in results if r.status in ("error", "timeout"))
                flags, risk = [], 0.0
                if lats:
                    # Robust to heavy-tailed benign latency, which a mean/std
                    # z-score is not: normal long-tail calls reach z>13 here,
                    # so z>4 pages on ordinary slowness. Page instead on a
                    # large multiple of the tool's p99 baseline; the hijacked
                    # endpoint sits ~6x p99, benign tails stay under ~3x.
                    p99 = pctl(base_lat, 0.99)
                    spike = max(lats)
                    if spike > self.LAT_SPIKE_MULT * p99:
                        flags.append(f"latency spike: {spike:.0f}ms "
                                     f"(> {self.LAT_SPIKE_MULT:.0f}x {tool} p99 {p99:.0f}ms)")
                        risk += 3
                # Failure rate is corroboration only, never a standalone page:
                # even a genuinely flaky window only means "attack" when the
                # same endpoint is also slow (the §4.1 hijack signature), so
                # this contributes risk below the emit floor. The test itself
                # is a Wilson lower bound on the observed rate vs. the tool's
                # baseline rate (see module docstring) -- small n shrinks the
                # bound toward 0 automatically, no minimum-sample gate needed.
                n = len(results)
                if n:
                    lower = wilson_lower(fails, n, z=self.FAIL_Z)
                    baseline = self.fail_rate.get(tool, 0.0)
                    if lower > baseline + self.FAIL_MARGIN:
                        fr = fails / n
                        flags.append(f"failure-rate spike: {fr:.0%} on {tool} over {n} calls "
                                     f"(Wilson lower bound {lower:.0%} > baseline {baseline:.0%} + {self.FAIL_MARGIN:.0%})")
                        risk += 2
                if risk >= 3:
                    findings.append(Finding(
                        detector=self.name, risk_score=risk, flags=flags,
                        ts=results[0].ts, agent_id=results[0].agent_id,
                        trace_id=results[0].trace_id, session_id=sid,
                        ground_truth=pick_ground_truth(evs)))
        return sorted(findings, key=lambda f: f.risk_score, reverse=True)
