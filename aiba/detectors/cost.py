"""
Cost per token (PLANNING.md §3.7).

Static pricing table (per-model in/out token rates); per-agent and per-trace
spend baselines; z-score anomalies.

Intentionally overlaps with loop detection: the injected retry spiral should
also be the top cost anomaly, which gives report.py cheap, legible
cross-detector corroboration.
"""

from __future__ import annotations

from collections import defaultdict

from aiba.detectors.base import Detector, Finding, zscore
from aiba.detectors.loops import group_sessions, pick_ground_truth
from aiba.schema import Event

# static, offline pricing (USD per 1k tokens), no API calls. Single default
# rate; the point here is the anomaly, not the invoice.
PRICING_PER_1K_TOKENS: dict[str, tuple[float, float]] = {
    "default": (0.003, 0.015),
}


def _session_cost(evs: list[Event]) -> float:
    r_in, r_out = PRICING_PER_1K_TOKENS["default"]
    tin = sum(e.tokens_in or 0 for e in evs)
    tout = sum(e.tokens_out or 0 for e in evs)
    return (tin * r_in + tout * r_out) / 1000.0


class CostDetector(Detector):
    name = "cost"

    def fit_baseline(self, events: list[Event]) -> None:
        # Per-agent session-spend baselines (z-score, same mechanic as loops'
        # token spike; the injected retry spiral should top both, §3.7).
        self.agent_costs: dict[str, list[float]] = defaultdict(list)
        for evs in group_sessions(events).values():
            agent = evs[0].agent_id
            self.agent_costs[agent].append(_session_cost(evs))

    def detect(self, events: list[Event]) -> list[Finding]:
        findings: list[Finding] = []
        for sid, evs in group_sessions(events).items():
            agent = evs[0].agent_id
            cost = _session_cost(evs)
            base = self.agent_costs.get(agent, [cost])
            z = zscore(cost, base)
            if z > 3 and cost > 0.05:
                findings.append(Finding(
                    detector=self.name, risk_score=min(5.0, 2 + z / 3),
                    flags=[f"cost spike: ${cost:.2f} this session (z={z:.1f} vs {agent} baseline)"],
                    ts=evs[0].ts, agent_id=agent, trace_id=evs[0].trace_id,
                    session_id=sid, ground_truth=pick_ground_truth(evs)))
        return sorted(findings, key=lambda f: f.risk_score, reverse=True)
