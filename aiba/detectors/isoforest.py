"""
Isolation Forest: a learned baseline over the same engineered features the
rule detectors use (PLANNING.md §3, §5 future-work).

Why this exists: every other detector in the suite is a hand-written rule or
a z-score, and a reviewer will fairly ask whether an unsupervised learner
would do as well or better. This detector answers instead of guessing. It
consumes one per-session behavioral feature vector (the union of the scalars
loops, cost, tool_health, memory, sequence, and action_hallucination
already compute), trains an Isolation Forest on the clean baseline window,
normal traffic only, and scores the eval window by anomaly.
scripts/eval_isoforest.py then sweeps the threshold into a PR curve and
reports average precision against the rule suite on the same session
universe.

Scope, stated plainly:
- The unit of analysis is the SESSION (matches loops/cost/memory).
  Card-modality attacks (typosquat, shadowing, capability_escalation,
  endpoint_hijack, rug_pull) have no event-stream session, so they are out
  of this detector's universe by construction. That is card_drift's beat,
  not this one's.
- Features are behavioral scalars only, no embeddings. A purely semantic
  attack (goal_mutation, whose planner text drifts while its tool/token
  footprint stays benign) is expected to be near-invisible here. That
  negative is deliberate: it shows where a behavioral learner cannot
  substitute for the embedding rule.
- Learned, but still baseline-derived and deterministic: trained on the
  clean window, seeded, no network.

Requires the [ml] extra (scikit-learn). Like [embed] for goal_drift, the
rest of the suite runs without it.
"""

from __future__ import annotations

from collections import defaultdict

from aiba.detectors.base import Detector, Finding, pctl, std
from aiba.detectors.cost import _session_cost
from aiba.detectors.loops import _session_features, group_sessions, pick_ground_truth
from aiba.detectors.memory import _dlp_hits
from aiba.detectors.sequence import _is_high_risk, _session_edges
from aiba.schema import Event

# Fixed feature order (the vector's columns). Kept as a module constant so the
# eval script can label a feature-importance readout and so the order is stable
# across baseline fit and eval scoring.
FEATURE_NAMES: list[str] = [
    "tokens",               # loops F4
    "duration_s",           # loops F5
    "span_count",           # loops F5
    "n_tool_calls",
    "bigram_dominance",     # loops F1 (repetition)
    "error_rate",           # loops F6 (0% over long session is suspicious)
    "n_errors",
    "n_empty",              # empty/0-record results (action_hallucination R3)
    "delegation_depth",     # loops F2
    "n_cycles",             # loops F3 (delegation cycles)
    "max_cycle_count",      # loops F3 (repeat count of a cycle)
    "cost_usd",             # cost §3.7
    "max_latency_ms",       # tool_health §3.6
    "n_memory_ops",
    "n_agent_deletes",      # memory M2
    "n_dlp_hits",           # memory M3 (secret written to memory)
    "n_cross_user",         # memory M1
    "n_cross_tenant",       # memory M1
    "n_high_risk_edges",    # sequence §3.4 enumerated edges
    "n_claim_contradictions",  # action_hallucination R1/R2/R3
    "n_claimed_actions",
]


def _session_vector(evs: list[Event]) -> list[float]:
    """One fixed-order behavioral feature vector for a session, drawing on the
    same helpers the rule detectors use (parity, not a parallel reimplementation).
    """
    f = _session_features(evs)   # tokens, duration, span_count, n_steps,
                                 # dominance, error_rate, depth, cycles, ...
    results = [e for e in evs if e.event_type == "tool_result"]
    n_errors = sum(1 for e in results if e.status in ("error", "timeout"))
    n_empty = sum(1 for e in results
                  if e.status == "empty" or e.records_returned == 0)
    latencies = [e.latency_ms for e in evs if e.latency_ms is not None]
    cycles = f["cycles"]

    # memory-op derived counts (M1/M2/M3 raw material)
    n_mem = n_del = n_dlp = n_cu = n_ct = 0
    for e in evs:
        if e.event_type != "memory_op" or not e.memory:
            continue
        m = e.memory
        n_mem += 1
        if m.op == "delete" and m.initiator == "agent":
            n_del += 1
        if m.op in ("create", "update") and m.content and _dlp_hits(m.content):
            n_dlp += 1
        if m.op == "read":
            if e.user_id and m.owner and m.owner != e.user_id:
                n_cu += 1
            if e.tenant and m.tenant and m.tenant != e.tenant:
                n_ct += 1

    # sequence: count enumerated high-risk transitions in the session
    n_hre = sum(1 for _agent, cur, nxt, *_ in _session_edges(evs)
                if _is_high_risk(cur, nxt))

    # action hallucination: claimed actions unsupported by the execution log
    called = {e.tool for e in evs if e.event_type == "tool_call" and e.tool}
    last_result: dict[str, Event] = {}
    for e in sorted(evs, key=lambda x: x.ts):
        if e.event_type == "tool_result" and e.tool:
            last_result[e.tool] = e
    n_claimed = n_contra = 0
    for e in evs:
        if e.event_type == "final_response":
            for tool in e.claimed_actions:
                n_claimed += 1
                res = last_result.get(tool)
                if tool not in called:
                    n_contra += 1
                elif res is not None and (res.status in ("error", "timeout", "empty")
                                          or res.records_returned == 0):
                    n_contra += 1

    return [
        float(f["tokens"]), float(f["duration"]), float(f["span_count"]),
        float(f["n_steps"]), float(f["dominance"]), float(f["error_rate"]),
        float(n_errors), float(n_empty), float(f["depth"]),
        float(len(cycles)), float(max(cycles.values(), default=0)),
        float(_session_cost(evs)), float(max(latencies, default=0.0)),
        float(n_mem), float(n_del), float(n_dlp), float(n_cu), float(n_ct),
        float(n_hre), float(n_contra), float(n_claimed),
    ]


def session_matrix(events: list[Event]) -> tuple[list[list[float]], list[dict]]:
    """Feature matrix + per-row metadata (session_id, agent, ts, trace_id,
    ground_truth) for every session in `events`, in a stable session order."""
    X, meta = [], []
    for sid, evs in sorted(group_sessions(events).items()):
        X.append(_session_vector(evs))
        evs_sorted = sorted(evs, key=lambda e: e.ts)
        meta.append({
            "session_id": sid,
            "agent_id": evs_sorted[0].agent_id,
            "trace_id": evs_sorted[0].trace_id,
            "ts": evs_sorted[0].ts,
            "ground_truth": pick_ground_truth(evs),
        })
    return X, meta


class IsolationForestDetector(Detector):
    """Unsupervised anomaly detector over the per-session behavioral vector.

    Trained on the clean baseline window; scores eval sessions by isolation
    depth. anomaly_score() exposes the continuous score the PR sweep consumes;
    detect() applies a baseline-derived operating threshold so it can also sit
    in the report alongside the rule detectors.
    """

    name = "isoforest"

    def __init__(self, n_estimators: int = 200, seed: int = 42) -> None:
        self.n_estimators = n_estimators
        self.seed = seed
        self.model = None

    def fit_baseline(self, events: list[Event]) -> None:
        from sklearn.ensemble import IsolationForest   # [ml] extra

        X, _ = session_matrix(events)
        self.model = IsolationForest(
            n_estimators=self.n_estimators, max_samples="auto",
            contamination="auto", random_state=self.seed)
        self.model.fit(X)
        # Higher score == more anomalous (negate sklearn's normal-positive
        # decision_function). Operating point = the 95th-percentile clean-window
        # score: a standard, robust novelty threshold. (The single clean-window
        # MAX is too brittle here: with hundreds of benign sessions it sits so
        # high nothing in eval clears it; p95 is the defensible deploy choice.)
        train_scores = [-s for s in self.model.decision_function(X)]
        self.threshold = pctl(train_scores, 0.95)
        self.train_max = max(train_scores)
        self.train_spread = std(train_scores) or 1.0

    def anomaly_score(self, events: list[Event]) -> tuple[list[float], list[dict]]:
        """Continuous anomaly score per session (higher = more anomalous),
        with row metadata. This is what the PR threshold sweep operates on."""
        assert self.model is not None, "fit_baseline first"
        X, meta = session_matrix(events)
        scores = [-s for s in self.model.decision_function(X)]
        return scores, meta

    def detect(self, events: list[Event]) -> list[Finding]:
        scores, meta = self.anomaly_score(events)
        findings: list[Finding] = []
        for score, m in zip(scores, meta):
            if score > self.threshold:
                risk = min(10.0, 3.0 + 4.0 * (score - self.threshold) / self.train_spread)
                findings.append(Finding(
                    detector=self.name, risk_score=round(risk, 1),
                    flags=[f"session anomaly score {score:.3f} > clean-window p95 "
                           f"{self.threshold:.3f} (isolation forest novelty)"],
                    ts=m["ts"], agent_id=m["agent_id"], trace_id=m["trace_id"],
                    session_id=m["session_id"], ground_truth=m["ground_truth"]))
        return sorted(findings, key=lambda f: f.risk_score, reverse=True)
