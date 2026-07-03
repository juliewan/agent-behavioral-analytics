"""
Merged ranked findings + cross-detector correlation (PLANNING.md §2, §4).

Two jobs:

1. Merge every detector's findings into one ranked table (risk_score desc)
   with per-detector and overall precision/recall vs injected_scenario:
   one summary across the whole suite.

2. Correlate findings by trace_id / agent_id / time proximity and reconstruct
   incident timelines. The payoff demo, and the reason these signals belong
   in one pipeline, is the two §4 chains re-emerging from independent
   detectors:

     endpoint hijack chain:   card_drift (endpoint change), then tool_health
       (latency/failure spike), then action_hallucination
       (errored-but-claimed), then loops + cost (same runaway session)

     memory poisoning chain:  sequence (web-read-then-exec edge), then
       memory M3 (credential write), then memory M2 (deletion
       burst chained off the M3 hit). goal_drift stays above its bound here
       and recovers mid-session, so it correctly does not fire on this
       chain; see PLANNING §4.2.

Scope: we detect the drift SYMPTOMS of memory poisoning, not the poisoning
mechanism itself.
"""

from __future__ import annotations

from datetime import timedelta

from aiba.detectors.base import Finding, evaluate
from aiba.schema import Event


def merge_findings(all_findings: dict[str, list[Finding]]) -> list[Finding]:
    """One ranked list across detectors (risk_score desc), ground truth kept."""
    flat = [f for fs in all_findings.values() for f in fs]
    return sorted(flat, key=lambda f: f.risk_score, reverse=True)


def correlate(all_findings: dict[str, list[Finding]],
              window_hours: int = 24) -> list[list[Finding]]:
    """Group findings into incidents by trace, then fold in same-agent findings
    (e.g. card_drift, which has no trace_id) within a time window. Returns
    clusters that span >=2 detectors: the §4 chains re-emerging."""
    flat = merge_findings(all_findings)
    clusters: list[dict] = []
    for f in sorted(flat, key=lambda x: x.ts):
        placed = False
        for c in clusters:
            same_trace = f.trace_id and f.trace_id == c["trace_id"]
            same_agent_near = (f.agent_id == c["agent_id"]
                               and abs((f.ts - c["ts"]).total_seconds()) <= window_hours * 3600)
            if same_trace or same_agent_near:
                c["findings"].append(f)
                c["trace_id"] = c["trace_id"] or f.trace_id
                placed = True
                break
        if not placed:
            clusters.append({"trace_id": f.trace_id, "agent_id": f.agent_id,
                             "ts": f.ts, "findings": [f]})
    incidents = [c["findings"] for c in clusters
                 if len({fd.detector for fd in c["findings"]}) >= 2]
    return sorted(incidents, key=lambda fs: sum(f.risk_score for f in fs), reverse=True)


def print_report(events: list[Event], all_findings: dict[str, list[Finding]]) -> None:
    """Ranked candidates, per-detector eval summary, incident timelines."""
    print(f"\n{'=' * 70}\nAGENT BEHAVIORAL ANALYTICS — detection report")
    print(f"{'=' * 70}\nEvaluated {len(events)} events in the detection window.\n")

    print("--- Per-detector eval vs ground truth ---")
    for name, fs in all_findings.items():
        ev = evaluate(fs, events)
        miss = f"  missed={ev['missed']}" if ev["missed"] else ""
        print(f"  {name:22s} caught {ev['caught']}/{ev['injected_total']}  "
              f"FP={ev['false_positives']}  P={ev['precision']:.2f} R={ev['recall']:.2f}{miss}")

    print("\n--- Top ranked findings ---")
    for f in merge_findings(all_findings)[:15]:
        gt = f.ground_truth or "BENIGN"
        print(f"  [risk={f.risk_score:>4.1f}] {f.detector:20s} {f.agent_id:18s} truth={gt}")
        for fl in f.flags:
            print(f"        - {fl}")

    print("\n--- Correlated incidents (>=2 detectors, §4 chains) ---")
    for i, inc in enumerate(correlate(all_findings), 1):
        dets = sorted({f.detector for f in inc})
        truth = next((f.ground_truth for f in inc if f.ground_truth), "?")
        total = sum(f.risk_score for f in inc)
        print(f"\n  Incident {i}: truth={truth}  total_risk={total:.1f}  detectors={dets}")
        for f in sorted(inc, key=lambda x: x.ts):
            print(f"    {f.ts:%m-%d %H:%M}  {f.detector:20s} {f.flags[0] if f.flags else ''}")
