"""
Detector interface: events in, ranked findings out (PLANNING.md §2, §3).

Scoring model: findings accumulate an additive risk score from independent
rule hits ("flags"). No single weak signal pages alone, but corroborating
signals stack (see the loops §3.2 scoring note). Every finding keeps its
ground-truth label alongside, so eval is a one-liner over the merged findings
table.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from aiba.schema import Event


# --- small shared stats helpers (stdlib only; keeps the core numpy-free) ---

def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def pctl(xs: list[float], q: float) -> float:
    """Linear-interpolation percentile (q in [0,1])."""
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    frac = pos - lo
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * frac


def zscore(x: float, xs: list[float]) -> float:
    s = std(xs)
    return (x - mean(xs)) / s if s else 0.0


def wilson_lower(k: int, n: int, z: float = 1.645) -> float:
    """One-sided lower confidence bound on a binomial proportion k/n.

    Unlike a raw rate (or phat +/- z*SE(phat), the naive normal approximation),
    this doesn't blow up at small n or phat near 0/1: it shrinks toward 0 as n
    shrinks, so "1 failure out of 2 calls" reads as genuinely uncertain
    (~12% lower bound) instead of a face-value 50% rate. z=1.645 is a
    one-sided ~95% bound (adjust for a stricter/looser test). See
    tool_health.py for why this replaced a hand-tuned min-sample gate.
    """
    if n == 0:
        return 0.0
    phat = k / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    adj = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5)
    return max(0.0, (center - adj) / denom)


@dataclass
class Finding:
    """One ranked detection candidate.

    `spans` lets a detector flag a *contradiction pair* (e.g. the failed
    tool_result span + the claiming final_response span in action
    hallucination) rather than a single point. report.py uses trace_id /
    agent_id / ts to correlate findings across detectors into incident
    timelines (§4).
    """
    detector: str
    risk_score: float
    flags: list[str]                      # human-readable rule hits, e.g. "endpoint_drift new=..."
    ts: datetime
    agent_id: str
    trace_id: Optional[str] = None
    session_id: Optional[str] = None
    spans: list[str] = field(default_factory=list)   # span_ids of the evidence
    ground_truth: Optional[str] = None    # injected_scenario at the flagged span(s)


class Detector(ABC):
    """Baseline on a clean window, then score an eval window.

    The two-phase split is a day<20 baseline / day>=20 detection window.
    Baselining is separate from detection so the same fitted baseline can score
    streaming batches later.
    """

    name: str = "base"

    @abstractmethod
    def fit_baseline(self, events: list[Event]) -> None:
        """Learn per-agent/per-tool baselines from the clean window."""

    @abstractmethod
    def detect(self, events: list[Event]) -> list[Finding]:
        """Score the eval window; return findings sorted by risk_score desc."""


def is_attack(label: Optional[str]) -> bool:
    """Ground-truth labels prefixed control_ are deliberate benign lookalikes
    (§3.x controls). They are NOT attacks, and a finding on one is a false
    positive."""
    return bool(label) and not label.startswith("control_")


def evaluate(findings: list[Finding], events: list[Event]) -> dict:
    """Precision/recall of findings vs injected_scenario ground truth.

    Injected total, caught, false positives: computed uniformly for every
    detector. Recall is over distinct attack scenarios present in `events`; a
    scenario counts as caught if any finding carries its label.
    """
    attacks_present = {e.injected_scenario for e in events if is_attack(e.injected_scenario)}
    caught_labels = {f.ground_truth for f in findings if is_attack(f.ground_truth)}
    caught = attacks_present & caught_labels
    tp = [f for f in findings if is_attack(f.ground_truth)]
    fp = [f for f in findings if not is_attack(f.ground_truth)]  # benign or control_
    n_flagged = len(tp) + len(fp)
    return {
        "injected_total": len(attacks_present),
        "caught": len(caught),
        "missed": sorted(attacks_present - caught),
        "false_positives": len(fp),
        "precision": len(tp) / n_flagged if n_flagged else 1.0,
        "recall": len(caught) / len(attacks_present) if attacks_present else 1.0,
    }
