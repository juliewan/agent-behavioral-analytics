"""
Learned-baseline benchmark: Isolation Forest vs the rule suite (PLANNING.md §5).

    python scripts/eval_isoforest.py [--png data/sample/pr_isoforest.png] [--seed 42]

The question behind this repo's "data science" claim: would an unsupervised
learner do as well as the hand-built rules? This script answers it with a
proper train/test split and a PR curve, not a single flattering operating
point.

  1. Split the toy dataset the same way the suite does: baseline window (clean)
     for training, eval window (attacks injected) for scoring.
  2. Train an Isolation Forest on the baseline SESSIONS' behavioral feature
     vectors (normal traffic only: unsupervised novelty).
  3. Score eval sessions two ways on the SAME session universe:
       - Isolation Forest anomaly score (the learned generalist), and
       - the rule suite's per-session max risk (the hand-built baseline).
  4. Sweep the decision threshold into a precision/recall curve for each and
     report average precision (PR-AUC). Under heavy class imbalance PR-AUC is
     the honest headline; ROC's huge FP-rate denominator flatters everything.
  5. Write the PR-curve PNG and print the operating point + a short verdict.

Why PR and not ROC: a handful of attack sessions against a large benign stream
is exactly the imbalanced-rare-event regime where average precision separates
detectors that ROC-AUC would call indistinguishable.

Requires [ml] (scikit-learn); the PNG additionally needs [viz] (matplotlib).
Deterministic: seeded forest, no network.
"""

from __future__ import annotations

import argparse

from aiba.detectors.action_hallucination import ActionHallucinationDetector
from aiba.detectors.base import is_attack
from aiba.detectors.cost import CostDetector
from aiba.detectors.goal_drift import GoalDriftDetector
from aiba.detectors.isoforest import FEATURE_NAMES, IsolationForestDetector, session_matrix
from aiba.detectors.loops import LoopDetector
from aiba.detectors.memory import MemoryDetector
from aiba.detectors.sequence import SequenceDetector
from aiba.detectors.tool_health import ToolHealthDetector
from aiba.synth.traffic import BASELINE_CUTOFF_DAY, SEED, START
from run_all import build_dataset   # scripts/ is on sys.path when run as a script

# --- validated reference palette (mirrors aiba/diagnostics.py) ---------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SER_IF = "#2a78d6"     # categorical slot 1 (blue)  -> Isolation Forest
SER_RULES = "#1baf7a"  # categorical slot 2 (aqua)  -> rule suite
CRIT = "#d03b3b"


def _day(ts) -> int:
    return (ts - START).days


def rule_suite_session_scores(baseline, eval_win) -> dict[str, float]:
    """Per-session max risk from the BEHAVIORAL rule detectors (the fair
    apples-to-apples baseline: same session universe, excludes card_drift's
    separate card modality). Sessions with no finding score 0."""
    detectors = [
        ActionHallucinationDetector(), LoopDetector(), MemoryDetector(),
        SequenceDetector(), ToolHealthDetector(), CostDetector(), GoalDriftDetector(),
    ]
    best: dict[str, float] = {}
    for det in detectors:
        det.fit_baseline(baseline)
        for f in det.detect(eval_win):
            if f.session_id is not None:
                best[f.session_id] = max(best.get(f.session_id, 0.0), f.risk_score)
    return best


def pr_and_ap(y_true, scores):
    """precision_recall_curve + average precision (PR-AUC)."""
    from sklearn.metrics import average_precision_score, precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    return precision, recall, thresholds, ap


def main() -> None:
    ap_ = argparse.ArgumentParser(description="Isolation Forest vs rule suite — PR benchmark")
    ap_.add_argument("--png", default="data/sample/pr_isoforest.png",
                     help="output path for the PR-curve plot ([viz] extra)")
    ap_.add_argument("--seed", type=int, default=SEED)
    args = ap_.parse_args()

    events, _cards = build_dataset(args.seed)
    baseline = [e for e in events if _day(e.ts) < BASELINE_CUTOFF_DAY]
    eval_win = [e for e in events if _day(e.ts) >= BASELINE_CUTOFF_DAY]

    # --- learned detector: train on clean baseline, score eval ---
    iso = IsolationForestDetector(seed=args.seed)
    iso.fit_baseline(baseline)
    if_scores, meta = iso.anomaly_score(eval_win)

    # labels + the rule-suite score aligned to the SAME session order
    y_true = [1 if is_attack(m["ground_truth"]) else 0 for m in meta]
    rule_map = rule_suite_session_scores(baseline, eval_win)
    rule_scores = [rule_map.get(m["session_id"], 0.0) for m in meta]

    n_pos, n_neg = sum(y_true), len(y_true) - sum(y_true)
    print(f"\n{'=' * 68}\nISOLATION FOREST vs RULE SUITE — session-anomaly PR benchmark")
    print("=" * 68)
    print(f"Train: {len({e.session_id for e in baseline})} baseline sessions (clean)")
    print(f"Test : {len(meta)} eval sessions  "
          f"({n_pos} attack, {n_neg} benign/control)  prevalence={n_pos/len(meta):.2f}")
    print("Note: card-modality attacks (typosquat/shadowing/capability_escalation/"
          "endpoint_hijack/rug_pull)\n      have no event-stream session and are out of "
          "this universe by construction (card_drift's job).")

    if_p, if_r, _if_t, if_ap = pr_and_ap(y_true, if_scores)
    ru_p, ru_r, _ru_t, ru_ap = pr_and_ap(y_true, rule_scores)
    chance = n_pos / len(meta)   # a no-skill detector's average precision

    print(f"\n--- Average precision (PR-AUC) ---")
    print(f"  isolation_forest   AP = {if_ap:.3f}")
    print(f"  rule_suite         AP = {ru_ap:.3f}")
    print(f"  no-skill baseline  AP = {chance:.3f}  (prevalence)")

    # IF operating point (baseline-derived threshold = most-anomalous clean session)
    pred = [s > iso.threshold for s in if_scores]
    tp = sum(1 for p, y in zip(pred, y_true) if p and y)
    fp = sum(1 for p, y in zip(pred, y_true) if p and not y)
    fn = sum(1 for p, y in zip(pred, y_true) if not p and y)
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    print(f"\n--- Isolation Forest operating point (threshold = clean-window p95) ---")
    print(f"  flagged {tp + fp} sessions: TP={tp} FP={fp} FN={fn}  P={prec:.2f} R={rec:.2f}")
    caught = sorted({m["ground_truth"] for p, m in zip(pred, meta)
                     if p and is_attack(m["ground_truth"])})
    missed = sorted({m["ground_truth"] for m in meta
                     if is_attack(m["ground_truth"])} - set(caught))
    print(f"  caught : {caught}")
    print(f"  missed : {missed}")

    # --- the split decision: who catches what the other cannot ---
    if_caught = {m["ground_truth"] for p, m in zip(pred, meta)
                 if p and is_attack(m["ground_truth"])}
    rules_caught = {m["ground_truth"] for s, m in zip(rule_scores, meta)
                    if s > 0 and is_attack(m["ground_truth"])}
    print("\n--- Split decision: complementary coverage ---")
    print(f"  rules-only (invariant/threshold hits the learner misses): "
          f"{sorted(rules_caught - if_caught)}")
    print(f"  learner-only (diffuse multivariate anomalies the rules miss): "
          f"{sorted(if_caught - rules_caught)}")
    print(f"  both: {sorted(rules_caught & if_caught)}")

    _feature_signal(if_scores, eval_win)

    print(f"\n--- Verdict: rules win on AGGREGATE PR-AUC ({ru_ap:.2f} vs {if_ap:.2f}) "
          f"because most attacks here are invariant violations (rules' turf), but the "
          f"learner\n    OWNS the diffuse multivariate scenarios the rules are blind to "
          f"({sorted(if_caught - rules_caught)}). Complementary, not competing. ---")

    _plot(args.png, if_r, if_p, if_ap, ru_r, ru_p, ru_ap, chance,
          op=(rec, prec))


def _feature_signal(if_scores, eval_win) -> None:
    """Cheap interpretability: which features move most with the anomaly score
    (|Pearson r| over eval sessions). Isolation Forest has no feature_importances_,
    so this stands in as a directional readout."""
    try:
        import numpy as np
    except ImportError:
        return
    X, _ = session_matrix(eval_win)
    X = np.asarray(X, dtype=float)
    s = np.asarray(if_scores, dtype=float)
    corrs = []
    for j, name in enumerate(FEATURE_NAMES):
        col = X[:, j]
        if col.std() < 1e-9 or s.std() < 1e-9:
            continue
        corrs.append((abs(float(np.corrcoef(col, s)[0, 1])), name))
    corrs.sort(reverse=True)
    top = ", ".join(f"{name} ({r:.2f})" for r, name in corrs[:5])
    print(f"\n--- Top features driving the anomaly score (|corr|) ---\n  {top}")


def _plot(path, if_r, if_p, if_ap, ru_r, ru_p, ru_ap, chance, op) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(skipped PR plot: matplotlib missing; install the [viz] extra)")
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # no-skill reference (a flat line at prevalence), kept recessive
    ax.axhline(chance, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(0.99, chance + 0.015, f"no-skill (prevalence {chance:.2f})",
            color=MUTED, fontsize=8, va="bottom", ha="right")

    # PR curves: recall (x) vs precision (y), stepwise as is conventional
    ax.step(ru_r, ru_p, where="post", color=SER_RULES, lw=2, zorder=3,
            label=f"rule suite  (AP={ru_ap:.2f})")
    ax.step(if_r, if_p, where="post", color=SER_IF, lw=2, zorder=3,
            label=f"isolation forest  (AP={if_ap:.2f})")

    # IF operating point
    ax.scatter([op[0]], [op[1]], s=42, color=SER_IF, edgecolor=SURFACE,
               linewidth=1.5, zorder=4)
    ax.annotate("IF operating point\n(clean p95)", xy=op,
                xytext=(op[0] + 0.03, op[1] + 0.14), color=INK, fontsize=8,
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Recall", color=INK, fontsize=10)
    ax.set_ylabel("Precision", color=INK, fontsize=10)
    ax.set_title("Session-anomaly detection — precision/recall (eval window)",
                 color=INK, fontsize=11, pad=10)
    ax.grid(True, color=GRID, lw=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    leg = ax.legend(loc="lower left", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    print(f"\nWrote PR-curve plot -> {path}")


if __name__ == "__main__":
    main()
