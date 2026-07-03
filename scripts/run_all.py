"""
One command: generate, detect, report (PLANNING.md §2).

    python scripts/run_all.py [--out data/sample/] [--seed 42]

Pipeline:
  1. synth: benign traffic + benign cards (traffic.py), then every scenario
     injector incl. controls and both correlated chains (scenarios.py)
  2. persist events.jsonl / agent_cards.jsonl (the committed toy dataset)
  3. fit each detector on the clean baseline window, detect on the eval window
  4. embedding-gated detectors (goal_drift, card_drift R6/R7) run only if the
     [embed] extra or a precomputed embeddings.npz is available; the rule-
     based core must complete without either
  5. report: merged ranking, per-detector precision/recall, incident timelines

Deterministic end to end: seeded RNG, no network, no API keys.
"""

from __future__ import annotations

import argparse
import os
import random

from aiba import diagnostics, report
from aiba.detectors.action_hallucination import ActionHallucinationDetector
from aiba.detectors.card_drift import CardDriftDetector
from aiba.detectors.cost import CostDetector
from aiba.detectors.goal_drift import GoalDriftDetector
from aiba.detectors.loops import LoopDetector
from aiba.detectors.memory import MemoryDetector
from aiba.detectors.sequence import SequenceDetector
from aiba.detectors.tool_health import ToolHealthDetector
from aiba.embed import embed_texts, save_npz, using_real_model
from aiba.schema import dump_cards, dump_events
from aiba.synth import scenarios
from aiba.synth.traffic import (
    BASELINE_CUTOFF_DAY, SEED, START, Gen, generate_benign_cards, populate_benign,
)


def _day(ts) -> int:
    return (ts - START).days


def build_dataset(seed: int = SEED):
    """Benign traffic + every scenario injector (incl. controls and the two
    correlated chains), sharing one Gen so ids never collide."""
    g = Gen(rng=random.Random(seed))
    populate_benign(g)
    for inject in scenarios.EVENT_INJECTORS:
        inject(g)

    cards = generate_benign_cards()
    for inject in scenarios.CARD_INJECTORS:
        inject(cards)
    scenarios.inject_endpoint_hijack_chain(g, cards)   # needs both streams

    events = sorted(g.events, key=lambda e: e.ts)
    cards = sorted(cards, key=lambda c: c.timestamp)
    return events, cards


def save_embeddings(events, cards, path: str) -> None:
    """Precompute the committed embeddings.npz (goal/planner/final + card
    descriptions) so cloners reproduce the eval without the model download."""
    texts = {e.content for e in events
             if e.event_type in ("goal_established", "planner_step", "final_response") and e.content}
    texts |= {c.description for c in cards}
    texts = sorted(texts)
    save_npz(path, texts, embed_texts(texts))


def main() -> None:
    ap = argparse.ArgumentParser(description="generate -> detect -> report")
    ap.add_argument("--out", default="data/sample", help="output dir for the toy dataset")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-embed-detectors", action="store_true",
                    help="skip goal_drift and card_drift's embedding rules")
    ap.add_argument("--png", nargs="?", const="data/sample/diagnostics.png",
                    help="also write the dashboard PNG (needs the [viz] extra; "
                         "default path data/sample/diagnostics.png)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    events, cards = build_dataset(args.seed)
    dump_events(events, os.path.join(args.out, "events.jsonl"))
    dump_cards(cards, os.path.join(args.out, "agent_cards.jsonl"))
    print(f"Generated {len(events)} events, {len(cards)} cards -> {args.out}/")
    print(f"Embeddings: {'real bge model' if using_real_model() else 'deterministic offline fallback'}")

    baseline = [e for e in events if _day(e.ts) < BASELINE_CUTOFF_DAY]
    eval_win = [e for e in events if _day(e.ts) >= BASELINE_CUTOFF_DAY]

    use_embed = not args.no_embed_detectors
    detectors = [
        ActionHallucinationDetector(), LoopDetector(), MemoryDetector(),
        SequenceDetector(), ToolHealthDetector(), CostDetector(),
        CardDriftDetector(cards, use_embed=use_embed),
    ]
    if use_embed:
        detectors.append(GoalDriftDetector())

    all_findings = {}
    for det in detectors:
        det.fit_baseline(baseline)
        all_findings[det.name] = det.detect(eval_win)

    report.print_report(eval_win, all_findings)

    # graphical diagnostics: terminal by default (dev loop), PNG on request
    diagnostics.print_terminal(eval_win, all_findings, cards)
    if args.png is not None:
        try:
            path = diagnostics.write_report(eval_win, all_findings, args.png, cards)
            print(f"\nWrote diagnostics dashboard -> {path}")
        except ImportError:
            print("\n(skipped --png: matplotlib missing; install the [viz] extra)")

    try:
        save_embeddings(events, cards, os.path.join(args.out, "embeddings.npz"))
    except Exception as exc:   # numpy optional; the rule-based run already finished
        print(f"\n(skipped embeddings.npz: {exc})")


if __name__ == "__main__":
    main()
