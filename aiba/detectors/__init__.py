"""
Detectors. Each consumes the unified event stream (or agent cards) and emits
ranked Finding objects. The rule-based core is deterministic and offline;
only goal_drift and card_drift's rug-pull/shadowing-upgrade rules need the
[embed] extra. See PLANNING.md §3 for per-detector design.
"""
