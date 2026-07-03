"""
agent-behavioral-analytics: detection over synthetic enterprise agent logs.

The shared funnel: synthesize data, learn a baseline, run rule/stat triggers,
rank the candidates, evaluate against ground truth.

Everything consumes one JSONL event stream (schema.py) and emits Finding
objects (detectors/base.py); report.py merges and correlates them.
See PLANNING.md for the full design.
"""
