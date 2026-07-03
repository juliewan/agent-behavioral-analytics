"""
Unified event schema, the backbone every detector consumes (PLANNING.md §1).

One JSONL trace/span stream, loosely modeled on OpenTelemetry GenAI semantic
conventions. Design commitments baked into the fields:

- `injected_scenario` is the ground-truth label so every detector can report
  precision/recall.
- `parent_span_id` chains give delegation depth for free (loops detector).
- `status="empty"` plus `records_returned` exist specifically so action
  hallucination can tell "errored" apart from "succeeded but found nothing".
- `args_hash` makes repetition cheap to detect without parsing args.
- `user_id`/`tenant` live at top level: the cross-user memory invariant (M1)
  needs them on the session, not just inside memory ops.
- No shared sensitivity registry. Sensitivity is derived where observed
  (DLP regex classes in memory, taint tags in sequence).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

TS_FMT = "%Y-%m-%dT%H:%M:%SZ"  # UTC, matches the OTel-flavored example in §1

# Enumerations kept as plain string constants: events round-trip through
# JSONL, and detectors match on values, not types.
EVENT_TYPES = (
    "llm_call",
    "tool_call",
    "tool_result",
    "delegation",
    "goal_established",   # carries original user goal text (embedded by goal_drift)
    "planner_step",       # carries planner step summary + tool justification text
    "memory_op",
    "final_response",     # carries claimed_actions for hallucination reconciliation
)

STATUSES = ("ok", "error", "timeout", "empty")

MEMORY_OPS = ("create", "read", "update", "delete")
MEMORY_INITIATORS = ("agent", "system_gc")   # M2 rules scope to initiator=agent
MEMORY_AUDIENCES = ("user", "workspace", "all_agents")


@dataclass
class MemoryOp:
    """Extra block carried by event_type=memory_op (PLANNING.md §1).

    Treated as an audited database write by memory: owner/tenant feed the
    M1 cross-user invariant, initiator scopes M2 deletion rules (system GC is
    routine hygiene, agent deletion is a decision), ttl_days/audience feed the
    M3 severity product.
    """
    memory_id: str
    op: str                       # MEMORY_OPS
    owner: str
    tenant: str
    retriever: Optional[str] = None       # who read it (M1: != owner -> flag)
    initiator: str = "agent"              # MEMORY_INITIATORS
    ttl_days: Optional[int] = None        # None = no TTL (scores worse in M3)
    audience: str = "user"                # MEMORY_AUDIENCES
    old_value_hash: Optional[str] = None
    new_value_hash: Optional[str] = None
    content: Optional[str] = None         # DLP regexes run over this on create/update


@dataclass
class Event:
    """One span in the unified trace stream (PLANNING.md §1)."""
    trace_id: str
    span_id: str
    ts: datetime
    session_id: str
    agent_id: str
    event_type: str                       # EVENT_TYPES
    parent_span_id: Optional[str] = None  # delegation depth via chain walk
    user_id: Optional[str] = None
    tenant: Optional[str] = None
    tool: Optional[str] = None
    args_hash: Optional[str] = None
    status: Optional[str] = None          # STATUSES
    latency_ms: Optional[float] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    records_returned: Optional[int] = None
    content: Optional[str] = None         # only free text; goal_drift embeds
                                          # goal/planner text ONLY, never CoT
    claimed_actions: list[str] = field(default_factory=list)
    memory: Optional[MemoryOp] = None     # present iff event_type == "memory_op"
    injected_scenario: Optional[str] = None  # ground-truth label, None = benign


@dataclass
class AgentSkill:
    """A2A-style agent skill: a structured capability unit, not a bare tool
    name. `id` is the invocable identifier (== the tool name in the event
    stream, so it doubles as the authorization boundary card_drift's scope rule
    checks). `tags` carry the capability category: governance metadata that
    lets card_drift reason over what a skill *is* (e.g. an `identity`/
    `privileged` tag) rather than only how many skills exist."""
    id: str
    name: str
    description: str
    tags: list[str] = field(default_factory=list)


@dataclass
class AgentCard:
    """A2A-style agent card registration (consumed by card_drift, §3.8)."""
    agent_id: str
    name: str
    skills: list[AgentSkill]
    description: str                      # embedded for rug-pull rule; names never are
    endpoint: str
    timestamp: datetime
    injected: Optional[str] = None


def _event_to_dict(e: Event) -> dict:
    d = asdict(e)                     # asdict recurses into the MemoryOp dataclass
    d["ts"] = e.ts.strftime(TS_FMT)
    # Drop None / empty fields so the committed JSONL stays readable by eye.
    d = {k: v for k, v in d.items() if v not in (None, [], {})}
    return d


def _event_from_dict(d: dict) -> Event:
    d = dict(d)
    d["ts"] = datetime.strptime(d["ts"], TS_FMT)
    mem = d.pop("memory", None)
    ev = Event(**d)
    if mem is not None:
        ev.memory = MemoryOp(**mem)
    return ev


def dump_events(events: list[Event], path: str) -> None:
    """Write events as JSONL (the committed data/sample/events.jsonl format)."""
    with open(path, "w") as fh:
        for e in events:
            fh.write(json.dumps(_event_to_dict(e), default=str) + "\n")


def load_events(path: str) -> list[Event]:
    """Read a JSONL event stream into Event objects."""
    with open(path) as fh:
        return [_event_from_dict(json.loads(line)) for line in fh if line.strip()]


def _card_to_dict(c: AgentCard) -> dict:
    d = asdict(c)
    d["timestamp"] = c.timestamp.strftime(TS_FMT)
    return {k: v for k, v in d.items() if v is not None}


def _card_from_dict(d: dict) -> AgentCard:
    d = dict(d)
    d["timestamp"] = datetime.strptime(d["timestamp"], TS_FMT)
    d["skills"] = [AgentSkill(**s) for s in d.get("skills", [])]
    return AgentCard(**d)


def dump_cards(cards: list[AgentCard], path: str) -> None:
    with open(path, "w") as fh:
        for c in cards:
            fh.write(json.dumps(_card_to_dict(c), default=str) + "\n")


def load_cards(path: str) -> list[AgentCard]:
    with open(path) as fh:
        return [_card_from_dict(json.loads(line)) for line in fh if line.strip()]
