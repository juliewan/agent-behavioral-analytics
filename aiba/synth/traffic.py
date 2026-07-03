"""
Benign multi-agent traffic generator (PLANNING.md §6 step 2).

This is the clean baseline window every detector fits on, and its statistical
properties are load-bearing. Each one exists so a specific detector's
thresholds hold (§3.2):

- Sessions finish in ~1-5s with a handful of spans and diverse
  (tool, args_hash) n-grams, so loops' duration/repetition thresholds hold.
- Occasional benign errors and retries keep the baseline error rate nonzero,
  so "suspiciously clean health" (0% errors) stays a meaningful loop signal.
- Baseline-normal A->B->A handoffs (scheduler and worker) give loops' cycle
  baseliner something legitimate to suppress.
- Routine memory ops, including RARE benign agent deletions and routine
  system_gc eviction of old cold memories, are what make M2's
  first-ever-deletion and initiator/temperature scoping earn their keep.
- Smooth goal-similarity sessions (goal_established plus on-topic
  planner_step text) give goal_drift's cliff test a flat baseline to
  contrast with.
- Per-agent tool vocabularies stay stable over time, for sequence's edge
  frequencies and card_drift's scope-violation rule.

The toy dataset must stay salient: ~2-5k events, every injected incident
findable by eye in the JSONL.

This module also owns the shared WORLD constants and the Gen event-builder,
which scenarios.py imports so injected events describe the same world.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from aiba.schema import AgentCard, AgentSkill, Event, MemoryOp

SEED = 42
START = datetime(2026, 6, 1)
N_DAYS = 30
BASELINE_CUTOFF_DAY = 20   # day < 20 = clean baseline window; day >= 20 = eval window

# --- the world -----------------------------------------------------------
# Skill names == tool names on purpose: the card's declared skills ARE the
# authorization boundary card_drift's scope-violation rule checks against.
AGENTS = {
    "agent-dataproc-01":  dict(name="DataProc-Agent",  tenant="acme-corp",
                               tools=["parse_csv", "transform_schema", "load_warehouse", "dedupe_rows"],
                               desc="Handles ETL of structured data into the warehouse."),
    "agent-triage-01":    dict(name="Triage-Agent",    tenant="acme-corp",
                               tools=["classify_alert", "enrich_context", "escalate_ticket"],
                               desc="Performs first-pass triage of security alerts."),
    "agent-scheduler-01": dict(name="Scheduler-Agent", tenant="acme-corp",
                               tools=["queue_task", "delegate_job", "report_status"],
                               desc="Coordinates task delegation across worker agents."),
    "agent-report-01":    dict(name="Reporting-Agent", tenant="acme-corp",
                               tools=["aggregate_metrics", "render_dashboard", "export_pdf"],
                               desc="Builds and distributes periodic reporting artifacts."),
    "agent-idp-01":       dict(name="Identity-Agent",  tenant="globex-inc",
                               tools=["verify_identity", "issue_token", "revoke_session"],
                               desc="Manages identity verification and session tokens."),
    # --- additional agents / tenants: cardinality so the benign feature cloud
    # is heterogeneous, not one tight ball (§5.1 / eval_isoforest). None ingest
    # untrusted content, so they add no taint edges to benign traffic.
    "agent-crm-01":       dict(name="CRM-Agent",       tenant="initech",
                               tools=["update_contact", "send_email", "sync_pipeline"],
                               desc="Keeps CRM records and outreach in sync."),
    "agent-finops-01":    dict(name="FinOps-Agent",    tenant="initech",
                               tools=["fetch_invoice", "reconcile_ledger", "post_journal"],
                               desc="Reconciles invoices and posts ledger entries."),
    "agent-support-01":   dict(name="Support-Agent",   tenant="acme-corp",
                               tools=["read_ticket", "draft_reply", "close_ticket"],
                               desc="Drafts first-line customer support replies."),
    "agent-mlops-01":     dict(name="MLOps-Agent",     tenant="globex-inc",
                               tools=["launch_training", "eval_model", "publish_artifact"],
                               desc="Runs training jobs and publishes model artifacts."),
    "agent-secops-01":    dict(name="SecOps-Agent",    tenant="acme-corp",
                               tools=["scan_host", "quarantine_asset", "open_case"],
                               desc="Sweeps hosts and opens security cases."),
}
ENDPOINTS = {aid: f"https://svc.internal.corp/agents/{aid.split('-')[1]}" for aid in AGENTS}

# per-agent principal + tenant; the cross-tenant M1 rule needs these paired
USERS = {
    "agent-dataproc-01": ("alice", "acme-corp"),
    "agent-triage-01":   ("bob", "acme-corp"),
    "agent-scheduler-01": ("carol", "acme-corp"),
    "agent-report-01":   ("dave", "acme-corp"),
    "agent-idp-01":      ("erin", "globex-inc"),
    "agent-crm-01":      ("frank", "initech"),
    "agent-finops-01":   ("grace", "initech"),
    "agent-support-01":  ("heidi", "acme-corp"),
    "agent-mlops-01":    ("ivan", "globex-inc"),
    "agent-secops-01":   ("judy", "acme-corp"),
}


def _tool_profile(tool: str) -> tuple[float, float, int, int]:
    """Deterministic but genuinely heterogeneous per-tool baseline: latency
    mean 80-420ms with a proportional (heavy-ish) std, varied token in/out.
    Derived from a stable hash so runs agree, but tools really differ; the
    point is that benign telemetry is not one tight gaussian ball (§5.1)."""
    h = int(hashlib.sha1(tool.encode()).hexdigest(), 16)
    mean = 80 + (h % 340)                          # 80..419 ms
    std = mean * (0.18 + (h >> 8) % 12 / 100.0)    # 18%..29% of mean
    tin = 90 + (h >> 16) % 320                      # 90..409
    tout = 40 + (h >> 24) % 200                     # 40..239
    return (float(mean), float(std), int(tin), int(tout))


# latency (ms) mean/std and per-step token cost: the benign baselines the
# tool_health / cost / loops z-scores are measured against.
TOOL_PROFILE = {t: _tool_profile(t) for a in AGENTS.values() for t in a["tools"]}
# two explicit profiles the worked examples reference
TOOL_PROFILE["load_warehouse"] = (300.0, 60.0, 220, 40)
TOOL_PROFILE["render_dashboard"] = (280.0, 50.0, 260, 90)


def _agent_profile(aid: str) -> dict:
    """Per-agent 'personality': benign heterogeneity ACROSS agents so the
    normal envelope is multi-modal (each agent its own region), which is what
    forces a learner to actually model 'normal' instead of a single ball."""
    h = int(hashlib.sha1(aid.encode()).hexdigest(), 16)
    return dict(
        err_rate=0.03 + (h % 8) / 100.0,             # 3%..10%
        steps=(3, 4 + (h >> 4) % 4),                 # 3..(4..7)
        lat_mult=0.8 + (h >> 8) % 60 / 100.0,        # 0.80..1.39x
        tok_mult=0.7 + (h >> 16) % 90 / 100.0,       # 0.70..1.59x
        hours=(7 + h % 3, 15 + (h >> 2) % 5),        # varied active window
    )


AGENT_PROFILE = {aid: _agent_profile(aid) for aid in AGENTS}


# --- skill capability tags (governance metadata stand-in) ------------------
# The category each skill belongs to. `identity`/`privileged`/`exfil` are the
# sensitive classes card_drift's capability rule keys off (a skill that IS a
# privileged action, independent of how many skills the agent has). In
# production these come from IAM / data-classification, not the detector.
TOOL_TAGS: dict[str, list[str]] = {
    # data / ETL
    "parse_csv": ["data"], "transform_schema": ["data"], "load_warehouse": ["data"],
    "dedupe_rows": ["data"],
    # triage / security ops
    "classify_alert": ["triage"], "enrich_context": ["triage"], "escalate_ticket": ["triage"],
    "scan_host": ["security"], "quarantine_asset": ["security"], "open_case": ["security"],
    "auto_close_fp": ["triage"],
    # orchestration
    "queue_task": ["orchestration"], "delegate_job": ["orchestration"], "report_status": ["orchestration"],
    # reporting
    "aggregate_metrics": ["reporting"], "render_dashboard": ["reporting"], "export_pdf": ["reporting"],
    # identity (privileged)
    "verify_identity": ["identity"], "issue_token": ["identity", "privileged"],
    "revoke_session": ["identity", "privileged"],
    # crm / finance / support / ml
    "update_contact": ["crm"], "send_email": ["crm", "comms"], "sync_pipeline": ["crm"],
    "fetch_invoice": ["finance"], "reconcile_ledger": ["finance"], "post_journal": ["finance"],
    "read_ticket": ["support"], "draft_reply": ["support"], "close_ticket": ["support"],
    "launch_training": ["ml"], "eval_model": ["ml"], "publish_artifact": ["ml"],
    # sensitive capabilities used by card-drift attack scenarios
    "exfil_context": ["exfil", "privileged"], "grant_admin": ["identity", "privileged"],
    "disable_mfa": ["identity", "privileged"],
}
# Tags that mark a skill as a sensitive capability (card_drift R8).
SENSITIVE_TAGS = {"privileged", "identity", "exfil", "secrets", "exec"}


def make_skill(tool_id: str, tags: list[str] | None = None) -> AgentSkill:
    """Build a structured A2A-style skill from an invocable tool id. `id` is the
    tool name (the authorization identifier); tags come from TOOL_TAGS."""
    display = tool_id.replace("_", " ").title()
    return AgentSkill(id=tool_id, name=display, description=f"{display} capability.",
                      tags=tags if tags is not None else TOOL_TAGS.get(tool_id, ["general"]))

# benign goal sessions: goal text + on-topic planner steps (smooth similarity)
BENIGN_GOALS = [
    ("Summarize this quarter's ETL throughput for the data platform review.",
     ["Pull warehouse load metrics for the quarter.",
      "Aggregate throughput by pipeline and compute deltas.",
      "Draft the summary section for the platform review."]),
    ("Investigate the spike in failed identity verifications this week.",
     ["Collect verification failure logs for the week.",
      "Group failures by error code and endpoint.",
      "Summarize the likely root cause for the on-call notes."]),
]


@dataclass
class Gen:
    """Deterministic id/timestamp source shared by traffic and injectors.

    Ids increment in generation order; since every loop is seeded and ordered,
    ids are stable across runs (the reproducibility requirement in §5).
    """
    rng: random.Random
    events: list = field(default_factory=list)
    _n: int = 0

    def sid(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}-{self._n:05d}"

    def hash(self, *parts) -> str:
        return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8]

    def emit(self, **kw) -> Event:
        ev = Event(**kw)
        self.events.append(ev)
        return ev


def ts_at(day: int, hour: int, sec: float = 0.0) -> datetime:
    return START + timedelta(days=day, hours=hour, seconds=sec)


def _step(g: Gen, *, trace, session, agent, tool, day, hour, t0,
          parent=None, status="ok", records=1, injected=None,
          lat_mult=1.0, tok_mult=1.0):
    """One tool step = tool_call span + tool_result span. Returns end offset.

    Latency is heavy-tailed (occasional benign slow call) and both latency and
    tokens are scaled by the caller's personality multipliers, so benign
    telemetry has real spread, not a single tight gaussian. tool_health / cost
    baselines absorb this (thresholds are z-scores over the widened spread), so
    the big attack spikes still stand out while benign variety no longer looks
    anomalous."""
    lat_mean, lat_std, tin, tout = TOOL_PROFILE[tool]
    lat = g.rng.gauss(lat_mean, lat_std)
    if g.rng.random() < 0.07:                      # heavy tail: benign slow call
        lat *= g.rng.uniform(1.7, 2.8)
    lat = min(lat, 3.0 * lat_mean)                 # hard cap: benign never nears
                                                   # tool_health's 4x-p99 bar (seed-robust)
    latency = max(5.0, lat * lat_mult)
    jitter = g.rng.uniform(0.6, 1.8)               # bursty per-call token size
    tin = max(1, int(tin * tok_mult * jitter))
    tout = max(1, int(tout * tok_mult * jitter))
    call = g.emit(trace_id=trace, span_id=g.sid("span"), parent_span_id=parent,
                  ts=ts_at(day, hour, t0), session_id=session, agent_id=agent,
                  event_type="tool_call", tool=tool,
                  args_hash=g.hash(tool, session, records),
                  tokens_in=tin, injected_scenario=injected)
    g.emit(trace_id=trace, span_id=g.sid("span"), parent_span_id=call.span_id,
           ts=ts_at(day, hour, t0 + latency / 1000.0), session_id=session, agent_id=agent,
           event_type="tool_result", tool=tool, status=status,
           latency_ms=round(latency, 1),
           records_returned=(0 if status in ("empty", "error", "timeout") else records),
           tokens_out=tout, injected_scenario=injected)
    return t0 + latency / 1000.0 + 0.05, call.span_id


def _benign_task_session(g: Gen, agent: str, day: int, hour: int):
    """A short, diverse, mostly-healthy task: the shape everything baselines on.
    Length / error-rate / latency / token size follow the agent's personality
    (AGENT_PROFILE), so different agents occupy different benign regions."""
    info = AGENTS[agent]
    prof = AGENT_PROFILE[agent]
    trace = g.sid("trace")
    session = g.sid("sess")
    pool = list(info["tools"])
    g.rng.shuffle(pool)
    n = g.rng.randint(prof["steps"][0], prof["steps"][1])
    # round-robin the shuffled vocab so a session can be longer than the tool
    # count without becoming a repetitive loop (diverse n-grams keep loops'
    # dominance well under its 0.5 bar)
    tools = [pool[i % len(pool)] for i in range(n)]
    tjit = g.rng.uniform(0.8, 1.3)   # session-level size scale
    t0 = 0.0
    for tool in tools:
        # benign error + immediate retry -> baseline error rate stays nonzero
        if g.rng.random() < prof["err_rate"]:
            t0, _ = _step(g, trace=trace, session=session, agent=agent, tool=tool,
                          day=day, hour=hour, t0=t0, status="error",
                          lat_mult=prof["lat_mult"], tok_mult=prof["tok_mult"] * tjit)
        t0, _ = _step(g, trace=trace, session=session, agent=agent, tool=tool,
                      day=day, hour=hour, t0=t0,
                      lat_mult=prof["lat_mult"], tok_mult=prof["tok_mult"] * tjit)


def _delegation_handoff(g: Gen, day: int, hour: int):
    """Baseline-normal A->B->A: scheduler delegates to a worker, gets status back.
    loops' cycle baseliner must learn this is business-as-usual and suppress it."""
    scheduler, worker = "agent-scheduler-01", "agent-dataproc-01"
    trace = g.sid("trace")
    session = g.sid("sess")
    t0, root = _step(g, trace=trace, session=session, agent=scheduler,
                     tool="delegate_job", day=day, hour=hour, t0=0.0)
    g.emit(trace_id=trace, span_id=g.sid("span"), parent_span_id=root,
           ts=ts_at(day, hour, t0), session_id=session, agent_id=scheduler,
           event_type="delegation", tool="delegate_job")
    for tool in ["parse_csv", "load_warehouse"]:
        t0, _ = _step(g, trace=trace, session=session, agent=worker, tool=tool,
                      day=day, hour=hour, t0=t0, parent=root)
    _step(g, trace=trace, session=session, agent=scheduler, tool="report_status",
          day=day, hour=hour, t0=t0, parent=root)


def _memory_activity(g: Gen, agent: str, day: int, hour: int):
    """Routine memory ops: create/read/update, owner==user, tenant matched."""
    user, tenant = USERS[agent]
    trace = g.sid("trace"); session = g.sid("sess")
    op = g.rng.choice(["create", "read", "read", "update"])
    mid = f"mem-{agent[6:12]}-{g.rng.randint(1, 40)}"
    g.emit(trace_id=trace, span_id=g.sid("span"), ts=ts_at(day, hour), session_id=session,
           agent_id=agent, user_id=user, tenant=tenant, event_type="memory_op",
           memory=MemoryOp(memory_id=mid, op=op, owner=user, tenant=tenant,
                           retriever=(user if op == "read" else None), initiator="agent",
                           ttl_days=30, audience="user",
                           content="user prefers dark mode dashboards"))


def _system_gc_sweep(g: Gen, day: int, hour: int):
    """Routine LRU/TTL eviction of old COLD memories (initiator=system_gc).
    CONTROL: M2 scopes deletion rules to initiator=agent, so this must NOT fire."""
    agent = "agent-report-01"; user, tenant = USERS[agent]
    trace = g.sid("trace"); session = g.sid("sess")
    for i in range(g.rng.randint(3, 6)):
        g.emit(trace_id=trace, span_id=g.sid("span"), ts=ts_at(day, hour, i),
               session_id=session, agent_id=agent, user_id=user, tenant=tenant,
               event_type="memory_op", injected_scenario="control_gc_sweep",
               memory=MemoryOp(memory_id=f"mem-cold-{day}-{i}", op="delete", owner=user,
                               tenant=tenant, initiator="system_gc", ttl_days=1,
                               audience="user"))


def _benign_agent_deletion(g: Gen, day: int, hour: int):
    """Reporting-Agent has a baseline history of benign agent-initiated deletes,
    so its eval-window deletion is a CONTROL (M2 first-ever must not fire)."""
    agent = "agent-report-01"; user, tenant = USERS[agent]
    trace = g.sid("trace"); session = g.sid("sess")
    label = "control_benign_delete" if day >= BASELINE_CUTOFF_DAY else None
    g.emit(trace_id=trace, span_id=g.sid("span"), ts=ts_at(day, hour), session_id=session,
           agent_id=agent, user_id=user, tenant=tenant, event_type="memory_op",
           injected_scenario=label,
           memory=MemoryOp(memory_id=f"mem-scratch-{day}", op="delete", owner=user,
                           tenant=tenant, initiator="agent", ttl_days=7, audience="user"))


def _goal_session(g: Gen, day: int, hour: int):
    """A goal-directed session with on-topic planner steps: smooth goal-drift
    baseline (goal_drift only embeds these free-text fields)."""
    agent = "agent-report-01"; user, tenant = USERS[agent]
    goal, steps = g.rng.choice(BENIGN_GOALS)
    trace = g.sid("trace"); session = g.sid("sess")
    g.emit(trace_id=trace, span_id=g.sid("span"), ts=ts_at(day, hour), session_id=session,
           agent_id=agent, user_id=user, tenant=tenant,
           event_type="goal_established", content=goal)
    t0 = 0.5
    for i, step in enumerate(steps):
        g.emit(trace_id=trace, span_id=g.sid("span"), ts=ts_at(day, hour, t0 + i),
               session_id=session, agent_id=agent, event_type="planner_step", content=step)
    g.emit(trace_id=trace, span_id=g.sid("span"), ts=ts_at(day, hour, t0 + len(steps)),
           session_id=session, agent_id=agent, event_type="final_response",
           content="Summary compiled and shared with the requested reviewers.")


def populate_benign(g: Gen, n_days: int = N_DAYS) -> None:
    """Fill a shared Gen with benign traffic. Injectors append to the same Gen
    afterwards so span/trace ids never collide."""
    for day in range(n_days):
        for agent in AGENTS:
            lo, hi = AGENT_PROFILE[agent]["hours"]
            for _ in range(g.rng.randint(1, 3)):
                _benign_task_session(g, agent, day, g.rng.randint(lo, hi))
        for _ in range(g.rng.randint(1, 2)):
            _delegation_handoff(g, day, g.rng.randint(8, 18))
        for agent in AGENTS:
            if g.rng.random() < 0.7:
                _memory_activity(g, agent, day, g.rng.randint(8, 18))
        if g.rng.random() < 0.5:
            _system_gc_sweep(g, day, g.rng.randint(0, 5))       # runs in every window
        if g.rng.random() < 0.4:
            _benign_agent_deletion(g, day, g.rng.randint(8, 18))
        if g.rng.random() < 0.6:
            _goal_session(g, day, g.rng.randint(8, 18))


def generate_benign_traffic(n_days: int = N_DAYS, seed: int = SEED) -> list[Event]:
    """Convenience wrapper: benign stream in a fresh Gen (baseline + benign
    portion of the eval window). run_all uses populate_benign directly so it
    can inject into the same Gen."""
    g = Gen(rng=random.Random(seed))
    populate_benign(g, n_days)
    return g.events


def generate_benign_cards(n_days: int = N_DAYS, seed: int = SEED) -> list[AgentCard]:
    """Benign card re-registration stream incl. rare legitimate skill drift."""
    rng = random.Random(seed + 1)
    benign_new = {"agent-dataproc-01": "dedupe_rows", "agent-triage-01": "auto_close_fp"}
    cards: list[AgentCard] = []
    for day in range(n_days):
        for aid, info in AGENTS.items():
            if rng.random() > 0.4:
                continue
            tool_ids = list(info["tools"])
            if day > 15 and rng.random() < 0.05 and aid in benign_new:
                if benign_new[aid] not in tool_ids:
                    tool_ids.append(benign_new[aid])
            cards.append(AgentCard(agent_id=aid, name=info["name"],
                                   skills=[make_skill(t) for t in tool_ids],
                                   description=info["desc"], endpoint=ENDPOINTS[aid],
                                   timestamp=ts_at(day, rng.randint(0, 23))))
    return cards
