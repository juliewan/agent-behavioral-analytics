"""
Memory: persistent memory treated as the database it is, with every
memory operation an audited write (PLANNING.md §3.3).

Three rule families, one detector, all consuming memory_op events.

M1, cross-user / cross-tenant retrieval. An invariant, not an anomaly: zero
baseline needed, zero FPs by construction, and the highest-impact detection
in the set.
  - memory.owner  != event.user_id on a read: flag.
  - memory.tenant != event.tenant  on a read: flag, higher severity.

M2, deletion anomalies. Every rule is scoped to initiator=agent, and the
scoping is the trick: system GC (LRU/TTL/FIFO) deletes old COLD memories as
routine hygiene, so an unscoped rule would page on garbage collection.
Agent-initiated deletion is a decision, and decisions are the signal.
  - First-ever deletion: zero agent-initiated deletes in baseline, then one.
    Reliable precisely because deletion's benign base rate is near zero.
  - Deletion burst after suspicious context (the anti-forensics signature):
    >=K agent deletions within N events of an error status, another
    detector's finding, or an M3 DLP hit. Chaining off M3 makes "write a
    secret, then erase history" causally explicit.
  - Old-and-hot deletion: memory age > baseline p95 AND a read within a
    recency window (computed from the stream, no new field). Age alone is
    FIFO/LRU bait; old-and-HOT means destroying a record someone still
    relied on.

M3, sensitive-data persistence (DLP-on-write): regexes over memory.content on
create/update. AWS key prefixes (AKIA...), PEM private-key headers, JWT
structure (eyJ...), SSN format, password=/api_key= assignments. Genuinely
high-precision patterns, not just toy-precision.
  - Severity comes from the regex class (credentials/keys > PII > generic).
    Sensitivity is a property of the content, observed at write time; no
    tool ontology required.
  - risk = severity x TTL x audience. An SSN stored forever and visible to
    all_agents pages immediately; a favorite color with a 30-day TTL scores
    about zero.

Build order note (§6): M2's burst rule consumes M3 hits, so M3 is
implemented first.
"""

from __future__ import annotations

import re
from collections import defaultdict

from aiba.detectors.base import Detector, Finding, pctl
from aiba.schema import Event

# regex class -> (pattern, severity) table; classes ARE the sensitivity source
# (no shared registry, §1). credentials/keys > PII > generic markers.
DLP_PATTERNS: dict[str, tuple[str, int]] = {
    "aws_key":      (r"AKIA[0-9A-Z]{16}", 3),
    "private_key":  (r"-----BEGIN [A-Z ]*PRIVATE KEY", 3),
    "jwt":          (r"eyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]+", 3),
    "ssn":          (r"\b\d{3}-\d{2}-\d{4}\b", 2),
    "secret_assign": (r"(?i)(password|api_key|secret)\s*=\s*\S+", 2),
}
_AUDIENCE_FACTOR = {"all_agents": 2.0, "workspace": 1.5, "user": 1.0}
BURST_K = 3          # >=K agent-initiated deletes...
RECENCY_DAYS = 3     # ...and "hot" = read within this many days before delete


def _dlp_hits(content: str) -> list[tuple[str, int]]:
    return [(cls, sev) for cls, (pat, sev) in DLP_PATTERNS.items() if re.search(pat, content)]


class MemoryDetector(Detector):
    name = "memory"

    def fit_baseline(self, events: list[Event]) -> None:
        # M1 and M3 are baseline-free invariants. M2 needs: which agents have
        # a benign agent-initiated deletion history, the memory-age p95, and
        # first-seen timestamps for age computation.
        self.agents_with_delete_history: set[str] = set()
        self.first_seen: dict[str, object] = {}
        ages = []
        life: dict[str, list] = defaultdict(list)
        for e in events:
            if e.event_type != "memory_op" or not e.memory:
                continue
            mid = e.memory.memory_id
            self.first_seen.setdefault(mid, e.ts)
            life[mid].append(e.ts)
            if e.memory.op == "delete" and e.memory.initiator == "agent":
                self.agents_with_delete_history.add(e.agent_id)
        for mid, tss in life.items():
            ages.append((max(tss) - min(tss)).days)
        self.age_p95 = max(pctl([float(a) for a in ages], 0.95), 7.0)

    def detect(self, events: list[Event]) -> list[Finding]:
        mem = [e for e in events if e.event_type == "memory_op" and e.memory]
        findings: list[Finding] = []

        # index reads per memory for the M2 old-and-hot "hot" test
        reads: dict[str, list] = defaultdict(list)
        for e in mem:
            if e.memory.op == "read":
                reads[e.memory.memory_id].append(e.ts)

        # sessions that carry a "suspicious context" for the M2 burst rule:
        # an error status OR an M3 DLP hit in the same session (report.py adds
        # cross-detector findings as further triggers, §4.2).
        m3_sessions, err_sessions = set(), set()
        for e in events:
            if e.event_type == "tool_result" and e.status in ("error", "timeout"):
                err_sessions.add(e.session_id)
        for e in mem:
            if e.memory.op in ("create", "update") and e.memory.content \
                    and _dlp_hits(e.memory.content):
                m3_sessions.add(e.session_id)

        # --- M3 DLP-on-write ------------------------------------------------
        for e in mem:
            m = e.memory
            if m.op in ("create", "update") and m.content:
                hits = _dlp_hits(m.content)
                if hits:
                    sev = max(s for _, s in hits)
                    ttl_factor = 2.0 if m.ttl_days is None else (1.8 if m.ttl_days >= 365 else 1.0)
                    risk = sev * ttl_factor * _AUDIENCE_FACTOR.get(m.audience, 1.0)
                    classes = ",".join(c for c, _ in hits)
                    findings.append(Finding(
                        detector=self.name, risk_score=risk,
                        flags=[f"M3 DLP {classes} in memory content; audience={m.audience} "
                               f"ttl={m.ttl_days} (severity {sev})"],
                        ts=e.ts, agent_id=e.agent_id, session_id=e.session_id,
                        spans=[e.span_id], ground_truth=e.injected_scenario))

        # --- M1 cross-user / cross-tenant retrieval (invariant) -------------
        for e in mem:
            m = e.memory
            if m.op != "read":
                continue
            flags, risk = [], 0.0
            if e.user_id and m.owner and m.owner != e.user_id:
                flags.append(f"M1 cross-user read: owner={m.owner} retriever={e.user_id}")
                risk += 3
            if e.tenant and m.tenant and m.tenant != e.tenant:
                flags.append(f"M1 cross-tenant read: mem tenant={m.tenant} caller tenant={e.tenant}")
                risk += 5
            if risk:
                findings.append(Finding(
                    detector=self.name, risk_score=risk, flags=flags, ts=e.ts,
                    agent_id=e.agent_id, session_id=e.session_id, spans=[e.span_id],
                    ground_truth=e.injected_scenario))

        # --- M2 deletion anomalies (scoped to initiator=agent) --------------
        by_session_deletes: dict[str, list[Event]] = defaultdict(list)
        for e in mem:
            m = e.memory
            if m.op != "delete" or m.initiator != "agent":
                continue  # system_gc eviction of cold memories is routine hygiene
            by_session_deletes[e.session_id].append(e)

            flags, risk = [], 0.0
            if e.agent_id not in self.agents_with_delete_history:
                flags.append("M2 first-ever agent deletion (no baseline deletion history)")
                risk += 3
            first = self.first_seen.get(m.memory_id, e.ts)
            age_days = (e.ts - first).days
            recent_read = any(0 <= (e.ts - r).days <= RECENCY_DAYS for r in reads.get(m.memory_id, []))
            if age_days > self.age_p95 and recent_read:
                flags.append(f"M2 old-and-hot deletion: age={age_days}d (> p95 {self.age_p95:.0f}d) "
                             f"and recently read")
                risk += 4
            if flags:
                findings.append(Finding(
                    detector=self.name, risk_score=risk, flags=flags, ts=e.ts,
                    agent_id=e.agent_id, session_id=e.session_id, spans=[e.span_id],
                    ground_truth=e.injected_scenario))

        # M2 burst: >=K agent deletes in a session with suspicious context.
        for sid, dels in by_session_deletes.items():
            if len(dels) >= BURST_K and (sid in m3_sessions or sid in err_sessions):
                trigger = "M3 DLP write" if sid in m3_sessions else "error status"
                e0 = dels[0]
                findings.append(Finding(
                    detector=self.name, risk_score=5,
                    flags=[f"M2 deletion burst: {len(dels)} agent deletions after {trigger} "
                           f"(anti-forensics)"],
                    ts=e0.ts, agent_id=e0.agent_id, session_id=sid,
                    spans=[d.span_id for d in dels], ground_truth=pick_gt(dels)))

        return sorted(findings, key=lambda f: f.risk_score, reverse=True)


def pick_gt(evs: list[Event]):
    labels = [e.injected_scenario for e in evs if e.injected_scenario]
    attacks = [l for l in labels if not l.startswith("control_")]
    return attacks[0] if attacks else (labels[0] if labels else None)
