"""Audit entry construction. Persistence to the `audit_log` table lands in Phase 8 — for now
each graph node builds an in-memory `AuditEntry` and appends it to `WorkflowState.audit_trail`,
which already gives Section 9's "one AuditEntry per state transition" guarantee within a run.
"""

from datetime import datetime, timezone

from app.models.domain import AuditEntry


def make_entry(
    request_id: str,
    state: str,
    input_snapshot: dict,
    output_snapshot: dict,
    actor: str = "system",
) -> AuditEntry:
    return AuditEntry(
        request_id=request_id,
        state=state,
        timestamp=datetime.now(timezone.utc),
        input_snapshot=input_snapshot,
        output_snapshot=output_snapshot,
        actor=actor,
    )
