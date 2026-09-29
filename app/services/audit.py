"""Audit entry construction and persistence.

Each graph node builds an in-memory `AuditEntry` and appends it to `WorkflowState.audit_trail`
during a run (Section 9's "one AuditEntry per state transition"); `persist_entries` writes the
full trail to the append-only `audit_log` table once a run completes, and `get_audit_trail`
reads it back for `GET /audit/{request_id}` (Phase 8/10).
"""

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models.domain import AuditEntry
from app.models.orm import AuditLog


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


def persist_entries(session: Session, entries: list[AuditEntry]) -> None:
    session.add_all(
        AuditLog(
            request_id=e.request_id,
            state=e.state,
            timestamp=e.timestamp,
            input_snapshot_json=e.input_snapshot,
            output_snapshot_json=e.output_snapshot,
            actor=e.actor,
        )
        for e in entries
    )
    session.commit()


def get_audit_trail(session: Session, request_id: str) -> list[AuditEntry]:
    rows = (
        session.query(AuditLog)
        .filter(AuditLog.request_id == request_id)
        .order_by(AuditLog.timestamp)
        .all()
    )
    return [
        AuditEntry(
            request_id=r.request_id,
            state=r.state,
            timestamp=r.timestamp,
            input_snapshot=r.input_snapshot_json,
            output_snapshot=r.output_snapshot_json,
            actor=r.actor,
        )
        for r in rows
    ]
