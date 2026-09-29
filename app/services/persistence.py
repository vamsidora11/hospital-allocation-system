"""Persists a completed/paused run's agent results + citations (plan Section 4:
`agent_results`, `policy_citations`) and reviewer actions (`approvals`) to the DB. Audit
entries and the `requests`/`patients` rows have their own helpers in app/services/audit.py
and app/services/requests.py respectively.
"""

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models.domain import AgentResult, ApprovalRecord
from app.models.orm import AgentResultRow, Approval, PolicyCitationRow, Request


def persist_agent_results(session: Session, request_id: str, agent_results: list[AgentResult]) -> None:
    for result in agent_results:
        row = AgentResultRow(
            request_id=request_id,
            agent_name=result.agent_name,
            decision_json=result.decision,
            confidence_score=result.confidence_score,
            policy_grounded=result.policy_grounded,
        )
        session.add(row)
        session.flush()  # populate row.id for the citation FK below
        session.add_all(
            PolicyCitationRow(
                agent_result_id=row.id,
                policy_name=c.policy_name,
                section=c.section,
                excerpt=c.excerpt,
                similarity_score=c.similarity_score,
            )
            for c in result.citations
        )
    session.commit()


def persist_approval(session: Session, request_id: str, approval: ApprovalRecord) -> None:
    session.add(
        Approval(
            request_id=request_id,
            reviewer_id=approval.reviewer_id,
            action=approval.action,
            notes=approval.notes,
            decided_at=approval.decided_at,
        )
    )
    session.commit()


def update_request_status(session: Session, request_id: str, status: str, closed: bool = False) -> None:
    request_row = session.get(Request, request_id)
    if request_row is None:
        return
    request_row.status = status
    if closed:
        request_row.closed_at = datetime.now(timezone.utc)
    session.commit()
