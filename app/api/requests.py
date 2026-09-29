"""POST /requests, GET /requests/{id}, GET /requests/{id}/recommendation,
POST /requests/{id}/approval (plan Section 10).

Live status/recommendation reads come from the compiled graph's checkpointer state
(`_GRAPH.get_state`), not the DB — `agent_results`/`policy_citations`/`audit_log`/`approvals`
are still persisted to the DB for durability, but re-querying live in-progress state from a
`MemorySaver` only works within this process (see CLAUDE.md's checkpointer TODO — a
Postgres-backed checkpointer would remove this limitation).
"""

from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from langgraph.types import Command
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.auth import verify_api_key
from app.db import get_session
from app.graph.build_graph import build_graph, initial_state
from app.models.domain import ApprovalRecord, PatientInfo
from app.services.audit import persist_entries
from app.services.persistence import persist_agent_results, persist_approval, update_request_status
from app.services.requests import create_request

router = APIRouter(prefix="/requests", tags=["requests"], dependencies=[Depends(verify_api_key)])

# Module-level singleton so a paused thread_id (== request_id) can be resumed by a later,
# separate HTTP request within this process's lifetime.
_GRAPH = build_graph()


class ApprovalActionRequest(BaseModel):
    action: Literal["approve", "reject", "override", "request_info"]
    reviewer_id: str
    notes: str | None = None
    override_decision: dict | None = None


class RequestStatusResponse(BaseModel):
    request_id: str
    workflow_status: str
    approval_status: str | None = None
    request_type: str | None = None


class RecommendationResponse(BaseModel):
    workflow_status: str
    final_recommendation: dict | None = None
    citations: list[dict] = []


def _config(request_id: str) -> dict:
    return {"configurable": {"thread_id": request_id}}


def _status_response(request_id: str, values: dict) -> RequestStatusResponse:
    return RequestStatusResponse(
        request_id=request_id,
        workflow_status=values.get("workflow_status", "unknown"),
        approval_status=values.get("approval_status"),
        request_type=values.get("request_type"),
    )


def _persist_run(session: Session, request_id: str, values: dict) -> None:
    if values.get("audit_trail"):
        persist_entries(session, values["audit_trail"])
    if values.get("agent_results"):
        persist_agent_results(session, request_id, values["agent_results"])
    status = values.get("workflow_status", "running")
    update_request_status(session, request_id, status, closed=(status == "decided"))


def _get_state_or_404(request_id: str):
    snapshot = _GRAPH.get_state(_config(request_id))
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="Unknown request_id")
    return snapshot


@router.post("", response_model=RequestStatusResponse)
def create_and_run_request(patient: PatientInfo, session: Session = Depends(get_session)) -> RequestStatusResponse:
    request_row = create_request(session, patient)
    request_id = request_row.id

    result = _GRAPH.invoke(initial_state(request_id, patient), config=_config(request_id))
    _persist_run(session, request_id, result)

    return _status_response(request_id, result)


@router.get("/{request_id}", response_model=RequestStatusResponse)
def get_request_status(request_id: str) -> RequestStatusResponse:
    snapshot = _get_state_or_404(request_id)
    return _status_response(request_id, snapshot.values)


@router.get("/{request_id}/recommendation", response_model=RecommendationResponse)
def get_recommendation(request_id: str) -> RecommendationResponse:
    values = _get_state_or_404(request_id).values
    citations = [
        citation.model_dump()
        for result in values.get("agent_results", [])
        for citation in result.citations
    ]
    return RecommendationResponse(
        workflow_status=values.get("workflow_status", "unknown"),
        final_recommendation=values.get("final_recommendation"),
        citations=citations,
    )


@router.post("/{request_id}/approval", response_model=RequestStatusResponse)
def submit_approval(
    request_id: str, payload: ApprovalActionRequest, session: Session = Depends(get_session)
) -> RequestStatusResponse:
    snapshot = _get_state_or_404(request_id)
    if not snapshot.next:
        raise HTTPException(status_code=409, detail="Request is not awaiting approval")

    approval_record = ApprovalRecord(
        reviewer_id=payload.reviewer_id,
        action=payload.action,
        notes=payload.notes,
        decided_at=datetime.now(timezone.utc),
    )
    persist_approval(session, request_id, approval_record)

    resume_payload = {"action": payload.action, "reviewer_id": payload.reviewer_id, "notes": payload.notes}
    if payload.override_decision:
        resume_payload["override_decision"] = payload.override_decision

    result = _GRAPH.invoke(Command(resume=resume_payload), config=_config(request_id))
    _persist_run(session, request_id, result)

    return _status_response(request_id, result)
