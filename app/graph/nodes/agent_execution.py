"""AGENT_EXECUTION: fans out to 1-3 agent nodes in parallel via LangGraph's `Send`, retries
only this state (up to `MAX_AGENT_RETRIES`, exponential backoff), and falls back to a
conservative default + forces human review after exhausting retries (plan Section 5)."""

import time

from langgraph.types import Send

from app.agents.bed_allocation import run_bed_allocation_agent
from app.agents.emergency_routing import PRIORITY_BY_SEVERITY, run_emergency_routing_agent
from app.agents.staff_allocation import required_specialty, run_staff_allocation_agent
from app.config import get_settings
from app.graph.state import WorkflowState
from app.models.domain import AgentResult, PatientInfo
from app.rag.retriever import retrieve
from app.services import capacity
from app.services.audit import make_entry

_DOMAIN_BY_AGENT = {
    "bed_allocation": "bed",
    "staff_allocation": "staffing",
    "emergency_routing": "emergency",
}

_AGENT_FUNCS = {
    "bed_allocation": run_bed_allocation_agent,
    "staff_allocation": run_staff_allocation_agent,
    "emergency_routing": run_emergency_routing_agent,
}

_BACKOFF_SECONDS = (5, 10, 15)  # attempts 1, 2, 3 — sums to the plan's "~30s"


def _retrieval_query(patient: PatientInfo) -> str:
    return f"{patient.diagnosis_category} {patient.presenting_complaint} severity {patient.severity}"


def _resource_snapshot_for(agent_name: str, patient: PatientInfo) -> dict:
    if agent_name == "bed_allocation":
        return capacity.bed_snapshot()
    if agent_name == "staff_allocation":
        specialty = required_specialty(patient)
        return capacity.staff_snapshot(specialty=specialty, shift="day")
    # emergency_routing: no live department-load tracking table yet — conservative
    # defaults (see PROGRESS.md Phase 5 TODO).
    return {"department_load": "normal", "specialist_capability_available": True}


def _fallback_result(agent_name: str, patient: PatientInfo) -> AgentResult:
    if agent_name == "bed_allocation":
        decision = {"bed_type": "general"}
    elif agent_name == "staff_allocation":
        decision = {"assigned_role": "no reassignment — pending human review"}
    else:
        decision = {"routing_decision": "accept", "priority_level": PRIORITY_BY_SEVERITY[patient.severity]}

    return AgentResult(
        agent_name=agent_name,
        decision=decision,
        reasoning=(
            "Agent execution failed after repeated transient failures; falling back to a "
            "conservative default and forcing human review regardless of confidence."
        ),
        confidence_score=0.0,
        citations=[],
        policy_grounded=False,
    )


def dispatch_agents(state: WorkflowState) -> list[Send]:
    """Conditional edge: fans REQUEST_CLASSIFICATION's `active_agents` out to parallel
    `run_agent` invocations."""
    return [
        Send(
            "run_agent",
            {
                "agent_name": agent_name,
                "request_id": state["request_id"],
                "patient_information": state["patient_information"],
            },
        )
        for agent_name in state["active_agents"]
    ]


def run_agent(payload: dict) -> dict:
    agent_name = payload["agent_name"]
    patient: PatientInfo = payload["patient_information"]
    request_id = payload["request_id"]

    max_retries = get_settings().max_agent_retries
    domain = _DOMAIN_BY_AGENT[agent_name]
    agent_func = _AGENT_FUNCS[agent_name]

    attempts = 0
    last_error: Exception | None = None
    result: AgentResult | None = None

    while attempts < max_retries:
        try:
            policies = retrieve(_retrieval_query(patient), domain)
            resource_snapshot = _resource_snapshot_for(agent_name, patient)
            result = agent_func(patient, policies, resource_snapshot)
            last_error = None
            break
        except Exception as exc:  # noqa: BLE001 - genuinely any tool/LLM failure is retryable here
            last_error = exc
            attempts += 1
            if attempts < max_retries:
                time.sleep(_BACKOFF_SECONDS[min(attempts - 1, len(_BACKOFF_SECONDS) - 1)])

    forced_review = False
    if result is None:
        result = _fallback_result(agent_name, patient)
        forced_review = True

    entry = make_entry(
        request_id=request_id,
        state="AGENT_EXECUTION",
        input_snapshot={"agent_name": agent_name, "patient_ref": patient.patient_ref},
        output_snapshot={
            "decision": result.decision,
            "confidence_score": result.confidence_score,
            "attempts": attempts + 1,
            "fallback": forced_review,
            "error": str(last_error) if last_error else None,
        },
    )

    return {
        "agent_results": [result],
        "confidence_scores": {agent_name: result.confidence_score},
        "retry_count": attempts,
        "force_human_review": forced_review,
        "audit_trail": [entry],
    }
