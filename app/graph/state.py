"""LangGraph-runtime state.

Same shape as app/models/domain.py's `WorkflowState` (the pinned domain-level contract — see
CLAUDE.md), but fields multiple parallel `AGENT_EXECUTION` branches write to need a merge
reducer so LangGraph can combine concurrent partial updates instead of overwriting each other.
"""

import operator
from typing import Annotated, TypedDict

from app.models.domain import AgentResult, AuditEntry, PatientInfo, PolicyCitation


def _merge_dicts(a: dict, b: dict) -> dict:
    """Reducer for concurrent partial updates to a dict field (parallel agent branches
    each set one key) — `operator.add` doesn't work on dicts."""
    return {**a, **b}


def _max_reducer(a: int, b: int) -> int:
    """`max` itself isn't introspectable by LangGraph's reducer signature check."""
    return max(a, b)


class WorkflowState(TypedDict):
    request_id: str
    patient_information: PatientInfo
    request_type: str | None
    workflow_status: str
    active_agents: list[str]
    agent_results: Annotated[list[AgentResult], operator.add]
    retrieved_policies: Annotated[list[PolicyCitation], operator.add]
    confidence_scores: Annotated[dict[str, float], _merge_dicts]
    approval_status: str | None
    final_recommendation: dict | None
    audit_trail: Annotated[list[AuditEntry], operator.add]
    retry_count: Annotated[int, _max_reducer]
    # Set by AGENT_EXECUTION when any agent exhausted its retries and fell back to a
    # conservative default — forces CONFIDENCE_EVALUATION to route to human review
    # regardless of the aggregate confidence score (plan Section 5). Not part of the
    # pinned domain-level WorkflowState (app/models/domain.py) since it's graph-internal
    # plumbing, not a field the API/DB layers need to know about.
    force_human_review: Annotated[bool, operator.or_]
    # Set by RESULT_CONSOLIDATION when two agents' decisions conflict (plan Section 8, e.g.
    # bed agent says ICU but staff agent can't confirm ICU nursing coverage) — forces human
    # review regardless of individual confidence scores. Single writer (consolidation runs
    # once, not in parallel), so no reducer needed.
    conflict_detected: bool
