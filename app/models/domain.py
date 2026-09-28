from datetime import datetime
from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class PatientInfo(BaseModel):
    """Synthetic operational patient signal — never a real MRN/SSN/insurance ID.

    See CLAUDE.md "Data-handling constraint": this is intentionally narrower than a
    full chart. Only enough clinical signal to route beds/staff.
    """

    patient_ref: str = Field(..., description='Synthetic id, e.g. "PT-2026-00931"')
    age: int = Field(..., ge=0, le=120)
    presenting_complaint: str
    diagnosis_category: str
    severity: Literal["critical", "high", "moderate", "minor"]
    requires_isolation: bool
    requires_monitoring_level: Literal["none", "standard", "icu"]
    arrival_mode: Literal["walk_in", "ambulance", "transfer", "mass_casualty"]


class RouterOutput(BaseModel):
    request_type: Literal["bed_allocation", "staff_allocation", "emergency_routing", "combination"]
    selected_agents: list[str]
    routing_reasoning: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)


class PolicyCitation(BaseModel):
    citation_id: str
    policy_name: str
    section: str
    excerpt: str = Field(..., description="Short, paraphrased excerpt — never the full source paragraph")
    similarity_score: float = Field(..., ge=0.0, le=1.0)


class AgentResult(BaseModel):
    agent_name: str
    decision: dict
    reasoning: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    citations: list[PolicyCitation]
    policy_grounded: bool


class ApprovalRecord(BaseModel):
    reviewer_id: str
    action: Literal["approve", "reject", "override", "request_info"]
    notes: str | None = None
    decided_at: datetime


class AuditEntry(BaseModel):
    request_id: str
    state: str
    timestamp: datetime
    input_snapshot: dict
    output_snapshot: dict
    actor: Literal["system", "human"]


class WorkflowState(TypedDict):
    request_id: str
    patient_information: PatientInfo
    request_type: str | None
    workflow_status: str
    active_agents: list[str]
    agent_results: list[AgentResult]
    retrieved_policies: list[PolicyCitation]
    confidence_scores: dict[str, float]
    approval_status: str | None
    final_recommendation: dict | None
    audit_trail: list[AuditEntry]
    retry_count: int
