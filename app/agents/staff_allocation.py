"""Staff Allocation Agent (plan Section 7).

Pure function of (patient_information, retrieved_policies, resource_snapshot) -> AgentResult.
`resource_snapshot` here is the shape returned by app.services.capacity.staff_snapshot():
{"available_on_shift": int, "specialist_available": int}.
"""

from pydantic import BaseModel

from app.agents.grounding import apply_grounding
from app.config import get_settings
from app.llm import get_chat_llm
from app.models.domain import AgentResult, PatientInfo, PolicyCitation

MAX_CITATIONS = 2

_ROLE_BY_SPECIALTY = {
    "cardiac": "Cardiac-credentialed nurse",
    "respiratory_infectious": "Respiratory/infectious-care nurse",
    "icu": "ICU nurse (1:2 ratio)",
    "general": "General ward nurse",
}

_CARDIAC_KEYWORDS = ("cardiac", "acute_mi", "myocardial")


class StaffAllocationLLMOutput(BaseModel):
    assigned_role: str
    reasoning: str
    confidence_score: float


_PROMPT = """You assign staff. Given specialty needs, current shift staffing, and
staffing-ratio policy excerpts, recommend a staff assignment and validate it against ratio
rules. Cite only the provided policy excerpts.

Patient: {patient}
Policy excerpts: {policies}
Staff availability: {resource_snapshot}
"""


def required_specialty(patient: PatientInfo) -> str:
    text = f"{patient.diagnosis_category} {patient.presenting_complaint}".lower()
    if any(k in text for k in _CARDIAC_KEYWORDS):
        return "cardiac"
    if patient.requires_isolation:
        return "respiratory_infectious"
    if patient.severity == "critical" or patient.requires_monitoring_level == "icu":
        return "icu"
    return "general"


def _rule_based_decision(
    patient: PatientInfo, resource_snapshot: dict
) -> tuple[str, str, float]:
    specialty = required_specialty(patient)
    role = _ROLE_BY_SPECIALTY[specialty]

    if specialty == "cardiac":
        reasoning = (
            "Confirmed cardiac diagnosis is matched to a cardiac-credentialed nurse ahead of "
            "general staff (Staffing Policy 5.3)."
        )
    elif specialty == "respiratory_infectious":
        reasoning = (
            "Isolation-room patients require a nurse credentialed in respiratory/infectious "
            "care for the duration of stay (Infection Control Standards 2.2 / Staffing "
            "Policy 5.4)."
        )
    elif specialty == "icu":
        reasoning = "ICU assignment requires validating the 1:2 nurse-to-patient ratio (Staffing Policy 5.1)."
    else:
        reasoning = "No specialty or elevated-ratio signal; matched to general ward staffing."

    available_on_shift = resource_snapshot.get("available_on_shift", 0)
    specialist_available = resource_snapshot.get("specialist_available", 0)

    if specialty != "general" and specialist_available <= 0:
        base_confidence = 0.52
        reasoning += (
            " Requested credentialed coverage is not confirmed for the current shift; escalate "
            "to human review before assignment (Staffing Policy 6.0)."
        )
    elif available_on_shift <= 0:
        base_confidence = 0.52
        reasoning += " No staff are available on the current shift; escalate before assignment."
    else:
        base_confidence = 0.88
        reasoning += " Matched against the current shift roster; coverage is confirmed."

    return role, reasoning, base_confidence


def run_staff_allocation_agent(
    patient: PatientInfo,
    retrieved_policies: list[PolicyCitation],
    resource_snapshot: dict,
) -> AgentResult:
    settings = get_settings()
    llm = get_chat_llm() if not settings.use_rule_based_agents else None
    if llm is None:
        assigned_role, reasoning, base_confidence = _rule_based_decision(patient, resource_snapshot)
    else:
        structured_llm = llm.with_structured_output(StaffAllocationLLMOutput)
        result = structured_llm.invoke(
            _PROMPT.format(
                patient=patient.model_dump(),
                policies=[p.model_dump() for p in retrieved_policies],
                resource_snapshot=resource_snapshot,
            )
        )
        assigned_role, reasoning, base_confidence = (
            result.assigned_role,
            result.reasoning,
            result.confidence_score,
        )

    citations = sorted(retrieved_policies, key=lambda c: c.similarity_score, reverse=True)[:MAX_CITATIONS]
    confidence, policy_grounded = apply_grounding(citations, base_confidence)

    return AgentResult(
        agent_name="staff_allocation",
        decision={"assigned_role": assigned_role},
        reasoning=reasoning,
        confidence_score=confidence,
        citations=citations,
        policy_grounded=policy_grounded,
    )
