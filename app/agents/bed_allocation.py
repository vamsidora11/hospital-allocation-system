"""Bed Allocation Agent (plan Section 7).

Pure function of (patient_information, retrieved_policies, resource_snapshot) -> AgentResult.
Never calls other agents or the graph; never retrieves policies itself (that's done upstream
and passed in as `retrieved_policies`).
"""

from typing import Literal

from pydantic import BaseModel

from app.agents.grounding import apply_grounding
from app.config import get_settings
from app.llm import get_chat_llm
from app.models.domain import AgentResult, PatientInfo, PolicyCitation

MAX_CITATIONS = 2


class BedAllocationLLMOutput(BaseModel):
    bed_type: Literal["icu", "general", "isolation"]
    reasoning: str
    confidence_score: float


_PROMPT = """You allocate hospital beds. Given patient severity, isolation needs, monitoring
needs, and current bed availability, recommend one of: icu, general, isolation. Cite only the
provided policy excerpts. If no excerpt supports a clear decision, say so and lower your
confidence.

Patient: {patient}
Policy excerpts: {policies}
Bed availability: {resource_snapshot}
"""


def _rule_based_decision(
    patient: PatientInfo, resource_snapshot: dict
) -> tuple[Literal["icu", "general", "isolation"], str, float]:
    if patient.severity == "critical" or patient.requires_monitoring_level == "icu":
        bed_type = "icu"
        reasoning = (
            "Critical acuity and/or ICU-level monitoring places this patient ahead of ward "
            "assignment (Bed Allocation Policy 3.2)."
        )
        if patient.requires_isolation:
            reasoning += " Isolation precautions apply within the ICU bed (Section 4.2)."
    elif patient.requires_isolation:
        bed_type = "isolation"
        reasoning = (
            "Isolation requirement overrides standard ward placement regardless of occupancy "
            "(Bed Allocation Policy 4.1)."
        )
    else:
        bed_type = "general"
        reasoning = (
            "No elevated monitoring or isolation signal; stable patients default to General "
            "Ward (Bed Allocation Policy 2.4)."
        )

    available = resource_snapshot.get(bed_type, {}).get("available", 0)
    if available > 0:
        base_confidence = 0.90
    else:
        base_confidence = 0.55
        reasoning += (
            f" No {bed_type} bed is currently available on this shift; escalate to the "
            "on-call bed manager per Section 5.2 rather than substituting a lower-acuity bed."
        )

    return bed_type, reasoning, base_confidence


def run_bed_allocation_agent(
    patient: PatientInfo,
    retrieved_policies: list[PolicyCitation],
    resource_snapshot: dict,
) -> AgentResult:
    settings = get_settings()
    llm = get_chat_llm() if not settings.use_rule_based_agents else None
    if llm is None:
        bed_type, reasoning, base_confidence = _rule_based_decision(patient, resource_snapshot)
    else:
        structured_llm = llm.with_structured_output(BedAllocationLLMOutput)
        result = structured_llm.invoke(
            _PROMPT.format(
                patient=patient.model_dump(),
                policies=[p.model_dump() for p in retrieved_policies],
                resource_snapshot=resource_snapshot,
            )
        )
        bed_type, reasoning, base_confidence = result.bed_type, result.reasoning, result.confidence_score

    citations = sorted(retrieved_policies, key=lambda c: c.similarity_score, reverse=True)[:MAX_CITATIONS]
    confidence, policy_grounded = apply_grounding(citations, base_confidence)

    return AgentResult(
        agent_name="bed_allocation",
        decision={"bed_type": bed_type},
        reasoning=reasoning,
        confidence_score=confidence,
        citations=citations,
        policy_grounded=policy_grounded,
    )
