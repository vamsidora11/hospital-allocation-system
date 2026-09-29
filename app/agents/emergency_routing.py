"""Emergency Routing Agent (plan Section 7).

Pure function of (patient_information, retrieved_policies, resource_snapshot) -> AgentResult.
`resource_snapshot` shape here: {"department_load": "normal"|"constrained"|"limited",
"specialist_capability_available": bool} (defaults applied when keys are absent).
"""

from typing import Literal

from pydantic import BaseModel

from app.agents.grounding import apply_grounding
from app.llm import get_chat_llm
from app.models.domain import AgentResult, PatientInfo, PolicyCitation

MAX_CITATIONS = 2

PRIORITY_BY_SEVERITY = {"critical": 1, "high": 2, "moderate": 3, "minor": 4}


class EmergencyRoutingLLMOutput(BaseModel):
    routing_decision: Literal["accept", "transfer", "divert"]
    priority_level: int
    reasoning: str
    confidence_score: float


_PROMPT = """You triage incoming emergencies. Given severity, hospital load, and specialist
availability, decide: accept, transfer, or divert. Cite only the provided policy excerpts.

Patient: {patient}
Policy excerpts: {policies}
Department status: {resource_snapshot}
"""


def _rule_based_decision(
    patient: PatientInfo, resource_snapshot: dict
) -> tuple[Literal["accept", "transfer", "divert"], int, str, float]:
    priority = PRIORITY_BY_SEVERITY[patient.severity]
    department_load = resource_snapshot.get("department_load", "normal")
    specialist_capability_available = resource_snapshot.get("specialist_capability_available", True)

    if not specialist_capability_available:
        return (
            "transfer",
            priority,
            "Required specialist capability is not available on-site; transferred to the "
            "designated regional facility independent of department load (Emergency Protocols "
            "3.0).",
            0.88,
        )

    if priority in (1, 2):
        routing_decision = "accept"
        reasoning = (
            f"Priority-{priority} classification is accepted regardless of current department "
            "load (Emergency Protocols 1.1/2.2)."
        )
        base_confidence = 0.90
    else:
        if department_load == "limited":
            routing_decision = "divert"
            reasoning = (
                f"Priority-{priority} case diverted to urgent care; department load is at the "
                "limited-capacity threshold (Emergency Protocols 2.3)."
            )
            base_confidence = 0.85
        else:
            routing_decision = "accept"
            reasoning = (
                f"Priority-{priority} case accepted on-site; department load ({department_load}) "
                "is below the diversion threshold (Emergency Protocols 2.1/2.2)."
            )
            base_confidence = 0.88

    return routing_decision, priority, reasoning, base_confidence


def run_emergency_routing_agent(
    patient: PatientInfo,
    retrieved_policies: list[PolicyCitation],
    resource_snapshot: dict,
) -> AgentResult:
    llm = get_chat_llm()
    if llm is None:
        routing_decision, priority_level, reasoning, base_confidence = _rule_based_decision(
            patient, resource_snapshot
        )
    else:
        structured_llm = llm.with_structured_output(EmergencyRoutingLLMOutput)
        result = structured_llm.invoke(
            _PROMPT.format(
                patient=patient.model_dump(),
                policies=[p.model_dump() for p in retrieved_policies],
                resource_snapshot=resource_snapshot,
            )
        )
        routing_decision, priority_level, reasoning, base_confidence = (
            result.routing_decision,
            result.priority_level,
            result.reasoning,
            result.confidence_score,
        )

    citations = sorted(retrieved_policies, key=lambda c: c.similarity_score, reverse=True)[:MAX_CITATIONS]
    confidence, policy_grounded = apply_grounding(citations, base_confidence)

    return AgentResult(
        agent_name="emergency_routing",
        decision={"routing_decision": routing_decision, "priority_level": priority_level},
        reasoning=reasoning,
        confidence_score=confidence,
        citations=citations,
        policy_grounded=policy_grounded,
    )
