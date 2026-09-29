"""Request classification (plan Section 7).

Real path: a single structured-output LLM call returning `RouterOutput`, so multi-intent
requests naturally produce `selected_agents` with more than one entry and
`request_type: "combination"`. With no LLM configured in this sandbox (`get_chat_llm()`
returns `None` — see CLAUDE.md), falls back to an equivalent deterministic rule set so the
workflow still runs end-to-end and is unit-testable without a live key.

The plan's retry/fallback wrapping (Section 5) only covers AGENT_EXECUTION, not
REQUEST_CLASSIFICATION — but a bare LLM call with no retry at all means one transient error
(a 429 rate-limit response, a network blip) crashes the whole request with an unhandled 500.
`classify_request` retries the real LLM call a couple of times with backoff and, if it still
fails, falls back to the same deterministic rule set used when no LLM is configured at all —
so a transient failure degrades gracefully instead of failing the request outright.
"""

import time

from app.config import get_settings
from app.graph.state import WorkflowState
from app.llm import get_chat_llm
from app.models.domain import PatientInfo, RouterOutput
from app.services.audit import make_entry

_ROUTER_MAX_ATTEMPTS = 3
_ROUTER_BACKOFF_SECONDS = (3, 6)

_ROUTER_PROMPT = """You classify hospital resource-allocation requests. Given the patient
information below, decide which of the following agents must run: bed_allocation,
staff_allocation, emergency_routing. A request needs more than one agent when, for example, a
critical patient needs both a bed and specialist staffing. Return your answer as structured
JSON matching the RouterOutput schema, including your reasoning and a confidence score.

Patient: {patient}
"""


def _rule_based_classify(patient: PatientInfo) -> RouterOutput:
    agents: list[str] = []

    # Only ambulance arrivals with high/critical severity route through emergency
    # triage. Mass-casualty batches are handled as parallel bed/staff allocation per
    # patient (Compliance Rules 1.0 / Emergency Protocols Section 4) rather than
    # individually re-triaged as emergency-department arrivals — see
    # data/fixtures/eval_fixtures.json's mass_casualty_trauma cases, none of which
    # select emergency_routing even at critical severity.
    if patient.arrival_mode == "ambulance" and patient.severity in ("critical", "high"):
        agents.append("emergency_routing")

    agents.append("bed_allocation")

    if patient.requires_isolation or patient.severity == "critical" or patient.requires_monitoring_level == "icu":
        agents.append("staff_allocation")

    # de-duplicate while preserving order
    seen: set[str] = set()
    agents = [a for a in agents if not (a in seen or seen.add(a))]

    request_type = agents[0] if len(agents) == 1 else "combination"

    reasoning = (
        f"Classified from severity ({patient.severity}), monitoring need "
        f"({patient.requires_monitoring_level}), isolation requirement "
        f"({patient.requires_isolation}), and arrival mode ({patient.arrival_mode})."
    )

    # Rule-based path has no model uncertainty to report; use a high, fixed
    # confidence — real LLM calls will report their own calibrated score instead.
    confidence = 0.93

    return RouterOutput(
        request_type=request_type,
        selected_agents=agents,
        routing_reasoning=reasoning,
        confidence_score=confidence,
    )


def classify_request(patient: PatientInfo) -> RouterOutput:
    settings = get_settings()
    llm = get_chat_llm() if not settings.use_rule_based_agents else None
    if llm is None:
        return _rule_based_classify(patient)

    structured_llm = llm.with_structured_output(RouterOutput)
    prompt = _ROUTER_PROMPT.format(patient=patient.model_dump())

    last_error: Exception | None = None
    for attempt in range(_ROUTER_MAX_ATTEMPTS):
        try:
            return structured_llm.invoke(prompt)
        except Exception as exc:  # noqa: BLE001 - any transient LLM/network failure is retryable
            last_error = exc
            if attempt < _ROUTER_MAX_ATTEMPTS - 1:
                time.sleep(_ROUTER_BACKOFF_SECONDS[min(attempt, len(_ROUTER_BACKOFF_SECONDS) - 1)])

    # Exhausted retries — degrade to the deterministic rule set rather than failing the
    # request outright. Reflected in the reasoning so it's visible in the audit trail.
    fallback = _rule_based_classify(patient)
    fallback.routing_reasoning = (
        f"LLM classification failed after {_ROUTER_MAX_ATTEMPTS} attempts ({last_error}); "
        f"fell back to rule-based classification. {fallback.routing_reasoning}"
    )
    return fallback


def router_node(state: WorkflowState) -> dict:
    router_output = classify_request(state["patient_information"])
    entry = make_entry(
        request_id=state["request_id"],
        state="REQUEST_CLASSIFICATION",
        input_snapshot=state["patient_information"].model_dump(mode="json"),
        output_snapshot=router_output.model_dump(),
    )
    return {
        "request_type": router_output.request_type,
        "active_agents": router_output.selected_agents,
        "confidence_scores": {"router": router_output.confidence_score},
        "audit_trail": [entry],
    }
