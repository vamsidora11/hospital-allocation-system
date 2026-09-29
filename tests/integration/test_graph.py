"""Full LangGraph workflow tests (plan Phase 5 definition of done):
1. an end-to-end graph run on a fixture request produces a FinalDecision.
2. a forced-failure test exercises retry -> fallback.

Retrieval/capacity/LLM are monkeypatched per-test so these run hermetically, without
depending on a live Chroma collection, seeded DB, or LLM key being present.
"""

import app.graph.nodes.agent_execution as agent_execution_module
import app.graph.nodes.router as router_module
from app.agents import bed_allocation, emergency_routing, staff_allocation
from app.graph.build_graph import build_graph
from app.models.domain import PatientInfo, PolicyCitation


def patient(**overrides) -> PatientInfo:
    base = dict(
        patient_ref="PT-2026-00931",
        age=65,
        presenting_complaint="acute myocardial infarction",
        diagnosis_category="acute_mi",
        severity="critical",
        requires_isolation=False,
        requires_monitoring_level="icu",
        arrival_mode="ambulance",
    )
    base.update(overrides)
    return PatientInfo(**base)


def initial_state(request_id: str, p: PatientInfo) -> dict:
    return {
        "request_id": request_id,
        "patient_information": p,
        "request_type": None,
        "workflow_status": "new",
        "active_agents": [],
        "agent_results": [],
        "retrieved_policies": [],
        "confidence_scores": {},
        "approval_status": None,
        "final_recommendation": None,
        "audit_trail": [],
        "retry_count": 0,
        "force_human_review": False,
        "conflict_detected": False,
        "override_decision": None,
    }


def _good_citation() -> PolicyCitation:
    return PolicyCitation(
        citation_id="CIT-TEST",
        policy_name="Bed allocation policy",
        section="3.2 Critical acuity placement",
        excerpt="Test excerpt.",
        similarity_score=0.9,
    )


def _force_offline_and_hermetic(monkeypatch):
    """No real LLM, no real Chroma/DB — ample capacity, well-grounded citations."""
    monkeypatch.setattr(router_module, "get_chat_llm", lambda: None)
    monkeypatch.setattr(bed_allocation, "get_chat_llm", lambda: None)
    monkeypatch.setattr(staff_allocation, "get_chat_llm", lambda: None)
    monkeypatch.setattr(emergency_routing, "get_chat_llm", lambda: None)
    monkeypatch.setattr(agent_execution_module, "retrieve", lambda query, domain, k=4: [_good_citation()])
    monkeypatch.setattr(
        agent_execution_module,
        "_resource_snapshot_for",
        lambda agent_name, patient: {
            "icu": {"total": 6, "available": 3},
            "general": {"total": 20, "available": 10},
            "isolation": {"total": 5, "available": 2},
            "available_on_shift": 5,
            "specialist_available": 2,
            "department_load": "normal",
            "specialist_capability_available": True,
        },
    )


def test_end_to_end_run_produces_final_decision(monkeypatch):
    _force_offline_and_hermetic(monkeypatch)

    graph = build_graph()
    p = patient()
    config = {"configurable": {"thread_id": "REQ-E2E-001"}}
    result = graph.invoke(initial_state("REQ-E2E-001", p), config=config)

    assert result["workflow_status"] == "decided"
    assert result["final_recommendation"] is not None
    assert result["final_recommendation"]["decisions"]["bed_allocation"]["bed_type"] == "icu"
    assert "__interrupt__" not in result or not result["__interrupt__"]

    states_hit = [e.state for e in result["audit_trail"]]
    for expected_state in (
        "INPUT_VALIDATION",
        "REQUEST_CLASSIFICATION",
        "AGENT_EXECUTION",
        "RESULT_CONSOLIDATION",
        "CONFIDENCE_EVALUATION",
        "HUMAN_APPROVAL_GATE",
        "FINAL_DECISION",
        "AUDIT_LOGGING",
    ):
        assert expected_state in states_hit


def test_forced_failure_retries_then_falls_back_to_conservative_default(monkeypatch):
    _force_offline_and_hermetic(monkeypatch)
    monkeypatch.setattr(agent_execution_module.time, "sleep", lambda seconds: None)

    call_count = {"n": 0}

    def _always_fails(patient, policies, resource_snapshot):
        call_count["n"] += 1
        raise RuntimeError("simulated transient tool failure")

    monkeypatch.setattr(agent_execution_module, "_AGENT_FUNCS", {"bed_allocation": _always_fails})

    graph = build_graph()
    # minor/walk_in -> router selects only bed_allocation, isolating the retry/fallback path
    p = patient(
        severity="minor",
        requires_monitoring_level="none",
        arrival_mode="walk_in",
        diagnosis_category="fracture_closed",
        presenting_complaint="closed fracture",
    )
    config = {"configurable": {"thread_id": "REQ-FAIL-001"}}
    result = graph.invoke(initial_state("REQ-FAIL-001", p), config=config)

    assert call_count["n"] == 3  # MAX_AGENT_RETRIES
    assert result["retry_count"] == 3
    assert result["force_human_review"] is True

    fallback_result = next(r for r in result["agent_results"] if r.agent_name == "bed_allocation")
    assert fallback_result.decision == {"bed_type": "general"}
    assert fallback_result.policy_grounded is False

    # Confidence was 0.0 and force_human_review is set, so this must NOT have
    # auto-approved straight through to a decision — it should be paused at the
    # human approval interrupt instead.
    assert "__interrupt__" in result
    assert result["workflow_status"] != "decided"


def test_conflicting_bed_and_staff_agents_forces_human_review(monkeypatch):
    """Phase 6 definition of done: bed agent recommends ICU, but staff agent can't confirm
    ICU-level coverage (no specialist on shift) — this must force human review even though
    the bed agent itself is confident and well-grounded."""
    monkeypatch.setattr(bed_allocation, "get_chat_llm", lambda: None)
    monkeypatch.setattr(staff_allocation, "get_chat_llm", lambda: None)
    monkeypatch.setattr(emergency_routing, "get_chat_llm", lambda: None)
    monkeypatch.setattr(agent_execution_module, "retrieve", lambda query, domain, k=4: [_good_citation()])

    def _snapshot(agent_name, patient):
        if agent_name == "bed_allocation":
            return {"icu": {"total": 6, "available": 3}}
        if agent_name == "staff_allocation":
            return {"available_on_shift": 5, "specialist_available": 0}  # coverage shortfall
        return {"department_load": "normal", "specialist_capability_available": True}

    monkeypatch.setattr(agent_execution_module, "_resource_snapshot_for", _snapshot)

    graph = build_graph()
    p = patient(severity="critical", requires_monitoring_level="icu", arrival_mode="ambulance")
    config = {"configurable": {"thread_id": "REQ-CONFLICT-001"}}
    result = graph.invoke(initial_state("REQ-CONFLICT-001", p), config=config)

    bed_result = next(r for r in result["agent_results"] if r.agent_name == "bed_allocation")
    staff_result = next(r for r in result["agent_results"] if r.agent_name == "staff_allocation")
    assert bed_result.decision["bed_type"] == "icu"
    assert bed_result.confidence_score >= 0.85  # bed agent itself is confident
    assert staff_result.confidence_score < 0.70  # staff agent can't confirm coverage

    assert result["conflict_detected"] is True
    assert "__interrupt__" in result
    assert result["workflow_status"] != "decided"
