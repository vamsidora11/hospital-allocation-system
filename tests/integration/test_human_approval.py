"""Phase 7 definition of done: a low-confidence run pauses, and each of the four reviewer
actions (approve/override/reject/request_info) resumes the graph correctly."""

from langgraph.types import Command

import app.graph.nodes.agent_execution as agent_execution_module
import app.graph.nodes.router as router_module
from app.agents import bed_allocation, emergency_routing, staff_allocation
from app.graph.build_graph import build_graph
from app.models.domain import PatientInfo, PolicyCitation


def patient(**overrides) -> PatientInfo:
    base = dict(
        patient_ref="PT-2026-00071",
        age=71,
        presenting_complaint="fall with comorbid cardiac history",
        diagnosis_category="moderate_trauma_fall",
        severity="moderate",
        requires_isolation=False,
        requires_monitoring_level="standard",
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


def _weak_citation() -> PolicyCitation:
    # Below GROUNDING_FLOOR (0.55) so every agent comes back ungrounded / low-confidence,
    # guaranteeing CONFIDENCE_EVALUATION routes to HUMAN_APPROVAL_GATE.
    return PolicyCitation(
        citation_id="CIT-WEAK",
        policy_name="Bed allocation policy",
        section="2.4 General ward defaults",
        excerpt="Weak match.",
        similarity_score=0.30,
    )


def _strong_citation() -> PolicyCitation:
    return PolicyCitation(
        citation_id="CIT-STRONG",
        policy_name="Bed allocation policy",
        section="3.2 Critical acuity placement",
        excerpt="Strong match.",
        similarity_score=0.9,
    )


def _mock_offline(monkeypatch, citation_fn):
    monkeypatch.setattr(router_module, "get_chat_llm", lambda: None)
    monkeypatch.setattr(bed_allocation, "get_chat_llm", lambda: None)
    monkeypatch.setattr(staff_allocation, "get_chat_llm", lambda: None)
    monkeypatch.setattr(emergency_routing, "get_chat_llm", lambda: None)
    monkeypatch.setattr(agent_execution_module, "retrieve", lambda query, domain, k=4: [citation_fn()])
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


def _run_to_pending(monkeypatch, thread_id: str):
    """Weak citations -> low confidence -> graph pauses at the human-approval interrupt."""
    _mock_offline(monkeypatch, _weak_citation)
    graph = build_graph()
    p = patient()
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(initial_state(thread_id, p), config=config)
    assert "__interrupt__" in result, "expected the low-confidence fixture to pause for review"
    return graph, config, result


def test_approve_resumes_to_decided(monkeypatch):
    graph, config, _ = _run_to_pending(monkeypatch, "REQ-APPROVE-001")

    result = graph.invoke(Command(resume={"action": "approve", "reviewer_id": "reviewer-01"}), config=config)

    assert result["workflow_status"] == "decided"
    assert result["approval_status"] == "approve"
    assert result["final_recommendation"] is not None
    assert "__interrupt__" not in result or not result["__interrupt__"]


def test_override_resumes_to_decided_with_substituted_decision_alongside_original(monkeypatch):
    graph, config, pending = _run_to_pending(monkeypatch, "REQ-OVERRIDE-001")
    original_bed_decision = next(r for r in pending["agent_results"] if r.agent_name == "bed_allocation").decision

    override_decision = {"bed_type": "isolation"}
    result = graph.invoke(
        Command(
            resume={
                "action": "override",
                "reviewer_id": "reviewer-01",
                "notes": "Reviewer judged isolation more appropriate given shift staffing.",
                "override_decision": override_decision,
            }
        ),
        config=config,
    )

    assert result["workflow_status"] == "decided"
    assert result["approval_status"] == "override"
    assert result["final_recommendation"]["reviewer_override"] == override_decision
    # Original agent recommendation must still be present, not replaced.
    assert result["final_recommendation"]["decisions"]["bed_allocation"] == original_bed_decision


def test_reject_resumes_to_rejected_and_still_logs(monkeypatch):
    graph, config, _ = _run_to_pending(monkeypatch, "REQ-REJECT-001")

    result = graph.invoke(
        Command(resume={"action": "reject", "reviewer_id": "reviewer-01", "notes": "Insufficient coverage."}),
        config=config,
    )

    assert result["workflow_status"] == "decided"  # closed via FINAL_DECISION/AUDIT_LOGGING, not silently dropped
    assert result["approval_status"] == "reject"
    assert result["final_recommendation"]["headline"] == "Request rejected"
    states_hit = [e.state for e in result["audit_trail"]]
    assert "FINAL_DECISION" in states_hit
    assert "AUDIT_LOGGING" in states_hit


def test_request_info_reruns_agents_and_can_then_auto_approve(monkeypatch):
    graph, config, pending = _run_to_pending(monkeypatch, "REQ-INFO-001")
    assert pending["approval_status"] == "pending"

    # Simulate the added context resolving the low-confidence signal: strengthen the
    # citations returned on the re-run so this pass is grounded and auto-approves.
    monkeypatch.setattr(agent_execution_module, "retrieve", lambda query, domain, k=4: [_strong_citation()])

    result = graph.invoke(
        Command(resume={"action": "request_info", "reviewer_id": "reviewer-01", "notes": "Add recent vitals."}),
        config=config,
    )

    assert result["workflow_status"] == "decided"
    assert result["approval_status"] == "auto_approved"
    states_hit = [e.state for e in result["audit_trail"]]
    assert states_hit.count("AGENT_EXECUTION") >= 2  # ran once before the interrupt, once after resume
