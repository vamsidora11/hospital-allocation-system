"""Phase 9 definition of done: all endpoints in plan Section 10 are live.

Hits app/api/requests.py's endpoints through the real FastAPI app + its module-level graph
singleton, so this also exercises the create -> run -> (maybe pause) -> approve -> resume
lifecycle end-to-end through HTTP, not just direct graph.invoke() calls.
"""

import app.api.requests as requests_api_module
import app.graph.nodes.agent_execution as agent_execution_module
import app.graph.nodes.router as router_module
from app.agents import bed_allocation, emergency_routing, staff_allocation
from tests.conftest import AUTH_HEADERS

_AMPLE_SNAPSHOT = {
    "icu": {"total": 6, "available": 3},
    "general": {"total": 20, "available": 10},
    "isolation": {"total": 5, "available": 2},
    "available_on_shift": 5,
    "specialist_available": 2,
    "department_load": "normal",
    "specialist_capability_available": True,
}


def _good_citation():
    from app.models.domain import PolicyCitation

    return PolicyCitation(
        citation_id="CIT-TEST",
        policy_name="Bed allocation policy",
        section="3.2 Critical acuity placement",
        excerpt="Test excerpt.",
        similarity_score=0.9,
    )


def _weak_citation():
    from app.models.domain import PolicyCitation

    return PolicyCitation(
        citation_id="CIT-WEAK",
        policy_name="Bed allocation policy",
        section="2.4 General ward defaults",
        excerpt="Weak match.",
        similarity_score=0.30,
    )


def _mock_offline(monkeypatch, citation_fn):
    monkeypatch.setattr(router_module, "get_chat_llm", lambda: None)
    monkeypatch.setattr(bed_allocation, "get_chat_llm", lambda: None)
    monkeypatch.setattr(staff_allocation, "get_chat_llm", lambda: None)
    monkeypatch.setattr(emergency_routing, "get_chat_llm", lambda: None)
    monkeypatch.setattr(agent_execution_module, "retrieve", lambda query, domain, k=4: [citation_fn()])
    monkeypatch.setattr(agent_execution_module, "_resource_snapshot_for", lambda agent_name, patient: _AMPLE_SNAPSHOT)


def _patient_payload(**overrides):
    payload = dict(
        patient_ref="PT-2026-00931",
        age=65,
        presenting_complaint="acute myocardial infarction",
        diagnosis_category="acute_mi",
        severity="critical",
        requires_isolation=False,
        requires_monitoring_level="icu",
        arrival_mode="ambulance",
    )
    payload.update(overrides)
    return payload


def test_create_request_auto_approves_and_reaches_decided(monkeypatch, client):
    _mock_offline(monkeypatch, _good_citation)

    response = client.post("/requests", json=_patient_payload(), headers=AUTH_HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["workflow_status"] == "decided"
    assert body["approval_status"] == "auto_approved"
    request_id = body["request_id"]

    status_response = client.get(f"/requests/{request_id}", headers=AUTH_HEADERS)
    assert status_response.status_code == 200
    assert status_response.json()["workflow_status"] == "decided"

    rec_response = client.get(f"/requests/{request_id}/recommendation", headers=AUTH_HEADERS)
    assert rec_response.status_code == 200
    rec_body = rec_response.json()
    assert rec_body["final_recommendation"]["decisions"]["bed_allocation"]["bed_type"] == "icu"
    assert len(rec_body["citations"]) > 0

    audit_response = client.get(f"/audit/{request_id}", headers=AUTH_HEADERS)
    assert audit_response.status_code == 200
    entries = audit_response.json()
    states_present = {e["state"] for e in entries}
    # this patient (critical, ambulance) triggers all 3 agents, so there are 3
    # AGENT_EXECUTION entries — check the 8 distinct states are present, not an exact count.
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
        assert expected_state in states_present


def test_low_confidence_request_pauses_then_approve_resumes_it(monkeypatch, client):
    _mock_offline(monkeypatch, _weak_citation)

    create_response = client.post(
        "/requests",
        json=_patient_payload(severity="moderate", requires_monitoring_level="standard", arrival_mode="ambulance"),
        headers=AUTH_HEADERS,
    )
    assert create_response.status_code == 200
    body = create_response.json()
    assert body["workflow_status"] == "pending_approval" or body["approval_status"] == "pending"
    request_id = body["request_id"]

    approval_response = client.post(
        f"/requests/{request_id}/approval",
        json={"action": "approve", "reviewer_id": "reviewer-01"},
        headers=AUTH_HEADERS,
    )
    assert approval_response.status_code == 200
    assert approval_response.json()["workflow_status"] == "decided"
    assert approval_response.json()["approval_status"] == "approve"


def test_approval_on_a_not_pending_request_is_rejected(monkeypatch, client):
    _mock_offline(monkeypatch, _good_citation)

    create_response = client.post("/requests", json=_patient_payload(), headers=AUTH_HEADERS)
    request_id = create_response.json()["request_id"]
    assert create_response.json()["workflow_status"] == "decided"

    approval_response = client.post(
        f"/requests/{request_id}/approval",
        json={"action": "approve", "reviewer_id": "reviewer-01"},
        headers=AUTH_HEADERS,
    )
    assert approval_response.status_code == 409


def test_unknown_request_id_returns_404(client):
    assert client.get("/requests/REQ-NOPE", headers=AUTH_HEADERS).status_code == 404
    assert client.get("/requests/REQ-NOPE/recommendation", headers=AUTH_HEADERS).status_code == 404


def test_agents_status_lists_all_three_agents(client):
    response = client.get("/agents/status", headers=AUTH_HEADERS)
    assert response.status_code == 200
    names = {a["agent_name"] for a in response.json()}
    assert names == {"bed_allocation", "staff_allocation", "emergency_routing"}


def test_requests_endpoints_reject_missing_api_key(client):
    assert client.post("/requests", json=_patient_payload()).status_code == 401
    assert client.get("/requests/whatever").status_code == 401
    assert client.get("/agents/status").status_code == 401
