"""Phase 8 definition of done: GET /audit/{request_id} shows all 8 states for a completed
run. Runs the graph hermetically (same pattern as test_graph.py), persists the resulting
audit trail to a throwaway SQLite DB, and reads it back through the real FastAPI endpoint.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.graph.nodes.agent_execution as agent_execution_module
import app.graph.nodes.router as router_module
from app.agents import bed_allocation, emergency_routing, staff_allocation
from app.db import get_session
from app.graph.build_graph import build_graph
from app.main import app as fastapi_app
from app.models.domain import PatientInfo, PolicyCitation
from app.models.orm import Base
from app.services.audit import persist_entries
from app.services.requests import create_request

ALL_8_STATES = [
    "INPUT_VALIDATION",
    "REQUEST_CLASSIFICATION",
    "AGENT_EXECUTION",
    "RESULT_CONSOLIDATION",
    "CONFIDENCE_EVALUATION",
    "HUMAN_APPROVAL_GATE",
    "FINAL_DECISION",
    "AUDIT_LOGGING",
]


@pytest.fixture
def session_factory(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test_audit.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


@pytest.fixture
def client(session_factory):
    def override_get_session():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    fastapi_app.dependency_overrides[get_session] = override_get_session
    yield TestClient(fastapi_app)
    fastapi_app.dependency_overrides.clear()


def _good_citation() -> PolicyCitation:
    return PolicyCitation(
        citation_id="CIT-TEST",
        policy_name="Bed allocation policy",
        section="3.2 Critical acuity placement",
        excerpt="Test excerpt.",
        similarity_score=0.9,
    )


def _run_hermetic_graph(monkeypatch, request_id: str) -> list:
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
            "available_on_shift": 5,
            "specialist_available": 2,
            "department_load": "normal",
            "specialist_capability_available": True,
        },
    )

    patient = PatientInfo(
        patient_ref="PT-2026-00931",
        age=65,
        presenting_complaint="acute myocardial infarction",
        diagnosis_category="acute_mi",
        severity="critical",
        requires_isolation=False,
        requires_monitoring_level="icu",
        arrival_mode="ambulance",
    )
    graph = build_graph()
    initial_state = {
        "request_id": request_id,
        "patient_information": patient,
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
    result = graph.invoke(initial_state, config={"configurable": {"thread_id": request_id}})
    assert result["workflow_status"] == "decided"
    return result["audit_trail"]


def test_audit_endpoint_shows_all_8_states_for_a_completed_run(monkeypatch, client, session_factory):
    session = session_factory()
    patient = PatientInfo(
        patient_ref="PT-2026-00931",
        age=65,
        presenting_complaint="acute myocardial infarction",
        diagnosis_category="acute_mi",
        severity="critical",
        requires_isolation=False,
        requires_monitoring_level="icu",
        arrival_mode="ambulance",
    )
    request_row = create_request(session, patient, request_type="combination")
    request_id = request_row.id

    audit_trail = _run_hermetic_graph(monkeypatch, request_id)
    persist_entries(session, audit_trail)
    session.close()

    response = client.get(f"/audit/{request_id}")
    assert response.status_code == 200

    entries = response.json()
    states_present = {e["state"] for e in entries}
    for expected_state in ALL_8_STATES:
        assert expected_state in states_present, f"missing {expected_state}"

    # ordered by timestamp — INPUT_VALIDATION first, AUDIT_LOGGING last
    assert entries[0]["state"] == "INPUT_VALIDATION"
    assert entries[-1]["state"] == "AUDIT_LOGGING"


def test_audit_endpoint_returns_empty_list_for_unknown_request(client):
    response = client.get("/audit/REQ-DOES-NOT-EXIST")
    assert response.status_code == 200
    assert response.json() == []
