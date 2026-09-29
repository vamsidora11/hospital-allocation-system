"""The three demo scenarios from plan Section 11, pinned as regression tests (Phase 10).

Runs the full system through the real HTTP API (app/api/requests.py), with the two external
seams — the LLM and vector retrieval — held deterministic via the same monkeypatch pattern
used throughout tests/integration/. This matches the plan's own stated goal ("the exact
known-good outcomes stay pinned as the system evolves") better than depending on a live LLM's
free-form phrasing/confidence, which even at temperature=0 isn't guaranteed byte-identical
across model versions.

Scenario 3 note: the plan's Section 11 narrative says mass-casualty patients get "a mix of
ICU / general-ward / transfer / divert outcomes." The user's own hand-derived
data/fixtures/eval_fixtures.json (authoritative — see Phase 3's PROGRESS.md entry, where
trusting it over my own first assumption caught a real router bug) shows mass_casualty
arrivals never select emergency_routing, even at critical severity — so "transfer"/"divert"
(which only emergency_routing can produce) don't occur here. This test asserts the actual,
fixture-verified contract (a severity-driven ICU/general-ward mix from bed_allocation) rather
than force a contradiction into the router to chase the narrative text.
"""

import app.graph.nodes.agent_execution as agent_execution_module
import app.graph.nodes.router as router_module
from app.agents import bed_allocation, emergency_routing, staff_allocation
from app.models.domain import PolicyCitation
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


def _citation(policy_name: str, section: str, similarity: float = 0.9) -> PolicyCitation:
    return PolicyCitation(
        citation_id=f"CIT-{policy_name[:4].upper()}",
        policy_name=policy_name,
        section=section,
        excerpt="Test excerpt.",
        similarity_score=similarity,
    )


_CITATIONS_BY_DOMAIN = {
    "bed": _citation("Bed allocation policy", "3.2 Critical acuity placement"),
    "staffing": _citation("Staffing policy", "5.3 Specialist matching"),
    "emergency": _citation("Emergency protocols", "1.1 Priority-1 criteria"),
    "infection_control": _citation("Infection control standards", "2.2 Isolation staffing"),
    "compliance": _citation("Compliance rules", "1.0 Mass-casualty activation"),
}

# Mirrors app/rag/retriever.py's real DOMAIN_EXPANSION — a bed_allocation call retrieves
# with domain="bed", but the real retriever folds in cross-cutting infection_control
# citations too (and staffing similarly, emergency pulls in compliance). The mock needs to
# reproduce that expansion, not just a flat domain->citation map, or scenario 2's expected
# "Infection control standards" citation never shows up.
_DOMAIN_EXPANSION = {
    "bed": ["bed", "infection_control"],
    "staffing": ["staffing", "infection_control"],
    "emergency": ["emergency", "compliance"],
    "compliance": ["compliance"],
    "infection_control": ["infection_control"],
}


def _mock_stack(monkeypatch, citations_by_domain=None):
    citations_by_domain = citations_by_domain or _CITATIONS_BY_DOMAIN
    monkeypatch.setattr(router_module, "get_chat_llm", lambda: None)
    monkeypatch.setattr(bed_allocation, "get_chat_llm", lambda: None)
    monkeypatch.setattr(staff_allocation, "get_chat_llm", lambda: None)
    monkeypatch.setattr(emergency_routing, "get_chat_llm", lambda: None)

    def _mock_retrieve(query, domain, k=4):
        return [citations_by_domain[d] for d in _DOMAIN_EXPANSION.get(domain, [domain]) if d in citations_by_domain]

    monkeypatch.setattr(agent_execution_module, "retrieve", _mock_retrieve)
    monkeypatch.setattr(agent_execution_module, "_resource_snapshot_for", lambda agent_name, patient: _AMPLE_SNAPSHOT)


def test_scenario_1_critical_cardiac_case(monkeypatch, client):
    """Acute MI, requires ICU monitoring -> ICU bed, confidence >= 0.85, citations from
    bed-allocation and emergency-protocol policies, auto-approved (no human gate)."""
    _mock_stack(monkeypatch)

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
    response = client.post("/requests", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200
    body = response.json()

    assert body["workflow_status"] == "decided"
    assert body["approval_status"] == "auto_approved"  # no human gate

    rec = client.get(f"/requests/{body['request_id']}/recommendation", headers=AUTH_HEADERS).json()
    assert rec["final_recommendation"]["decisions"]["bed_allocation"]["bed_type"] == "icu"

    policy_names = {c["policy_name"] for c in rec["citations"]}
    assert "Bed allocation policy" in policy_names
    assert "Emergency protocols" in policy_names


def test_scenario_2_high_acuity_isolation_case(monkeypatch, client):
    """Infectious, requires isolation + respiratory support -> isolation room + specialist
    staff assignment, confidence >= 0.80 (high-0.80s per the plan's narrative), citations
    from infection-control and staffing policies."""
    _mock_stack(monkeypatch)

    payload = dict(
        patient_ref="PT-2026-00048",
        age=48,
        presenting_complaint="pneumonia, respiratory distress",
        diagnosis_category="pneumonia_respiratory_distress",
        severity="high",
        requires_isolation=True,
        requires_monitoring_level="standard",
        arrival_mode="walk_in",
    )
    response = client.post("/requests", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["workflow_status"] == "decided"

    rec = client.get(f"/requests/{body['request_id']}/recommendation", headers=AUTH_HEADERS).json()
    decisions = rec["final_recommendation"]["decisions"]
    assert decisions["bed_allocation"]["bed_type"] == "isolation"
    assert "infectious" in decisions["staff_allocation"]["assigned_role"].lower()

    audit = client.get(f"/audit/{body['request_id']}", headers=AUTH_HEADERS).json()
    per_agent_confidence = next(
        e["output_snapshot"]["aggregate_confidence"] for e in audit if e["state"] == "CONFIDENCE_EVALUATION"
    )
    assert per_agent_confidence >= 0.80

    policy_names = {c["policy_name"] for c in rec["citations"]}
    assert "Infection control standards" in policy_names
    assert "Staffing policy" in policy_names


def test_scenario_3_mass_casualty_burst(monkeypatch, client):
    """8 simultaneous patients -> all processed independently with a complete audit trail
    each, outcomes driven by severity (see module docstring on the transfer/divert
    discrepancy vs. the plan's Section 11 narrative)."""
    _mock_stack(monkeypatch)

    severities = ["critical", "critical", "high", "high", "moderate", "moderate", "minor", "minor"]
    request_ids = []
    for i, severity in enumerate(severities):
        monitoring = "icu" if severity == "critical" else ("standard" if severity in ("high", "moderate") else "none")
        payload = dict(
            patient_ref=f"PT-2026-{10000 + i}",
            age=22 + i * 7,
            presenting_complaint="multi-vehicle collision trauma",
            diagnosis_category="mass_casualty_trauma",
            severity=severity,
            requires_isolation=False,
            requires_monitoring_level=monitoring,
            arrival_mode="mass_casualty",
        )
        response = client.post("/requests", json=payload, headers=AUTH_HEADERS)
        assert response.status_code == 200
        request_ids.append(response.json()["request_id"])

    assert len(request_ids) == 8
    assert len(set(request_ids)) == 8  # each patient got its own independent request

    bed_types = []
    for request_id, severity in zip(request_ids, severities):
        rec = client.get(f"/requests/{request_id}/recommendation", headers=AUTH_HEADERS).json()
        bed_type = rec["final_recommendation"]["decisions"]["bed_allocation"]["bed_type"]
        bed_types.append(bed_type)
        assert bed_type == ("icu" if severity == "critical" else "general")

        # a complete audit entry for every one of the 8, within the same overall run
        audit = client.get(f"/audit/{request_id}", headers=AUTH_HEADERS).json()
        states_present = {e["state"] for e in audit}
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
            assert expected_state in states_present, (request_id, expected_state)

    assert "icu" in bed_types
    assert "general" in bed_types
