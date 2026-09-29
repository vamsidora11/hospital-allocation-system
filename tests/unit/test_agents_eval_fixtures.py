"""Bed/emergency agent assertions driven by the shared eval fixture set.

Assumes adequate capacity and a well-grounded citation for each call — these fixtures pin the
*decision logic*, not the capacity-shortage/ungrounded paths (covered in test_agents.py).
"""

import json
from pathlib import Path

import pytest

from app.agents.bed_allocation import run_bed_allocation_agent
from app.agents.emergency_routing import run_emergency_routing_agent
from app.models.domain import PatientInfo, PolicyCitation

FIXTURES_PATH = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "eval_fixtures.json"
FIXTURES = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))

_ANY_GROUNDED_CITATION = PolicyCitation(
    citation_id="CIT-TEST",
    policy_name="Bed allocation policy",
    section="3.2 Critical acuity placement",
    excerpt="Test excerpt.",
    similarity_score=0.85,
)
_AMPLE_BED_SNAPSHOT = {
    "icu": {"total": 10, "available": 5},
    "general": {"total": 20, "available": 10},
    "isolation": {"total": 5, "available": 3},
}


@pytest.mark.parametrize("fixture", FIXTURES, ids=[f["input"]["patient_ref"] for f in FIXTURES])
def test_bed_agent_matches_eval_fixture(fixture):
    patient = PatientInfo(**fixture["input"])
    result = run_bed_allocation_agent(patient, [_ANY_GROUNDED_CITATION], _AMPLE_BED_SNAPSHOT)
    assert result.decision["bed_type"] == fixture["expected_bed_allocation"], fixture["input"]["patient_ref"]


@pytest.mark.parametrize(
    "fixture",
    [f for f in FIXTURES if f["expected_emergency_routing"] is not None],
    ids=lambda f: f["input"]["patient_ref"],
)
def test_emergency_agent_matches_eval_fixture(fixture):
    patient = PatientInfo(**fixture["input"])
    result = run_emergency_routing_agent(
        patient, [_ANY_GROUNDED_CITATION], {"department_load": "normal", "specialist_capability_available": True}
    )

    expected = fixture["expected_emergency_routing"]  # e.g. "accept_priority_1"
    expected_decision, expected_priority = expected.rsplit("_priority_", 1)
    assert result.decision["routing_decision"] == expected_decision, fixture["input"]["patient_ref"]
    assert result.decision["priority_level"] == int(expected_priority), fixture["input"]["patient_ref"]
