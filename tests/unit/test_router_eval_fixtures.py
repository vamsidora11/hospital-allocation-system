"""Router assertions driven by the shared eval fixture set.

Only checks `expected_router` — `expected_bed_allocation` / `expected_emergency_routing`
are for the Phase 4 agent tests, once those agents exist.
"""

import json
from pathlib import Path

import pytest

from app.graph.nodes.router import classify_request
from app.models.domain import PatientInfo

FIXTURES_PATH = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "eval_fixtures.json"
FIXTURES = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("fixture", FIXTURES, ids=[f["input"]["patient_ref"] for f in FIXTURES])
def test_router_matches_eval_fixture(fixture):
    patient = PatientInfo(**fixture["input"])
    result = classify_request(patient)

    expected = fixture["expected_router"]
    assert result.request_type == expected["request_type"], fixture["input"]["patient_ref"]
    assert set(result.selected_agents) == set(expected["selected_agents"]), fixture["input"]["patient_ref"]
