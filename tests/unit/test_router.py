import pytest

from app.graph.nodes.router import classify_request
from app.models.domain import PatientInfo


def patient(**overrides) -> PatientInfo:
    base = dict(
        patient_ref="PT-2026-00001",
        age=40,
        presenting_complaint="unspecified",
        diagnosis_category="general",
        severity="moderate",
        requires_isolation=False,
        requires_monitoring_level="standard",
        arrival_mode="walk_in",
    )
    base.update(overrides)
    return PatientInfo(**base)


def test_stable_walk_in_routes_to_bed_allocation_only():
    p = patient(severity="minor", requires_monitoring_level="none")
    result = classify_request(p)
    assert result.selected_agents == ["bed_allocation"]
    assert result.request_type == "bed_allocation"


def test_critical_cardiac_ambulance_is_a_combination_request():
    p = patient(
        presenting_complaint="acute myocardial infarction",
        diagnosis_category="acute_mi",
        severity="critical",
        requires_monitoring_level="icu",
        arrival_mode="ambulance",
    )
    result = classify_request(p)
    assert result.request_type == "combination"
    assert set(result.selected_agents) == {"emergency_routing", "bed_allocation", "staff_allocation"}


def test_isolation_case_without_urgent_arrival_skips_emergency_routing():
    p = patient(
        presenting_complaint="pneumonia, respiratory distress",
        diagnosis_category="pneumonia_respiratory_distress",
        severity="high",
        requires_isolation=True,
        arrival_mode="walk_in",
    )
    result = classify_request(p)
    assert result.request_type == "combination"
    assert set(result.selected_agents) == {"bed_allocation", "staff_allocation"}
    assert "emergency_routing" not in result.selected_agents


def test_high_severity_ambulance_triggers_emergency_routing():
    p = patient(severity="high", arrival_mode="ambulance")
    result = classify_request(p)
    assert "emergency_routing" in result.selected_agents


def test_mass_casualty_arrival_does_not_trigger_emergency_routing():
    # Per data/fixtures/eval_fixtures.json: mass-casualty patients are routed as
    # parallel bed/staff allocation, not re-triaged through emergency_routing, even
    # at critical severity.
    p = patient(severity="critical", requires_monitoring_level="icu", arrival_mode="mass_casualty")
    result = classify_request(p)
    assert "emergency_routing" not in result.selected_agents
    assert set(result.selected_agents) == {"bed_allocation", "staff_allocation"}


@pytest.mark.parametrize("severity", ["critical", "high", "moderate", "minor"])
def test_bed_allocation_always_selected(severity):
    p = patient(severity=severity)
    result = classify_request(p)
    assert "bed_allocation" in result.selected_agents


def test_confidence_score_is_in_valid_range():
    result = classify_request(patient())
    assert 0.0 <= result.confidence_score <= 1.0
