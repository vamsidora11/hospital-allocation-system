from app.agents.bed_allocation import run_bed_allocation_agent
from app.agents.emergency_routing import run_emergency_routing_agent
from app.agents.staff_allocation import run_staff_allocation_agent
from app.models.domain import PatientInfo, PolicyCitation
from app.rag.retriever import GROUNDING_FLOOR


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


def citation(similarity_score: float = 0.85, section: str = "3.2 Critical acuity placement") -> PolicyCitation:
    return PolicyCitation(
        citation_id="CIT-TEST",
        policy_name="Bed allocation policy",
        section=section,
        excerpt="Test excerpt.",
        similarity_score=similarity_score,
    )


# ---------- Bed allocation agent ----------


def test_bed_agent_clear_cut_icu_case():
    p = patient(severity="critical", requires_monitoring_level="icu")
    snapshot = {"icu": {"total": 6, "available": 2}, "general": {"total": 20, "available": 10}}
    result = run_bed_allocation_agent(p, [citation(0.85)], snapshot)

    assert result.decision["bed_type"] == "icu"
    assert result.policy_grounded is True
    assert result.confidence_score >= 0.85


def test_bed_agent_low_confidence_when_no_capacity():
    p = patient(severity="critical", requires_monitoring_level="icu")
    snapshot = {"icu": {"total": 6, "available": 0}}
    result = run_bed_allocation_agent(p, [citation(0.85)], snapshot)

    assert result.decision["bed_type"] == "icu"
    assert result.confidence_score < 0.70
    assert "escalate" in result.reasoning.lower()


def test_bed_agent_ungrounded_when_no_citations():
    p = patient(severity="minor")
    result = run_bed_allocation_agent(p, [], {"general": {"total": 20, "available": 10}})

    assert result.policy_grounded is False
    assert result.confidence_score < 0.70


def test_bed_agent_ungrounded_when_citations_below_floor():
    p = patient(severity="minor")
    weak_citation = citation(GROUNDING_FLOOR - 0.1)
    result = run_bed_allocation_agent(p, [weak_citation], {"general": {"total": 20, "available": 10}})

    assert result.policy_grounded is False


# ---------- Staff allocation agent ----------


def test_staff_agent_clear_cut_cardiac_case():
    p = patient(diagnosis_category="acute_mi", presenting_complaint="acute myocardial infarction", severity="critical")
    snapshot = {"available_on_shift": 5, "specialist_available": 2}
    result = run_staff_allocation_agent(p, [citation(0.85, "5.3 Specialist matching")], snapshot)

    assert result.decision["assigned_role"] == "Cardiac-credentialed nurse"
    assert result.policy_grounded is True
    assert result.confidence_score >= 0.85


def test_staff_agent_low_confidence_on_coverage_shortfall():
    p = patient(requires_isolation=True, severity="high")
    snapshot = {"available_on_shift": 5, "specialist_available": 0}
    result = run_staff_allocation_agent(p, [citation(0.85, "5.4 Isolation staffing cross-reference")], snapshot)

    assert result.decision["assigned_role"] == "Respiratory/infectious-care nurse"
    assert result.confidence_score < 0.70
    assert "escalate" in result.reasoning.lower()


def test_staff_agent_ungrounded_when_no_citations():
    p = patient()
    result = run_staff_allocation_agent(p, [], {"available_on_shift": 5, "specialist_available": 5})

    assert result.policy_grounded is False


# ---------- Emergency routing agent ----------


def test_emergency_agent_clear_cut_priority_1_accept():
    p = patient(severity="critical", arrival_mode="ambulance")
    snapshot = {"department_load": "limited", "specialist_capability_available": True}
    result = run_emergency_routing_agent(p, [citation(0.85, "1.1 Priority-1 criteria")], snapshot)

    assert result.decision["routing_decision"] == "accept"
    assert result.decision["priority_level"] == 1
    assert result.policy_grounded is True


def test_emergency_agent_diverts_minor_case_under_limited_load():
    p = patient(severity="minor", arrival_mode="ambulance")
    snapshot = {"department_load": "limited"}
    result = run_emergency_routing_agent(p, [citation(0.85, "2.3 Diversion thresholds")], snapshot)

    assert result.decision["routing_decision"] == "divert"
    assert result.decision["priority_level"] == 4


def test_emergency_agent_transfers_when_specialist_capability_unavailable():
    p = patient(severity="high", arrival_mode="ambulance")
    snapshot = {"department_load": "normal", "specialist_capability_available": False}
    result = run_emergency_routing_agent(p, [citation(0.85, "3.0 Regional transfer")], snapshot)

    assert result.decision["routing_decision"] == "transfer"


def test_emergency_agent_ungrounded_when_no_citations():
    p = patient(severity="critical")
    result = run_emergency_routing_agent(p, [], {"department_load": "normal"})

    assert result.policy_grounded is False
    assert result.confidence_score < 0.70
