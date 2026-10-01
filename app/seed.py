"""Seed synthetic bed/staff/patient/policy-document rows for local dev and tests.

Run with: py -3 -m app.seed
All identifiers here are synthetic per CLAUDE.md's data-handling constraint.
Bed, staff, and patient rows are loaded from data/seed/*.csv rather than inlined here so the
seed set stays in one place (README_DATA.md documents its shape and provenance).
"""

import csv
from pathlib import Path

from app.db import SessionLocal
from app.models.orm import Bed, Patient, PolicyDocument, Staff

REPO_ROOT = Path(__file__).resolve().parents[1]
BEDS_CSV = REPO_ROOT / "data" / "seed" / "beds.csv"
STAFF_CSV = REPO_ROOT / "data" / "seed" / "staff.csv"
PATIENTS_CSV = REPO_ROOT / "data" / "seed" / "synthetic_patient_requests.csv"

EXPECTED_BED_COUNT = 120
EXPECTED_STAFF_COUNT = 48
EXPECTED_PATIENT_COUNT = 98


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_beds() -> list[dict]:
    rows = _read_csv(BEDS_CSV)
    return [
        {
            "id": r["id"],
            "bed_type": r["bed_type"],
            "ward": r["ward"],
            "floor": int(r["floor"]),
            "status": r["status"],
        }
        for r in rows
    ]


def _load_staff() -> list[dict]:
    rows = _read_csv(STAFF_CSV)
    return [
        {
            "id": r["id"],
            "role": r["role"],
            "specialty": r["specialty"] or None,
            "shift": r["shift"],
            "status": r["status"],
        }
        for r in rows
    ]


def _load_patients() -> list[dict]:
    rows = _read_csv(PATIENTS_CSV)
    return [
        {
            "patient_ref": r["patient_ref"],
            "age": int(r["age"]),
            "diagnosis_category": r["diagnosis_category"],
            "severity": r["severity"],
            "requires_isolation": r["requires_isolation"].strip().lower() == "true",
            "requires_monitoring_level": r["requires_monitoring_level"],
        }
        for r in rows
    ]


POLICY_DOCUMENTS = [
    {"name": "Bed allocation policy", "source_path": "data/policies/bed_allocation_policy.md"},
    {"name": "Staffing policy", "source_path": "data/policies/staffing_policy.md"},
    {"name": "Emergency protocols", "source_path": "data/policies/emergency_protocols.md"},
    {"name": "Infection control standards", "source_path": "data/policies/infection_control_standards.md"},
    {"name": "Compliance rules", "source_path": "data/policies/compliance_rules.md"},
]


def seed() -> None:
    session = SessionLocal()
    try:
        if session.query(Bed).count() == 0:
            session.add_all(Bed(**b) for b in _load_beds())
        if session.query(Staff).count() == 0:
            session.add_all(Staff(**s) for s in _load_staff())
        if session.query(Patient).count() == 0:
            session.add_all(Patient(**p) for p in _load_patients())
        if session.query(PolicyDocument).count() == 0:
            session.add_all(PolicyDocument(version="1.0", **d) for d in POLICY_DOCUMENTS)
        session.commit()

        bed_count = session.query(Bed).count()
        staff_count = session.query(Staff).count()
        patient_count = session.query(Patient).count()
        print(
            f"Seeded {bed_count} beds, {staff_count} staff, {patient_count} patients, "
            f"{session.query(PolicyDocument).count()} policy documents."
        )
        assert bed_count == EXPECTED_BED_COUNT, f"expected {EXPECTED_BED_COUNT} beds, got {bed_count}"
        assert staff_count == EXPECTED_STAFF_COUNT, f"expected {EXPECTED_STAFF_COUNT} staff, got {staff_count}"
        assert patient_count == EXPECTED_PATIENT_COUNT, (
            f"expected {EXPECTED_PATIENT_COUNT} patients, got {patient_count}"
        )
    finally:
        session.close()


if __name__ == "__main__":
    seed()
