"""Seed synthetic bed/staff/policy-document rows for local dev and tests.

Run with: py -3 -m app.seed
All identifiers here are synthetic per CLAUDE.md's data-handling constraint.
"""

from app.db import SessionLocal
from app.models.orm import Bed, PolicyDocument, Staff

BEDS = [
    *[{"bed_type": "icu", "ward": "ICU-A", "floor": 3} for _ in range(6)],
    *[{"bed_type": "general", "ward": "Ward-2E", "floor": 2} for _ in range(20)],
    *[{"bed_type": "isolation", "ward": "Isolation-1", "floor": 1} for _ in range(5)],
]

STAFF = [
    {"role": "nurse", "specialty": "icu", "shift": "day"},
    {"role": "nurse", "specialty": "icu", "shift": "night"},
    {"role": "nurse", "specialty": "respiratory_infectious", "shift": "day"},
    {"role": "nurse", "specialty": "cardiac", "shift": "day"},
    {"role": "nurse", "specialty": "general", "shift": "day"},
    {"role": "nurse", "specialty": "general", "shift": "night"},
    {"role": "physician", "specialty": "cardiology", "shift": "day"},
    {"role": "physician", "specialty": "emergency", "shift": "day"},
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
            session.add_all(Bed(status="available", **b) for b in BEDS)
        if session.query(Staff).count() == 0:
            session.add_all(Staff(status="available", **s) for s in STAFF)
        if session.query(PolicyDocument).count() == 0:
            session.add_all(PolicyDocument(version="1.0", **d) for d in POLICY_DOCUMENTS)
        session.commit()
        print(
            f"Seeded {session.query(Bed).count()} beds, "
            f"{session.query(Staff).count()} staff, "
            f"{session.query(PolicyDocument).count()} policy documents."
        )
    finally:
        session.close()


if __name__ == "__main__":
    seed()
