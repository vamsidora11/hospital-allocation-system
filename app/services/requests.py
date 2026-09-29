"""Creates the `patients`/`requests` rows a workflow run's audit trail attaches to.

Full request-lifecycle management (status transitions, closing) is Phase 9's API layer;
this is just the minimal, reusable piece Phase 8 needs to persist audit entries against a
real FK-valid request_id.
"""

from sqlalchemy.orm import Session

from app.models.domain import PatientInfo
from app.models.orm import Patient, Request


def create_request(session: Session, patient: PatientInfo, request_type: str | None = None) -> Request:
    existing_patient = session.get(Patient, patient.patient_ref)
    if existing_patient is None:
        session.add(
            Patient(
                patient_ref=patient.patient_ref,
                age=patient.age,
                diagnosis_category=patient.diagnosis_category,
                severity=patient.severity,
                requires_isolation=patient.requires_isolation,
                requires_monitoring_level=patient.requires_monitoring_level,
            )
        )

    request_row = Request(patient_ref=patient.patient_ref, request_type=request_type, status="running")
    session.add(request_row)
    session.commit()
    session.refresh(request_row)
    return request_row
