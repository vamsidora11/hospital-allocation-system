from app.graph.state import WorkflowState
from app.services.audit import make_entry


def input_validation(state: WorkflowState) -> dict:
    patient = state["patient_information"]
    valid = bool(patient.presenting_complaint.strip()) and patient.patient_ref.startswith("PT-")

    status = "validated" if valid else "rejected"
    entry = make_entry(
        request_id=state["request_id"],
        state="INPUT_VALIDATION",
        input_snapshot=patient.model_dump(mode="json"),
        output_snapshot={"valid": valid},
    )
    return {"workflow_status": status, "audit_trail": [entry]}
