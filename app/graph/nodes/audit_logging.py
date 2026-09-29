from app.graph.state import WorkflowState
from app.services.audit import make_entry


def audit_logging_node(state: WorkflowState) -> dict:
    entry = make_entry(
        request_id=state["request_id"],
        state="AUDIT_LOGGING",
        input_snapshot={"prior_entries": len(state["audit_trail"])},
        output_snapshot={"complete": True},
    )
    return {"audit_trail": [entry]}
