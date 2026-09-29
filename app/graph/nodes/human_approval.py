"""HUMAN_APPROVAL_GATE (plan Section 5/8).

Auto-approves when CONFIDENCE_EVALUATION found no reason for review. Otherwise pauses the
graph via LangGraph's `interrupt()` — a first-class state transition with its own audit entry,
not a side workflow — awaiting one of the four reviewer actions (`approve`/`reject`/
`override`/`request_info`). Phase 5 only exercises the auto-approve and pending/fallback
branches in its own tests; resuming the interrupt with each of the four actions is Phase 7's
definition of done.
"""

from langgraph.types import interrupt

from app.graph.state import WorkflowState
from app.services.audit import make_entry


def human_approval_node(state: WorkflowState) -> dict:
    if state["approval_status"] == "auto_approved":
        entry = make_entry(
            request_id=state["request_id"],
            state="HUMAN_APPROVAL_GATE",
            input_snapshot={"approval_status": "auto_approved"},
            output_snapshot={"decision": "auto_approved"},
        )
        return {"workflow_status": "approved", "audit_trail": [entry]}

    reason = (
        "Agent execution fell back to a conservative default after exhausting retries."
        if state.get("force_human_review")
        else "Aggregate confidence is below the human-approval threshold."
    )

    approval = interrupt({"request_id": state["request_id"], "reason": reason})
    action = approval["action"]

    entry = make_entry(
        request_id=state["request_id"],
        state="HUMAN_APPROVAL_GATE",
        input_snapshot={"reason": reason},
        output_snapshot={"action": action, "reviewer_id": approval.get("reviewer_id")},
        actor="human",
    )

    if action in ("approve", "override"):
        return {"workflow_status": "approved", "approval_status": action, "audit_trail": [entry]}
    if action == "reject":
        return {"workflow_status": "rejected", "approval_status": "reject", "audit_trail": [entry]}
    # request_info: caller (build_graph's conditional edge) routes back to AGENT_EXECUTION
    return {"workflow_status": "needs_more_info", "approval_status": "request_info", "audit_trail": [entry]}
