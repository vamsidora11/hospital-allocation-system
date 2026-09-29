"""HUMAN_APPROVAL_GATE (plan Section 5/8).

Auto-approves when CONFIDENCE_EVALUATION found no reason for review. Otherwise pauses the
graph via LangGraph's `interrupt()` — a first-class state transition with its own audit entry,
not a side workflow — awaiting one of the four reviewer actions:
- `approve` / `override`: proceeds to FINAL_DECISION. `override` additionally carries the
  reviewer's substituted decision (`override_decision` in the resume payload), stored
  *alongside* the original `agent_results`, never replacing them (plan Section 8).
- `reject`: still proceeds to FINAL_DECISION/AUDIT_LOGGING ("logged, closed" — plan Section 5)
  but produces a "Request rejected" recommendation instead of an allocation.
- `request_info`: routes back to AGENT_EXECUTION to re-run with the same active agents.
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

    if state.get("force_human_review"):
        reason = "Agent execution fell back to a conservative default after exhausting retries."
    elif state.get("conflict_detected"):
        reason = "Agents disagree: bed and staff recommendations don't coordinate."
    else:
        reason = "Aggregate confidence is below the human-approval threshold."

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
        return {
            "workflow_status": "approved",
            "approval_status": action,
            "override_decision": approval.get("override_decision") if action == "override" else None,
            "audit_trail": [entry],
        }
    if action == "reject":
        return {"workflow_status": "rejected", "approval_status": "reject", "audit_trail": [entry]}
    # request_info: caller (build_graph's conditional edge) routes back to AGENT_EXECUTION
    return {"workflow_status": "needs_more_info", "approval_status": "request_info", "audit_trail": [entry]}
