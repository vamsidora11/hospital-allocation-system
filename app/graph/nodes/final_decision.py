from app.graph.state import WorkflowState
from app.services.audit import make_entry


def _headline(agent_results) -> str:
    bed = next((r for r in agent_results if r.agent_name == "bed_allocation"), None)
    if bed is not None:
        return f"{bed.decision['bed_type']} bed assigned"
    return agent_results[0].reasoning[:80]


def final_decision_node(state: WorkflowState) -> dict:
    if state["workflow_status"] == "rejected":
        recommendation = {"headline": "Request rejected", "reasoning": "Closed on reviewer rejection."}
    else:
        agent_results = state["agent_results"]
        recommendation = {
            "headline": _headline(agent_results),
            "reasoning": " ".join(f"{r.agent_name}: {r.reasoning}" for r in agent_results),
            "decisions": {r.agent_name: r.decision for r in agent_results},
        }

    entry = make_entry(
        request_id=state["request_id"],
        state="FINAL_DECISION",
        input_snapshot={"workflow_status": state["workflow_status"]},
        output_snapshot=recommendation,
    )

    return {
        "final_recommendation": recommendation,
        "workflow_status": "decided",
        "audit_trail": [entry],
    }
