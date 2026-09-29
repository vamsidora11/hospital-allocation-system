"""RESULT_CONSOLIDATION (plan Section 8).

Phase 5 keeps this a straightforward merge; real conflict detection between agents lands in
Phase 6 (its own plan section/phase) — `conflict_detected` here is always False for now.
"""

from app.graph.state import WorkflowState
from app.services.audit import make_entry


def consolidation_node(state: WorkflowState) -> dict:
    agent_results = state["agent_results"]
    summary = (
        f"Combined {len(agent_results)} agent output(s) into a single recommendation."
        if len(agent_results) > 1
        else "Single-agent recommendation carried forward without conflict."
    )

    entry = make_entry(
        request_id=state["request_id"],
        state="RESULT_CONSOLIDATION",
        input_snapshot={"agent_names": [r.agent_name for r in agent_results]},
        output_snapshot={"summary": summary, "conflict_detected": False},
    )
    return {"audit_trail": [entry]}
