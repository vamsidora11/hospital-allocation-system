"""RESULT_CONSOLIDATION (plan Section 8).

Merges agent outputs into one recommendation for combination requests, and detects the
conflict case the plan calls out explicitly: the bed agent recommends ICU but the staff agent
can't confirm ICU-level nursing coverage. A conflict forces human review regardless of
individual confidence scores (checked in confidence_eval.py alongside `force_human_review`).
"""

from app.graph.state import WorkflowState
from app.models.domain import AgentResult
from app.services.audit import make_entry

# Below this, the staff agent's own output says it couldn't confirm the ratio/credential
# coverage a bed-type-ICU decision requires (see app/agents/staff_allocation.py's coverage
# shortfall path, base_confidence 0.52 in that branch) — not a generic "low confidence"
# threshold, specifically "can this recommendation be trusted to coordinate with the bed
# decision."
STAFF_COVERAGE_CONFIRMATION_FLOOR = 0.70


def detect_conflict(agent_results: list[AgentResult]) -> bool:
    bed = next((r for r in agent_results if r.agent_name == "bed_allocation"), None)
    staff = next((r for r in agent_results if r.agent_name == "staff_allocation"), None)

    if bed is None or staff is None:
        return False

    if bed.decision.get("bed_type") == "icu" and staff.confidence_score < STAFF_COVERAGE_CONFIRMATION_FLOOR:
        return True

    return False


def consolidation_node(state: WorkflowState) -> dict:
    agent_results = state["agent_results"]
    conflict_detected = detect_conflict(agent_results)

    if conflict_detected:
        summary = (
            "Bed allocation recommends ICU but staff allocation cannot confirm ICU-level "
            "nursing coverage — flagged as a conflict, forcing human review."
        )
    elif len(agent_results) > 1:
        summary = f"Combined {len(agent_results)} agent output(s) into a single recommendation. No conflicts detected."
    else:
        summary = "Single-agent recommendation carried forward without conflict."

    entry = make_entry(
        request_id=state["request_id"],
        state="RESULT_CONSOLIDATION",
        input_snapshot={"agent_names": [r.agent_name for r in agent_results]},
        output_snapshot={"summary": summary, "conflict_detected": conflict_detected},
    )
    return {"conflict_detected": conflict_detected, "audit_trail": [entry]}
