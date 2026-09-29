"""CONFIDENCE_EVALUATION (plan Section 8).

Aggregation defaults to the *minimum* across active agents' confidence scores — a single
low-confidence agent pulls the whole decision below threshold. Also routes to human review,
regardless of the resulting confidence number, when either:
- AGENT_EXECUTION had to fall back after exhausting retries (plan Section 5), or
- RESULT_CONSOLIDATION flagged a conflict between agents (plan Section 8).
"""

from app.config import get_settings
from app.graph.state import WorkflowState
from app.services.audit import make_entry


def confidence_eval_node(state: WorkflowState) -> dict:
    agent_results = state["agent_results"]
    aggregate_confidence = min((r.confidence_score for r in agent_results), default=0.0)

    threshold = get_settings().human_approval_threshold
    conflict_detected = state.get("conflict_detected", False)
    needs_approval = (
        aggregate_confidence < threshold or state.get("force_human_review", False) or conflict_detected
    )

    entry = make_entry(
        request_id=state["request_id"],
        state="CONFIDENCE_EVALUATION",
        input_snapshot={"per_agent": {r.agent_name: r.confidence_score for r in agent_results}},
        output_snapshot={
            "aggregate_confidence": aggregate_confidence,
            "threshold": threshold,
            "conflict_detected": conflict_detected,
            "needs_approval": needs_approval,
        },
    )

    return {
        "confidence_scores": {"aggregate": aggregate_confidence},
        "approval_status": "pending" if needs_approval else "auto_approved",
        "audit_trail": [entry],
    }
