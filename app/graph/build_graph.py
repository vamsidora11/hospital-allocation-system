"""Assembles the 8-state workflow graph (plan Section 5)."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph.nodes.agent_execution import dispatch_agents, run_agent
from app.graph.nodes.audit_logging import audit_logging_node
from app.graph.nodes.confidence_eval import confidence_eval_node
from app.graph.nodes.consolidation import consolidation_node
from app.graph.nodes.final_decision import final_decision_node
from app.graph.nodes.human_approval import human_approval_node
from app.graph.nodes.input_validation import input_validation
from app.graph.nodes.router import router_node
from app.graph.state import WorkflowState
from app.models.domain import PatientInfo


def initial_state(request_id: str, patient: PatientInfo) -> WorkflowState:
    """The zero-value `WorkflowState` a new run starts from."""
    return {
        "request_id": request_id,
        "patient_information": patient,
        "request_type": None,
        "workflow_status": "new",
        "active_agents": [],
        "agent_results": [],
        "retrieved_policies": [],
        "confidence_scores": {},
        "approval_status": None,
        "final_recommendation": None,
        "audit_trail": [],
        "retry_count": 0,
        "force_human_review": False,
        "conflict_detected": False,
        "override_decision": None,
    }


def _route_after_validation(state: WorkflowState) -> str:
    return "router" if state["workflow_status"] == "validated" else END


def _route_after_approval(state: WorkflowState):
    status = state["workflow_status"]
    if status in ("approved", "rejected"):
        # Rejection is still "logged, closed" (plan Section 5) — final_decision builds
        # the "Request rejected" recommendation and audit_logging still runs, it just
        # doesn't reach an allocation decision.
        return "final_decision"
    # needs_more_info: re-run AGENT_EXECUTION with the same active_agents
    return dispatch_agents(state)


def build_graph(checkpointer=None):
    """`checkpointer` defaults to an in-memory saver — this sandbox has no Postgres (see
    CLAUDE.md); swap in a Postgres-backed checkpointer for anything beyond local dev/tests."""
    graph = StateGraph(WorkflowState)

    graph.add_node("input_validation", input_validation)
    graph.add_node("router", router_node)
    graph.add_node("run_agent", run_agent)
    graph.add_node("consolidation", consolidation_node)
    graph.add_node("confidence_eval", confidence_eval_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("final_decision", final_decision_node)
    graph.add_node("audit_logging", audit_logging_node)

    graph.add_edge(START, "input_validation")
    graph.add_conditional_edges("input_validation", _route_after_validation, ["router", END])
    graph.add_conditional_edges("router", dispatch_agents, ["run_agent"])
    graph.add_edge("run_agent", "consolidation")
    graph.add_edge("consolidation", "confidence_eval")
    graph.add_edge("confidence_eval", "human_approval")
    graph.add_conditional_edges("human_approval", _route_after_approval, ["final_decision", "run_agent", END])
    graph.add_edge("final_decision", "audit_logging")
    graph.add_edge("audit_logging", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())
