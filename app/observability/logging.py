"""Structured logging (plan Section 12): one JSON log line per state transition, correlated
by `request_id`. Call `configure_logging()` once at process startup (app/main.py).
"""

import logging
import sys

import structlog


def configure_logging() -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger():
    return structlog.get_logger()


def log_state_transition(request_id: str, state: str, **fields) -> None:
    """One structured log line per graph state transition, per Section 12."""
    get_logger().info("workflow_state_transition", request_id=request_id, state=state, **fields)
