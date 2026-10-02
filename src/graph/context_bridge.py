"""AgentCore Platform v1.0"""

# CMN-C2-298 - carries the caller's request context across the outer -> inner
# graph boundary.
#
# Why this module exists: the framework's GraphNode invokes the inner graph as
# `subgraph.invoke(user_input, session_id=..., ctx=...)`. That call forwards
# the question string and the invocation context, but NOT the outer state's
# `input_context`, so an inner node reading `state["input_context"]` would see
# an empty mapping on every real invocation while unit tests that seed the
# inner state directly kept passing. The two sanctioned subclass hooks bridge
# it without touching the framework:
#
#   IntegrationQAGraphNode.extract_input(state)     [runs BEFORE subgraph.invoke]
#       -> set_caller_context(state["input_context"])
#   DomainWorkflowGraph._extra_initial_state()      [runs INSIDE subgraph.invoke]
#       -> returns {"input_context": get_caller_context()}
#
# A ContextVar rather than a module global: it is per-thread and per-task, so
# two invocations running concurrently in one process cannot read each other's
# context.

from contextvars import ContextVar
from typing import Any, Dict, Optional

_CALLER_CONTEXT: ContextVar[Optional[Dict[str, Any]]] = ContextVar("cmn_c2_298_caller_context", default=None)


def set_caller_context(input_context: Optional[Dict[str, Any]]) -> None:
    """Stash the outer graph's request context for the imminent inner invoke."""
    _CALLER_CONTEXT.set(dict(input_context) if isinstance(input_context, dict) else {})


def get_caller_context() -> Dict[str, Any]:
    """Read (without consuming) the stashed context; {} when none was set."""
    return _CALLER_CONTEXT.get() or {}
