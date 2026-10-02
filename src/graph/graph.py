"""AgentCore Platform v1.0"""

# CMN-C2-298 - outer graph (AgentBaseGraph; two-layer nested Cat 2 architecture)
#
# WorkspaceSkillsIntegrationQAAgent - retrieval-augmented Q&A over three
# seeded, pre-indexed knowledge bases (Google Agent Skills documentation,
# enterprise integration patterns, Japan compliance requirements) that
# answers *how to integrate* Google's official Agent Skills into a Japanese
# enterprise. It never connects to, authenticates against, or calls any
# Google API, and makes no configuration changes.
#
# Architecture:
#
#   Outer backbone (fixed - do NOT override add_edges()):
#     START -> initialize -> pre_process -> main -> {route} -> post_process -> finalize -> END
#                                             |  (retry)
#                                             -> pre_process
#
#   The `main` slot is a GraphNode subclass (IntegrationQAGraphNode) that
#   delegates the whole integration-Q&A workflow to DomainWorkflowGraph
#   (inner BaseGraph: input_validate -> retrieve -> rerank_filter ->
#   generate_answer -> output_format).
#
#   Domain complexity is fully encapsulated inside the inner graph; the outer
#   backbone is never modified.
#
# Directory layout:
#   src/graph/graph.py                 <- outer graph (this file)
#   src/graph/domain_workflow_graph.py <- inner graph (multi-step topology)
#   src/graph/context_bridge.py        <- carries input_context across the boundary
#
# Class-name contract:
#   graph.py class:           WorkspaceSkillsIntegrationQAAgent (this file)
#   config/agent.yaml class:  "src.graph.graph.WorkspaceSkillsIntegrationQAAgent"
#   src/api/server.py import: from src.graph.graph import WorkspaceSkillsIntegrationQAAgent
#
# Rules enforced:
#   - WorkspaceSkillsIntegrationQAAgent inherits AgentBaseGraph (L1 Base - direct inheritance)
#   - super().register_nodes() called first (fills initialize + finalize)
#   - IntegrationQAGraphNode assigned to self._nodes["main"]
#   - _parent_config() forwards the runtime retrieval settings (never {})
#   - merge_output() returns only changed keys
#   - add_edges() NOT overridden on the outer graph
#   - get_output() EXTENDS super().get_output(); structured keys surfaced on
#     SUCCESS only, fail-closed
#   - No platform SDK imports

from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import set_caller_context
from src.nodes.post_process_node import PostProcessNode, _security_gate_output
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State, from_json

# Runtime settings path: src/graph/graph.py -> parents[2] = repo root.
# config/config.yaml holds the RUNTIME parameters; config/agent.yaml is the
# static registration manifest and carries no tuning values.
_RUNTIME_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"

# Fallbacks mirror the `retrieval` block in config/config.yaml so
# _parent_config() never forwards an empty mapping even if the settings file
# is unreadable in an exotic deployment layout.
_FALLBACK_RETRIEVAL: Dict[str, Any] = {
    "top_k": 6,
    "score_threshold": 0.15,
    "kb_paths": [
        "config/kb/google_agent_skills_kb.json",
        "config/kb/integration_patterns_kb.json",
        "config/kb/japan_compliance_kb.json",
    ],
}

# Structured-answer keys surfaced by get_output() below - a fixed allowlist.
# Nothing outside this list is ever surfaced, even if structured_answer
# somehow carried an extra key.
_STRUCTURED_KEYS = (
    "integration_steps",
    "oauth_scopes",
    "compliance_flags",
    "testing_checklist",
    "related_skills",
    "citations",
)


def runtime_config() -> Dict[str, Any]:
    """Read the runtime parameters from config/config.yaml.

    This is the same file the platform registry loads and hands to the graph
    as ``Graph(config=...)``; the standalone server (src/api/server.py) reads
    it here so a registry-loaded agent and a standalone one see identical
    settings. Returns an empty mapping - never raises - when the file is
    absent, unreadable, not valid YAML, or not a mapping; the graph then runs
    on the built-in defaults below.
    """
    try:
        import yaml

        loaded = yaml.safe_load(_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


class IntegrationQAGraphNode(GraphNode):
    """GraphNode subclass assigned to the `main` slot of the outer agent.

    Wraps DomainWorkflowGraph (the inner retrieval pipeline). Called by the
    backbone after pre_process and before post_process.

    Contracts:
      get_subgraph()    - instantiate DomainWorkflowGraph with the forwarded
                          runtime settings (_parent_config())
      extract_input()   - pull validated_input (identifier-stripped) from outer
                          state, and stash the caller's request context for the
                          inner graph (see src/graph/context_bridge.py)
      merge_output()    - map sub_result fields into the outer state delta
                          (changed keys only)
      error_strategy    - "propagate": re-raise inner errors as SubgraphError

    execute() is NOT overridden - the base GraphNode.execute() already
    implements the subgraph invoke plus the merge_output() call.
    """

    # "propagate": re-raise inner graph exceptions as SubgraphError (fail fast).
    # "handle": call on_subgraph_error() instead - for graceful degradation.
    error_strategy: ClassVar[str] = "propagate"

    # False: interrupts are handled inside the inner graph only.
    propagate_hitl: ClassVar[bool] = False

    def _parent_config(self) -> Dict[str, Any]:
        """Forward the runtime `retrieval` settings to the inner graph.

        Reads config/config.yaml and returns the tuning block under
        ``config["configurable"]`` - never an empty mapping. The inner graph
        republishes it into inner state (DomainWorkflowGraph.
        _extra_initial_state()) so RetrieveNode and RerankFilterNode read live
        top_k / score_threshold / kb_paths values rather than dead
        declarations.
        """
        settings = runtime_config()
        retrieval = settings.get("retrieval")
        if not isinstance(retrieval, dict) or not retrieval:
            retrieval = dict(_FALLBACK_RETRIEVAL)
        return {"configurable": {"retrieval": retrieval}}

    def get_subgraph(self) -> "Any":
        """Instantiate and return the inner domain workflow graph.

        DomainWorkflowGraph is imported inside the method to avoid a circular
        import at module load time.

        The inner graph receives the runtime settings via its BaseGraph
        constructor; its domain NODES still take no constructor arguments and
        read configuration per call from State exclusively.
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._parent_config())

    def extract_input(self, state: AgentState) -> str:
        """Return the string input passed into inner_graph.invoke().

        PreProcessNode validates and identifier-strips the raw user_input and
        writes the result to validated_input. Prefer that; fall back to
        user_input when validated_input is absent (unit tests seeding state
        directly).

        This hook is also where the caller's request context is stashed for
        the inner graph: GraphNode.execute() calls extract_input() immediately
        before subgraph.invoke(), and that invoke does not forward
        input_context on its own.
        """
        set_caller_context(state.get("input_context") or {})
        value = state.get("validated_input") or state.get("user_input", "")
        return value if isinstance(value, str) else ""

    def merge_output(self, state: AgentState, sub_result: Dict[str, Any]) -> Dict[str, Any]:
        """Map the inner graph's sub_result back into the outer state delta.

        sub_result is the dict returned by DomainWorkflowGraph.get_output().
        Returns ONLY changed keys - never the full state.

        Key coupling (designed together with DomainWorkflowGraph.get_output()):
          inner get_output() emits  -> "formatted_answer", "structured_answer", "status", ...
          this merge_output() reads -> sub_result.get("formatted_answer"),
                                       sub_result.get("structured_answer"),
                                       sub_result.get("status")

        integration_answer: the final rendered integration answer, written by
          OutputFormatNode inside the inner graph.
        result: PostProcessNode (the outer post_process slot) reads
          state.get("result"), and the inner graph emits the rendered answer
          under "formatted_answer" - so it is mapped to "result" as well.
          Without that mapping the output the gate sees is always empty.
        structured_answer: forwarded as-is (still a JSON string) so
          PostProcessNode can gate it and get_output() below can surface it.
        status: terminal status value from the inner graph run.
        """
        delta: Dict[str, Any] = {
            "integration_answer": sub_result.get("formatted_answer"),
            "result": sub_result.get("formatted_answer"),
            "structured_answer": sub_result.get("structured_answer"),
            "status": sub_result.get("status"),
        }
        return delta


class WorkspaceSkillsIntegrationQAAgent(AgentBaseGraph):
    """Outer graph for CMN-C2-298.

    Inherits AgentBaseGraph directly (L1 Base). Domain logic is fully
    encapsulated in IntegrationQAGraphNode (the main slot), which delegates to
    DomainWorkflowGraph.

    Backbone (fixed):
        START -> initialize -> pre_process -> main -> post_process -> finalize -> END

    register_nodes() is the ONLY structural override:
      - super().register_nodes() fills: initialize, finalize (framework defaults)
      - pre_process:  PreProcessNode (trust gate, caller-contract validation,
                      injection screen, identifier strip)
      - main:         IntegrationQAGraphNode (delegates to DomainWorkflowGraph)
      - post_process: PostProcessNode (output gate)

    get_output() is ALSO overridden to surface the structured answer
    (integration steps / OAuth scopes / compliance flags / testing checklist /
    related skills / citations) on top of the base envelope.

    add_edges() is NOT overridden - backbone wiring belongs to the framework.
    """

    @property
    def name(self) -> str:
        """Agent identifier registered with the platform registry."""
        return "WorkspaceSkillsIntegrationQAAgent"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill all 5 backbone slots.

        super().register_nodes() MUST be called first - it injects the
        framework's default initialize node (schema_version, session_id, trust
        level) and finalize node (response metadata, total time).
        """
        super().register_nodes()  # fills: initialize, finalize

        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = IntegrationQAGraphNode()
        self._nodes["post_process"] = PostProcessNode()

    # add_edges() is NOT overridden - backbone wiring belongs to the framework.

    def get_output(self, state: AgentState) -> Dict[str, Any]:
        """EXTEND the base envelope with the structured answer.

        Surfaces integration_steps / oauth_scopes / compliance_flags /
        testing_checklist / related_skills / citations on top of the base
        {output, status, trace_id, correlation_id, node_history} envelope -
        never replaces it.

        Fail-closed, in order:
          1. Non-SUCCESS status -> base envelope only, no structured keys.
          2. structured_answer missing / not valid JSON / not a mapping ->
             base envelope only.
          3. The recursive output gate finds a violation in structured_answer
             -> base envelope only. This is a defence-in-depth re-check:
             PostProcessNode has already gated (and clears structured_answer on
             a violation, which step 2 catches) - this step protects the true
             response boundary even if that upstream gate were ever bypassed.
          4. Otherwise: allowlist EXACTLY the six known keys - nothing else in
             the object is surfaced, even if it somehow carried an extra key.
        """
        # The framework base is untyped, so the envelope arrives as Any; bind
        # it to a concrete mapping before extending it.
        base: Dict[str, Any] = dict(super().get_output(state))
        if base.get("status") != AgentStatus.SUCCESS.value:
            return base

        structured = from_json(state.get("structured_answer"), None)
        if not isinstance(structured, dict):
            return base

        if _security_gate_output(structured) is not None:
            return base

        base.update({key: structured.get(key, []) for key in _STRUCTURED_KEYS})
        return base


def build_agent(config: Optional[Dict[str, Any]] = None) -> WorkspaceSkillsIntegrationQAAgent:
    """Construct the agent with the runtime settings from config/config.yaml.

    The registry passes the loaded settings itself; this helper gives the
    standalone entry point the same construction so the two deployments are
    not configured differently.
    """
    return WorkspaceSkillsIntegrationQAAgent(config=config if config is not None else runtime_config())
