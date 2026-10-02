"""AgentCore Platform v1.0"""

# CMN-C2-298 - DomainWorkflowGraph (inner BaseGraph)
#
# The INNER graph of the two-layer nested architecture. It encapsulates the
# whole Google Agent Skills integration Q&A workflow:
#
#   START -> input_validate -> retrieve -> rerank_filter
#         -> generate_answer -> output_format -> END
#
# Called by IntegrationQAGraphNode.get_subgraph() (graph.py).
# get_output() shapes the sub_result dict consumed by merge_output() there.
#
# Rules enforced:
#   - Inherits BaseGraph (fully custom topology - no forced backbone)
#   - Implements all 7 BaseGraph ABC methods
#   - register_nodes() does NOT call super() (abstract in BaseGraph)
#   - register_nodes() instantiates every domain node with NO ctor args
#   - Does NOT register initialize / finalize (outer backbone concerns)
#   - get_output() designed together with IntegrationQAGraphNode.merge_output()
#   - No platform SDK imports
#   - Not placed under src/subagents/

from typing import Any, Dict

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import get_caller_context
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State, to_json


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for CMN-C2-298.

    Inherits BaseGraph directly for a fully custom node topology. Called by
    IntegrationQAGraphNode.get_subgraph() in graph.py, which passes the runtime
    settings into the constructor.

    Pipeline (linear):
        START
          -> input_validate  (InputValidateNode)  - parse + normalise the question
          -> retrieve        (RetrieveNode)       - keyword-score the 3 seeded KBs
          -> rerank_filter   (RerankFilterNode)   - boost / threshold / top_k cut
          -> generate_answer (GenerateAnswerNode) - grounded answer + structured extract
          -> output_format   (OutputFormatNode)   - final format + scope notice
          -> END

    All nodes are FunctionNode subclasses returning partial-dict state updates
    via execute(self, state) -> dict; they take no config parameter, so
    configuration reaches them through _extra_initial_state() below.
    initialize / finalize are outer backbone concerns - not registered here.
    """

    # -- Identity --------------------------------------------------------------

    @property
    def name(self) -> str:
        """Unique identifier for this inner graph."""
        return "cmn_c2_298_integration_qa_workflow"

    @property
    def state_schema(self) -> type:
        """TypedDict subclass shared across the inner and outer graph."""
        return State

    # -- Config validation -----------------------------------------------------

    def _validate_config(self) -> None:
        """Validate inner graph config before compilation.

        The forwarded `retrieval` block (top_k / score_threshold / kb_paths) is
        read per call by the domain nodes with safe defaults, so absence is
        non-fatal and validation is permissive rather than raising.
        """
        pass

    # -- Config + caller context forwarding into state -------------------------

    def _extra_initial_state(self) -> Dict[str, Any]:
        """Seed the inner initial state with the settings and caller context.

        Two things travel through this hook, because neither reaches an inner
        node any other way:

        `retrieval_config` - IntegrationQAGraphNode._parent_config() forwards
        the runtime `retrieval` block under config["configurable"]; this hook
        republishes it as a JSON-string state field, which is the only channel
        left once nodes lost their per-call config parameter. RetrieveNode and
        RerankFilterNode read it (module defaults as the fallback).

        `input_context` - the framework's GraphNode invokes this graph without
        forwarding the outer state's input_context, so the caller's request
        context is handed over through the ContextVar bridge that
        IntegrationQAGraphNode.extract_input() writes immediately before the
        invoke (see src/graph/context_bridge.py).
        """
        retrieval = (self.config or {}).get("configurable", {}).get("retrieval") or {}
        return {
            "retrieval_config": to_json(retrieval),
            "input_context": get_caller_context(),
        }

    # -- Node registration -----------------------------------------------------

    def register_nodes(self) -> None:
        """Register all 5 domain nodes.

        No super() call - BaseGraph.register_nodes() is abstract. Do NOT
        register initialize or finalize; those are outer backbone concerns
        handled by AgentBaseGraph in graph.py.

        Every node is instantiated with NO constructor arguments - domain nodes
        are stateless; configuration flows in via _extra_initial_state() above.
        Every key registered here is referenced in add_edges().
        """
        self._nodes["input_validate"] = InputValidateNode()
        self._nodes["retrieve"] = RetrieveNode()
        self._nodes["rerank_filter"] = RerankFilterNode()
        self._nodes["generate_answer"] = GenerateAnswerNode()
        self._nodes["output_format"] = OutputFormatNode()

    # -- Edge wiring -----------------------------------------------------------

    def add_edges(self) -> None:
        """Wire the linear integration Q&A topology.

        Each step passes its partial-dict output into the shared State. The
        topology is intentionally linear - there is no conditional branching
        between domain nodes, so add_conditional_edges() is not used and
        route() below is never reached at runtime.
        """
        self._sg.add_edge(START, "input_validate")
        self._sg.add_edge("input_validate", "retrieve")
        self._sg.add_edge("retrieve", "rerank_filter")
        self._sg.add_edge("rerank_filter", "generate_answer")
        self._sg.add_edge("generate_answer", "output_format")
        self._sg.add_edge("output_format", END)

    # -- Routing ---------------------------------------------------------------

    def route(self, state: State) -> str:
        """Conditional routing - required by the BaseGraph ABC.

        The parameter is annotated with THIS graph's own State, not the shared
        base state type. The graph library reads a path callable's annotation
        as the callable's input schema and projects away every field the
        annotation does not declare - so a route annotated with the base type
        would be handed a state with all of this graph's domain fields removed,
        and any branch keyed on one of them would never be taken. That failure
        is invisible in unit tests, which call route() directly with a full
        mapping.

        This topology is linear, so add_conditional_edges() is not used and
        this method is never called at runtime; it is implemented to satisfy
        the ABC and annotated correctly so that wiring it later is safe.
        Returns END on error so an unexpected call does not re-enter a
        processing node.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return END
        return "output_format"

    # -- Output shape ----------------------------------------------------------

    def get_output(self, state: State) -> Dict[str, Any]:
        """Shape the output dict returned to the outer graph as sub_result.

        Received by IntegrationQAGraphNode.merge_output() in graph.py as its
        `sub_result` argument. Both methods are designed together to guarantee
        field-name consistency:

            inner get_output()   emits: "formatted_answer", "structured_answer",
                                        "status", ...
            outer merge_output() reads: sub_result.get("formatted_answer"),
                                        sub_result.get("structured_answer"),
                                        sub_result.get("status")

        The additional fields (intake_notes, trace_id, correlation_id,
        node_history) are surfaced for observability and downstream extension.
        """
        return {
            "formatted_answer": state.get("formatted_answer"),
            "structured_answer": state.get("structured_answer"),
            "status": state.get("status"),
            "intake_notes": state.get("intake_notes"),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
        }
