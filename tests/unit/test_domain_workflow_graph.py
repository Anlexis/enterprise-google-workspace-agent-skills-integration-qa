# CMN-C2-298 — Unit Tests: DomainWorkflowGraph (inner BaseGraph)
#
# Inner-graph composition plus a full inner invoke() over the seeded knowledge
# bases. The inner graph runs the 5 domain nodes, all at ANONYMOUS — the outer
# trust boundary is the backbone's concern, covered in
# test_graph_composition.py and tests/proof_of_boundary.
#
# Mirrors docs/03_test_spec.md Section 3.
# Deterministic — no model call, no network. framework.* / src.* imports only.

from langgraph.graph import END

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_status import AgentStatus

from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.graph import IntegrationQAGraphNode
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State, from_json

# Byte-equal to deploy/invoke_payload.json "input" / PB-6 _VALID_PAYLOAD.
_INTEGRATION_QUERY = (
    "How do I integrate the Gmail skill for our enterprise, and what APPI " "compliance steps are required?"
)


class TestInnerGraphConstruction:
    def test_int_01_inherits_base_graph(self):
        assert issubclass(DomainWorkflowGraph, BaseGraph)

    def test_int_01_registers_the_five_domain_nodes(self):
        inner = DomainWorkflowGraph()
        inner.register_nodes()
        assert set(inner._nodes.keys()) == {
            "input_validate",
            "retrieve",
            "rerank_filter",
            "generate_answer",
            "output_format",
        }
        assert isinstance(inner._nodes["input_validate"], InputValidateNode)
        assert isinstance(inner._nodes["retrieve"], RetrieveNode)
        assert isinstance(inner._nodes["rerank_filter"], RerankFilterNode)
        assert isinstance(inner._nodes["generate_answer"], GenerateAnswerNode)
        assert isinstance(inner._nodes["output_format"], OutputFormatNode)

    def test_inner_graph_name_and_schema(self):
        inner = DomainWorkflowGraph()
        assert inner.name == "cmn_c2_298_integration_qa_workflow"
        assert inner.state_schema is State

    def test_initialize_finalize_are_not_registered(self):
        # Outer backbone concerns must not leak into the inner topology.
        inner = DomainWorkflowGraph()
        inner.register_nodes()
        assert "initialize" not in inner._nodes
        assert "finalize" not in inner._nodes


class TestConfigForwarding:
    def test_int_02_extra_initial_state_republishes_retrieval_block(self):
        inner = DomainWorkflowGraph(config={"configurable": {"retrieval": {"top_k": 2}}})
        extra = inner._extra_initial_state()
        assert set(extra.keys()) == {"retrieval_config", "input_context"}
        # Structured State fields travel as JSON strings, never bare dicts.
        assert isinstance(extra["retrieval_config"], str)
        assert from_json(extra["retrieval_config"]) == {"top_k": 2}

    def test_int_02b_extra_initial_state_carries_the_bridged_caller_context(self):
        """The framework's GraphNode invokes this graph without forwarding the
        outer input_context, so the ContextVar bridge is the only way a caller
        filter reaches an inner node. Seeded here the way extract_input() does
        it immediately before the invoke."""
        from src.graph.context_bridge import set_caller_context

        set_caller_context({"skill": "gmail", "top_k": 3})
        try:
            extra = DomainWorkflowGraph()._extra_initial_state()
            assert extra["input_context"] == {"skill": "gmail", "top_k": 3}
        finally:
            set_caller_context({})

    def test_extra_initial_state_with_no_config_is_empty_block(self):
        assert from_json(DomainWorkflowGraph()._extra_initial_state()["retrieval_config"]) == {}

    def test_route_is_annotated_with_this_graphs_own_state(self):
        """The graph library reads a path callable's annotation as its input
        schema and projects away every field the annotation does not declare.
        A route annotated with the shared base state would therefore be handed
        a state with this graph's domain fields removed, and any branch keyed
        on one of them would never be taken — invisibly, because a unit test
        calls route() directly with a full mapping. The topology here is
        linear, but the annotation is kept correct so wiring it later is safe.
        """
        import typing

        from src.schemas.state import State

        hints = typing.get_type_hints(DomainWorkflowGraph.route)
        assert hints["state"] is State


class TestOutputShape:
    def test_int_03_get_output_shapes_the_merge_contract(self):
        inner = DomainWorkflowGraph()
        out = inner.get_output(
            {
                "formatted_answer": "ANSWER",
                "structured_answer": "{}",
                "status": AgentStatus.SUCCESS.value,
                "intake_notes": "[]",
                "trace_id": "t-1",
                "correlation_id": "c-1",
                "node_history": ["InputValidateNode"],
            }
        )
        assert out["formatted_answer"] == "ANSWER"
        assert out["structured_answer"] == "{}"
        assert out["status"] == AgentStatus.SUCCESS.value
        assert out["intake_notes"] == "[]"
        assert out["trace_id"] == "t-1"
        assert out["correlation_id"] == "c-1"
        assert out["node_history"] == ["InputValidateNode"]

    def test_route_returns_end_on_error(self):
        inner = DomainWorkflowGraph()
        assert inner.route({"status": AgentStatus.ERROR.value}) == END
        assert inner.route({"status": AgentStatus.SUCCESS.value}) == "output_format"


class TestInnerEndToEnd:
    def _invoke(self, payload: str) -> dict:
        # Same construction path the outer GraphNode uses: manifest-derived
        # config via _parent_config(); domain nodes take NO ctor args.
        inner = DomainWorkflowGraph(config=IntegrationQAGraphNode()._parent_config())
        return inner.invoke(payload, session_id="inner-e2e")

    def test_int_04_full_inner_run_produces_the_formatted_answer(self):
        result = self._invoke(_INTEGRATION_QUERY)
        assert result["status"] == AgentStatus.SUCCESS.value
        answer = result["formatted_answer"]
        assert answer.startswith("# Google Agent Skills Integration Guidance")
        assert "[1]" in answer and "[2]" in answer and "[3]" in answer and "[4]" in answer
        # Verified retrieval outcome for this exact payload by actually
        # running DomainWorkflowGraph against the real config/kb/*.json
        # content (not hand-simulated): gas-001 + gas-003 + jc-001 + jc-002
        # (4 hits — gas-003's "Docs Agent Skill" content contains "required",
        # overlapping a query token). Note get_output() here does NOT expose
        # a top-level "citations" key (that shaping happens one layer up, in
        # the outer get_output() override) — citations live inside
        # structured_answer at this inner layer.
        structured = from_json(result["structured_answer"])
        assert len(structured["citations"]) == 4
        assert {c["id"] for c in structured["citations"]} == {
            "gas-001",
            "gas-003",
            "jc-001",
            "jc-002",
        }

    def test_int_04_inner_node_history_is_the_linear_topology(self):
        history = self._invoke(_INTEGRATION_QUERY)["node_history"]
        assert history == [
            "InputValidateNode",
            "RetrieveNode",
            "RerankFilterNode",
            "GenerateAnswerNode",
            "OutputFormatNode",
        ]

    def test_no_coverage_query_still_terminates_success(self):
        result = self._invoke("purple elephant zeppelin recipes")
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "does not contain sufficient coverage" in result["formatted_answer"]
