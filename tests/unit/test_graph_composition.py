# CMN-C2-298 — Unit Tests: nested Cat-2 graph composition (outer + end-to-end)
#
# Drives the REAL outer agent (WorkspaceSkillsIntegrationQAAgent) end-to-end
# via AgentBaseGraph.invoke(). The e2e context is
# InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL) — the
# manifest's declared caller level; for_internal() is NEVER used (it would
# over-privilege the run and hide trust-gate regressions).
#
# _AML_QUERY below is byte-equal to deploy/invoke_payload.json "input" (also
# the PB-6 _VALID_PAYLOAD) — verified by actually running DomainWorkflowGraph
# against the real config/kb/*.json content (not hand-simulated) to retrieve
# exactly 4 ranked passages: gas-001 (Gmail Agent Skill), gas-003 (Docs Agent
# Skill — "required" in its content overlaps a query token), jc-001 and
# jc-002 (both APPI). Assertions below reflect that real, verified retrieval
# outcome rather than a guessed one.
#
# Mirrors docs/03_test_spec.md Section 3 (INT-05..INT-14).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import pathlib

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

import src.graph.graph
from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.graph import IntegrationQAGraphNode, WorkspaceSkillsIntegrationQAAgent
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State, from_json, to_json

# Byte-equal to deploy/invoke_payload.json "input" / PB-6 _VALID_PAYLOAD.
_INTEGRATION_QUERY = (
    "How do I integrate the Gmail skill for our enterprise, and what APPI " "compliance steps are required?"
)


def _run(user_input: str, trust: TrustLevel = TrustLevel.VERIFIED_EXTERNAL) -> dict:
    ctx = InvocationContext(caller_trust_level=trust, caller_id="unit-suite")
    return WorkspaceSkillsIntegrationQAAgent().invoke(user_input=user_input, ctx=ctx)


class TestOuterGraphConstruction:
    def test_int_05_inherits_agent_base_graph_directly(self):
        assert issubclass(WorkspaceSkillsIntegrationQAAgent, AgentBaseGraph)

    def test_state_schema_is_state(self):
        assert WorkspaceSkillsIntegrationQAAgent().state_schema is State

    def test_int_06_compile_fills_all_backbone_slots(self):
        agent = WorkspaceSkillsIntegrationQAAgent()
        agent.compile()
        for slot in ("initialize", "pre_process", "main", "post_process", "finalize"):
            assert agent._nodes.get(slot) is not None, f"backbone slot not filled: {slot}"
        assert isinstance(agent._nodes["pre_process"], PreProcessNode)
        assert isinstance(agent._nodes["main"], IntegrationQAGraphNode)
        assert isinstance(agent._nodes["post_process"], PostProcessNode)

    def test_add_edges_is_not_overridden(self):
        # Backbone wiring belongs to the framework — the template must not
        # redefine it.
        assert "add_edges" not in WorkspaceSkillsIntegrationQAAgent.__dict__


class TestMainSlotGraphNode:
    def test_int_07_get_subgraph_returns_the_inner_graph(self):
        subgraph = IntegrationQAGraphNode().get_subgraph()
        assert isinstance(subgraph, DomainWorkflowGraph)
        assert subgraph.config["configurable"]["retrieval"][
            "kb_paths"
        ], "inner config must carry the retrieval kb_paths list"

    def test_int_08_extract_input_prefers_validated_input(self):
        node = IntegrationQAGraphNode()
        assert node.extract_input({"validated_input": "VI", "user_input": "UI"}) == "VI"
        assert node.extract_input({"user_input": "UI"}) == "UI"

    def test_int_09_merge_output_maps_the_inner_contract(self):
        node = IntegrationQAGraphNode()
        structured = to_json({"integration_steps": ["step"]})
        delta = node.merge_output(
            {},
            {"formatted_answer": "ANSWER", "structured_answer": structured, "status": AgentStatus.SUCCESS.value},
        )
        # formatted_answer surfaces as BOTH integration_answer and result
        # (PostProcessNode's output gate reads state["result"]).
        assert delta == {
            "integration_answer": "ANSWER",
            "result": "ANSWER",
            "structured_answer": structured,
            "status": AgentStatus.SUCCESS.value,
        }

    def test_error_strategy_is_propagate_and_hitl_is_contained(self):
        assert IntegrationQAGraphNode.error_strategy == "propagate"
        assert IntegrationQAGraphNode.propagate_hitl is False

    def test_int_10_parent_config_never_empty_without_settings_file(self, monkeypatch):
        # With an unreadable settings file the forwarded config still carries
        # the fallback retrieval block — never an empty mapping, which would
        # leave the inner nodes with nothing to read at all.
        monkeypatch.setattr(src.graph.graph, "_RUNTIME_CONFIG_PATH", pathlib.Path("/nonexistent/config.yaml"))
        cfg = IntegrationQAGraphNode()._parent_config()
        assert cfg["configurable"]["retrieval"]["kb_paths"] == [
            "config/kb/google_agent_skills_kb.json",
            "config/kb/integration_patterns_kb.json",
            "config/kb/japan_compliance_kb.json",
        ]


class TestEndToEndInvoke:
    """Full agent run: outer backbone + inner domain workflow, no LLM."""

    def test_int_11_invoke_returns_success(self):
        result = _run(_INTEGRATION_QUERY)
        assert (
            result.get("status") == AgentStatus.SUCCESS.value
        ), f"Expected success, got {result.get('status')}. result={result!r}"

    def test_int_12_output_is_the_gated_formatted_answer(self):
        output = _run(_INTEGRATION_QUERY).get("output")
        assert isinstance(output, str) and output.strip()
        assert output.startswith("# Google Agent Skills Integration Guidance")
        assert "[1]" in output and "[2]" in output and "[3]" in output and "[4]" in output
        assert "does NOT connect to, authenticate against, or call any Google" in output

    def test_int_13_structured_product_surfaced_on_success(self):
        result = _run(_INTEGRATION_QUERY)
        # get_output() extends the base envelope with the allowlisted
        # structured keys, on success only.
        assert result["oauth_scopes"], "expected an oauth scope from the gmail-skill KB hit"
        assert any("gmail" in str(s.get("scope", "")) for s in result["oauth_scopes"])
        assert len(result["compliance_flags"]) == 2  # jc-001 + jc-002, both APPI
        assert all(f["framework"] == "APPI" for f in result["compliance_flags"])
        assert len(result["citations"]) == 4  # gas-001, gas-003, jc-001, jc-002

    def test_int_14_e2e_traverses_the_post_process_gate(self):
        history = _run(_INTEGRATION_QUERY).get("node_history", [])
        for cls_name in ("PreProcessNode", "IntegrationQAGraphNode", "PostProcessNode"):
            assert cls_name in history, f"node_history missing {cls_name}: {history}"

    def test_no_coverage_query_still_terminates_success(self):
        result = _run("purple elephant zeppelin recipes")
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert "does not contain sufficient coverage" in result.get("output", "")

    def test_int_15_anonymous_caller_is_denied_at_the_outer_boundary(self):
        """The trust gate at graph level: an ANONYMOUS invoke is refused by the
        VERIFIED_EXTERNAL pre_process slot. No domain answer is ever
        produced and post_process is skipped."""
        result = _run(_INTEGRATION_QUERY, trust=TrustLevel.ANONYMOUS)
        assert result.get("status") == AgentStatus.ERROR.value
        assert not result.get("output")
        history = result.get("node_history", [])
        assert "PostProcessNode" not in history
        assert history[:2] == ["InitializeNode", "PreProcessNode"]


class TestStateRoundTrip:
    """The JSON-string helpers: producers to_json(), consumers from_json()."""

    def test_to_from_json_list_round_trip(self):
        original = [{"id": "gas-001", "score": 0.69, "title": "gmail agent skill"}]
        assert from_json(to_json(original)) == original

    def test_to_from_json_dict_round_trip(self):
        original = {"skill": "gmail", "top_k": 3}
        assert from_json(to_json(original)) == original

    def test_to_json_none_passes_through(self):
        assert to_json(None) is None

    def test_from_json_malformed_returns_default(self):
        assert from_json("{not valid json", default=[]) == []
        assert from_json(None, default={}) == {}
        assert from_json("", default=[]) == []
