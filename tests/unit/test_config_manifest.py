# CMN-C2-298 — Unit Tests: the configuration contract
#
# Two files, two jobs, and the tests keep them apart:
#
#   config/agent.yaml   the static registration manifest — flat, read at ROOT
#                       level by the platform registry. It carries identity and
#                       compile-time requirements, and NO tuning values.
#   config/config.yaml  the runtime parameters, handed to the graph as
#                       Graph(config=...) and forwarded from there into the
#                       inner graph.
#
# The second file is the one that can go quietly dead: a declared value whose
# reader still points at the old location produces no error, no failing test
# and no log line — the agent simply runs on its built-in defaults while the
# file says otherwise. The end-to-end test at the bottom is the guard against
# that: it changes a declared value and asserts the change reaches the node
# that consumes it.
#
# Mirrors docs/03_test_spec.md Section 2.8.
# Deterministic — file reads only, no model call, no network.

import pathlib

import yaml

_CONFIG_DIR = pathlib.Path(__file__).resolve().parents[2] / "config"
_MANIFEST_PATH = _CONFIG_DIR / "agent.yaml"
_RUNTIME_PATH = _CONFIG_DIR / "config.yaml"


def _load(path: pathlib.Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class TestManifestIdentity:
    """The registration manifest is FLAT: every key is read at root level."""

    def test_cfg_01_registry_identity_keys_present(self):
        manifest = _load(_MANIFEST_PATH)
        assert manifest["id"] == "CMN-C2-298"
        assert manifest["name"] == "WorkspaceSkillsIntegrationQAAgent"
        assert manifest["enabled"] is True

    def test_cfg_02_class_path_resolves_to_the_graph_class(self):
        manifest = _load(_MANIFEST_PATH)
        assert manifest["class"] == "src.graph.graph.WorkspaceSkillsIntegrationQAAgent"
        from src.graph.graph import WorkspaceSkillsIntegrationQAAgent

        assert manifest["class"].endswith(WorkspaceSkillsIntegrationQAAgent.__name__)

    def test_cfg_03_required_trust_level_is_verified_external(self):
        assert _load(_MANIFEST_PATH)["required_trust_level"] == "VERIFIED_EXTERNAL"

    def test_cfg_04_category_and_industry(self):
        manifest = _load(_MANIFEST_PATH)
        assert manifest["category"] == "Cat 2"
        assert manifest["industry"] == "CMN"

    def test_manifest_declares_no_secrets_or_extras(self):
        """Both are code-derived and both are empty here: the agent calls no
        model and requires no secret. Declaring either one unnecessarily makes
        the agent fail to compile when the value is not provisioned."""
        requires = _load(_MANIFEST_PATH)["requires"]
        assert requires["secrets"] == []
        assert requires["extras"] == []

    def test_manifest_carries_no_tuning_values(self):
        """Tuning lives in config/config.yaml. A copy here would be the kind of
        second source that drifts without anything failing."""
        manifest = _load(_MANIFEST_PATH)
        assert "retrieval" not in manifest
        assert "config" not in manifest


class TestRuntimeSettings:
    def test_cfg_05_retrieval_block_matches_node_defaults(self):
        retrieval = _load(_RUNTIME_PATH)["retrieval"]
        assert retrieval["top_k"] == 6
        assert retrieval["score_threshold"] == 0.15
        assert retrieval["kb_paths"] == [
            "config/kb/google_agent_skills_kb.json",
            "config/kb/integration_patterns_kb.json",
            "config/kb/japan_compliance_kb.json",
        ]

    def test_cfg_06_every_declared_kb_path_exists_on_disk(self):
        repo_root = _RUNTIME_PATH.parents[1]
        for kb_path in _load(_RUNTIME_PATH)["retrieval"]["kb_paths"]:
            assert (repo_root / kb_path).is_file(), f"declared kb_path missing on disk: {kb_path}"

    def test_runtime_config_helper_reads_the_file(self):
        from src.graph.graph import runtime_config

        settings = runtime_config()
        assert settings["max_retry"] == 3
        assert settings["timeout_s"] == 30
        assert settings["retrieval"]["top_k"] == 6

    def test_runtime_config_helper_degrades_to_empty_when_unreadable(self, monkeypatch):
        import src.graph.graph as graph_module

        monkeypatch.setattr(graph_module, "_RUNTIME_CONFIG_PATH", pathlib.Path("/nonexistent/config.yaml"))
        assert graph_module.runtime_config() == {}


class TestDeclaredValuesAreLive:
    """A declared value must reach the node that consumes it.

    This is the test that catches a dead declaration: the reader is re-pointed
    at a temporary settings file, and the assertion is on the value the inner
    node would actually use — not on the file's contents, which prove nothing
    about whether anything reads them.
    """

    def test_a_changed_retrieval_value_reaches_the_inner_graph(self, tmp_path, monkeypatch):
        import src.graph.graph as graph_module
        from src.schemas.state import from_json

        settings = tmp_path / "config.yaml"
        settings.write_text(
            "retrieval:\n  top_k: 2\n  score_threshold: 0.99\n  kb_paths:\n"
            '    - "config/kb/japan_compliance_kb.json"\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(graph_module, "_RUNTIME_CONFIG_PATH", settings)

        node = graph_module.IntegrationQAGraphNode()
        forwarded = node._parent_config()["configurable"]["retrieval"]
        assert forwarded["top_k"] == 2
        assert forwarded["score_threshold"] == 0.99

        inner = node.get_subgraph()
        seeded = from_json(inner._extra_initial_state()["retrieval_config"])
        assert seeded["top_k"] == 2
        assert seeded["score_threshold"] == 0.99
        assert seeded["kb_paths"] == ["config/kb/japan_compliance_kb.json"]

    def test_the_inner_node_uses_the_seeded_threshold(self, tmp_path, monkeypatch):
        """End to end through the node that reads it: a threshold of 0.99 must
        drop a candidate scored 0.5, which the default 0.15 would keep."""
        import src.graph.graph as graph_module
        from framework.schemas.trust_level import TrustLevel

        from src.nodes.rerank_filter_node import RerankFilterNode
        from src.schemas.state import from_json, to_json

        settings = tmp_path / "config.yaml"
        settings.write_text("retrieval:\n  top_k: 6\n  score_threshold: 0.99\n", encoding="utf-8")
        monkeypatch.setattr(graph_module, "_RUNTIME_CONFIG_PATH", settings)

        seeded = graph_module.IntegrationQAGraphNode().get_subgraph()._extra_initial_state()
        state = {
            "retrieval_config": seeded["retrieval_config"],
            "retrieved_documents": to_json([{"id": "a", "score": 0.5, "title": "t"}]),
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
            "node_history": [],
            "error_log": [],
            "session_id": "unit-session",
            "execution_time": {},
        }
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert kept == []
