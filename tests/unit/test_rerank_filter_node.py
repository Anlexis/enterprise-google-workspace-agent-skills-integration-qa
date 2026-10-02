# CMN-C2-298 — Unit Tests: RerankFilterNode (inner domain node 3)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS
# caller. C2 is FULLY RETIRED for this template — RerankFilterNode.execute()
# takes exactly (self, state); config arrives exclusively via the state field
# retrieval_config (docs/02_design.md), so every call below is node(state).
#
# Defaults under test (config/agent.yaml): top_k=6, score_threshold=0.15,
# skill-match boost=0.1 (module defaults in rerank_filter_node.py mirror
# these — distinct from a golden peer template pattern's 0.25/4/category).
#
# Mirrors docs/03_test_spec.md Section 2.4 (RRF-01..RRF-09).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from framework.schemas.trust_level import TrustLevel

from src.nodes.rerank_filter_node import RerankFilterNode
from src.schemas.state import from_json, to_json


def _doc(doc_id, score, skill=None):
    return {
        "id": doc_id,
        "kb": "google_agent_skills" if skill else "japan_compliance",
        "skill": skill,
        "title": f"entry {doc_id}",
        "source": "seeded kb",
        "score": score,
        "excerpt": "excerpt text",
    }


def _make_state(candidates, **extra) -> dict:
    state = {
        "retrieved_documents": to_json(candidates),
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestThresholdAndCap:
    def test_rrf_01_default_threshold_drops_weak_candidates(self):
        result = RerankFilterNode()(_make_state([_doc("gas-a", 0.9), _doc("gas-b", 0.1)]))
        kept = from_json(result["ranked_documents"])
        assert [d["id"] for d in kept] == ["gas-a"]  # 0.1 < default 0.15 floor

    def test_rrf_02_state_score_threshold_override(self):
        state = _make_state(
            [_doc("gas-a", 0.9), _doc("gas-b", 0.3)],
            retrieval_config=to_json({"score_threshold": 0.5}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["gas-a"]

    def test_rrf_03_state_top_k_override(self):
        state = _make_state(
            [_doc("gas-a", 0.9), _doc("gas-b", 0.8), _doc("gas-c", 0.7)],
            retrieval_config=to_json({"top_k": 1}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["gas-a"]

    def test_ranked_documents_is_json_string(self):
        # List-shaped State fields travel as JSON strings.
        result = RerankFilterNode()(_make_state([_doc("gas-a", 0.9)]))
        assert isinstance(result["ranked_documents"], str)


class TestSkillBoost:
    def test_rrf_04_matching_skill_is_boosted_and_reranked(self):
        state = _make_state(
            [_doc("gas-a", 0.30, skill="drive"), _doc("gas-b", 0.25, skill="gmail")],
            query_filters=to_json({"skill": "gmail", "top_k": None}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["gas-b", "gas-a"]
        assert kept[0]["score"] == 0.35  # 0.25 + 0.1 skill boost

    def test_rrf_05_boost_is_capped_at_one(self):
        state = _make_state(
            [_doc("gas-a", 0.95, skill="gmail")],
            query_filters=to_json({"skill": "gmail", "top_k": None}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert kept[0]["score"] == 1.0

    def test_rrf_06_non_google_skill_entries_are_never_boosted(self):
        # Japan-compliance / integration-pattern candidates have no `skill`
        # field — a caller skill filter must not affect their score.
        state = _make_state(
            [_doc("jc-a", 0.40, skill=None)],
            query_filters=to_json({"skill": "gmail", "top_k": None}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert kept[0]["score"] == 0.40


class TestCallerTopK:
    def test_rrf_07_stricter_caller_top_k_wins(self):
        state = _make_state(
            [_doc("gas-a", 0.9), _doc("gas-b", 0.8), _doc("gas-c", 0.7)],
            query_filters=to_json({"skill": None, "top_k": 1}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["gas-a"]

    def test_rrf_07_looser_caller_top_k_does_not_widen(self):
        state = _make_state(
            [_doc("gas-a", 0.9), _doc("gas-b", 0.8), _doc("gas-c", 0.7)],
            query_filters=to_json({"skill": None, "top_k": 10}),
            retrieval_config=to_json({"top_k": 2, "score_threshold": 0.15}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["gas-a", "gas-b"]


class TestRobustness:
    def test_rrf_08_garbage_candidates_are_skipped_or_dropped(self):
        candidates = [
            "not-a-dict",
            {"id": "gas-bad", "title": "b", "skill": None, "source": "s", "score": "NaN?", "excerpt": "e"},
            _doc("gas-a", 0.9),
        ]
        kept = from_json(RerankFilterNode()(_make_state(candidates))["ranked_documents"])
        # The string entry is skipped; the uncoercible score becomes 0.0 and
        # falls below the relevance floor.
        assert [d["id"] for d in kept] == ["gas-a"]

    def test_rrf_09_deterministic_tie_break_by_id(self):
        kept = from_json(RerankFilterNode()(_make_state([_doc("gas-b", 0.5), _doc("gas-a", 0.5)]))["ranked_documents"])
        assert [d["id"] for d in kept] == ["gas-a", "gas-b"]
