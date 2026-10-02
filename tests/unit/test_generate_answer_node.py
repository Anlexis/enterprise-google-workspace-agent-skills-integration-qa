# CMN-C2-298 — Unit Tests: GenerateAnswerNode (inner domain node 4)
#
# Invoked as node(state) via BaseNode.__call__ with an ANONYMOUS caller. The
# grounded answer, citations and structured extract are domain fields, never
# targets of the framework's input mask (which covers only the request text),
# so Title-Case knowledge-base titles inside them are safe to assert on
# exactly.
#
# Mirrors docs/03_test_spec.md Section 2.5. Deterministic — rule-assembled and
# verbatim-aggregated from ranked_documents only, grounded by construction; no
# model call, no network. framework.* / src.* imports only.

from framework.schemas.trust_level import TrustLevel

from src.nodes.generate_answer_node import GenerateAnswerNode
from src.schemas.state import from_json, to_json


def _gas_doc(doc_id, title, excerpt, skill="gmail", scopes=None, related=None, notes=None):
    return {
        "id": doc_id,
        "kb": "google_agent_skills",
        "skill": skill,
        "title": title,
        "source": "seeded kb",
        "score": 0.9,
        "excerpt": excerpt,
        "integration_steps": ["register an oauth client", "request the narrowest scope"],
        "oauth_scopes": scopes
        if scopes is not None
        else [{"scope": "https://www.googleapis.com/auth/gmail.readonly", "purpose": "read"}],
        "related_skills": related if related is not None else [{"skill": "drive", "relation": "attachments"}],
        "testing_notes": notes if notes is not None else ["confirm consent screen scope list matches manifest"],
    }


def _jc_doc(doc_id, title, excerpt, framework="APPI", guidance="confirm the transfer basis"):
    return {
        "id": doc_id,
        "kb": "japan_compliance",
        "skill": None,
        "framework": framework,
        "title": title,
        "source": "seeded kb",
        "score": 0.8,
        "excerpt": excerpt,
        "guidance": guidance,
        "testing_notes": ["confirm the documented transfer basis is on file"],
    }


def _make_state(ranked_documents, query="gmail integration guidance", **extra) -> dict:
    state = {
        "ranked_documents": ranked_documents,
        "search_query": query,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestGroundedAnswer:
    def test_gen_01_answer_carries_numbered_citation_markers(self):
        ranked = to_json(
            [
                _gas_doc(
                    "gas-001", "Gmail Agent Skill — read, draft, and send", "oauth 2.0 user-consent based access."
                ),
                _jc_doc(
                    "jc-001",
                    "APPI — cross-border transfer of personal data",
                    "restricts transferring personal data outside japan.",
                ),
            ]
        )
        result = GenerateAnswerNode()(_make_state(ranked))
        answer = result["grounded_answer"]
        assert "[1] (Google Agent Skills) Gmail Agent Skill" in answer
        assert "[2] (Japan Compliance) APPI" in answer

    def test_gen_02_lead_sentence_quotes_the_query(self):
        ranked = to_json([_gas_doc("gas-001", "Gmail Agent Skill", "excerpt.")])
        result = GenerateAnswerNode()(_make_state(ranked, query="gmail integration guidance"))
        assert 'the question: "gmail integration guidance"' in result["grounded_answer"]

    def test_gen_03_citations_mirror_ranked_order(self):
        ranked = to_json(
            [
                _gas_doc("gas-001", "Gmail Agent Skill", "a.", scopes=[]),
                _jc_doc("jc-001", "APPI cross-border", "b."),
            ]
        )
        citations = from_json(GenerateAnswerNode()(_make_state(ranked))["citations"])
        assert [c["ref"] for c in citations] == [1, 2]
        assert [c["id"] for c in citations] == ["gas-001", "jc-001"]
        assert [c["kb"] for c in citations] == ["google_agent_skills", "japan_compliance"]

    def test_citations_is_json_string(self):
        ranked = to_json([_gas_doc("gas-001", "Gmail Agent Skill", "a.")])
        result = GenerateAnswerNode()(_make_state(ranked))
        assert isinstance(result["citations"], str)


class TestStructuredExtractAggregation:
    """The structured extract is a pure, verbatim AGGREGATION of pre-authored
    KB fields — never synthesised (docs/02_design.md)."""

    def test_gen_04_oauth_scopes_and_related_skills_only_from_google_agent_skills_kb(self):
        ranked = to_json(
            [
                _gas_doc("gas-001", "Gmail Agent Skill", "a."),
                _jc_doc("jc-001", "APPI cross-border", "b."),
            ]
        )
        extract = from_json(GenerateAnswerNode()(_make_state(ranked))["structured_extract"])
        assert extract["oauth_scopes"] == [
            {"skill": "gmail", "scope": "https://www.googleapis.com/auth/gmail.readonly", "purpose": "read"}
        ]
        assert extract["related_skills"] == [{"skill": "drive", "relation": "attachments"}]

    def test_gen_05_compliance_flags_only_from_japan_compliance_kb(self):
        ranked = to_json(
            [
                _gas_doc("gas-001", "Gmail Agent Skill", "a."),
                _jc_doc("jc-001", "APPI cross-border", "b.", framework="APPI", guidance="confirm the transfer basis"),
            ]
        )
        extract = from_json(GenerateAnswerNode()(_make_state(ranked))["structured_extract"])
        assert extract["compliance_flags"] == [
            {"framework": "APPI", "requirement": "APPI cross-border", "guidance": "confirm the transfer basis"}
        ]

    def test_gen_06_integration_steps_pool_gas_and_pattern_kbs_and_dedupe(self):
        ranked = to_json(
            [
                _gas_doc("gas-001", "Gmail Agent Skill", "a."),
                _gas_doc("gas-002", "Drive Agent Skill", "b."),  # same integration_steps text -> deduped
            ]
        )
        extract = from_json(GenerateAnswerNode()(_make_state(ranked))["structured_extract"])
        assert extract["integration_steps"] == ["register an oauth client", "request the narrowest scope"]

    def test_gen_07_testing_checklist_pools_every_kb(self):
        ranked = to_json(
            [
                _gas_doc("gas-001", "Gmail Agent Skill", "a.", notes=["gas note"]),
                _jc_doc("jc-001", "APPI cross-border", "b."),
            ]
        )
        extract = from_json(GenerateAnswerNode()(_make_state(ranked))["structured_extract"])
        assert "gas note" in extract["testing_checklist"]
        assert "confirm the documented transfer basis is on file" in extract["testing_checklist"]

    def test_gen_08_lists_are_capped(self):
        docs = [_gas_doc(f"gas-{i:03d}", f"Skill {i}", "x.", notes=[f"note-{i}"]) for i in range(15)]
        extract = from_json(GenerateAnswerNode()(_make_state(to_json(docs)))["structured_extract"])
        assert len(extract["testing_checklist"]) <= 10
        assert len(extract["oauth_scopes"]) <= 12


class TestNoCoverage:
    def test_gen_09_empty_ranked_set_yields_no_coverage_answer(self):
        result = GenerateAnswerNode()(_make_state(to_json([])))
        assert "does not contain sufficient coverage" in result["grounded_answer"]
        assert from_json(result["citations"]) == []
        extract = from_json(result["structured_extract"])
        assert extract["oauth_scopes"] == []
        assert extract["compliance_flags"] == []

    def test_missing_ranked_field_is_treated_as_no_coverage(self):
        state = _make_state(None)
        del state["ranked_documents"]
        result = GenerateAnswerNode()(state)
        assert "does not contain sufficient coverage" in result["grounded_answer"]
