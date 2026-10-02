# CMN-C2-298 — Unit Tests: RetrieveNode (inner domain node 2)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS
# caller. C2 is FULLY RETIRED for this template — every node.execute() takes
# exactly one argument (self, state); there is no execute(state, config=...)
# carve-out here (RetrieveNode reads config exclusively from the state field
# retrieval_config, republished by DomainWorkflowGraph._extra_initial_state()
# — docs/02_design.md). Every call below therefore goes through node(state).
#
# Queries are hand-picked so their tokens appear ONLY in one target KB
# entry's title (verified against the real config/kb/*.json content), giving
# a deterministic top-1 without depending on retrieval-internals knowledge —
# e.g. "Gmail Agent Skill read draft send" reproduces gas-001's title tokens
# exactly (score 1.0, the maximum), so it cannot tie with any other entry.
#
# Mirrors docs/03_test_spec.md Section 2.3 (RET-01..RET-10).
# Deterministic — keyword scoring over the seeded config/kb/*.json; no LLM,
# no network. framework.* / src.* imports only.

from framework.schemas.trust_level import TrustLevel

from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import from_json, to_json

_GMAIL_QUERY = "Gmail Agent Skill read draft send"
_COMPLIANCE_QUERY = "APPI cross-border transfer personal data"


def _make_state(query, **extra) -> dict:
    state = {
        "search_query": query,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestRetrieveHappyPath:
    def test_ret_01_top_hit_is_the_gmail_skill_entry(self):
        result = RetrieveNode()(_make_state(_GMAIL_QUERY))
        docs = from_json(result["retrieved_documents"])
        assert docs, "expected candidates for the gmail-skill query"
        assert docs[0]["id"] == "gas-001"
        assert docs[0]["score"] == 1.0
        assert docs[0]["kb"] == "google_agent_skills"

    def test_ret_02_scores_sorted_descending(self):
        docs = from_json(RetrieveNode()(_make_state(_GMAIL_QUERY))["retrieved_documents"])
        scores = [d["score"] for d in docs]
        assert scores == sorted(scores, reverse=True)
        assert all(s > 0.0 for s in scores)

    def test_ret_03_entry_shape_and_excerpt_cap(self):
        docs = from_json(RetrieveNode()(_make_state(_GMAIL_QUERY))["retrieved_documents"])
        for doc in docs:
            assert {
                "id",
                "kb",
                "skill",
                "framework",
                "title",
                "source",
                "score",
                "excerpt",
                "integration_steps",
                "oauth_scopes",
                "related_skills",
                "testing_notes",
                "guidance",
            } <= set(doc.keys())
            assert len(doc["excerpt"]) <= 400

    def test_retrieved_documents_is_json_string(self):
        # List-shaped State fields travel as JSON strings.
        result = RetrieveNode()(_make_state(_GMAIL_QUERY))
        assert isinstance(result["retrieved_documents"], str)

    def test_ret_04_pools_across_all_three_seeded_kbs(self):
        """The compliance-only query must surface a japan_compliance hit even
        though search_query has no google_agent_skills vocabulary — proves
        the pool spans all three config/kb/*.json files, not just one."""
        docs = from_json(RetrieveNode()(_make_state(_COMPLIANCE_QUERY))["retrieved_documents"])
        assert docs[0]["id"] == "jc-001"
        assert docs[0]["kb"] == "japan_compliance"
        assert docs[0]["framework"] == "APPI"


class TestRetrieveSkillFilter:
    """RET-05: a skill filter narrows the google_agent_skills-specific pool
    only — compliance/pattern entries (no `skill` field) always stay in
    play, since a skill-specific question still needs the cross-cutting
    compliance/pattern guidance alongside it (docs/02_design.md)."""

    def test_ret_05_skill_filter_excludes_non_matching_google_skills(self):
        state = _make_state(
            "gmail agent skill",
            query_filters=to_json({"skill": "drive", "top_k": None}),
        )
        docs = from_json(RetrieveNode()(state)["retrieved_documents"])
        gas_ids = {d["id"] for d in docs if d["kb"] == "google_agent_skills"}
        assert "gas-001" not in gas_ids  # gmail skill excluded by the drive filter

    def test_ret_05_skill_filter_keeps_non_google_skills_entries_in_play(self):
        state = _make_state(
            "calendar agent skill event scheduling",
            query_filters=to_json({"skill": "calendar", "top_k": None}),
        )
        docs = from_json(RetrieveNode()(state)["retrieved_documents"])
        kbs = {d["kb"] for d in docs}
        assert "japan_compliance" in kbs or "integration_patterns" in kbs

    def test_ret_06_empty_query_yields_no_candidates(self):
        docs = from_json(RetrieveNode()(_make_state(""))["retrieved_documents"])
        assert docs == []


class TestRetrieveConfigFromState:
    """Config plumbing is state-seeding only — there is no
    execute(state, config=...) route at node level."""

    def test_ret_07_state_retrieval_config_kb_paths_override(self):
        state = _make_state(
            _GMAIL_QUERY,
            retrieval_config=to_json({"kb_paths": ["config/kb/does_not_exist.json"]}),
        )
        result = RetrieveNode()(state)
        assert from_json(result["retrieved_documents"]) == []
        notes = from_json(result.get("intake_notes"), [])
        assert any("not readable" in n for n in notes)

    def test_ret_08_malformed_kb_path_list_degrades_gracefully_not_fatal(self):
        # One bad path among otherwise-default paths still yields the other
        # two KBs' candidates (graceful per-file degradation).
        state = _make_state(
            _GMAIL_QUERY,
            retrieval_config=to_json(
                {
                    "kb_paths": [
                        "config/kb/does_not_exist.json",
                        "config/kb/google_agent_skills_kb.json",
                    ]
                }
            ),
        )
        result = RetrieveNode()(state)
        docs = from_json(result["retrieved_documents"])
        assert docs and docs[0]["id"] == "gas-001"
        notes = from_json(result.get("intake_notes"), [])
        assert any("not readable" in n for n in notes)


class TestRetrieveNotesAccumulation:
    def test_ret_09_notes_append_never_clobber(self):
        state = _make_state(
            _GMAIL_QUERY,
            intake_notes=to_json(["earlier note from input validation"]),
            retrieval_config=to_json({"kb_paths": ["config/kb/bogus.json"]}),
        )
        result = RetrieveNode()(state)
        notes = from_json(result["intake_notes"])
        assert notes[0] == "earlier note from input validation"
        assert len(notes) == 2

    def test_ret_10_falls_back_to_user_input_when_search_query_absent(self):
        state = {
            "user_input": _GMAIL_QUERY,
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
            "node_history": [],
            "error_log": [],
            "session_id": "unit-session",
            "execution_time": {},
        }
        docs = from_json(RetrieveNode()(state)["retrieved_documents"])
        assert docs and docs[0]["id"] == "gas-001"
