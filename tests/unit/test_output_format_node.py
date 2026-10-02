# CMN-C2-298 — Unit Tests: OutputFormatNode (inner domain node 5, terminal)
#
# Invoked as node(state) via BaseNode.__call__ with an ANONYMOUS caller.
# formatted_answer and structured_answer are domain fields, not targets of the
# framework's input mask; composing the scope/limitations notice belongs to
# THIS node, while enforcing it belongs to the output boundary
# (docs/02_design.md).
#
# Over-claiming guard: the standing notice is appended to EVERY answer this
# agent emits. TestScopeNotice asserts it is present unconditionally, including
# on the no-citations and missing-answer paths.
#
# Mirrors docs/03_test_spec.md Section 2.6.
# Deterministic — no model call, no network. framework.* / src.* imports only.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.output_format_node import OutputFormatNode
from src.schemas.state import from_json, to_json

_NOTICE_FRAGMENT = (
    "does NOT connect to, authenticate against, or call any Google " "Workspace/Cloud/Maps/YouTube/Search API"
)


def _make_state(grounded_answer, citations, structured_extract=None, **extra) -> dict:
    state = {
        "grounded_answer": grounded_answer,
        "citations": citations,
        "structured_extract": to_json(structured_extract or {}),
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestFormattedAnswer:
    def test_fmt_01_composes_header_body_sources_notice(self):
        citations = to_json(
            [
                {
                    "ref": 1,
                    "id": "gas-001",
                    "title": "Gmail Agent Skill",
                    "source": "google/skills",
                    "kb": "google_agent_skills",
                }
            ]
        )
        result = OutputFormatNode()(_make_state("[1] the grounded answer body.", citations))
        answer = result["formatted_answer"]
        assert answer.startswith("# Google Agent Skills Integration Guidance")
        assert "[1] the grounded answer body." in answer
        assert "## Sources" in answer
        assert "- [1] Gmail Agent Skill (google/skills) [Google Agent Skills]" in answer
        assert "## Scope & Limitations" in answer
        assert result["status"] == AgentStatus.SUCCESS.value
        # State carries the plain status string, never the bare enum.
        assert result["status"].__class__ is str

    def test_fmt_02_source_and_kb_suffix_omitted_when_blank(self):
        citations = to_json([{"ref": 1, "id": "gas-001", "title": "Gmail Agent Skill", "source": "", "kb": ""}])
        answer = OutputFormatNode()(_make_state("body.", citations))["formatted_answer"]
        assert "- [1] Gmail Agent Skill\n" in answer + "\n"
        assert "()" not in answer
        assert "[]" not in answer


class TestScopeNotice:
    """The over-claiming guard: present unconditionally, every path."""

    def test_fmt_03_notice_present_with_citations(self):
        citations = to_json([{"ref": 1, "id": "gas-001", "title": "t", "source": "s", "kb": "google_agent_skills"}])
        answer = OutputFormatNode()(_make_state("a body.", citations))["formatted_answer"]
        assert _NOTICE_FRAGMENT in answer

    def test_fmt_04_notice_present_with_no_citations(self):
        answer = OutputFormatNode()(_make_state("no coverage body.", to_json([])))["formatted_answer"]
        assert _NOTICE_FRAGMENT in answer

    def test_fmt_05_notice_present_even_when_grounded_answer_missing(self):
        state = _make_state("", to_json([]))
        del state["grounded_answer"]
        answer = OutputFormatNode()(state)["formatted_answer"]
        assert _NOTICE_FRAGMENT in answer


class TestDegradedInputs:
    def test_fmt_06_no_citations_renders_explicit_none_line(self):
        answer = OutputFormatNode()(_make_state("no coverage body.", to_json([])))["formatted_answer"]
        assert "- none (no knowledge-base passage cleared the relevance threshold)" in answer

    def test_fmt_07_missing_grounded_answer_uses_fallback_text(self):
        state = _make_state("", to_json([]))
        del state["grounded_answer"]
        result = OutputFormatNode()(state)
        assert "No answer is available for this request." in result["formatted_answer"]
        assert result["status"] == AgentStatus.SUCCESS.value


class TestStructuredAnswerAssembly:
    def test_fmt_08_combines_structured_extract_and_citations(self):
        citations = to_json([{"ref": 1, "id": "gas-001", "title": "t", "source": "s", "kb": "google_agent_skills"}])
        extract = {
            "integration_steps": ["step one"],
            "oauth_scopes": [
                {"skill": "gmail", "scope": "https://www.googleapis.com/auth/gmail.readonly", "purpose": "read"}
            ],
            "compliance_flags": [],
            "testing_checklist": ["check one"],
            "related_skills": [],
        }
        result = OutputFormatNode()(_make_state("body.", citations, structured_extract=extract))
        structured = from_json(result["structured_answer"])
        assert structured["integration_steps"] == ["step one"]
        assert structured["oauth_scopes"][0]["scope"] == "https://www.googleapis.com/auth/gmail.readonly"
        assert structured["citations"] == from_json(citations)

    def test_structured_answer_is_json_string(self):
        result = OutputFormatNode()(_make_state("body.", to_json([])))
        assert isinstance(result["structured_answer"], str)
