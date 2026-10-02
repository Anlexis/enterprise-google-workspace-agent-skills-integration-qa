# CMN-C2-298 — Unit Tests: PostProcessNode (the outer output boundary)
#
# Every behavioural test invokes the node as node(state), through
# BaseNode.__call__. PostProcessNode requires VERIFIED_EXTERNAL like
# PreProcessNode; the ANONYMOUS rejection is covered in
# tests/unit/test_trust_gate.py.
#
# The discrimination this node has to get right, in both directions: the
# structured answer NAMES OAuth scope identifiers (for example
# "https://www.googleapis.com/auth/gmail.readonly") — plain permission
# identifiers, not secrets, which must NEVER be blanked. An issued token shape
# (a Google OAuth access token, a JWT, an AWS key id, a Bearer header, a
# credential assignment) DOES trip the gate, including when it is nested inside
# the structured_answer object rather than sitting in a top-level string.
#
# Mirrors docs/03_test_spec.md Section 2.7.
# Deterministic — no model call, no network. framework.* / src.* imports only.

import pytest
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.output_format_node import _SCOPE_NOTICE
from src.nodes.post_process_node import PostProcessNode
from src.schemas.state import to_json

# Every released answer carries the standing scope notice; the gate refuses one
# that does not, so the fixture below is a realistic answer, not a fragment.
_CLEAN_ANSWER = (
    "# Google Agent Skills Integration Guidance\n\n"
    "[1] the gmail skill requires oauth consent before it can read mail.\n\n"
    "## Scope & Limitations\n\n"
    f"*{_SCOPE_NOTICE}*\n"
)

# Real OAuth scope identifiers this agent legitimately surfaces — the output
# gate must never blank them.
_GMAIL_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
_DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.file"

# Credential-shaped literals assembled at runtime so no real-looking secret
# sits in the repository as a literal.
_FAKE_JWT = "eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12
_FAKE_GOOGLE_TOKEN = "ya29." + "x" * 24
_FAKE_AWS_KEY = "AKIA" + "Q" * 16


def _make_state(result_text, **extra) -> dict:
    state = {
        "result": result_text,
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestPostProcessClean:
    def test_post_01_clean_output_passes_through(self):
        result = PostProcessNode()(_make_state(_CLEAN_ANSWER))
        assert result["status"] == AgentStatus.SUCCESS.value
        # State carries the plain status string, never the bare enum.
        assert result["status"].__class__ is str
        assert result["formatted_output"] == _CLEAN_ANSWER

    def test_post_02_empty_result_is_non_fatal(self):
        result = PostProcessNode()(_make_state(""))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == ""


class TestScopeSurvives:
    """OAuth scope identifiers are permission names, not secrets — the gate
    must let them through, in the flat result text and nested inside
    structured_answer alike."""

    def test_post_03_scope_url_in_result_text_is_not_blocked(self):
        text = f"# Report\n\nRequest scope {_GMAIL_SCOPE} for read access.\n\n*{_SCOPE_NOTICE}*\n"
        result = PostProcessNode()(_make_state(text))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert _GMAIL_SCOPE in result["formatted_output"]

    def test_post_04_scope_url_nested_in_structured_answer_is_not_blocked(self):
        structured = {
            "oauth_scopes": [
                {"skill": "gmail", "scope": _GMAIL_SCOPE, "purpose": "read"},
                {"skill": "drive", "scope": _DRIVE_SCOPE, "purpose": "file access"},
            ],
        }
        result = PostProcessNode()(_make_state(_CLEAN_ANSWER, structured_answer=to_json(structured)))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Clean path never returns structured_answer (unchanged in State).
        assert "structured_answer" not in result


class TestNestedCredentialCaught:
    """A credential-shaped value nested inside structured_answer must still
    be caught (RECURSIVE scan) — the discrimination cuts both ways."""

    def test_post_05_credential_nested_in_structured_answer_is_blocked(self):
        structured = {
            "oauth_scopes": [{"skill": "gmail", "scope": _GMAIL_SCOPE, "purpose": "read"}],
            "citations": [{"ref": 1, "id": "x", "title": "t", "source": _FAKE_GOOGLE_TOKEN, "kb": "k"}],
        }
        result = PostProcessNode()(_make_state(_CLEAN_ANSWER, structured_answer=to_json(structured)))
        assert result["status"] == AgentStatus.ERROR.value
        assert result["structured_answer"] is None
        assert _FAKE_GOOGLE_TOKEN not in str(result["formatted_output"])


class TestCredentialShapesBlocked:
    def _assert_blocked(self, result, secret):
        assert result["status"] == AgentStatus.ERROR.value
        assert any("output blocked" in str(e) for e in result["error_log"])
        # The raw secret must not survive into either surfaced field.
        assert secret not in str(result.get("formatted_output", ""))
        assert secret not in str(result.get("result", ""))
        assert "[OUTPUT BLOCKED" in result["formatted_output"]

    def test_post_06_api_key_is_blocked(self):
        secret = "sk-ABCDEF0123456789abcdef"
        result = PostProcessNode()(_make_state(f"# Report\n\n<!-- debug api_key={secret} -->\n"))
        self._assert_blocked(result, secret)

    def test_post_07_credential_assignment_is_blocked(self):
        secret = "password=super_secret_value_123"
        result = PostProcessNode()(_make_state(f"# Report\n\ninternal note: {secret}\n"))
        self._assert_blocked(result, "super_secret_value_123")

    def test_post_08_jwt_is_blocked(self):
        result = PostProcessNode()(_make_state(f"# Report\n\nsession token {_FAKE_JWT}\n"))
        self._assert_blocked(result, _FAKE_JWT)

    def test_post_09_google_oauth_access_token_is_blocked(self):
        # Distinct from a scope URL — an ISSUED token, never present in this
        # agent's output by construction, but caught if one ever appeared.
        result = PostProcessNode()(_make_state(f"# Report\n\ntoken: {_FAKE_GOOGLE_TOKEN}\n"))
        self._assert_blocked(result, _FAKE_GOOGLE_TOKEN)

    def test_post_10_aws_access_key_is_blocked(self):
        result = PostProcessNode()(_make_state(f"# Report\n\nkey {_FAKE_AWS_KEY} leaked\n"))
        self._assert_blocked(result, _FAKE_AWS_KEY)


class TestScopeNoticeInvariant:
    """The agent's other stated output invariant: every released answer says,
    at the point of use, that it did not call a Google API.

    Composing the notice is not enforcing it. A renderer change, a truncation,
    or a partially assembled answer would drop the line silently, and what
    ships then reads as if the integration had been performed."""

    def test_answer_without_the_notice_is_refused(self):
        result = PostProcessNode()(
            _make_state("# Google Agent Skills Integration Guidance\n\n[1] configure oauth consent.\n")
        )
        assert result["status"] == AgentStatus.ERROR.value
        assert "scope_notice_missing" in " ".join(str(e) for e in result["error_log"])

    def test_answer_with_the_notice_is_released(self):
        result = PostProcessNode()(_make_state(_CLEAN_ANSWER))
        assert result["status"] == AgentStatus.SUCCESS.value


class TestStructuredShapeInvariant:
    """The structured product is documented as scalars only. A non-scalar leaf
    would be released without ever having been pattern-scanned, because the
    credential scan skips non-string leaves by design."""

    def test_opaque_leaf_is_refused(self):
        # A shape the JSON round-trip cannot produce, reached by handing the
        # node a pre-deserialized object through the same field.
        import json as _json

        class _Opaque:
            def __repr__(self):
                return "<opaque>"

        payload = _json.dumps({"integration_steps": ["step one"]})
        state = _make_state(_CLEAN_ANSWER, structured_answer=payload)
        # Substitute an object leaf the way a future producer bug would.
        from src.nodes import post_process_node as module

        original = module.from_json
        module.from_json = lambda *_args, **_kwargs: {"integration_steps": [_Opaque()]}
        try:
            result = PostProcessNode()(state)
        finally:
            module.from_json = original
        assert result["status"] == AgentStatus.ERROR.value
        assert "structured_shape" in " ".join(str(e) for e in result["error_log"])


class TestContainmentOnViolation:
    """Returning an error status is not containment.

    The response envelope falls back to state["result"] even on a non-success
    status, so a gate that flips the status without clearing the fields that
    carry text still ships the un-gated answer inside the error envelope."""

    @pytest.mark.parametrize(
        "field",
        [
            "integration_answer",
            "formatted_answer",
            "grounded_answer",
            "structured_answer",
            "citations",
            "structured_extract",
        ],
    )
    def test_every_output_bearing_field_is_cleared(self, field):
        leaked = "zqx_released_text_zqx"
        result = PostProcessNode()(
            _make_state(
                f"# Report\n\nkey {_FAKE_AWS_KEY} leaked\n",
                integration_answer=leaked,
                formatted_answer=leaked,
                grounded_answer=leaked,
                structured_answer=to_json({"integration_steps": [leaked]}),
                citations=to_json([{"ref": 1, "title": leaked}]),
                structured_extract=to_json({"integration_steps": [leaked]}),
            )
        )
        assert result["status"] == AgentStatus.ERROR.value
        assert field in result and result[field] is None

    def test_the_error_delta_carries_no_released_text_or_paths(self):
        import json as _json

        leaked = "zqx_released_text_zqx"
        result = PostProcessNode()(
            _make_state(
                f"# Report\n\nkey {_FAKE_AWS_KEY} leaked\n",
                integration_answer=leaked,
                formatted_answer=leaked,
            )
        )
        serialized = _json.dumps(result, default=str)
        assert leaked not in serialized
        assert _FAKE_AWS_KEY not in serialized
        assert "Traceback" not in serialized
        assert "/src/" not in serialized
