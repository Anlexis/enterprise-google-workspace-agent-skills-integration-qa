# CMN-C2-298 — Unit Tests: the caller trust gate on the two outer slots
#
# Covers the two outer backbone gate nodes (PreProcessNode, PostProcessNode —
# both required_trust_level = VERIFIED_EXTERNAL). The inner domain nodes run
# behind this boundary at ANONYMOUS.
#
# Tests invoke nodes as node(state), through BaseNode.__call__, which runs the
# trust gate, the input masking, execute(), and the output scan in order —
# never via node.execute(state) directly, which would bypass the gate under
# test. A denial RETURNS an error dict (it never raises) with an error status
# and "trust gate denied" in error_log; execute() never runs, so the keys only
# execute() writes are ABSENT from the returned dict.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.output_format_node import _SCOPE_NOTICE
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode

# A released answer must carry the standing scope notice, so the fixture below
# is a realistic answer rather than a bare sentence.
_CLEAN_ANSWER = f"a clean Google Agent Skills integration answer\n\n*{_SCOPE_NOTICE}*"


def _make_state(trust_value: str, user_input: str = "how do I make the Gmail skill APPI-compliant?", **extra) -> dict:
    state = {
        "user_input": user_input,
        "caller_trust_level": trust_value,
        "node_history": [],
        "error_log": [],
        "session_id": "test-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestTrustGate:
    """Trust gate tests — all invocations go through node(state) / __call__."""

    def test_anonymous_caller_denied_on_pre_process(self):
        """ANONYMOUS caller on the VERIFIED_EXTERNAL PreProcessNode.

        __call__ must RETURN an error dict (never raise) with status ERROR and
        'trust gate denied' in the error_log. execute() never ran, so the
        execute-only output key (validated_input) must be ABSENT.
        """
        node = PreProcessNode()  # required_trust_level = VERIFIED_EXTERNAL
        result = node(_make_state(TrustLevel.ANONYMOUS.value))
        assert result.get("status") == AgentStatus.ERROR.value
        error_log = result.get("error_log", [])
        assert any(
            "trust gate denied" in str(e) for e in error_log
        ), f"Expected 'trust gate denied' in error_log, got: {error_log}"
        assert "validated_input" not in result, "execute() must not run on a trust denial — validated_input leaked"

    def test_verified_external_caller_passes_pre_process(self):
        """A VERIFIED_EXTERNAL caller clears the pre_process gate and the node
        writes the identifier-stripped validated_input."""
        node = PreProcessNode()
        result = node(_make_state(TrustLevel.VERIFIED_EXTERNAL.value))
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("validated_input")

    def test_pre_process_empty_input_rejected_after_gate(self):
        """The gate passes, then the node's own validation rejects empty input."""
        node = PreProcessNode()
        result = node(_make_state(TrustLevel.VERIFIED_EXTERNAL.value, user_input=""))
        assert result.get("status") == AgentStatus.ERROR.value
        assert any("empty" in str(e) for e in result.get("error_log", []))

    def test_anonymous_caller_denied_on_post_process(self):
        """Rejection on the other VERIFIED_EXTERNAL outer slot (post_process).

        The denial dict carries no execute-only key (formatted_output ABSENT).
        """
        node = PostProcessNode()  # required_trust_level = VERIFIED_EXTERNAL
        result = node(
            _make_state(
                TrustLevel.ANONYMOUS.value,
                result=_CLEAN_ANSWER,
            )
        )
        assert result.get("status") == AgentStatus.ERROR.value
        assert any("trust gate denied" in str(e) for e in result.get("error_log", []))
        assert "formatted_output" not in result, "execute() must not run on a trust denial — formatted_output leaked"

    def test_verified_external_caller_passes_post_process(self):
        """A VERIFIED_EXTERNAL caller clears the post_process gate."""
        node = PostProcessNode()
        result = node(
            _make_state(
                TrustLevel.VERIFIED_EXTERNAL.value,
                result=_CLEAN_ANSWER,
            )
        )
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("formatted_output")


class TestTrustLevelMatrix:
    """The declared trust matrix (docs/02_design.md, security section).

    The outer gate slots require VERIFIED_EXTERNAL, the level the manifest
    declares. The inner domain nodes run behind that boundary and declare
    ANONYMOUS: a stricter inner level would deny a real verified invoke.
    """

    def test_outer_gate_nodes_require_verified_external(self):
        assert PreProcessNode.required_trust_level is TrustLevel.VERIFIED_EXTERNAL
        assert PostProcessNode.required_trust_level is TrustLevel.VERIFIED_EXTERNAL
