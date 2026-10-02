"""AgentCore Platform v1.0"""

# CMN-C2-298 - PostProcessNode (outer post_process slot; the output boundary)
#
# Reads the final integration-advisory answer from state["result"] (mapped by
# IntegrationQAGraphNode.merge_output() from the inner graph's
# formatted_answer) and the structured_answer JSON, and enforces the agent's
# two output invariants on BOTH before either is surfaced further.
#
# Invariant 1 - nothing credential-shaped is released. The scan below RECURSES
# into dict/list/tuple content, so a credential hidden inside the
# structured_answer object is still caught. This is separate from, and in
# addition to, the framework's own automatic per-node credential scan, which
# inspects only the TOP-LEVEL STRING values of a node's own return dict and
# therefore never sees anything nested inside a payload.
#
# Invariant 2 - every released answer carries the standing scope notice. This
# agent advises on how to integrate Google's Agent Skills; it never connects
# to, authenticates against, or calls a Google API. OutputFormatNode composes
# that notice into every answer, and this gate REFUSES to release an answer
# that does not carry it. Composing an invariant is not enforcing it: a future
# renderer change, a truncation, or a partially assembled answer would drop the
# line silently, and the resulting output reads as if the agent had performed
# the integration.
#
# Scope-versus-credential discrimination (this agent's specific risk): the
# structured answer NAMES OAuth scope identifiers (for example
# "https://www.googleapis.com/auth/gmail.send"). Those carry no secret material
# and must NOT be mistaken for credentials and blanked. An issued token shape
# (a Google OAuth access token "ya29....", a JWT, an AWS key id, a Bearer
# header, or a "token: <value>" / "client_secret: <value>" assignment) DOES
# trip the gate. Both directions are pinned by tests.
#
# On a violation the node returns ERROR **and clears every output-bearing
# state field**. Returning an error status is not by itself containment: the
# base envelope falls back to state["result"] even on a non-success status, so
# a gate that only raised - or that flipped the status without clearing - would
# still ship the un-gated inner answer inside the error envelope.
#
# No _extra_security_gate_input/_extra_security_gate_output instance methods
# are defined on this (or any) node here - those are the framework's own
# FunctionNode extension hooks, a different, framework-owned mechanism from
# this manual gate.

import logging
import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json

logger = logging.getLogger(__name__)

# Disallowed output content. Each tuple: (name, compiled regex) - order
# matters (most specific first). None of these match a bare OAuth scope URL
# ("https://www.googleapis.com/auth/drive.readonly") or a short scope name
# ("gmail.send") - see the module header.
_DISALLOWED_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    # Generic API key shapes: sk-..., pk-..., ak-...
    ("api_key", re.compile(r"\b(?:sk|pk|ak)-[A-Za-z0-9]{16,}", re.IGNORECASE)),
    # JWT: three base64url segments separated by dots
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    # Bearer token in an authorization-like context
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._~+/]{20,}", re.IGNORECASE)),
    # Google OAuth 2.0 ISSUED access-token shape - distinct from a scope URL or
    # scope name. Never present in this agent's advisory-only output (it holds
    # no token and runs no OAuth flow), but caught explicitly if one appeared.
    ("google_oauth_token", re.compile(r"\bya29\.[A-Za-z0-9_-]{20,}")),
    # AWS IAM access key id
    ("aws_access_key", re.compile(r"\bAKIA[A-Z0-9]{16}\b")),
    # Credential assignment shapes. Both the bare keywords (token, secret, ...)
    # and the compound identifiers (refresh_token, client_secret) are listed
    # explicitly: an underscore joins into one \w+ run, so \btoken\b alone does
    # NOT reach into "refresh_token" - there is no word boundary before "token"
    # when it is preceded by "_", so each compound needs its own alternative.
    (
        "credential_assignment",
        re.compile(
            r"\b(?:password|passwd|secret|api_key|token|access_key|private_key|"
            r"refresh_token|client_secret)\s*[:=]\s*\S{8,}",
            re.IGNORECASE,
        ),
    ),
]

# The substring every released answer must carry. Kept deliberately short and
# stable: it is the load-bearing clause of the scope notice OutputFormatNode
# composes, so a reworded notice still satisfies the gate as long as the
# no-live-call statement survives.
SCOPE_NOTICE_MARKER = "does NOT connect to, authenticate against, or call any Google"

_SANITISED_STUB = (
    "[OUTPUT BLOCKED - the generated answer did not clear the output gate. " "Retry without credential-like strings.]"
)

# Every state field that can carry released text. All of them are cleared on a
# violation: the response envelope falls back through several of these, so
# clearing only the one the gate happened to read is not containment.
_OUTPUT_BEARING_FIELDS = (
    "integration_answer",
    "formatted_answer",
    "grounded_answer",
    "structured_answer",
    "citations",
    "structured_extract",
)


def _security_gate_output(content: Any) -> Optional[str]:
    """Scan released content for disallowed values.

    RECURSES into dict/list/tuple so a credential nested inside a structured
    payload is still caught. Only string leaves are pattern-scanned;
    int/float/bool/None leaves carry no credential pattern by construction, so
    they are skipped rather than stringified.

    Returns the name of the first matched violation, or None when clean.
    Shared by PostProcessNode.execute() (which scans the flat `result` text and
    the structured_answer object) and by
    WorkspaceSkillsIntegrationQAAgent.get_output(), which re-scans
    structured_answer at the true response boundary.
    """
    if content is None:
        return None
    if isinstance(content, str):
        for name, pattern in _DISALLOWED_PATTERNS:
            if pattern.search(content):
                return name
        return None
    if isinstance(content, dict):
        for value in content.values():
            violation = _security_gate_output(value)
            if violation:
                return violation
        return None
    if isinstance(content, (list, tuple)):
        for item in content:
            violation = _security_gate_output(item)
            if violation:
                return violation
        return None
    # int / float / bool / other scalars carry no credential pattern.
    return None


def _structured_leaves_are_scalars(content: Any, _depth: int = 0) -> bool:
    """True when every leaf of the structured product is a plain scalar.

    The structured answer is documented as scalars only: strings and numbers
    copied verbatim from a knowledge-base entry. An opaque object reaching a
    leaf would be released without ever having been pattern-scanned, because
    the credential scan skips non-string leaves by design.
    """
    if _depth > 4:
        return False
    if content is None or isinstance(content, (str, int, float, bool)):
        return True
    if isinstance(content, dict):
        return all(
            isinstance(key, str) and _structured_leaves_are_scalars(value, _depth + 1) for key, value in content.items()
        )
    if isinstance(content, (list, tuple)):
        return all(_structured_leaves_are_scalars(item, _depth + 1) for item in content)
    return False


class PostProcessNode(FunctionNode):
    """Output gate: enforce both invariants on the answer and structured product.

    The outer backbone post_process slot. Reads state["result"] (the merged
    formatted_answer) and state["structured_answer"] (JSON), and applies the
    module-level gate to both before the response is returned to the caller.

    Input state keys:
        result:            final formatted answer (from merge_output)
        structured_answer: JSON structured product (from merge_output)

    Output state keys (partial dict):
        formatted_output:  the released answer, or the blocked stub
        result:            gated alongside formatted_output
        the output-bearing fields: cleared on a violation
        status:            success or error value (plain strings)
        error_log:         (on error) list of error messages
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: AgentState) -> Dict[str, Any]:
        result = state.get("result") or ""
        structured = from_json(state.get("structured_answer"), None)

        if not result or not str(result).strip():
            # No answer was generated - forward as-is (non-fatal).
            return {
                "formatted_output": result,
                "status": AgentStatus.SUCCESS.value,
            }

        answer = str(result)
        violation = _security_gate_output(answer) or _security_gate_output(structured)
        if violation is None and structured is not None and not _structured_leaves_are_scalars(structured):
            violation = "structured_shape"
        if violation is None and SCOPE_NOTICE_MARKER not in answer:
            violation = "scope_notice_missing"

        if violation:
            logger.error("PostProcessNode: output blocked - violation type: %s", violation)
            return self._blocked(violation)

        # Clean - audit that a finalized answer was emitted.
        emit_trace_event(
            "post_process_complete",
            {"output_chars": len(answer)},
            state,
        )

        return {
            "formatted_output": result,
            "status": AgentStatus.SUCCESS.value,
        }

    @staticmethod
    def _blocked(violation: str) -> Dict[str, Any]:
        """Return the error delta, clearing every field that could carry text.

        The error message names the violation CLASS - fixed vocabulary from
        this module - and never the matched text, the answer, or a path.
        """
        delta: Dict[str, Any] = {
            "formatted_output": _SANITISED_STUB,
            "result": _SANITISED_STUB,
            "status": AgentStatus.ERROR.value,
            "error_log": [
                f"PostProcessNode: output blocked - the generated answer did not "
                f"clear the output gate ({violation})"
            ],
        }
        for field in _OUTPUT_BEARING_FIELDS:
            delta[field] = None
        return delta
