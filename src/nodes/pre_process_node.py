"""AgentCore Platform v1.0"""

# CMN-C2-298 - PreProcessNode (outer pre_process slot)
#
# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus.<X>.value strings for status assignments
#  - Read input_context via state.get("input_context", {}) - read-only
#  - Never import from mediator/, api/, or other agents
#
# This node owns the caller contract for the whole agent. It is the outer
# backbone gate, so required_trust_level is raised to VERIFIED_EXTERNAL and an
# unauthenticated or unverified caller is denied before the retrieval workflow
# runs at all.
#
# What it enforces, in order:
#
#   1. A non-empty question.
#   2. Prompt-injection refusal on the RAW question. The template owns this
#      guarantee rather than delegating it to the framework input gate: that
#      gate is active only where the deployment configures it, and where it is
#      absent the payload would reach the answer path and return a normal
#      success - fail-open on exactly the input that must not succeed. Enforced
#      here, refusal holds in every deployment, and it is observable by calling
#      execute() directly with no framework wrapper in front.
#   3. The request-context contract: each accepted field parsed against
#      explicit bounds, refused by NAME when it does not fit. The rejected
#      value is never echoed - not into error_log, not into an audit event.
#   4. A surface redaction pass over the question (e-mail addresses, long digit
#      runs that look like phone or account numbers) on top of the framework's
#      own masking.
#   5. Prompt-injection refusal AGAIN, on the redacted text. A redaction pass
#      is not a refusal, and it can make an attack harder to see rather than
#      easier: stripping a control token out of a hostile string leaves the
#      directive behind as ordinary prose. Screening both forms catches the
#      token before it is removed and the directive after the removal splices
#      it back together.

import re
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.caller_contract import (
    screen_injection,
    screen_structure,
    validate_context,
)

# Surface-level identifier patterns redacted before validated_input is written.
_PII_PATTERNS: List["re.Pattern[str]"] = [
    re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{2,11}\b"),  # long digit runs (phone/account-like)
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),  # e-mail addresses
]
_PII_REPLACEMENT = "[REDACTED]"


def _surface_strip_identifiers(text: str) -> str:
    """Redact obvious direct-identifier tokens from a free-text string."""
    for pattern in _PII_PATTERNS:
        text = pattern.sub(_PII_REPLACEMENT, text)
    return text


class PreProcessNode(FunctionNode):
    """Trust gate, caller-contract validation, injection screen, identifier strip."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: AgentState) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        input_context = state.get("input_context", {})  # read-only

        if not isinstance(user_input, str) or not user_input.strip():
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["PreProcessNode: user_input is empty or missing"],
            }

        # Screen the question as it arrived.
        violation = screen_injection(user_input)
        if violation:
            return self._refuse_injection("user_input", violation)

        # Screen the request context depth first, keys included. A directive
        # can ride in a field NAME, and a scan that walks only values will not
        # see it; running on the parsed structure also means JSON \u escapes
        # are already ordinary characters by the time they are inspected.
        violation = screen_structure(input_context)
        if violation:
            return self._refuse_injection("input_context", violation)

        filters, invalid_fields = validate_context(input_context)
        if invalid_fields:
            # Name the fields, never the values: a rejected value is caller
            # content and echoing it would put it back into the response.
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [
                    "PreProcessNode: request context rejected - out-of-contract "
                    f"field(s): {', '.join(invalid_fields)}"
                ],
            }

        validated_input = _surface_strip_identifiers(user_input.strip())

        # Screen again after redaction - see the module header.
        violation = screen_injection(validated_input)
        if violation:
            return self._refuse_injection("user_input", violation)

        # Audit: a request was accepted and surface-redacted.
        emit_trace_event(
            "pre_process_complete",
            {
                "input_chars": len(validated_input),
                "context_fields": len(filters),
            },
            state,
        )

        enriched: Dict[str, Any] = {
            "source": "WorkspaceSkillsIntegrationQAAgent",
            # Inert by construction: validate_context() only ever returns a
            # lowercase [a-z0-9_] identifier for this field.
            "channel": filters.get("channel", "unknown"),
        }
        return {
            "validated_input": validated_input,
            "enriched_context": enriched,
            "status": AgentStatus.SUCCESS.value,
        }

    @staticmethod
    def _refuse_injection(field: str, violation: str) -> Dict[str, Any]:
        """Fail closed on hostile content: nothing carried forward, no echo.

        The message carries the field and the violation CLASS - both fixed
        vocabulary from this module - and never the matched text.
        """
        return {
            "status": AgentStatus.ERROR.value,
            "error_log": [
                f"PreProcessNode: request refused - {field} contains " f"disallowed instruction content ({violation})"
            ],
        }
