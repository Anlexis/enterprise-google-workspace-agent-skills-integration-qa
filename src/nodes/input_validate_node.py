"""AgentCore Platform v1.0"""

# CMN-C2-298 - InputValidateNode
# Domain node 1: parse and normalise the incoming Google Agent Skills
# integration question, and resolve the caller's retrieval filters.
#
# Two channels reach this node, and both are validated by the same contract
# module (src/schemas/caller_contract.py) so they cannot drift apart:
#
#   input_context   the structured channel - {"skill", "frameworks", "top_k",
#                   "channel"}. Carried across the outer -> inner graph
#                   boundary by the ContextVar bridge, because the framework's
#                   GraphNode does not forward it. This is the preferred
#                   channel and it wins on conflict.
#
#   the question    plain text, or a JSON envelope
#                   {"query": "...", "skill": "...", "top_k": N} for callers
#                   that have only the single string field available.
#
# Both are refused fail-closed when a field is out of contract: a caller who
# asked for a specific narrowing and got it wrong is not served by silently
# answering a different question.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

import json
import re
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.caller_contract import (
    MAX_QUERY_CHARS,
    TOP_K_MAX,
    TOP_K_MIN,
    finite_int_in_range,
    inert_identifier,
    screen_injection,
    validate_context,
)
from src.schemas.state import to_json

_WHITESPACE_RE = re.compile(r"\s+")

# Known Google Agent Skills names. A filter outside this set is still an inert
# identifier and is not refused - it simply matches no knowledge-base entry, so
# retrieval degrades to the unfiltered pool.
_KNOWN_SKILLS = frozenset(
    {
        "workspace",
        "cloud",
        "maps",
        "youtube",
        "calendar",
        "gmail",
        "sheets",
        "drive",
        "docs",
        "search",
    }
)

# Compliance frameworks carried by the Japan compliance knowledge base.
_KNOWN_FRAMEWORKS = frozenset({"appi", "fisc", "isms"})


class InputValidateNode(FunctionNode):
    """Parse the request into a normalised question plus bounded filters.

    Input state keys:
        validated_input | user_input: the request payload
        input_context:                structured caller filters (bridged in)

    Output state keys (partial dict):
        search_query:  normalised free-text integration question
        query_filters: JSON dict {"skill": str|None, "frameworks": [str],
                       "top_k": int|None}
        intake_notes:  (when anomalies were seen) JSON list[str]
        status/error_log: on a fail-closed refusal
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        raw = state.get("validated_input") or state.get("user_input", "")
        notes: List[str] = []

        # Structured channel first - it wins on conflict.
        context_filters, invalid_fields = validate_context(state.get("input_context", {}))
        if invalid_fields:
            return self._refuse("request context rejected - out-of-contract field(s): " + ", ".join(invalid_fields))

        query = ""
        envelope_skill: Optional[str] = None
        envelope_top_k: Optional[int] = None

        if isinstance(raw, str) and raw.strip():
            text = raw.strip()
            payload: Any = None
            if text.startswith("{"):
                try:
                    payload = json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    notes.append(
                        "InputValidateNode: JSON-looking input did not parse - " "treated as plain text question."
                    )
            if isinstance(payload, dict):
                query = str(payload.get("query") or payload.get("question") or "")
                if payload.get("skill") is not None:
                    envelope_skill = inert_identifier(payload.get("skill"))
                    if envelope_skill is None:
                        return self._refuse("request rejected - out-of-contract field(s): skill")
                if payload.get("top_k") is not None:
                    envelope_top_k = finite_int_in_range(payload.get("top_k"), TOP_K_MIN, TOP_K_MAX)
                    if envelope_top_k is None:
                        return self._refuse("request rejected - out-of-contract field(s): top_k")
            else:
                query = text
        else:
            notes.append("InputValidateNode: empty request - no question to answer.")

        # Defence in depth: the outer gate screens the question already, but a
        # direct inner-graph invocation has no outer gate in front of it.
        violation = screen_injection(query)
        if violation:
            return self._refuse(
                f"request refused - the question contains disallowed " f"instruction content ({violation})"
            )

        query = _WHITESPACE_RE.sub(" ", query).strip()
        if len(query) > MAX_QUERY_CHARS:
            query = query[:MAX_QUERY_CHARS]
            notes.append(f"InputValidateNode: query truncated to {MAX_QUERY_CHARS} chars.")

        skill = context_filters.get("skill", envelope_skill)
        top_k = context_filters.get("top_k", envelope_top_k)
        frameworks: List[str] = list(context_filters.get("frameworks", []))

        # Unrecognised-but-inert selectors are reported by FIELD, never by
        # value: the value is caller content and this note travels with the
        # answer.
        if skill is not None and skill not in _KNOWN_SKILLS:
            notes.append(
                "InputValidateNode: the skill filter is not a recognised Google "
                "Agent Skills name - retrieval falls back to the unfiltered pool."
            )
        if frameworks and not set(frameworks) & _KNOWN_FRAMEWORKS:
            notes.append(
                "InputValidateNode: no requested compliance framework is present "
                "in the knowledge base - compliance guidance is not narrowed."
            )

        filters = {"skill": skill, "frameworks": frameworks, "top_k": top_k}

        # Audit: request parsed and normalised.
        emit_trace_event(
            "input_validate_complete",
            {
                "query_chars": len(query),
                "has_skill_filter": skill is not None,
                "framework_filters": len(frameworks),
                "has_top_k_override": top_k is not None,
            },
            state,
        )

        out: Dict[str, Any] = {
            "search_query": query,
            "query_filters": to_json(filters),
        }
        if notes:
            out["intake_notes"] = to_json(notes)
        return out

    @staticmethod
    def _refuse(reason: str) -> Dict[str, Any]:
        """Fail closed. The reason names fields and classes, never values."""
        return {
            "status": AgentStatus.ERROR.value,
            "error_log": [f"InputValidateNode: {reason}"],
        }
