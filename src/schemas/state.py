"""AgentCore Platform v1.0"""

# State must be a flat TypedDict - never a Pydantic model. Checkpoints use
# msgpack serialization, and a Pydantic object there corrupts silently. Extend
# the framework state with agent-specific fields only, and never add
# credentials, secrets, or model objects.
#
# Msgpack safety: structured fields (dict / list[dict]) are stored as JSON
# STRINGS, not bare Python containers - a bare dict or list in a checkpointed
# State field does not survive the round trip. Producers serialize with
# to_json() on write; consumers deserialize with from_json() on read.
#
# CMN-C2-298 - WorkspaceSkillsIntegrationQAAgent (Cat 2 RAG). Two-layer
# nested graph: outer backbone (AgentBaseGraph) plus an inner domain workflow
# (BaseGraph). The fields below cover both layers.
#
# PII / confidentiality note: this template answers *how to integrate*
# Google's Agent Skills - it never calls a Google API and never holds a
# live OAuth token or credential. State only ever carries the caller's
# free-text question, retrieved KB passages, and the assembled advisory
# answer (which itself only ever names scope IDENTIFIERS, never issued
# tokens - see src/nodes/post_process_node.py _security_gate_output()).

import json
from typing import Any, NotRequired, Optional

from framework.schemas.agent_state import AgentState


def to_json(value: Any) -> Optional[str]:
    """Serialize a dict/list State field to a JSON string (msgpack safety).

    None passes through unchanged so an 'unset' field stays distinguishable
    from an empty container.
    """
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialize a JSON-string State field back to its dict/list.

    None / empty / malformed input -> the supplied ``default`` so a missing or
    corrupt field is non-fatal for the consuming node.
    """
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class State(AgentState):
    """Flat TypedDict for CMN-C2-298.

    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, etc.) are inherited from AgentState.
    Domain fields are NotRequired so the TypedDict is valid at graph
    initialisation, before any node has written a value.
    """

    # ------------------------------------------------------------------
    # Outer layer - set by PreProcessNode / IntegrationQAGraphNode.merge_output
    # ------------------------------------------------------------------

    # Identifier-stripped, validated question payload produced by
    # PreProcessNode. Raw input is NOT persisted beyond that node.
    validated_input: NotRequired[str]

    # Final integration-advisory answer, mapped from the inner graph's
    # formatted_answer output via merge_output.
    integration_answer: NotRequired[str]

    # JSON STRING (to_json) of the structured answer surfaced by
    # WorkspaceSkillsIntegrationQAAgent.get_output() on SUCCESS only.
    # Deserialised shape: {"integration_steps": [str, ...],
    # "oauth_scopes": [{"skill", "scope", "purpose"}, ...],
    # "compliance_flags": [{"framework", "requirement", "guidance"}, ...],
    # "testing_checklist": [str, ...],
    # "related_skills": [{"skill", "relation"}, ...],
    # "citations": [{"ref", "id", "title", "source", "kb"}, ...]}.
    # Mapped from the inner graph's structured_answer output via
    # merge_output. Every leaf value is a whitelisted scalar (str/int) -
    # never a raw or opaque object - the output gate refuses those.
    structured_answer: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Inner layer - domain nodes (DomainWorkflowGraph)
    # ------------------------------------------------------------------

    # InputValidateNode outputs
    # Normalised free-text integration question (whitespace-collapsed,
    # length-capped).
    search_query: NotRequired[str]

    # JSON STRING (to_json) of parsed structured query params. Deserialised
    # dict shape: {"skill": str | None, "top_k": int | None}. "skill" is a
    # google/skills name filter (e.g. "gmail", "drive", "calendar").
    # Consumers (RetrieveNode, RerankFilterNode) read it back via from_json().
    query_filters: NotRequired[Optional[str]]

    # Manifest `retrieval` block forwarded by IntegrationQAGraphNode.
    # _parent_config() -> DomainWorkflowGraph._extra_initial_state().
    # JSON STRING (to_json) of {"top_k": int, "score_threshold": float,
    # "kb_paths": [str, ...]}. Consumers (RetrieveNode, RerankFilterNode)
    # read it back via from_json().
    retrieval_config: NotRequired[Optional[str]]

    # RetrieveNode output
    # JSON STRING (to_json) of scored KB candidates pooled across the three
    # seeded KBs. Deserialised shape: list[dict], each entry {"id": str,
    # "kb": str, "skill": str | None, "framework": str | None, "title": str,
    # "source": str, "score": float, "excerpt": str}.
    # Consumers (RerankFilterNode) read it back via from_json().
    retrieved_documents: NotRequired[Optional[str]]

    # RerankFilterNode output
    # JSON STRING (to_json) of reranked + threshold-filtered passages, capped
    # at top_k. Same entry shape as retrieved_documents.
    # Consumers (GenerateAnswerNode) read it back via from_json().
    ranked_documents: NotRequired[Optional[str]]

    # GenerateAnswerNode outputs
    # Rule-assembled grounded answer body with numbered citation markers.
    grounded_answer: NotRequired[str]

    # JSON STRING (to_json) of citations. Deserialised shape: list[dict],
    # each entry {"ref": int, "id": str, "title": str, "source": str,
    # "kb": str}. Consumers (OutputFormatNode) read it back via from_json().
    citations: NotRequired[Optional[str]]

    # JSON STRING (to_json) of the deterministic aggregation of the ranked
    # KB entries' own pre-authored structured fields (integration_steps /
    # oauth_scopes / compliance_flags / testing_checklist / related_skills -
    # everything except citations, which stays its own field above).
    # Nothing here is synthesised - every value is copied verbatim from a
    # retrieved KB entry (grounded by construction). Deserialised shape:
    # {"integration_steps": [str, ...], "oauth_scopes": [{"skill", "scope",
    # "purpose"}, ...], "compliance_flags": [{"framework", "requirement",
    # "guidance"}, ...], "testing_checklist": [str, ...], "related_skills":
    # [{"skill", "relation"}, ...]}.
    # Consumers (OutputFormatNode) read it back via from_json().
    structured_extract: NotRequired[Optional[str]]

    # OutputFormatNode output
    # Final formatted answer (body + sources + scope/limitations notice).
    # Written by OutputFormatNode; surfaced to the outer graph via
    # get_output() -> merge_output().
    formatted_answer: NotRequired[str]

    # Validation / parse notes accumulated during intake (no PII).
    # JSON STRING (to_json) of list[str].
    intake_notes: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Tracing / audit - framework-managed; do NOT write from node code
    # ------------------------------------------------------------------

    trace_id: Optional[str]
    correlation_id: Optional[str]
    # node_history inherited from AgentState; listed here for clarity
    # node_history: Optional[List[str]]
