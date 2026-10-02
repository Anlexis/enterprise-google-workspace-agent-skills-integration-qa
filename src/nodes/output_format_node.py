"""AgentCore Platform v1.0"""

# CMN-C2-298 - OutputFormatNode
# Domain node 5 (terminal): compose the final formatted answer - the grounded
# answer body, the Sources list, and the standing scope/limitations notice -
# and assemble the combined structured_answer object (the structured_extract
# fields plus citations) that WorkspaceSkillsIntegrationQAAgent.get_output()
# surfaces as the structured product. Composing the notice is THIS node's
# responsibility; the outer post_process slot gates, it does not compose.
#
# Over-claiming guard: this agent answers *how to integrate* Google's Agent
# Skills - it never connects to, authenticates against, or calls a Google API,
# and makes no configuration changes. The notice below is appended to EVERY
# answer so that the boundary is visible at the point of use rather than only
# in the documentation, and the output gate REFUSES to release an answer that
# does not carry it (see src/nodes/post_process_node.py).
#
# Wired by the inner graph (DomainWorkflowGraph). The inner graph's
# get_output() surfaces formatted_answer + structured_answer + status to the
# outer merge_output(). Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.nodes.post_process_node import SCOPE_NOTICE_MARKER
from src.schemas.state import from_json, to_json

# Standing scope/limitations notice - appended to EVERY answer this agent
# emits. PostProcessNode enforces its presence at the output boundary and
# matches on SCOPE_NOTICE_MARKER, which is spliced into the sentence below
# rather than duplicated: the notice cannot be reworded past the gate without
# the no-live-call clause travelling with it.
_SCOPE_NOTICE = (
    "This answer is advisory integration guidance only. It "
    f"{SCOPE_NOTICE_MARKER} Workspace/Cloud/Maps/YouTube/Search "
    "API, and it makes no configuration changes on your behalf. Verify scopes, "
    "consent-screen contents, and compliance posture against your Google Workspace "
    "Admin console, Google Cloud IAM, and your compliance/legal team before "
    "implementation."
)

_KB_LABELS = {
    "google_agent_skills": "Google Agent Skills",
    "integration_patterns": "Integration Pattern",
    "japan_compliance": "Japan Compliance",
}


class OutputFormatNode(FunctionNode):
    """Compose the final answer: body + sources + scope notice, and the
    combined structured_answer object.

    Input state keys:
        grounded_answer:    answer body with [n] citation markers
        citations:          JSON list [{ref, id, title, source, kb}]
        structured_extract: JSON dict {integration_steps, oauth_scopes,
                             compliance_flags, testing_checklist,
                             related_skills}

    Output state keys (partial dict):
        formatted_answer: final rendered answer string
        structured_answer: JSON dict - structured_extract + citations
                            combined into the single structured product
                            object (State field shared with the outer
                            WorkspaceSkillsIntegrationQAAgent.get_output()
                            override)
        status: the success value as a plain string (never the bare enum,
                which State cannot serialize)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        grounded_answer = state.get("grounded_answer") or ("No answer is available for this request.")
        citations: List[Dict[str, Any]] = from_json(state.get("citations"), []) or []
        structured_extract: Dict[str, Any] = from_json(state.get("structured_extract"), {}) or {}

        lines: List[str] = []
        lines.append("# Google Agent Skills Integration Guidance")
        lines.append("")
        lines.append(grounded_answer)
        lines.append("")
        lines.append("## Sources")
        if citations:
            for citation in citations:
                if not isinstance(citation, dict):
                    continue
                ref = citation.get("ref", "?")
                title = str(citation.get("title", "")).strip()
                source = str(citation.get("source", "")).strip()
                kb_label = _KB_LABELS.get(str(citation.get("kb", "")), "")
                suffix = f" ({source})" if source else ""
                kb_suffix = f" [{kb_label}]" if kb_label else ""
                lines.append(f"- [{ref}] {title}{suffix}{kb_suffix}")
        else:
            lines.append("- none (no knowledge-base passage cleared the relevance threshold)")
        lines.append("")
        lines.append("## Scope & Limitations")
        lines.append("")
        lines.append(f"*{_SCOPE_NOTICE}*")

        formatted_answer = "\n".join(lines)

        # The combined structured product: structured_extract's five
        # aggregated lists plus citations, all allowlisted and scalars-only at
        # the leaf. The outer get_output() override re-scans this before
        # surfacing it, fail-closed.
        structured_answer: Dict[str, Any] = {
            "integration_steps": structured_extract.get("integration_steps", []),
            "oauth_scopes": structured_extract.get("oauth_scopes", []),
            "compliance_flags": structured_extract.get("compliance_flags", []),
            "testing_checklist": structured_extract.get("testing_checklist", []),
            "related_skills": structured_extract.get("related_skills", []),
            "citations": citations,
        }

        # Audit: final answer + structured product composed.
        emit_trace_event(
            "output_format_complete",
            {
                "answer_chars": len(formatted_answer),
                "citation_count": len(citations),
            },
            state,
        )

        return {
            "formatted_answer": formatted_answer,
            "structured_answer": to_json(structured_answer),
            "status": AgentStatus.SUCCESS.value,
        }
