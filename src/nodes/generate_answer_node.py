"""AgentCore Platform v1.0"""

# CMN-C2-298 - GenerateAnswerNode
# Domain node 4: assemble the grounded answer AND the structured extract
# (integration steps / OAuth scopes / compliance flags / testing checklist /
# related skills) from the ranked KB passages.
#
# v1 is DETERMINISTIC (no live LLM call, no live call to any Google API):
# grounded_answer is rule-assembled from the ranked passages only - a lead
# sentence plus one cited point per passage, each carrying a numbered
# citation marker [n]. The structured extract is a pure AGGREGATION of
# fields the KB entries already carry (integration_steps / oauth_scopes /
# compliance_flags / testing_checklist / related_skills are pre-authored in
# config/kb/*.json, never synthesised here) - nothing outside the
# ranked_documents input reaches the output, so both are grounded by
# construction. The LLM synthesis upgrade seam is documented in
# docs/02_design.md ("v1 Implementation Note - LLM synthesis") and
# config/prompts/answer_synthesis_prompt.md: a v2 node swaps the assembly
# for an LLM call over the same input and emits the same state contract.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json, to_json

# Answer body used when no KB passage cleared the relevance threshold.
_NO_COVERAGE_ANSWER = (
    "The seeded Google Agent Skills / integration-pattern / Japan compliance "
    "knowledge base does not contain sufficient coverage to answer this "
    "question. Rephrase the query with a more specific skill name (e.g. "
    "gmail, drive, calendar) or compliance framework (APPI, FISC, ISMS), or "
    "escalate to the integration/compliance team for a manual review."
)

# Cited excerpt length per passage inside the answer body.
_POINT_EXCERPT_CHARS = 240

# Caps on aggregated structured-extract lists (keeps State small and the
# answer readable; every item is still grounded - this only trims volume).
_MAX_INTEGRATION_STEPS = 10
_MAX_OAUTH_SCOPES = 12
_MAX_COMPLIANCE_FLAGS = 10
_MAX_TESTING_ITEMS = 10
_MAX_RELATED_SKILLS = 8

# Human-readable label per source KB, used in the rendered answer body.
_KB_LABELS = {
    "google_agent_skills": "Google Agent Skills",
    "integration_patterns": "Integration Pattern",
    "japan_compliance": "Japan Compliance",
}


def _first_sentences(text: str, limit: int) -> str:
    """Trim an excerpt at a sentence boundary where possible, else hard-cap."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    period = cut.rfind(". ")
    if period > limit // 2:
        return cut[: period + 1]
    return cut.rstrip() + "..."


def _dedupe_strings(items: List[Any]) -> List[str]:
    """Order-preserving de-duplication of a list of strings."""
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _dedupe_dicts(items: List[Any], key: str) -> List[Dict[str, Any]]:
    """Order-preserving de-duplication of a list of dicts by a string key."""
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        marker = str(item.get(key, ""))
        if not marker or marker in seen:
            continue
        seen.add(marker)
        out.append(item)
    return out


class GenerateAnswerNode(FunctionNode):
    """Rule-based grounded answer + structured-extract assembly.

    Input state keys:
        ranked_documents: JSON list of surviving passages (from RerankFilterNode)
        search_query:     normalised query (for the lead sentence)

    Output state keys (partial dict):
        grounded_answer:    answer body with [n] citation markers
        citations:          JSON list [{ref, id, title, source, kb}]
        structured_extract: JSON dict {integration_steps, oauth_scopes,
                             compliance_flags, testing_checklist,
                             related_skills} - aggregated verbatim from the
                             ranked KB entries, nothing synthesised
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        ranked: List[Dict[str, Any]] = from_json(state.get("ranked_documents"), []) or []
        query = state.get("search_query") or ""

        citations: List[Dict[str, Any]] = []
        integration_steps: List[str] = []
        oauth_scopes: List[Dict[str, Any]] = []
        compliance_flags: List[Dict[str, Any]] = []
        testing_checklist: List[str] = []
        related_skills: List[Dict[str, Any]] = []

        if not ranked:
            grounded_answer = _NO_COVERAGE_ANSWER
        else:
            lines: List[str] = []
            if query:
                lines.append(
                    f"Based on the seeded Google Agent Skills integration knowledge "
                    f'base, the following passages answer the question: "{query}"'
                )
            else:
                lines.append(
                    "Based on the seeded Google Agent Skills integration knowledge "
                    "base, the most relevant passages are:"
                )
            lines.append("")
            for ref, doc in enumerate(ranked, start=1):
                if not isinstance(doc, dict):
                    continue
                title = str(doc.get("title", "")).strip()
                excerpt = _first_sentences(str(doc.get("excerpt", "")), _POINT_EXCERPT_CHARS)
                kb = str(doc.get("kb", ""))
                label = _KB_LABELS.get(kb, kb or "KB")
                lines.append(f"[{ref}] ({label}) {title}: {excerpt}")
                citations.append(
                    {
                        "ref": ref,
                        "id": str(doc.get("id", "")),
                        "title": title,
                        "source": str(doc.get("source", "")),
                        "kb": kb,
                    }
                )
            grounded_answer = "\n".join(lines)

            # ── Deterministic aggregation of pre-authored KB fields ──────────
            # Every value below is copied verbatim from a ranked candidate,
            # which RetrieveNode already carried through from the KB entry
            # (config/kb/*.json) - nothing is generated here. This is
            # deliberately not a synthesis step: the agent holds no model client.
            for doc in ranked:
                if not isinstance(doc, dict):
                    continue
                kb = str(doc.get("kb", ""))
                if kb in ("google_agent_skills", "integration_patterns"):
                    integration_steps.extend(doc.get("integration_steps") or [])
                if kb == "google_agent_skills":
                    for scope in doc.get("oauth_scopes") or []:
                        if isinstance(scope, dict):
                            oauth_scopes.append(
                                {
                                    "skill": doc.get("skill"),
                                    "scope": scope.get("scope"),
                                    "purpose": scope.get("purpose"),
                                }
                            )
                    for rel in doc.get("related_skills") or []:
                        if isinstance(rel, dict):
                            related_skills.append(rel)
                if kb == "japan_compliance":
                    compliance_flags.append(
                        {
                            "framework": doc.get("framework"),
                            "requirement": doc.get("title"),
                            "guidance": doc.get("guidance"),
                        }
                    )
                testing_checklist.extend(doc.get("testing_notes") or [])

        structured_extract: Dict[str, Any] = {
            "integration_steps": _dedupe_strings(integration_steps)[:_MAX_INTEGRATION_STEPS],
            "oauth_scopes": _dedupe_dicts(oauth_scopes, "scope")[:_MAX_OAUTH_SCOPES],
            "compliance_flags": _dedupe_dicts(compliance_flags, "requirement")[:_MAX_COMPLIANCE_FLAGS],
            "testing_checklist": _dedupe_strings(testing_checklist)[:_MAX_TESTING_ITEMS],
            "related_skills": _dedupe_dicts(related_skills, "skill")[:_MAX_RELATED_SKILLS],
        }

        # Audit: grounded answer + structured extract assembled.
        emit_trace_event(
            "generate_answer_complete",
            {
                "citation_count": len(citations),
                "answer_chars": len(grounded_answer),
                "no_coverage": not ranked,
                "oauth_scope_count": len(structured_extract["oauth_scopes"]),
                "compliance_flag_count": len(structured_extract["compliance_flags"]),
            },
            state,
        )

        return {
            "grounded_answer": grounded_answer,
            "citations": to_json(citations),
            "structured_extract": to_json(structured_extract),
        }
