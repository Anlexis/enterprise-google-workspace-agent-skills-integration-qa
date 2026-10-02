"""AgentCore Platform v1.0"""

# CMN-C2-298 - RetrieveNode
# Domain node 2: deterministic keyword retrieval pooled across the THREE
# seeded, pre-indexed knowledge bases (config/kb/*.json): Google Agent Skills
# documentation, enterprise integration patterns, and Japan compliance
# requirements (APPI/FISC/ISMS). Retrieval is fully deterministic - no
# embedding model, no vector store, and no live call to any Google API; the
# retrieval contract (retrieved_documents JSON) is store-agnostic, so swapping
# in a vector store changes only this node's internals.
#
# Configuration reaches this node through State: the outer graph forwards the
# `retrieval` block from config/config.yaml, and the inner graph republishes it
# as the retrieval_config field. Module defaults, which mirror the same file,
# apply when it is absent.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

import json
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json, to_json

# Defaults mirror the `retrieval` block in config/config.yaml.
_DEFAULT_RETRIEVAL: Dict[str, Any] = {
    "top_k": 6,
    "score_threshold": 0.15,
    "kb_paths": [
        "config/kb/google_agent_skills_kb.json",
        "config/kb/integration_patterns_kb.json",
        "config/kb/japan_compliance_kb.json",
    ],
}

# Repo root: src/nodes/retrieve_node.py -> parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Minimal stopword set for query tokenisation (deterministic, no NLP deps).
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "is",
        "are",
        "be",
        "with",
        "under",
        "what",
        "which",
        "when",
        "how",
        "do",
        "does",
        "must",
        "should",
        "before",
        "after",
        "by",
        "at",
        "from",
        "that",
        "this",
        "it",
        "as",
        "was",
        "were",
        "can",
        "may",
        "any",
        "our",
        "we",
        "i",
        "you",
        "your",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Per-field match weights: a query token found in the title counts more than
# one found only in the body content.
_TITLE_WEIGHT = 1.0
_TAG_WEIGHT = 0.8
_CONTENT_WEIGHT = 0.5

# Excerpt length carried into retrieved_documents (keeps State small).
_EXCERPT_CHARS = 400


def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens, stopwords and 1-2 char noise removed."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2 and t not in _STOPWORDS]


def _resolve_retrieval_config(state: AgentState) -> Dict[str, Any]:
    """Effective retrieval config: state retrieval_config > module defaults.

    execute(self, state) takes no config parameter, so State seeding is the
    only channel from the `retrieval` block in config/config.yaml:
    IntegrationQAGraphNode._parent_config() ->
    DomainWorkflowGraph._extra_initial_state() -> this state field.
    """
    effective = dict(_DEFAULT_RETRIEVAL)  # local copy - never mutate the module default
    from_state = from_json(state.get("retrieval_config"), None)
    if isinstance(from_state, dict):
        effective.update(from_state)
    return effective


def _load_kb(kb_paths: List[str]) -> tuple[List[Dict[str, Any]], List[str]]:
    """Load and pool all seeded KB JSON files. A missing/malformed file
    degrades gracefully (skipped with a note) rather than failing the run -
    the other two KBs still serve the request."""
    notes: List[str] = []
    entries: List[Dict[str, Any]] = []
    for kb_path in kb_paths:
        path = Path(kb_path)
        if not path.is_absolute():
            path = _REPO_ROOT / path
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            notes.append(f"RetrieveNode: knowledge base not readable at {kb_path}.")
            continue
        if not isinstance(loaded, list):
            notes.append(f"RetrieveNode: knowledge base root must be a JSON list ({kb_path}).")
            continue
        entries.extend(e for e in loaded if isinstance(e, dict))
    return entries, notes


def _score_entry(entry: Dict[str, Any], query_tokens: List[str]) -> float:
    """Per-entry relevance: best field-weight per query token, averaged."""
    if not query_tokens:
        return 0.0
    title_tokens = set(_tokenize(str(entry.get("title", ""))))
    tag_tokens = set(_tokenize(" ".join(str(t) for t in entry.get("tags", []))))
    content_tokens = set(_tokenize(str(entry.get("content", ""))))
    total = 0.0
    for token in query_tokens:
        if token in title_tokens:
            total += _TITLE_WEIGHT
        elif token in tag_tokens:
            total += _TAG_WEIGHT
        elif token in content_tokens:
            total += _CONTENT_WEIGHT
    return round(total / len(query_tokens), 4)


class RetrieveNode(FunctionNode):
    """Score the three seeded KBs against the search query and emit candidates.

    Input state keys:
        search_query:     normalised query (from InputValidateNode)
        query_filters:    JSON dict with the optional skill / frameworks filters
        retrieval_config: forwarded runtime retrieval block (JSON)

    Output state keys (partial dict):
        retrieved_documents: JSON list of scored candidates (score desc),
                              each tagged with its source `kb`
        intake_notes:        (on KB anomalies) JSON list[str]
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        query = state.get("search_query") or state.get("validated_input") or state.get("user_input", "")
        filters = from_json(state.get("query_filters"), {}) or {}
        retrieval_cfg = _resolve_retrieval_config(state)

        try:
            top_k = int(retrieval_cfg.get("top_k", _DEFAULT_RETRIEVAL["top_k"]))
        except (TypeError, ValueError):
            top_k = int(_DEFAULT_RETRIEVAL["top_k"])
        top_k = max(1, min(20, top_k))

        kb_paths = retrieval_cfg.get("kb_paths", _DEFAULT_RETRIEVAL["kb_paths"])
        if not isinstance(kb_paths, list) or not kb_paths:
            kb_paths = list(_DEFAULT_RETRIEVAL["kb_paths"])
        entries, notes = _load_kb(kb_paths)

        # Skill filter narrows the google_agent_skills-specific pool only -
        # integration_patterns / japan_compliance entries have no `skill`
        # field and always stay in play (score-gated only), since a
        # skill-specific question ("the Gmail skill") still needs the
        # cross-cutting compliance/pattern guidance alongside it.
        skill = filters.get("skill")
        if skill:
            entries = [
                e for e in entries if not e.get("skill") or str(e.get("skill", "")).lower() == str(skill).lower()
            ]

        # Compliance-framework filter narrows the japan_compliance pool only.
        # Entries from the other two knowledge bases carry no `framework` field
        # and stay in play, score-gated: a question scoped to one framework
        # still needs the skill and integration-pattern guidance beside it.
        frameworks = filters.get("frameworks") or []
        if isinstance(frameworks, list) and frameworks:
            wanted = {str(f).lower() for f in frameworks}
            entries = [e for e in entries if not e.get("framework") or str(e.get("framework", "")).lower() in wanted]

        query_tokens = _tokenize(query if isinstance(query, str) else "")

        candidates: List[Dict[str, Any]] = []
        for entry in entries:
            score = _score_entry(entry, query_tokens)
            if score <= 0.0:
                continue
            candidates.append(
                {
                    "id": str(entry.get("id", "")),
                    "kb": str(entry.get("kb", "")),
                    "skill": entry.get("skill"),
                    "framework": entry.get("framework"),
                    "title": str(entry.get("title", "")),
                    "source": str(entry.get("source", "")),
                    "score": score,
                    "excerpt": str(entry.get("content", ""))[:_EXCERPT_CHARS],
                    # Carried through verbatim so GenerateAnswerNode can
                    # aggregate the structured extract without a second KB
                    # file read - every value here is copied from the KB
                    # entry, never synthesised.
                    "integration_steps": entry.get("integration_steps") or [],
                    "oauth_scopes": entry.get("oauth_scopes") or [],
                    "related_skills": entry.get("related_skills") or [],
                    "testing_notes": entry.get("testing_notes") or [],
                    "guidance": entry.get("guidance"),
                }
            )

        # Deterministic ordering: score desc, then id asc for stable ties.
        candidates.sort(key=lambda c: (-c["score"], c["id"]))
        # Keep a candidate pool wider than top_k - RerankFilterNode makes
        # the final cut after the skill boost + threshold.
        pool_size = max(top_k * 3, 10)
        candidates = candidates[:pool_size]

        # Audit: retrieval pass completed.
        emit_trace_event(
            "retrieve_complete",
            {
                "candidates": len(candidates),
                "kb_entries": len(entries),
                "kb_count": len(kb_paths),
                "query_tokens": len(query_tokens),
                "framework_filters": len(frameworks) if isinstance(frameworks, list) else 0,
                "top_k": top_k,
            },
            state,
        )

        out: Dict[str, Any] = {"retrieved_documents": to_json(candidates)}
        if notes:
            # Append to (never clobber) the notes accumulated upstream.
            prior = from_json(state.get("intake_notes"), []) or []
            out["intake_notes"] = to_json(list(prior) + notes)
        return out
