"""AgentCore Platform v1.0"""

# CMN-C2-298 - the caller-facing request contract.
#
# One place that answers "what may a caller send, in what form, and what is
# refused" - shared by the outer gate node (PreProcessNode), the inner intake
# node (InputValidateNode) and the HTTP adapter, so the two graph layers can
# never drift apart on what they accept.
#
# Three rules, in the order they are applied:
#
#   1. Every caller-controlled NUMBER is parsed by a finite + bounded parser.
#      float("nan") and float("inf") survive a plain int()/float() coercion,
#      and every comparison against NaN is False - so an unchecked numeric
#      silently disables the bound it was supposed to enforce instead of
#      failing. `_finite_int_in_range` rejects bools, non-numerics, the
#      non-finite values (both as raw JSON floats and as the literal strings
#      "NaN"/"Infinity") and anything outside the declared range.
#
#   2. Every caller-controlled STRING that selects or labels content is locked
#      to an inert identifier - lowercase alphanumerics and underscore, at most
#      32 characters. Free text on those fields would be caller-controlled
#      content in a document the agent renders; an inert alphabet cannot carry
#      markup, a directive, or a credential shape.
#
#   3. Free-text question content is screened for prompt-injection BEFORE and
#      AFTER any redaction pass. A redactor is not a refusal: removing a
#      control token from a hostile string leaves the directive behind as
#      ordinary prose, which is harder to detect, not easier. Screening the raw
#      text catches the token forms; screening the redacted text catches
#      directives that only become contiguous once something between them is
#      removed.
#
# Refusals name the FIELD, never the value - a rejected value is caller
# content and must not be echoed into an error message, a log line or an
# audit event.

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

# Inert alphabet for every caller string that selects or labels content.
_INERT_ID_RE = re.compile(r"^[a-z0-9_]{1,32}$")

# Field names are caller-controlled too. A name is quoted back only when it is
# short and inert; anything else is referred to by position.
_SAFE_FIELD_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

# Bounds for the caller result-count override.
TOP_K_MIN = 1
TOP_K_MAX = 20

# Cap on the compliance-framework filter list.
MAX_FRAMEWORKS = 5

# Cap on the free-text question after whitespace normalisation.
MAX_QUERY_CHARS = 2000

# Fields this agent reads off the request context. Anything else a caller (or
# the hosting runtime) puts there is ignored rather than rejected: the context
# mapping is not exclusively ours, and refusing unknown keys would break every
# request the moment the runtime adds one of its own.
CONTEXT_FIELDS = ("skill", "frameworks", "top_k", "channel")

# ── Prompt-injection screen ──────────────────────────────────────────────────
#
# Two families, kept deliberately narrow. This agent answers questions about
# enterprise integration and security configuration, so its own subject matter
# is full of words like "rules", "policy", "system" and "disable" - a loose
# screen would refuse the questions the agent exists to answer. Every pattern
# below therefore requires the directive to be addressed AT the agent.
_INJECTION_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    # Chat-template control tokens, as a CLASS rather than a list of known
    # spellings: any <|...|> delimiter, the bracketed instruction markers, and
    # the system-block markers. These carry no meaning in a question about
    # integrating a skill, and a payload that contains one is trying to open a
    # new turn inside the message rather than ask something.
    ("control_token", re.compile(r"<\|[^|<>\n]{0,64}\|>")),
    ("control_token", re.compile(r"\[/?INST\]", re.IGNORECASE)),
    ("control_token", re.compile(r"<</?SYS>>", re.IGNORECASE)),
    # "Ignore the previous instructions" and its close variants: the verb, an
    # anaphoric reference, and an instruction-like object, in that order and
    # within one sentence.
    (
        "instruction_override",
        re.compile(
            r"\b(?:ignore|disregard|forget)\b[^.\n]{0,40}?"
            r"\b(?:previous|prior|preceding|above|earlier|foregoing|all)\b[^.\n]{0,40}?"
            r"\b(?:instruction|instructions|rule|rules|prompt|prompts|"
            r"direction|directions|guideline|guidelines)\b",
            re.IGNORECASE,
        ),
    ),
    # Asking the agent to disclose its own instructions. Anchored on the
    # possessive "your" so that a legitimate question about configuring a
    # system prompt for a skill is unaffected.
    (
        "prompt_disclosure",
        re.compile(
            r"\b(?:reveal|show|print|repeat|output|disclose|dump|leak)\b[^.\n]{0,20}?"
            r"\byour\s+(?:system\s+|initial\s+|original\s+|hidden\s+|internal\s+)?"
            r"(?:prompt|instructions?|message|rules?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt_disclosure",
        re.compile(
            r"\b(?:what\s+(?:is|are)|tell\s+me)\b[^.\n]{0,20}?"
            r"\byour\s+(?:system\s+|initial\s+|original\s+|hidden\s+|internal\s+)?"
            r"(?:prompt|instructions|rules)\b",
            re.IGNORECASE,
        ),
    ),
    # Persona replacement.
    (
        "persona_override",
        re.compile(r"\byou\s+are\s+now\s+(?:a|an|the|no\s+longer)\b", re.IGNORECASE),
    ),
]


def screen_injection(text: object) -> Optional[str]:
    """Return the violation class for a hostile string, or None when clean.

    Only strings are screened; other leaf types carry no directive. The caller
    decides what to do with the class name - it is a category, never the
    matched text, so it is safe to log and to return.
    """
    if not isinstance(text, str) or not text:
        return None
    for name, pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return name
    return None


def screen_structure(value: object, _depth: int = 0) -> Optional[str]:
    """Screen every string leaf AND every mapping KEY, depth first.

    Keys matter as much as values: a payload can carry its directive in a field
    name, and a scan that only walks values will not see it. Escaping the
    payload as JSON \\u sequences does not help an attacker either, because this
    runs on the PARSED structure - by the time it is reached, the escapes are
    ordinary characters.
    """
    if _depth > 8:
        # Deeper than any shape this agent accepts. Refuse rather than recurse.
        return "nesting_depth"
    if isinstance(value, str):
        return screen_injection(value)
    if isinstance(value, dict):
        for key, item in value.items():
            violation = screen_injection(key) if isinstance(key, str) else None
            if violation:
                return violation
            violation = screen_structure(item, _depth + 1)
            if violation:
                return violation
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            violation = screen_structure(item, _depth + 1)
            if violation:
                return violation
        return None
    return None


# ── Bounded parsers ──────────────────────────────────────────────────────────


def field_reference(name: object, index: int) -> str:
    """Render a caller-supplied field name in a form that is safe to return."""
    if isinstance(name, str) and _SAFE_FIELD_NAME_RE.match(name):
        return name
    return f"field #{index}"


def finite_int_in_range(value: object, lo: int, hi: int) -> Optional[int]:
    """Parse a caller integer that must be real, finite and inside [lo, hi].

    Returns None for everything else - bools (which are ints in Python and
    would otherwise read as 0/1), non-numeric types, fractional floats, the
    non-finite floats, their string spellings, and out-of-range magnitudes.
    None means "refuse", never "use a default": the caller asked for something
    specific and got it wrong, and silently substituting a value hides that.
    """
    if isinstance(value, bool):
        return None
    number: float
    if isinstance(value, int):
        number = float(value)
    elif isinstance(value, float):
        number = value
    elif isinstance(value, str):
        text = value.strip()
        # A digit run only. "NaN", "Infinity", "1e400" and "  12 " with an
        # inner space all fail here before float() ever sees them.
        if not re.fullmatch(r"[+-]?\d{1,9}", text):
            return None
        number = float(text)
    else:
        return None
    if not math.isfinite(number) or number != int(number):
        return None
    parsed = int(number)
    if parsed < lo or parsed > hi:
        return None
    return parsed


def inert_identifier(value: object) -> Optional[str]:
    """Return the value as an inert identifier, or None when it is not one.

    Case is folded first so that a caller may send "Gmail"; everything after
    that must already be within the inert alphabet. Anything carrying a space,
    punctuation, markup or non-ASCII text is refused rather than sanitised -
    sanitising a rejected value produces a third string that the caller never
    sent and nobody validated.
    """
    if not isinstance(value, str):
        return None
    folded = value.strip().lower()
    if not _INERT_ID_RE.match(folded):
        return None
    return folded


def validate_context(input_context: object) -> Tuple[Dict[str, Any], List[str]]:
    """Validate the caller's request context into a bounded filter mapping.

    Returns ``(filters, errors)``. ``errors`` is a list of field NAMES that
    failed; the offending values never appear in it, in the returned mapping,
    or anywhere they could be rendered. A non-empty ``errors`` list means the
    request must be refused - the caller asked for a narrowing this agent
    cannot honour, and answering the un-narrowed question instead would be a
    different question than the one that was asked.

    Fields outside ``CONTEXT_FIELDS`` are ignored. Their VALUES are still
    screened for injection content by the caller (via ``screen_structure``)
    before this runs.
    """
    filters: Dict[str, Any] = {}
    errors: List[str] = []
    if input_context is None:
        return filters, errors
    if not isinstance(input_context, dict):
        return filters, ["input_context"]

    for index, name in enumerate(CONTEXT_FIELDS, start=1):
        if name not in input_context:
            continue
        raw = input_context[name]
        if raw is None:
            continue
        if name == "top_k":
            parsed_top_k = finite_int_in_range(raw, TOP_K_MIN, TOP_K_MAX)
            if parsed_top_k is None:
                errors.append(field_reference(name, index))
            else:
                filters["top_k"] = parsed_top_k
        elif name == "frameworks":
            if not isinstance(raw, list) or len(raw) > MAX_FRAMEWORKS:
                errors.append(field_reference(name, index))
                continue
            parsed_list = [inert_identifier(item) for item in raw]
            if any(item is None for item in parsed_list):
                errors.append(field_reference(name, index))
            else:
                filters["frameworks"] = [item for item in parsed_list if item]
        else:
            parsed_id = inert_identifier(raw)
            if parsed_id is None:
                errors.append(field_reference(name, index))
            else:
                filters[name] = parsed_id

    return filters, errors
