"""AgentCore Platform v1.0"""

# Standalone HTTP entry point for the agent.
# Entry points are adapters only — no business logic here.
# For platform-level routing, the gateway calls agent.invoke() directly.

import json
import os
import re
import secrets
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from framework.security.credential_detector import detect_credentials_in_value
from shared.secrets import factory as secrets_factory
from shared.utils.audit_logger import emit_trace_event
from src.graph.graph import WorkspaceSkillsIntegrationQAAgent, runtime_config

app = FastAPI(title="Agent")

# The platform registry loads config/config.yaml and passes it as
# Graph(config=...); the standalone server mirrors that exactly, so declared
# runtime parameters are live in both deployments rather than only one.
agent = WorkspaceSkillsIntegrationQAAgent(config=runtime_config())
agent.compile()
# Namespace/agent_name match the agent's manifest values.
agent.provision_secrets(secrets_factory(namespace="cmn-c2-298", agent_name="WorkspaceSkillsIntegrationQAAgent"))

# Upper bound on the serialized request context (bytes). The graph enforces the
# per-field bounds (inert identifier alphabet, finite numeric ranges, list
# caps); this is the coarse adapter guard that keeps an oversized payload from
# reaching the graph at all.
_MAX_INPUT_CONTEXT_BYTES = 262_144

# ── Request-context credential screen ────────────────────────────────────────
# Why this runs before invoke(), not inside a node:
#
# The framework's mandatory output gate scans every value of every node result
# for credential patterns, and the backbone's initialize node copies
# input_context verbatim into its own result. So a credential-shaped string
# anywhere in the request context makes the FIRST node of the graph fail,
# before any of this agent's code runs. What the caller receives is an error
# status with the whole result withheld and nothing naming input_context, the
# field, or the reason. On a hosted conversation the same context is replayed
# on every turn, so the session does not recover on its own.
#
# The request cannot succeed either way. Refusing it here does not change what
# is accepted; it turns an opaque failure into an actionable one.
#
# The screen calls the SAME detector the gate calls, on the SAME assembled
# object, so what this adapter refuses and what the gate blocks are one set by
# construction — there is no local pattern list that could drift from it.
# Scanning field by field composes exactly to scanning the whole mapping (the
# detector's result on a dict is the union over its values), which is what lets
# the refusal name the offending field without widening or narrowing the match.
#
# Field NAMES are caller-controlled too, so a name is echoed only when it is
# short, inert, and carries no credential shape of its own; anything else is
# reported by position. The rejected value and the matched text are never
# echoed, in the response or in the audit record.
_SAFE_FIELD_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def _field_reference(name: object, index: int) -> str:
    """Render a caller-supplied context field name safe to put in a message."""
    if isinstance(name, str) and _SAFE_FIELD_NAME_RE.match(name) and not detect_credentials_in_value(name):
        return f"input_context.{name}"
    return f"input_context field #{index}"


def screen_input_context(input_context: Dict[str, Any]) -> Optional[str]:
    """Return a reference to the first credential-bearing field, else None.

    Walks the top-level fields in caller order and hands each value to the
    framework credential detector, which recurses through nested dicts and
    lists on its own. Only the first offending field is reported: one is enough
    to act on, and the message stays bounded however many fields were sent.
    """
    for index, (name, value) in enumerate(input_context.items(), start=1):
        if detect_credentials_in_value(value):
            return _field_reference(name, index)
    return None


class InvokeRequest(BaseModel):
    input: str
    session_id: str = ""
    # Structured request parameters (the SDK's input_context invoke argument):
    # skill, frameworks, top_k, channel — each validated against explicit
    # bounds inside the graph (PreProcessNode / InputValidateNode).
    input_context: Optional[Dict[str, Any]] = None


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request) -> Any:
    trust = getattr(request.state, "trust_level", TrustLevel.ANONYMOUS)
    # Standalone-deployment caller auth: when INVOKE_AUTH_TOKEN is set on the
    # server environment, callers that no upstream middleware vouched for
    # (still ANONYMOUS) must present it as a Bearer token and run at
    # VERIFIED_EXTERNAL. Middleware-established trust is never demoted.
    # This adapter is the entry-point auth boundary (the standalone equivalent
    # of the platform auth middleware) — a deployment-level caller credential,
    # not an agent secret, so ctx.secrets does not apply: no InvocationContext
    # exists before auth.
    #
    # Required here specifically: PreProcessNode occupies the pre_process
    # backbone slot and declares required_trust_level = VERIFIED_EXTERNAL.
    # Nothing else sets request.state.trust_level in a standalone deployment,
    # so without this boundary every deployed invoke arrives ANONYMOUS, the
    # trust gate denies it, and the agent returns an error status.
    expected = os.environ.get("INVOKE_AUTH_TOKEN")
    if expected and trust is TrustLevel.ANONYMOUS:
        supplied = request.headers.get("authorization", "")
        # Compare bytes: compare_digest raises TypeError on non-ASCII str input
        # (headers decode as latin-1), which would 500 instead of the generic 401.
        if not secrets.compare_digest(supplied.encode(), f"Bearer {expected}".encode()):
            # Generic body on purpose — do not leak whether the token was absent,
            # malformed, or wrong.
            raise HTTPException(status_code=401, detail="Token is invalid or expired.")
        trust = TrustLevel.VERIFIED_EXTERNAL

    input_context: Dict[str, Any] = req.input_context or {}
    if input_context and len(json.dumps(input_context, default=str)) > _MAX_INPUT_CONTEXT_BYTES:
        raise HTTPException(status_code=413, detail="input_context exceeds the maximum allowed size.")
    offending_field = screen_input_context(input_context)
    if offending_field is not None:
        emit_trace_event(
            "input_context_credential_refused",
            {"field": offending_field},
            {"session_id": req.session_id},
        )
        raise HTTPException(
            status_code=400,
            detail=(
                f"{offending_field} contains a credential-shaped value. Remove API keys, "
                "tokens and connection strings from input_context and retry."
            ),
        )

    with bound_secrets(agent._secrets_provider):
        ctx = InvocationContext(
            session_id=req.session_id or str(uuid4()),
            caller_trust_level=trust,
            caller_id=getattr(request.state, "caller_id", ""),
        )
        return agent.invoke(req.input, ctx=ctx, input_context=input_context)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "agent": "WorkspaceSkillsIntegrationQAAgent"}
