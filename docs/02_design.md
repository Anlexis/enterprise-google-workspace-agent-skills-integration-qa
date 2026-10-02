# Template Design Specification — CMN-C2-298

**Template ID:** CMN-C2-298
**Template Name:** WorkspaceSkillsIntegrationQAAgent
**Category:** Cat 2 (multi-step domain workflow — RAG pattern)
**Industry:** CMN

## Position in AgentCore Architecture

| Position | Value |
|---|---|
| Agent class | `WorkspaceSkillsIntegrationQAAgent` |
| L1 Base (framework base class) | `AgentBaseGraph` — direct framework inheritance |
| Inner graph base | `BaseGraph` — `DomainWorkflowGraph` |
| Pattern | Cat 2 two-layer nested (outer fixed 5-node backbone + a `GraphNode` in the `main` slot wrapping an inner `BaseGraph` domain workflow) |

- **Three-Layer Separation:**
  - State: flat `TypedDict` composition (no Pydantic — msgpack incompatible);
    structured fields stored as JSON strings via `to_json()` / `from_json()`
  - Node: framework inheritance via `FunctionNode` (override
    `execute(self, state) -> dict` only — there is no `config` parameter; see
    the Design Decision Record)
  - Graph: composition (`register_nodes()` for node substitution); outer
    `add_edges()` is NOT overridden

## Purpose

Enterprise IT/AI integration teams and security/compliance officers ask
natural-language questions about integrating Google's official Agent Skills
(google/skills — Workspace, Cloud, Maps, YouTube, Calendar, Gmail, Sheets,
Drive, Docs, Search) into a Japanese enterprise: retrieve → rerank/filter →
grounded, structured answer (integration steps, required OAuth scopes, a
compliance-flag summary, a testing checklist, a related-skills dependency
map, and source citations), assembled from three seeded, pre-indexed
knowledge bases (Google Agent Skills documentation, enterprise integration
patterns, Japan compliance requirements — APPI/FISC/ISMS). **This template
answers *how to integrate* the skills — it does NOT call any Google API,
does not authenticate against one, and makes no configuration changes.**
It is fully deterministic: keyword retrieval plus rule-based grounded answer
assembly, with no model call and no network call of any kind in the runtime
path — see the implementation note below.

## Architecture Overview

### Outer backbone (AgentBaseGraph)

```
START → initialize → pre_process → main → {route} → post_process → finalize → END
                                     ↓ (retry, max 3)
                                   pre_process
```

| Slot | Class | Responsibility | required_trust_level |
|------|-------|----------------|----------------------|
| initialize | InitializeNode (framework default) | session_id, trust level, schema version | — (framework) |
| pre_process | `PreProcessNode` | the caller contract: validate non-empty input, refuse prompt-injection content, validate every request-context field against explicit bounds, surface-strip direct identifiers → `validated_input` | `TrustLevel.VERIFIED_EXTERNAL` |
| main | `IntegrationQAGraphNode` (`GraphNode`) | delegates to the inner `DomainWorkflowGraph`; bridges `input_context` across the boundary; maps inner `formatted_answer` / `structured_answer` → outer `result` / `structured_answer` | — (delegation) |
| post_process | `PostProcessNode` | the output boundary — recursively scans `result` and `structured_answer` for credential shapes, enforces the standing scope notice and the scalars-only structured shape; on violation returns ERROR **and clears every output-bearing field** | `TrustLevel.VERIFIED_EXTERNAL` |
| finalize | FinalizeNode (framework default) | response metadata, total time | — (framework) |

### Inner graph (DomainWorkflowGraph — BaseGraph, linear)

```
START → input_validate → retrieve → rerank_filter → generate_answer → output_format → END
```

All five inner domain nodes declare `required_trust_level = TrustLevel.ANONYMOUS`
(the external trust gate lives on the outer backbone gate node; a stricter inner
level would deny a real VERIFIED_EXTERNAL invoke at runtime).

| Node | Responsibility | required_trust_level | Input State | Output State |
|------|----------------|----------------------|-------------|--------------|
| `InputValidateNode` | Parse the question (plain text or JSON envelope); normalise whitespace; cap length; resolve the caller's filters from `input_context` (which wins) or the envelope, each through the shared bounded parser — fail closed | `TrustLevel.ANONYMOUS` | `validated_input` \| `user_input`, `input_context` | `search_query`, `query_filters`, `intake_notes` |
| `RetrieveNode` | Deterministic keyword retrieval pooled across the THREE seeded KBs (`config/kb/google_agent_skills_kb.json`, `integration_patterns_kb.json`, `japan_compliance_kb.json`): tokenise query, score title/tags/content overlap, apply the optional `skill` filter (narrows the google_agent_skills pool only) and the optional `frameworks` filter (narrows the japan_compliance pool only) | `TrustLevel.ANONYMOUS` | `search_query`, `query_filters`, `retrieval_config` | `retrieved_documents`, `intake_notes` |
| `RerankFilterNode` | Rerank candidates (skill-match boost), drop entries below `score_threshold`, cap at `top_k` | `TrustLevel.ANONYMOUS` | `retrieved_documents`, `query_filters`, `retrieval_config` | `ranked_documents` |
| `GenerateAnswerNode` | Rule-based grounded answer assembly from the ranked KB passages only, with numbered citation markers; AND deterministic aggregation (dedupe + cap, never synthesis) of each ranked entry's own pre-authored `integration_steps` / `oauth_scopes` / `related_skills` / `testing_notes` / compliance `guidance` fields into `structured_extract` | `TrustLevel.ANONYMOUS` | `ranked_documents`, `search_query` | `grounded_answer`, `citations`, `structured_extract` |
| `OutputFormatNode` | Compose the final answer: body + Sources list + the standing scope/limitations notice (notice is part of this node, NOT post_process); combine `structured_extract` + `citations` into the single `structured_answer` object | `TrustLevel.ANONYMOUS` | `grounded_answer`, `citations`, `structured_extract` | `formatted_answer`, `structured_answer`, `status` |

### Data Flow

```
user_input + input_context
  → PreProcessNode                               → validated_input (or a refusal)
  → IntegrationQAGraphNode.extract_input         → inner DomainWorkflowGraph.invoke(validated_input)
        (also stashes input_context on the ContextVar bridge)
        → input_validate                         → search_query / query_filters
        → retrieve                               → retrieved_documents (pooled, 3 KBs)
        → rerank_filter                          → ranked_documents
        → generate_answer                        → grounded_answer / citations / structured_extract
        → output_format                          → formatted_answer + structured_answer (+ scope notice)
     get_output() → {formatted_answer, structured_answer, status, ...}
  → IntegrationQAGraphNode.merge_output          → result / structured_answer = (mapped)
  → PostProcessNode                              → formatted_output (gated); output-bearing fields
                                                   cleared on a violation
  → WorkspaceSkillsIntegrationQAAgent.get_output() → base envelope EXTENDED with the
                                                     structured product on SUCCESS only,
                                                     after a fail-closed re-scan
```

### The caller-data contract

`/invoke` accepts three things: the question (`input`), an optional
`session_id`, and an optional `input_context` mapping. `input_context` is the
structured channel and it carries four fields, every one of them bounded:

| Field | Form | Effect |
|---|---|---|
| `skill` | inert identifier, `[a-z0-9_]{1,32}` | narrows the Google Agent Skills pool to one skill |
| `frameworks` | up to 5 inert identifiers | narrows the compliance pool (`appi`, `fisc`, `isms`) |
| `top_k` | integer, finite, 1–20 | caps the number of passages the answer is built from |
| `channel` | inert identifier | audit label only |

A field outside its bounds is REFUSED, not clamped: clamping answers a
different question than the one that was asked, and does it silently. The
refusal names the field and never the value. Fields the agent does not read are
ignored rather than refused — the hosting runtime puts its own keys on the same
mapping, and refusing those would break every request the moment one appears.

Every caller-controlled number goes through `finite_int_in_range()` in
`src/schemas/caller_contract.py`. That function is the reason NaN and Infinity
are named explicitly in the tests: both survive a plain `int()`/`float()`
coercion, both arrive through bare JSON literals, and every comparison against
NaN is False — so an unchecked numeric disables the bound it was supposed to
enforce rather than failing.

For callers that have only a single string field available, the question may
instead be a JSON envelope (`{"query": ..., "skill": ..., "top_k": ...}`),
parsed back by `InputValidateNode` and validated by exactly the same functions.
`input_context` wins where both supply a field.

**Why a bridge is needed.** The framework's `GraphNode` invokes the inner graph
as `subgraph.invoke(user_input, session_id=..., ctx=...)` — it does not forward
the outer state's `input_context`. Inner reads of `state["input_context"]`
would therefore always see an empty mapping on a real invocation while unit
tests that seed inner state directly kept passing. `src/graph/context_bridge.py`
closes that gap through the two sanctioned hooks: `extract_input()` stashes the
context on a ContextVar immediately before the invoke, and the inner graph's
`_extra_initial_state()` seeds it into the inner state. A ContextVar rather than
a module global, so concurrent invocations in one process cannot see each
other's context.

### Runtime settings forwarding (`_parent_config`)

Configuration lives in `config/config.yaml`, which the platform registry loads
and hands to the graph as `Graph(config=...)`; `src/api/server.py` reads the
same file through `runtime_config()` so a registry-loaded agent and a
standalone one are configured identically. `config/agent.yaml` is the static
registration manifest and carries no tuning values at all — one home per
setting, so there is no second copy to drift.

`IntegrationQAGraphNode._parent_config()` forwards the `retrieval` block under
`config["configurable"]` (never `{}`):

```
{"configurable": {"retrieval": {top_k, score_threshold, kb_paths: [...]}}}
```

`get_subgraph()` passes that into `DomainWorkflowGraph(config=...)`; the inner
graph republishes it into the inner initial state as the JSON-string field
`retrieval_config` (via `_extra_initial_state()`), so the declared `top_k` /
`score_threshold` / `kb_paths` are live at runtime. `RetrieveNode` and
`RerankFilterNode` read `retrieval_config` from State — the only channel, since
`execute(self, state)` carries no `config` parameter.

`tests/unit/test_config_manifest.py::TestDeclaredValuesAreLive` is the guard
that this stays true: it changes a declared value and asserts the change
reaches the node that consumes it. A declared value whose reader points at the
wrong place produces no error and no failing test — the agent just runs on its
defaults while the file says otherwise.

### Structured Product

`WorkspaceSkillsIntegrationQAAgent.get_output()` **extends**
`super().get_output()` (the base `{output, status, trace_id, correlation_id,
node_history}` envelope) with six whitelisted keys, surfaced **on SUCCESS
only**, after a **fail-closed** defense-in-depth re-scan of `structured_answer`
via the same recursive `_security_gate_output()` PostProcessNode already
applied:

| Key | Shape | Source |
|-----|-------|--------|
| `integration_steps` | `list[str]` | Deduped, capped aggregation of ranked `google_agent_skills` / `integration_patterns` entries' own `integration_steps` |
| `oauth_scopes` | `list[{skill, scope, purpose}]` | Deduped (by `scope`), capped aggregation of ranked `google_agent_skills` entries' own `oauth_scopes` |
| `compliance_flags` | `list[{framework, requirement, guidance}]` | One entry per ranked `japan_compliance` KB entry (`framework` ∈ APPI/FISC/ISMS) |
| `testing_checklist` | `list[str]` | Deduped, capped aggregation of ranked entries' own `testing_notes` (any KB) |
| `related_skills` | `list[{skill, relation}]` | Deduped (by `skill`), capped aggregation of ranked `google_agent_skills` entries' own `related_skills` |
| `citations` | `list[{ref, id, title, source, kb}]` | Same citation list used in the rendered `grounded_answer` |

Every leaf value above is a scalar (`str`) copied verbatim from a
`config/kb/*.json` entry — nothing is generated, and nothing outside this
fixed six-key allowlist is ever surfaced (fail-closed: nothing else in
`structured_answer` reaches the caller even if it somehow carried an extra
key). The output boundary refuses a structured product whose leaves are not
scalars, since a non-scalar leaf would bypass the credential scan.

## Security Boundaries

### Caller trust

Every node declares `required_trust_level` (see the tables above). The two
outer slots require VERIFIED_EXTERNAL; the inner domain nodes run behind that
boundary at ANONYMOUS, because a stricter inner level would deny a real
verified invoke. `PreProcessNode` additionally rejects empty or non-string
`user_input` before the workflow runs. The standalone server promotes an
authenticated Bearer caller to VERIFIED_EXTERNAL (`INVOKE_AUTH_TOKEN`); where
no token is configured, requests arrive ANONYMOUS and the trust gate refuses
them.

### Input screening

`PreProcessNode` owns the input guarantees rather than delegating them to the
framework. That matters: the framework's own input screen is active only where
the deployment configures it, and where it is absent a hostile payload would
reach the answer path and return an ordinary success. Enforced in the node, the
refusal holds in every deployment, and it is observable by calling `execute()`
directly with nothing in front of it.

What it screens, and in what order:

1. The question **as it arrived**, for prompt-injection content.
2. The request context, **depth first including keys** — a directive can ride
   in a field name, and running on the parsed structure means unicode escapes
   are already ordinary characters by the time they are inspected.
3. Each context field against its bounds (see the caller-data contract above).
4. A surface redaction pass over the question (e-mail addresses, long digit
   runs) on top of the framework's own masking of `user_input` /
   `validated_input`.
5. The question **again, after redaction**. A redaction pass is not a refusal,
   and it can make an attack harder to see rather than easier: removing a
   control token from a hostile string leaves the directive behind as ordinary
   prose that no token screen will match. Screening both forms catches the
   token before removal and the directive after the removal splices it back
   together.

The injection screen (`src/schemas/caller_contract.py`) covers two families:
**chat-template control tokens as a class** — any `<|...|>` delimiter, `[INST]`,
`<<SYS>>` — and directives addressed at the agent (ignore/disregard the previous
instructions; disclose your system prompt; persona replacement). It is
deliberately narrow in the other direction. This agent's own subject matter is
full of "rules", "policy", "system" and "disable", so every pattern requires the
directive to be aimed AT the agent; `tests/unit/test_pre_process_node.py`
probes both directions, and the false-positive cases are real questions from
this domain rather than invented ones.

### The output boundary

`PostProcessNode` enforces two stated invariants on the released answer and on
the structured product, from `execute()`:

- **Nothing credential-shaped is released.** The module-level
  `_security_gate_output()` scan RECURSES into nested dict/list/tuple content.
  That recursion is the point: the framework's own automatic per-node
  credential scan inspects only the TOP-LEVEL STRING values of a node's return
  dict, so a credential nested inside `structured_answer` would never be seen
  by it.
- **Every released answer carries the standing scope notice.** Composing the
  notice is not enforcing it — a renderer change, a truncation, or a partially
  assembled answer would drop the line silently, and what ships then reads as
  if the integration had been performed. `OutputFormatNode` splices
  `SCOPE_NOTICE_MARKER` into the notice text rather than duplicating it, so the
  notice cannot be reworded past the gate without the no-live-call clause
  travelling with it.

A third structural check refuses a structured product whose leaves are not
plain scalars: a non-scalar leaf would be released without ever having been
pattern-scanned, because the credential scan skips non-string leaves by design.

**Containment on a violation.** The node returns an error status AND clears
every output-bearing field (`integration_answer`, `formatted_answer`,
`grounded_answer`, `structured_answer`, `citations`, `structured_extract`),
replacing the released text with a fixed stub. Returning an error status is not
by itself containment: the base envelope falls back to `state["result"]` even
on a non-success status, so a gate that merely raised — or that flipped the
status without clearing — would still ship the un-gated answer inside the error
envelope. `tests/unit/test_post_process_node.py::TestContainmentOnViolation`
asserts the error delta carries no released text, no traceback and no paths.

**Scope-versus-credential discrimination (this agent's specific risk):** the
structured answer NAMES OAuth scope identifiers (for example
`https://www.googleapis.com/auth/gmail.send`) — plain permission identifiers,
not secrets. None of the gate's patterns (`sk-`/`pk-`/`ak-` prefixes, JWT,
`Bearer <token>`, the `ya29.` Google access-token prefix, AWS `AKIA…`, a
`token:`/`secret:`/`client_secret:` assignment) match a bare scope URL or short
scope name, so a legitimate scope list is never blanked. An issued token shape
DOES trip the gate. Both directions are pinned by tests.

`WorkspaceSkillsIntegrationQAAgent.get_output()` re-applies the same recursive
scan to `structured_answer` at the true response boundary before surfacing the
structured keys — defence in depth on top of the node's own gate.

### Credential shapes on the request context

`src/api/server.py` screens the assembled `input_context` for credential shapes
BEFORE `invoke()`, and refuses with a message naming the offending field.

The reason is mechanical rather than defensive. The framework's mandatory
output gate scans every value of every node result, and the backbone's
initialize node copies `input_context` verbatim into its own result — so a
credential-shaped string anywhere on that mapping makes the FIRST node of the
graph fail, before any of this agent's code runs. What the caller receives is an
error status with the answer withheld and nothing naming `input_context`, the
field, or the reason. On a hosted conversation the same context is replayed on
every turn, so the session does not recover on its own.

The request cannot succeed either way. The screen does not change what is
accepted; it changes an opaque failure into an actionable one. It calls the
same detector the framework gate calls, on the same assembled object, so the
adapter's refusal set and the gate's block set are one set by construction —
there is no local pattern list that could drift from it. Field names are
caller-controlled too, so a name is quoted back only when it is short, inert
and carries no credential shape of its own; anything else is reported by
position, and the value is never echoed.

This agent's own context fields are locked to inert identifiers and bounded
integers, so they cannot carry a credential shape at all. The screen exists for
the fields it does NOT read — the ones a runtime or a caller adds.

### Audit

Every node's `execute()` emits exactly ONE domain-specific
`emit_trace_event("<node>_complete", {small non-identifying payload}, state)`
on its success path. Nodes do NOT emit `node_start` / `node_complete` /
`node_error` — `BaseNode.__call__()` emits those. Domain event names:

  - `pre_process_complete`
  - `input_validate_complete`
  - `retrieve_complete`
  - `rerank_filter_complete`
  - `generate_answer_complete`
  - `output_format_complete`
  - `post_process_complete`
  - `input_context_credential_refused` (the HTTP adapter, on a refusal)

No `_extra_security_gate_input` / `_extra_security_gate_output` instance
methods are defined on any node — those are the framework's own `FunctionNode`
extension hooks, a different mechanism from the manual gate described above.

### Numeric output

This agent renders no monetary aggregates and no computed numbers: every value
in the answer and in the structured product is copied verbatim from a
knowledge-base entry, and the only numbers in the rendered output are citation
reference markers. There is therefore no rounding grid to enforce and no
numeric rewriting anywhere in the output path — the invariants this boundary
enforces are the two stated above.

## Scope & Limitations Notice

Every answer carries a standing scope/limitations line: this agent does NOT
connect to, authenticate against, or call any Google Workspace/Cloud/Maps/
YouTube/Search API, and makes no configuration changes; scopes and compliance
posture must be verified against the customer's own admin console, IAM and
compliance team.

`OutputFormatNode` composes it as part of the domain output contract, and
`PostProcessNode` REFUSES to release an answer that does not carry it. Both
sides share one constant (`SCOPE_NOTICE_MARKER`), spliced into the notice text
rather than duplicated, so the notice cannot be reworded past the gate without
the no-live-call clause travelling with it.

The risk this addresses is specific and easy to trip over: the title and
overview of an agent like this one read as though a live, OAuth-connected
integration were being performed. The notice keeps the boundary visible at the
point of use rather than only in the documentation.

## Implementation Note — answer synthesis

This agent is **deterministic end to end**. Retrieval is keyword scoring pooled
across the three seeded knowledge bases, `GenerateAnswerNode` assembles the
grounded answer by rule from the ranked passages (lead sentence plus cited
passage excerpts), and the structured extract (`integration_steps` /
`oauth_scopes` / `compliance_flags` / `testing_checklist` / `related_skills`)
is a deterministic aggregation of each ranked entry's own pre-authored fields —
never synthesis. There is no model call and no model client dependency, and no
system prompt is read at runtime.

Because nothing reads them, `config/config.yaml` declares NO model settings: a
declared value with no reader is indistinguishable from a value that is being
ignored, and the campaign that produced this shape kept finding exactly that.
`config/prompts/answer_synthesis_prompt.md` states the contract a model-backed
`GenerateAnswerNode` must keep — same inputs, same state contract, same
grounding and no-live-API rules — so that swapping the assembly for a model
call changes only the inside of that one node. Add the model settings to
`config/config.yaml` at the same time as the client, and forward them the way
`retrieval` is forwarded.

The no-live-API boundary is unconditional and does not change with that swap.

## Composition Pattern

- **Pattern:** `GraphNode` (subgraph) in the outer `main` slot.
- **Composition target:** `DomainWorkflowGraph` (inner `BaseGraph`).
- **Error propagation strategy:** `propagate` (inner errors re-raised as `SubgraphError`).
- Inner domain nodes run at `TrustLevel.ANONYMOUS`; outer pre/post_process run
  at `TrustLevel.VERIFIED_EXTERNAL`.

## Import Isolation Confirmation
- [x] The agent does not import the platform SDK directly.
- [x] Import targets: `framework/` and `shared/` only.
- [x] The base position names a framework base class, never a higher-level agent class.

## Design Decision Record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| L1 base type | AgentBaseGraph | AutonomousBaseGraph | **AgentBaseGraph** | Fixed multi-step RAG workflow, no autonomous loop |
| Composition pattern | Standalone Cat 1 slots | GraphNode → inner BaseGraph | **GraphNode → inner BaseGraph** | 5-step domain workflow exceeds a single `main` node; nested keeps the outer backbone untouched |
| `execute()` signature | `execute(state, config=None)` | `execute(self, state) -> dict`, config via State seeding | **`execute(self, state) -> dict`** | `BaseNode.__call__` calls `self.execute(state)` with ONE argument, so a `config` parameter would be dead code. Configuration reaches the nodes by State seeding (`_parent_config()` → `_extra_initial_state()` → `state["retrieval_config"]`); no constructor injection is needed, since no node holds an immutable dependency such as a client or transport |
| Answer synthesis | Rule-based assembly | Model call | **Rule-based** | Deterministic assembly is testable and grounded by construction; a model-backed node can be swapped in at the documented contract without touching another node |
| KB storage | External vector store | Seeded JSON KB (single file) | **Seeded JSON KB, THREE files** | Self-contained, deterministic CI; three separate files mirror the three distinct corpora in the proposal (Google Agent Skills docs / integration patterns / Japan compliance) rather than merging them into one undifferentiated index; the retrieval contract (`retrieved_documents` JSON, each entry tagged with its source `kb`) is store-agnostic for a later vector-store upgrade |
| Output shape | Rendered text only | Structured product | **Structured product** | The agent commits to a structured answer (integration steps, OAuth scopes, compliance-flag summary, testing checklist, related-skills dependency map, citations); `get_output()` extends the base envelope rather than replacing it, fail-closed |
| Output gate scope | Top-level string scan | Recursive scan over dict/list/tuple | **Recursive** | The structured product is a nested object; a top-level-string-only scan would never inspect it. `_security_gate_output(content: Any)` is shared by PostProcessNode and the outer `get_output()` re-check |
| Caller filters channel | JSON envelope inside the question string only | `input_context`, envelope kept as a fallback | **Both, `input_context` first** | `input_context` is the framework's own structured channel and keeps parameters out of the text that gets masked and screened as prose. The envelope stays supported for callers with only one string field; both go through the same bounded parsers |
| Out-of-bounds caller value | Clamp to the nearest valid value | Refuse | **Refuse** | Clamping answers a different question than the one that was asked, silently. A refusal that names the field is something the caller can act on |
| Where the caller contract is enforced | The inner intake node only | The outer gate node, re-checked inside | **Both** | The outer gate stops a bad request before the workflow runs; the inner check still holds when the inner graph is invoked directly, which is how the domain tests drive it |
