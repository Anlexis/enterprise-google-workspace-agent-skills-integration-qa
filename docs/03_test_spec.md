# Test Specification — CMN-C2-298

**Template ID:** CMN-C2-298
**Template Name:** WorkspaceSkillsIntegrationQAAgent
**Category:** Cat 2 (nested) — Google Agent Skills integration Q&A (RAG pattern)

## Test Strategy

- Coverage: every node's `execute()` success path plus at least one rejection
  or degradation path; both outer gate nodes' trust behaviour; the caller
  contract in both directions; the output boundary's two invariants in both
  directions; the full nested-graph invoke order; and the whole stack through
  the real HTTP entry point.
- Test types: unit (`tests/unit/`) and proof-of-boundary
  (`tests/proof_of_boundary/`). The end-to-end coverage lives in the
  proof-of-boundary suite rather than a separate integration suite: this agent
  makes no network or model call, so there is no external system to integrate
  against — the boundaries worth proving are its own.
- Per-node tests invoke the node as `node(state)` (`BaseNode.__call__`), so the
  trust gate, the framework masking and the framework output scan all run.
  The **injection tests are the deliberate exception**: they call `execute()`
  directly, with nothing in front. With the framework wrapper in place a
  refusal proves only that something refused; called directly, it proves this
  agent refuses on its own — which is the property that has to hold in a
  deployment where the framework screen is absent or configured off.
- Deterministic throughout — no model call, no network. `framework.*` /
  `src.*` imports only.

## 1. Framework Compliance Tests

| TC-ID | Test | Expected Result | Result |
|-------|------|----------------|--------|
| TC-01 | State contract: flat TypedDict | Type check pass, no Pydantic/dataclass (`src/schemas/state.py`) | PASS |
| TC-02 | Invalid input rejected | `PreProcessNode` returns an error status on empty/non-string input (`tests/unit/test_pre_process_node.py::TestPreProcessRejection`) | PASS |
| TC-03 | No credential in State | Credential scan: 0 violations | PASS |
| TC-04 | Invocation context reaches nodes through the framework only | Direct access raises (framework-enforced) | PASS |
| TC-05 | No duplicate lifecycle events in `execute()` | `node_start`/`node_complete`/`node_error` absent from every `src/nodes/*.py` `execute()` body — each node emits its OWN domain event only | PASS |
| TC-06 | The default input gate is not overridden | `TypeError` at class definition if overridden (`@final`) — `tests/unit/test_framework_compliance_tc06_tc07.py` | PASS |
| TC-07 | The default output gate is not overridden | `TypeError` at class definition if overridden (`@final`) — `tests/unit/test_framework_compliance_tc06_tc07.py` | PASS |
| TC-08 | `required_trust_level` enforced | ANONYMOUS caller denied on both VERIFIED_EXTERNAL outer gates — `tests/unit/test_trust_gate.py::TestTrustGate` | PASS |
| TC-09 | Domain input checks | Enforced inside `PreProcessNode.execute()` (injection screen + request-context contract), not as a framework hook override — see docs/02_design.md | PASS |
| TC-10 | Domain output gate is non-trivial | `PostProcessNode`'s module-level `_security_gate_output()` recursively scans `result` and `structured_answer`; the scope-versus-credential discrimination is verified in both directions — `tests/unit/test_post_process_node.py` | PASS |
| TC-11 | At least one domain `emit_trace_event()` inside each `execute()` | 7 domain events, one per node (Section 5) — asserted for `PreProcessNode` via the spy pattern, reused per node | PASS |
| TC-12 | Audit-trace coverage gate | `python3 scripts/check_audit_trace.py src/` — every boundary node emits a reachable domain event, examples included | PASS |

## 2. Unit Tests

### 2.1 PreProcessNode — `tests/unit/test_pre_process_node.py`

The trust gate (VERIFIED_EXTERNAL), valid-query acceptance, `enriched_context`
channel passthrough, empty/whitespace/missing/non-string rejection, direct unit
coverage of `_surface_strip_identifiers()`, identifier screening through the
full `node(state)` call, and the audit payload assertion (on `call.args[1]`,
never the whole-call repr).

**The request-context contract** (`TestRequestContextContract`): the accepted
shapes, and a parametrized out-of-contract matrix — `"NaN"`, `"Infinity"`,
`"-Infinity"`, raw `float("nan")`, raw `float("inf")`, `0`, `21`, `True`, a
non-inert `skill`, an over-length `skill`, a non-list `frameworks`, a
non-inert framework entry, an over-cap framework list, a non-inert `channel`.
Each is refused with no `validated_input`. One case asserts the rejected value
never appears in the error, and one asserts that an unknown context key does
not refuse an otherwise valid request.

**Injection refusal** (`TestInjectionRefusal`, via direct `execute()`): the
control-token forms (`<|im_start|>`, `<|endoftext|>`, `[INST]`, `<<SYS>>`), the
override directives, the disclosure requests, persona replacement, a hostile
field NAME, a control token nested inside the context, and a unicode-escaped
payload — all refused. The other direction matters as much and is tested with
real questions from this domain: "what system prompt should our own synthesis
service use", "can a new delegation policy override the previous configuration
rules", "which audit rules apply after we disable the legacy security policy",
"how do I act as a delegated service account" — all unaffected.

**Redaction is not refusal** (`TestRedactionIsNotRefusal`): with a DELETING
redactor installed, a control token removed by redaction is still caught (the
raw screen sees it first), and a directive that only becomes contiguous AFTER
redaction is caught too (the post-redaction screen sees it). The second test
first asserts that the raw form is NOT matched, so it cannot pass for the wrong
reason.

### 2.2 InputValidateNode — `tests/unit/test_input_validate_node.py`

Plain-text and JSON-envelope parsing (`query`/`question`, `skill`, `top_k`),
`skill` normalisation, unrecognised-but-inert skill degradation with a note
that does NOT echo the value, malformed-JSON fallback to plain text,
oversize-query truncation, empty-request note (not an error), and the
`user_input` fallback when `validated_input` is absent.

`TestTopKGuard`: the out-of-contract matrix (out of range, zero, non-numeric,
`"NaN"`/`"Infinity"`/`"-Infinity"`, bare `NaN`/`Infinity`/`-Infinity` JSON
literals, `True`, a fractional float, a list, a mapping) is refused rather than
clamped, and the rejected value is never echoed. In-contract values (`1`, `4`,
`20`, `"7"`) are accepted.

`TestRequestContextChannel`: the structured channel is read, wins over the
envelope on conflict, is bounded by the same parser, and ignores unknown keys.

`TestInjectionRefusal`: the same two-direction screen, proving the inner node
holds when the inner graph is invoked directly.

### 2.3 RetrieveNode — `tests/unit/test_retrieve_node.py`

Deterministic keyword scoring against the REAL `config/kb/*.json` content
(queries hand-picked to hit one title's exact token set, so the top result is
provable without depending on retrieval internals). Covers the top hit and
descending-score ordering, candidate entry shape and excerpt cap, pooling
across all three seeded knowledge bases, the skill filter (narrows the
`google_agent_skills` pool only), empty query, the state-seeded `kb_paths`
override with graceful per-file degradation, notes accumulation (append, never
clobber), and the `user_input` fallback.

### 2.4 RerankFilterNode — `tests/unit/test_rerank_filter_node.py`

Default threshold and `top_k` from `config/config.yaml`, the state-seeded
overrides, the skill-match boost (+0.1, capped at 1.0, never applied to a
candidate from another knowledge base), a stricter caller `top_k` winning while
a looser one never widens, garbage-candidate robustness, and the deterministic
id-ascending tie-break.

### 2.5 GenerateAnswerNode — `tests/unit/test_generate_answer_node.py`

Numbered citation markers with knowledge-base label prefixes, the lead sentence
quoting the query, citations mirroring ranked order, and the structured-extract
aggregation as a pure VERBATIM copy of each ranked entry's own pre-authored
fields: `oauth_scopes`/`related_skills` from `google_agent_skills` only,
`compliance_flags` from `japan_compliance` only, `integration_steps` pooling
`google_agent_skills` + `integration_patterns`, `testing_checklist` pooling
every knowledge base. Dedup and list caps; the no-coverage path with an empty
`ranked_documents` and with the field absent entirely.

### 2.6 OutputFormatNode — `tests/unit/test_output_format_node.py`

Header / body / Sources / scope-notice composition, source and label suffix
formatting (omitted when blank), and **the over-claiming guard**: the standing
scope notice is asserted present on EVERY path unconditionally — with
citations, without citations, and with `grounded_answer` missing. The
no-citations "- none" line, the missing-answer fallback text, and the
`structured_answer` assembly (`structured_extract` + `citations`).

### 2.7 PostProcessNode — `tests/unit/test_post_process_node.py`

The trust gate (VERIFIED_EXTERNAL), clean pass-through, empty result non-fatal.

**Scope versus credential**, both directions: a real OAuth scope URL survives in
the flat result text AND nested inside `structured_answer`
(`TestScopeSurvives`); a credential-shaped value nested inside
`structured_answer` is still caught by the recursive scan
(`TestNestedCredentialCaught`) — proving the gate is neither too strict nor too
loose. Flat-text blocking covers an API key, a credential assignment, a JWT, an
issued Google OAuth access token (`ya29.…`, distinct from a scope URL) and an
AWS access key id.

**The scope-notice invariant** (`TestScopeNoticeInvariant`): an answer without
the notice is refused; one with it is released.

**The structured shape invariant** (`TestStructuredShapeInvariant`): an opaque
object at a leaf is refused, because the credential scan skips non-string
leaves and would release it unscanned.

**Containment** (`TestContainmentOnViolation`): on a violation, every
output-bearing field is cleared (parametrized per field), and the error delta
carries no released text, no traceback and no source paths.

### 2.8 Configuration contract — `tests/unit/test_config_manifest.py`

`config/agent.yaml` is flat and carries identity, trust level, category,
industry, and empty `requires.secrets`/`requires.extras` — and NO tuning
values. `config/config.yaml` carries the runtime parameters, every declared
`kb_path` resolves to a real file, and `runtime_config()` reads it and degrades
to an empty mapping when it cannot.

`TestDeclaredValuesAreLive` is the guard against a dead declaration: it points
the reader at a temporary settings file, asserts the changed value is forwarded
and seeded, and then drives `RerankFilterNode` to show the declared threshold
actually decides what survives. A declaration whose reader points at the wrong
place produces no error and no failing test — only this shape catches it.

## 3. Graph Composition Tests

### 3.1 Outer graph — `tests/unit/test_graph_composition.py`

`WorkspaceSkillsIntegrationQAAgent` inherits `AgentBaseGraph` directly;
`compile()` fills all 5 backbone slots with the correct classes; `add_edges()`
is NOT overridden. The `IntegrationQAGraphNode` contract
(`get_subgraph`/`extract_input`/`merge_output`/`_parent_config`), including the
settings-file-unreadable fallback (a `kb_paths` list, never `{}`). A full
end-to-end `invoke()` over the sign-off question: success status, gated output
(header, citation markers, the scope notice), the structured product surfaced
on success only, backbone traversal, a no-coverage query still terminating
success, and an ANONYMOUS caller denied at the outer boundary with
`node_history[:2] == ["InitializeNode", "PreProcessNode"]` (post_process never
runs). The `to_json`/`from_json` round-trip.

### 3.2 Inner graph — `tests/unit/test_domain_workflow_graph.py`

`DomainWorkflowGraph` inherits `BaseGraph`; registers exactly the 5 domain
nodes (no `initialize`/`finalize` leak); `_extra_initial_state()` republishes
the `retrieval` block as `retrieval_config` (a JSON string) AND carries the
bridged caller context; `get_output()` shapes the inner-to-outer merge
contract; `route()` returns `END` on error, else `output_format`.

`test_route_is_annotated_with_this_graphs_own_state` pins a subtle one: the
graph library reads a path callable's annotation as its input schema and
projects away every field the annotation does not declare, so a route annotated
with the shared base state would be handed a state with this graph's domain
fields removed, and any branch keyed on one of them would never be taken —
invisibly, because a unit test calls `route()` directly with a full mapping.
This topology is linear and does not use `add_conditional_edges()`, but the
annotation is kept correct so that wiring it later is safe.

A full inner `invoke()` over the sign-off question reproduces the same
citation outcome; the inner `node_history` is the exact linear topology; a
no-coverage query still terminates successfully.

## 4. Proof-of-Boundary Tests

| PB-ID | Boundary | Test | Expected Result | Result |
|-------|----------|------|----------------|--------|
| PB-1 | Node → event emitter | `emit_trace_event()` fires on every invocation path | No silent failures | PASS |
| PB-2 | State serialization | Post-invoke State is primitives only (structured fields are JSON strings) | No Pydantic/dataclass — `tests/proof_of_boundary/test_state_safety.py` | PASS |
| PB-3 | Agent → external service | N/A — the agent makes NO live external call by design | Auto-waived — no external service | Auto-waived |
| PB-4 | Import isolation | No platform SDK imports | AST scan: 0 violations — `tests/proof_of_boundary/test_import_isolation.py` | PASS |
| PB-5 | Checkpoint safety | No credential or Pydantic object in the checkpoint | Inspection pass — `tests/proof_of_boundary/test_state_safety.py` | PASS |
| PB-6 | Invoke execution order | Full `Graph().invoke()` at VERIFIED_EXTERNAL over the sign-off payload; backbone order `[InitializeNode, PreProcessNode, IntegrationQAGraphNode, PostProcessNode, FinalizeNode]` | Order verified — `tests/proof_of_boundary/test_pb_invoke_order.py` | PASS |
| PB-7 | Human-review interrupt propagation *(conditional)* | `IntegrationQAGraphNode.propagate_hitl = False` — no cross-boundary interrupt wired | Auto-waived — skip stub present — `tests/proof_of_boundary/test_pb7_hitl_interrupt_propagation.py` | Auto-waived |
| PB-8 | The real HTTP entry point | The whole stack through `/invoke` with Bearer auth — `tests/proof_of_boundary/test_pb_invoke_endpoint.py` | PASS | PASS |
| PB-BOOT | Standalone entry point | `import src.api.server` must not raise; module-level agent compiled; `/health` reports this agent | No import-time failure — `tests/proof_of_boundary/test_server_boot.py` | PASS |

### PB-8 in detail — `tests/proof_of_boundary/test_pb_invoke_endpoint.py`

The full stack exercised the way an external caller reaches it: the HTTP
adapter, the Bearer trust promotion, the runtime settings load, the compiled
two-layer graph, and the output gate.

- an authenticated request returns a real, cited answer with the structured
  product, and the answer changes with the question;
- **the caller's context reaches the inner graph** — `top_k: 1` yields exactly
  one citation where the default yields several, and a `frameworks` filter
  changes which compliance flags come back. This is the assertion that the
  ContextVar bridge works; the framework does not forward `input_context` into
  a subgraph on its own, and nothing at node level would reveal that;
- an unauthenticated caller is refused (401 with a token configured; refused by
  the trust gate with no token configured), and a wrong token gets the generic
  refusal that does not say which way it was wrong;
- the out-of-contract matrix is refused through the full stack with an empty
  output, and the rejected value never appears in the response body;
- bare `NaN` / `Infinity` / `-Infinity` literals in the raw request body fail
  closed wherever in the stack they are stopped;
- an oversized `input_context` is refused at the adapter (413);
- a **credential-shaped context value** is refused at the adapter (400) with a
  message naming the field, and the value never appears in the response; the
  control case — ordinary domain text on the same field — still succeeds;
- injection content is refused with nothing published (no output, no structured
  keys);
- **every released answer carries the scope notice**, across three questions
  including one with no knowledge-base coverage;
- the structured product is exactly the six-key allowlist and every leaf is a
  scalar.

## 5. Audit Events

| Event | Emitter |
|-------|---------|
| `pre_process_complete` | PreProcessNode |
| `input_validate_complete` | InputValidateNode |
| `retrieve_complete` | RetrieveNode |
| `rerank_filter_complete` | RerankFilterNode |
| `generate_answer_complete` | GenerateAnswerNode |
| `output_format_complete` | OutputFormatNode |
| `post_process_complete` | PostProcessNode |
| `input_context_credential_refused` | the HTTP adapter, on a refusal |

## 6. Business Logic Tests

Covered inline within the per-node unit suites in Section 2 rather than as a
separate flat table — each node's business logic is domain-specific enough that
the dedicated per-node test class, with its own fixtures, is the clearer unit
of organisation.

## Test Execution Summary

- Total tests: 275 across `tests/unit/` and `tests/proof_of_boundary/`
- Pass: 274 / Fail: 0 / Skip: 1
  (`tests/proof_of_boundary/test_pb7_hitl_interrupt_propagation.py` —
  auto-waived, no cross-boundary interrupt: `propagate_hitl = False`)
- Run against the real installed framework wheel, not an import shim.
