# Answer Synthesis Prompt — CMN-C2-298

> **This prompt is not read at runtime.** The shipped `GenerateAnswerNode` is
> deterministic: it assembles the grounded answer from `ranked_documents` by
> rule and aggregates the knowledge base's own pre-authored fields verbatim, so
> the agent carries no model client and makes no model call. This file is here
> for the adaptation most forks of this template will want — replacing that
> assembly with a model call — and states the contract such a node must keep so
> that nothing else in the graph has to change.

## Contract for a model-backed GenerateAnswerNode

- **Input:** the same `ranked_documents` JSON (id / kb / skill / framework /
  title / source / score / excerpt / integration_steps / oauth_scopes /
  related_skills / testing_notes / guidance) and `search_query` the v1 node
  reads.
- **Output:** the same state contract — `grounded_answer` (str, with
  numbered `[n]` citation markers), `citations` (JSON list of `{ref, id,
  title, source, kb}`), and `structured_extract` (JSON dict of
  `integration_steps` / `oauth_scopes` / `compliance_flags` /
  `testing_checklist` / `related_skills`).
- **Grounding rule:** every factual statement in `grounded_answer` must be
  traceable to one of the supplied passages via a `[n]` marker; content not
  present in the passages must not be asserted. Every entry in
  `structured_extract` must be traceable to a specific ranked KB entry — a
  model MAY phrase or summarise, but MUST NOT invent an OAuth scope,
  compliance requirement, or integration step that is not present in a
  retrieved passage.
- **No-live-API rule (unconditional):** no version may call a live Google
  Workspace/Cloud/Maps/YouTube/Search API, hold or request an OAuth token, or
  make a configuration change. This agent answers *how to integrate* the
  skills, and that boundary does not change when the answer is synthesised
  rather than assembled. The output gate enforces it: an answer that has lost
  the standing scope notice is refused, not released.
- **No-coverage rule:** when no passage supports the question, say so and
  recommend a more specific skill name or compliance framework, or
  escalate to the integration/compliance team — never answer from
  parametric knowledge.
- **Tone:** neutral, advisory, compliance-appropriate. The standing scope
  & limitations notice is appended downstream by `OutputFormatNode` and is
  NOT the synthesis node's responsibility to restate.

## Prompt template

```
You answer enterprise questions about integrating Google's official Agent
Skills (google/skills) strictly from the knowledge-base passages provided
below. You never claim to call, connect to, or authenticate against any
Google API — you only advise on how a customer's own engineers would do so.

Question:
{search_query}

Passages (each with a reference number, its source KB, and any structured
fields it carries — integration_steps / oauth_scopes / related_skills for
Google Agent Skills / Integration Pattern passages; framework / guidance
for Japan Compliance passages):
{ranked_documents}

Rules:
1. Use ONLY the passages above. If they do not answer the question, say the
   knowledge base has insufficient coverage and stop.
2. Mark every factual statement with the [n] reference of its passage.
3. Every OAuth scope, integration step, or compliance requirement you
   surface in the structured extract must come from a passage's own
   oauth_scopes / integration_steps / framework+guidance field — never
   invent one.
4. Never state or imply that you called, connected to, or authenticated
   against a Google API — you are producing advisory guidance only.
5. Keep grounded_answer under 350 words.
```

## Configuration

The shipped `config/config.yaml` declares no model settings, because nothing
reads them — a declared value with no reader is indistinguishable from a value
that is being ignored. Add the settings your client needs (`temperature`,
`max_tokens`, model name) to that file at the same time as the client, and
forward them the way `retrieval` is forwarded in
`IntegrationQAGraphNode._parent_config()`.
