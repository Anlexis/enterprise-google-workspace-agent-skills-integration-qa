# CMN-C2-298 — WorkspaceSkillsIntegrationQAAgent

> **Category**: Cat 2 (domain-specific retrieval-augmented Q&A pipeline)
> **Industry**: CMN (cross-industry)

## Overview

Answers natural-language questions about integrating Google's official Agent Skills
(Workspace, Cloud, Maps, YouTube, Calendar, Gmail, Sheets, Drive, Docs, Search) into a Japanese
enterprise. The agent retrieves over three seeded knowledge bases — Agent Skills documentation,
enterprise integration patterns, and Japan compliance requirements (APPI, FISC, ISMS) — and
returns a cited advisory answer plus a structured product: integration steps, the OAuth scopes
each skill needs, compliance flags, a testing checklist, related skills and source citations.
Callers can narrow retrieval with a skill filter and a result count passed as request context.

It answers *how to integrate* the skills. It never connects to, authenticates against, or calls
a Google API, and it makes no configuration changes on your behalf; every answer carries that
limitation in a standing scope notice.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent fails
during graph compile / start-up preflight rather than starting in a partially
working state. This is intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit, integration and boundary tests
config/       agent configuration
docs/         design and test documentation
```

See `docs/` for the design document and the test specification.

## Customising

1. Adjust `config/` for your own environment and policies.
2. Replace the knowledge sources and sample data with your own.
3. Review the node implementations under `src/nodes/` for domain-specific logic.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
