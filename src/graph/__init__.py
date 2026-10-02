"""AgentCore Platform v1.0"""

# AgentRegistry entry-point exposure: config/agent.yaml declares
# module: "src.graph" / class: "WorkspaceSkillsIntegrationQAAgent" — the registry
# resolves this via getattr(import_module("src.graph"), "WorkspaceSkillsIntegrationQAAgent"),
# so the package __init__ must re-export the real graph class (RULES A4).
from .graph import WorkspaceSkillsIntegrationQAAgent

__all__ = ["WorkspaceSkillsIntegrationQAAgent"]
