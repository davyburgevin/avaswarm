"""Agent management tools — create, update, and list agents from within a conversation."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from swarm.tools.base import BaseTool, ToolSchema

logger = logging.getLogger(__name__)

# Default path — same formula as web_gateway.py (parent.parent.parent == avaswarm/)
_DEFAULT_AGENTS_DIR = Path(__file__).resolve().parent.parent.parent / "agents"

_DEFAULT_AGENT_INSTRUCTIONS = (
    "You are a helpful assistant. Answer clearly and concisely."
)


class ListAgentsTool(BaseTool):
    """Return the list of all available agents with their descriptions."""

    name = "list_agents"
    description = (
        "List all available sub-agents with their names and descriptions. "
        "Use this before creating a new agent to avoid duplicates."
    )

    def __init__(self) -> None:
        self._agents_dir: Path = _DEFAULT_AGENTS_DIR

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def execute(self, **_: Any) -> str:
        self._agents_dir.mkdir(parents=True, exist_ok=True)
        agents = []
        for p in sorted(self._agents_dir.iterdir()):
            if not p.is_dir():
                continue
            cfg_path = p / "config.json"
            desc = ""
            if cfg_path.exists():
                try:
                    cfg = json.loads(cfg_path.read_text("utf-8"))
                    desc = cfg.get("description", "")
                except Exception:
                    pass
            agents.append(f"- **{p.name}**: {desc}" if desc else f"- **{p.name}**")

        if not agents:
            return "No agents found."
        return "Available agents:\n" + "\n".join(agents)


class CreateAgentTool(BaseTool):
    """Create a new agent with name, instructions, and optional config."""

    name = "create_agent"
    description = (
        "Create a new sub-agent with a name, system instructions, and optional description. "
        "Returns a confirmation message."
    )

    def __init__(self) -> None:
        self._agents_dir: Path = _DEFAULT_AGENTS_DIR
        self._cache: dict[str, Any] | None = None  # injected by web_gateway

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Unique identifier for the agent (slug, no spaces).",
                    },
                    "instructions": {
                        "type": "string",
                        "description": "Full system-prompt / instructions for the agent.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Short human-readable description shown in the UI.",
                    },
                    "model": {
                        "type": "string",
                        "description": "LLM model to use (e.g. gpt-4o-mini). Defaults to gpt-4o-mini.",
                    },
                    "provider": {
                        "type": "string",
                        "description": "LLM provider name (e.g. openai, anthropic). Defaults to openai.",
                    },
                },
                "required": ["name", "instructions"],
            },
        )

    async def execute(self, **kwargs: Any) -> str:
        agent_name: str = kwargs.get("name", "").strip()
        instructions: str = kwargs.get("instructions", "").strip()
        description: str = kwargs.get("description", "").strip()
        model: str = kwargs.get("model", "gpt-4o-mini").strip()
        provider: str = kwargs.get("provider", "openai").strip()

        if not agent_name:
            return "Error: 'name' is required."
        if not instructions:
            instructions = _DEFAULT_AGENT_INSTRUCTIONS

        # Sanitize name
        safe_name = agent_name.lower().replace(" ", "-")

        target = self._agents_dir / safe_name
        if target.exists():
            return f"Error: Agent '{safe_name}' already exists. Use update_agent_instructions to modify it."

        try:
            target.mkdir(parents=True, exist_ok=True)
            (target / "memory").mkdir(exist_ok=True)
            (target / "docs").mkdir(exist_ok=True)
            (target / "sessions").mkdir(exist_ok=True)

            config = {
                "name": safe_name,
                "description": description,
                "model": model,
                "provider": provider,
            }
            (target / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            (target / "agent.md").write_text(instructions, encoding="utf-8")

            logger.info("Agent '%s' created via tool.", safe_name)
            return f"Agent '{safe_name}' created successfully."
        except Exception as exc:
            logger.error("CreateAgentTool error: %s", exc)
            return f"Error creating agent: {exc}"


class SaveMyInstructionsTool(BaseTool):
    """Save the agent's own instructions to its agent.md file (self-configuration tool)."""

    name = "save_my_instructions"
    description = (
        "Persist your own system instructions to your agent.md file. "
        "Call this whenever the user defines, refines, or updates your role, identity, "
        "behaviour or instructions. The saved content becomes your permanent system prompt "
        "and will be reloaded on the next session. "
        "Pass the COMPLETE instructions you want to keep, not just the new part."
    )

    def __init__(self) -> None:
        self._agent_md_path: Path | None = None   # injected by web_gateway
        self._agent_name: str = ""                 # injected by web_gateway
        self._cache: dict[str, Any] | None = None  # injected by web_gateway

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "instructions": {
                        "type": "string",
                        "description": "Complete instructions text to save as permanent system prompt.",
                    },
                },
                "required": ["instructions"],
            },
        )

    async def execute(self, **kwargs: Any) -> str:
        instructions: str = kwargs.get("instructions", "").strip()
        if not instructions:
            return "Error: 'instructions' cannot be empty."
        if self._agent_md_path is None:
            return "Error: agent path not configured."
        try:
            self._agent_md_path.write_text(instructions, encoding="utf-8")
            # Evict from cache so next WS connect reloads fresh instructions
            if self._cache is not None:
                self._cache.pop(self._agent_name, None)
                logger.info("Instructions saved and cache invalidated for agent '%s'.", self._agent_name)
            return f"Instructions saved successfully to {self._agent_md_path.name}."
        except Exception as exc:
            logger.error("SaveMyInstructionsTool error: %s", exc)
            return f"Error saving instructions: {exc}"


class UpdateAgentInstructionsTool(BaseTool):
    """Update (replace or append) the instructions of an existing agent."""

    name = "update_agent_instructions"
    description = (
        "Update the system instructions of an existing agent. "
        "Use mode='replace' to overwrite or mode='append' to add to existing instructions."
    )

    def __init__(self) -> None:
        self._agents_dir: Path = _DEFAULT_AGENTS_DIR
        self._cache: dict[str, Any] | None = None  # injected by web_gateway

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the agent to update.",
                    },
                    "instructions": {
                        "type": "string",
                        "description": "New instructions text.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["replace", "append"],
                        "description": "Whether to replace the instructions or append to them. Defaults to 'replace'.",
                    },
                },
                "required": ["name", "instructions"],
            },
        )

    async def execute(self, **kwargs: Any) -> str:
        agent_name: str = kwargs.get("name", "").strip()
        instructions: str = kwargs.get("instructions", "").strip()
        mode: str = kwargs.get("mode", "replace").strip()

        if not agent_name:
            return "Error: 'name' is required."
        if not instructions:
            return "Error: 'instructions' cannot be empty."

        target = self._agents_dir / agent_name
        if not target.exists():
            return f"Error: Agent '{agent_name}' not found. Use list_agents to see available agents."

        instr_path = target / "agent.md"
        try:
            if mode == "append" and instr_path.exists():
                existing = instr_path.read_text("utf-8").rstrip()
                new_content = existing + "\n\n" + instructions
            else:
                new_content = instructions

            instr_path.write_text(new_content, encoding="utf-8")

            # Invalidate the sub-agent cache so the next call reloads fresh instructions
            if self._cache is not None:
                self._cache.pop(agent_name, None)
                logger.debug("Cache invalidated for agent '%s'.", agent_name)

            return f"Instructions updated for agent '{agent_name}' (mode={mode})."
        except Exception as exc:
            logger.error("UpdateAgentInstructionsTool error: %s", exc)
            return f"Error updating agent: {exc}"
