"""Meta tools — orchestrate other agents and introspect the system."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from swarm.tools.base import BaseTool, ToolSchema

if TYPE_CHECKING:
    from swarm.tools.registry import ToolRegistry


class ListToolsTool(BaseTool):
    name = "list_tools"
    description = "Return the list of all tools available to this agent."

    def __init__(self, tool_registry: "ToolRegistry") -> None:
        self._registry = tool_registry

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def execute(self, **_: Any) -> list[str]:
        return self._registry.list()


class SpawnSubAgentTool(BaseTool):
    """
    Dynamically spawns a sub-agent. The parent agent must inject itself
    after construction (see Agent.tools for the circular reference trick).
    """
    name = "spawn_sub_agent"
    description = (
        "Spawn a sub-agent to handle an isolated task. "
        "Returns the sub-agent's final response."
    )

    def __init__(self) -> None:
        self._parent_agent: Any | None = None

    def bind(self, agent: Any) -> None:
        self._parent_agent = agent

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Description of the task for the sub-agent."},
                    "provider": {"type": "string", "description": "LLM provider name (optional)."},
                    "model": {"type": "string", "description": "Model name (optional)."},
                },
                "required": ["task"],
            },
        )

    async def execute(self, task: str, provider: str | None = None, model: str | None = None, **_: Any) -> str:
        if not self._parent_agent:
            return "SpawnSubAgentTool not bound to a parent agent."
        return await self._parent_agent.spawn_sub_agent(task, provider_name=provider, model=model)
