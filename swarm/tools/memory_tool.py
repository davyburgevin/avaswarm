"""Memory tools — let the LLM read/write long-term memory."""
from __future__ import annotations

from typing import Any

from swarm.tools.base import BaseTool, ToolSchema


class SaveMemoryTool(BaseTool):
    """Persist a fact to long-term memory (MEMORY.md)."""

    name = "save_memory"
    description = (
        "Save an important fact to long-term memory so it is remembered across all future sessions. "
        "Use this whenever the user shares personal information (name, preferences, job, location, etc.) "
        "or whenever a key fact should be retained permanently."
    )

    def __init__(self, memory_manager=None) -> None:
        self._memory = memory_manager  # injected by Agent after init

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The fact to remember, written as a concise statement. E.g. 'The user's name is Alice.'",
                    }
                },
                "required": ["fact"],
            },
        )

    async def execute(self, fact: str, **_: Any) -> str:
        if self._memory is None:
            return "Memory manager not available."
        await self._memory.append_long_term(f"- {fact}")
        return f"Saved to long-term memory: {fact}"


class ReadMemoryTool(BaseTool):
    """Read the current long-term memory."""

    name = "read_memory"
    description = "Read the contents of long-term memory. Use this to check what facts are already stored."

    def __init__(self, memory_manager=None) -> None:
        self._memory = memory_manager

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def execute(self, **_: Any) -> str:
        if self._memory is None:
            return ""
        content = await self._memory.read_long_term()
        return content or "(long-term memory is empty)"
