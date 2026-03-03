"""Tool Registry — manages available tools and exposes their schemas."""
from __future__ import annotations

import logging
from typing import Any, Type

from swarm.tools.base import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def register_cls(self, cls: Type[BaseTool], **kwargs) -> None:
        self.register(cls(**kwargs))

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found. Available: {list(self._tools)}")
        return self._tools[name]

    def list(self) -> list[str]:
        return list(self._tools.keys())

    def schemas(self) -> list[dict]:
        return [t.schema().to_oai() for t in self._tools.values()]

    async def call(self, name: str, **kwargs: Any) -> Any:
        tool = self.get(name)
        try:
            return await tool.execute(**kwargs)
        except Exception as exc:
            logger.error("Tool %s raised: %s", name, exc)
            return {"error": str(exc)}

    @classmethod
    def default(cls) -> "ToolRegistry":
        """Return a registry pre-loaded with all built-in tools."""
        from swarm.tools.file_tools import ReadFileTool, WriteFileTool, ListDirTool
        from swarm.tools.shell_tools import ShellTool
        from swarm.tools.web_search import BraveSearchTool
        from swarm.tools.meta_tools import SpawnSubAgentTool, ListToolsTool
        from swarm.tools.memory_tool import SaveMemoryTool, ReadMemoryTool
        from swarm.tools.agent_tools import CreateAgentTool, UpdateAgentInstructionsTool, ListAgentsTool, SaveMyInstructionsTool

        reg = cls()
        for tool in [
            ReadFileTool(),
            WriteFileTool(),
            ListDirTool(),
            ShellTool(),
            BraveSearchTool(),
            ListToolsTool(reg),
            SaveMemoryTool(),   # memory manager injected later by Agent
            ReadMemoryTool(),
            ListAgentsTool(),
            CreateAgentTool(),
            UpdateAgentInstructionsTool(),
            SaveMyInstructionsTool(),
        ]:
            reg.register(tool)
        return reg
