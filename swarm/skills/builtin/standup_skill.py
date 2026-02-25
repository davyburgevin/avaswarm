"""Example built-in skill: daily stand-up reminder."""
from __future__ import annotations

from swarm.skills.base import BaseSkill, SkillMeta
from swarm.tools.base import BaseTool, ToolSchema


class StandUpTool(BaseTool):
    name = "standup_summary"
    description = "Generate a short daily stand-up summary from yesterday's memory."

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def execute(self, **_) -> str:
        from datetime import date, timedelta
        from swarm.memory.manager import MemoryManager

        memory = MemoryManager()
        yesterday = date.today() - timedelta(days=1)
        notes = await memory.read_daily(yesterday)
        if not notes:
            return "No activity recorded yesterday."
        return f"## Stand-up summary for {yesterday}\n\n{notes[:2000]}"


class Skill(BaseSkill):
    meta = SkillMeta(
        name="standup",
        version="0.1.0",
        description="Daily stand-up skill — summarises yesterday's work.",
        author="project-swarm",
        tags=["productivity", "daily"],
    )

    async def activate(self, agent) -> None:
        agent.tools.register(StandUpTool())
