"""Base skill interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SkillMeta:
    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    tags: list[str] = field(default_factory=list)


class BaseSkill(ABC):
    """
    A Skill is a self-contained capability that can be activated and
    disabled at runtime. Skills may expose new tools to the agent.
    """
    meta: SkillMeta

    @abstractmethod
    async def activate(self, agent: "Any") -> None:  # noqa: F821
        """Called when the skill is loaded. Register tools here."""

    async def deactivate(self, agent: "Any") -> None:  # noqa: F821
        """Called when the skill is unloaded (optional cleanup)."""
