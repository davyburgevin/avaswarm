"""Skills Registry — manage installed skills."""
from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Type

from swarm.skills.base import BaseSkill

if TYPE_CHECKING:
    from swarm.core.agent import Agent

logger = logging.getLogger(__name__)


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, BaseSkill] = {}

    async def load(self, skill: BaseSkill, agent: "Agent") -> None:
        name = skill.meta.name
        if name in self._skills:
            logger.warning("Skill '%s' already loaded, skipping.", name)
            return
        await skill.activate(agent)
        self._skills[name] = skill
        logger.info("Skill loaded: %s v%s", name, skill.meta.version)

    async def unload(self, name: str, agent: "Agent") -> None:
        skill = self._skills.pop(name, None)
        if skill:
            await skill.deactivate(agent)
            logger.info("Skill unloaded: %s", name)

    def list(self) -> list[dict]:
        return [
            {"name": s.meta.name, "version": s.meta.version, "description": s.meta.description}
            for s in self._skills.values()
        ]

    def get(self, name: str) -> BaseSkill | None:
        return self._skills.get(name)

    # ------------------------------------------------------------------
    # Load from a Python module path
    # ------------------------------------------------------------------

    async def load_from_module(self, module_path: str, agent: "Agent") -> None:
        """
        Load a skill from a dotted module path.
        The module must expose a 'Skill' class that inherits BaseSkill.
        """
        mod = importlib.import_module(module_path)
        skill_cls: Type[BaseSkill] = getattr(mod, "Skill")
        await self.load(skill_cls(), agent)

    # ------------------------------------------------------------------
    # Load all skills from a directory
    # ------------------------------------------------------------------

    async def load_from_dir(self, directory: Path, agent: "Agent") -> None:
        for py_file in directory.glob("*.py"):
            if py_file.stem.startswith("_"):
                continue
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            if not spec or not spec.loader:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            if hasattr(mod, "Skill"):
                await self.load(mod.Skill(), agent)
