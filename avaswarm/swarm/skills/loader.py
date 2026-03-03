"""ClawHub — public skill marketplace loader."""
from __future__ import annotations

import importlib
import logging
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from swarm.config import settings
from swarm.skills.registry import SkillRegistry

if TYPE_CHECKING:
    from swarm.core.agent import Agent

logger = logging.getLogger(__name__)


class ClawHub:
    """
    Minimal ClawHub client.
    Skills are Python packages published to ClawHub; this client
    fetches their metadata and source, then installs them at runtime.
    """

    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry
        self._base = settings.clawhub_base_url
        self._headers = (
            {"X-API-Key": settings.clawhub_api_key}
            if settings.clawhub_api_key
            else {}
        )

    async def search(self, query: str) -> list[dict]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self._base}/skills/search",
                params={"q": query},
                headers=self._headers,
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json().get("results", [])

    async def install(self, skill_name: str, agent: "Agent") -> None:
        """Download a skill from ClawHub and activate it."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self._base}/skills/{skill_name}/download",
                headers=self._headers,
                timeout=30,
            )
            resp.raise_for_status()
            source_code: str = resp.text

        # Write to a temp file and load
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(source_code)
            tmp_path = Path(f.name)

        try:
            spec = importlib.util.spec_from_file_location(skill_name, tmp_path)
            if not spec or not spec.loader:
                raise ImportError(f"Cannot load skill from {tmp_path}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            if not hasattr(mod, "Skill"):
                raise AttributeError(f"Module {skill_name} has no 'Skill' class")
            await self._registry.load(mod.Skill(), agent)
            logger.info("ClawHub skill installed: %s", skill_name)
        finally:
            tmp_path.unlink(missing_ok=True)
