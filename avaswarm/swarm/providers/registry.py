"""Provider Registry — discover and instantiate LLM providers at runtime."""
from __future__ import annotations

from typing import Type

from swarm.providers.base import BaseProvider


class ProviderRegistry:
    """Lightweight registry: map name → provider class."""

    def __init__(self) -> None:
        self._registry: dict[str, Type[BaseProvider]] = {}

    def register(self, provider_cls: Type[BaseProvider]) -> Type[BaseProvider]:
        """Register a provider class (usable as a decorator)."""
        self._registry[provider_cls.name] = provider_cls
        return provider_cls

    def get(self, name: str) -> Type[BaseProvider]:
        if name not in self._registry:
            raise KeyError(
                f"Provider '{name}' not found. Available: {list(self._registry)}"
            )
        return self._registry[name]

    def list(self) -> list[str]:
        return list(self._registry.keys())

    def build(self, name: str, **kwargs) -> BaseProvider:
        """Instantiate a provider by name."""
        cls = self.get(name)
        return cls(**kwargs)


# Global singleton
registry = ProviderRegistry()


def _auto_register() -> None:
    """Import all built-in providers so they self-register."""
    from swarm.providers import (  # noqa: F401
        openai_provider,
        anthropic_provider,
        openrouter_provider,
        github_copilot_provider,
        vllm_provider,
    )


_auto_register()
