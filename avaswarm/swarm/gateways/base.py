"""Base gateway interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swarm.core.agent import Agent


class BaseGateway(ABC):
    """All gateways must implement start() and stop()."""

    name: str = "base"

    def __init__(self, agent: "Agent") -> None:
        self.agent = agent

    @abstractmethod
    async def start(self) -> None:
        """Start accepting input from this channel."""

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully stop the gateway."""
