"""Base LLM provider interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict] | None = None  # set on assistant messages that invoke tools


@dataclass
class CompletionRequest:
    messages: list[Message]
    model: str | None = None
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int | None = None
    tools: list[dict] | None = None


@dataclass
class CompletionResponse:
    content: str
    model: str
    usage: dict = field(default_factory=dict)
    finish_reason: str = "stop"
    tool_calls: list[dict] | None = None


class BaseProvider(ABC):
    """Abstract base class every LLM provider must implement."""

    name: str = "base"

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Return a full completion."""

    @abstractmethod
    async def stream(self, request: CompletionRequest) -> AsyncIterator[str]:
        """Yield completion chunks as strings."""

    def to_dict(self) -> dict:
        return {"name": self.name}
