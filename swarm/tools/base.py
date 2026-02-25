"""Base tool interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolSchema:
    """OpenAI-compatible function schema."""
    name: str
    description: str
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}, "required": []})

    def to_oai(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class BaseTool(ABC):
    """All tools inherit from this class."""

    name: str = ""
    description: str = ""

    @abstractmethod
    def schema(self) -> ToolSchema:
        """Return the JSON schema for this tool."""

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool and return a JSON-serialisable result."""
