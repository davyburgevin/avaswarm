"""Providers package — import registry to trigger auto-registration."""
from swarm.providers.base import BaseProvider, Message, CompletionRequest, CompletionResponse
from swarm.providers.registry import registry

__all__ = ["BaseProvider", "Message", "CompletionRequest", "CompletionResponse", "registry"]
