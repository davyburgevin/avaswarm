"""Anthropic provider (Claude 3.x …)."""
from __future__ import annotations

from typing import AsyncIterator

import anthropic

from swarm.config import settings
from swarm.providers.base import (
    BaseProvider,
    CompletionRequest,
    CompletionResponse,
    Message,
)

_DEFAULT_MODEL = "claude-3-5-sonnet-20241022"


def _split_system(messages: list[Message]) -> tuple[str, list[dict]]:
    system = ""
    turns: list[dict] = []
    for m in messages:
        if m.role == "system":
            system += m.content + "\n"
        else:
            turns.append({"role": m.role, "content": m.content})
    return system.strip(), turns


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, api_key: str | None = None):
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or settings.anthropic_api_key
        )

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        system, turns = _split_system(request.messages)
        kwargs: dict = dict(
            model=request.model or _DEFAULT_MODEL,
            max_tokens=request.max_tokens or 4096,
            messages=turns,
            temperature=request.temperature,
        )
        if system:
            kwargs["system"] = system
        if request.tools:
            kwargs["tools"] = request.tools

        resp = await self._client.messages.create(**kwargs)
        content_block = resp.content[0]
        text = content_block.text if hasattr(content_block, "text") else ""
        return CompletionResponse(
            content=text,
            model=resp.model,
            usage={"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens},
            finish_reason=resp.stop_reason or "stop",
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[str]:
        system, turns = _split_system(request.messages)
        kwargs: dict = dict(
            model=request.model or _DEFAULT_MODEL,
            max_tokens=request.max_tokens or 4096,
            messages=turns,
            temperature=request.temperature,
        )
        if system:
            kwargs["system"] = system

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text


from swarm.providers.registry import registry  # noqa: E402
registry.register(AnthropicProvider)
