"""OpenAI provider (GPT-4o, GPT-4, GPT-3.5 …)."""
from __future__ import annotations

from typing import AsyncIterator

import openai

from swarm.config import settings
from swarm.providers.base import (
    BaseProvider,
    CompletionRequest,
    CompletionResponse,
    Message,
)


def _to_oai(msg: Message) -> dict:
    d: dict = {"role": msg.role, "content": msg.content}
    if msg.name:
        d["name"] = msg.name
    if msg.tool_call_id:
        d["tool_call_id"] = msg.tool_call_id
    return d


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self._client = openai.AsyncOpenAI(
            api_key=api_key or settings.openai_api_key,
            base_url=base_url or settings.openai_base_url,
        )

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        kwargs: dict = dict(
            model=request.model or settings.default_model,
            messages=[_to_oai(m) for m in request.messages],
            temperature=request.temperature,
        )
        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens
        if request.tools:
            kwargs["tools"] = request.tools

        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [tc.model_dump() for tc in choice.message.tool_calls]
        return CompletionResponse(
            content=choice.message.content or "",
            model=resp.model,
            usage=resp.usage.model_dump() if resp.usage else {},
            finish_reason=choice.finish_reason or "stop",
            tool_calls=tool_calls,
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[str]:
        kwargs: dict = dict(
            model=request.model or settings.default_model,
            messages=[_to_oai(m) for m in request.messages],
            temperature=request.temperature,
            stream=True,
        )
        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens

        async with await self._client.chat.completions.create(**kwargs) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta


# self-register
from swarm.providers.registry import registry  # noqa: E402
registry.register(OpenAIProvider)
