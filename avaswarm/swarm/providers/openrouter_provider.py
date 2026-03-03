"""OpenRouter provider (multi-model gateway)."""
from __future__ import annotations

from typing import AsyncIterator

import httpx

from swarm.config import settings
from swarm.providers.base import (
    BaseProvider,
    CompletionRequest,
    CompletionResponse,
    Message,
)


def _to_oai(msg: Message) -> dict:
    return {"role": msg.role, "content": msg.content}


class OpenRouterProvider(BaseProvider):
    name = "openrouter"

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or settings.openrouter_api_key
        self._base_url = settings.openrouter_base_url

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": "https://github.com/project-swarm",
            "X-Title": "project-swarm",
        }

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        payload = {
            "model": request.model or "openai/gpt-4o",
            "messages": [_to_oai(m) for m in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        return CompletionResponse(
            content=choice["message"]["content"],
            model=data.get("model", ""),
            usage=data.get("usage", {}),
            finish_reason=choice.get("finish_reason", "stop"),
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[str]:
        import json

        payload = {
            "model": request.model or "openai/gpt-4o",
            "messages": [_to_oai(m) for m in request.messages],
            "temperature": request.temperature,
            "stream": True,
        }
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=120,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta


from swarm.providers.registry import registry  # noqa: E402
registry.register(OpenRouterProvider)
