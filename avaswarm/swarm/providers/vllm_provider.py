"""vLLM provider — OpenAI-compatible local inference server."""
from __future__ import annotations

from typing import AsyncIterator

from swarm.config import settings
from swarm.providers.base import CompletionRequest, CompletionResponse
from swarm.providers.openai_provider import OpenAIProvider


class VLLMProvider(OpenAIProvider):
    """
    vLLM exposes an OpenAI-compatible API, so we simply reuse the
    OpenAI provider pointed at the local server.
    """
    name = "vllm"

    def __init__(self, base_url: str | None = None, model: str | None = None):
        super().__init__(
            api_key="EMPTY",  # vLLM doesn't require a real key
            base_url=base_url or settings.vllm_base_url,
        )
        self._default_model = model or settings.vllm_model

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        if not request.model:
            request.model = self._default_model
        return await super().complete(request)

    async def stream(self, request: CompletionRequest) -> AsyncIterator[str]:
        if not request.model:
            request.model = self._default_model
        async for chunk in super().stream(request):
            yield chunk


from swarm.providers.registry import registry  # noqa: E402
registry.register(VLLMProvider)
