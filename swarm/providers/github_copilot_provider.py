"""GitHub Copilot provider (uses the Copilot OAuth token to call the Copilot chat API)."""
from __future__ import annotations

from typing import AsyncIterator
import json

import httpx

from swarm.config import settings
from swarm.providers.base import (
    BaseProvider,
    CompletionRequest,
    CompletionResponse,
    Message,
)

_COPILOT_API = "https://api.githubcopilot.com"
_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"


class GitHubCopilotProvider(BaseProvider):
    """
    Uses a GitHub OAuth token (ghu_...) to obtain a short-lived Copilot
    session token, then calls the Copilot chat completions endpoint.
    """
    name = "github_copilot"

    def __init__(self, oauth_token: str | None = None):
        self._oauth_token = oauth_token or settings.github_copilot_token
        self._session_token: str | None = None

    async def _get_session_token(self) -> str:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                _TOKEN_URL,
                headers={
                    "Authorization": f"token {self._oauth_token}",
                    "Accept": "application/json",
                },
                timeout=30,
            )
            resp.raise_for_status()
            self._session_token = resp.json()["token"]
        return self._session_token

    def _headers(self, session_token: str) -> dict:
        return {
            "Authorization": f"Bearer {session_token}",
            "Content-Type": "application/json",
            "Editor-Version": "vscode/1.90.0",
            "Copilot-Integration-Id": "vscode-chat",
        }

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        token = await self._get_session_token()
        payload = {
            "model": request.model or "gpt-4o",
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{_COPILOT_API}/chat/completions",
                json=payload,
                headers=self._headers(token),
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        return CompletionResponse(
            content=choice["message"]["content"],
            model=data.get("model", "copilot"),
            usage=data.get("usage", {}),
            finish_reason=choice.get("finish_reason", "stop"),
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[str]:
        token = await self._get_session_token()
        payload = {
            "model": request.model or "gpt-4o",
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "stream": True,
        }
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{_COPILOT_API}/chat/completions",
                json=payload,
                headers=self._headers(token),
                timeout=120,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta


from swarm.providers.registry import registry  # noqa: E402
registry.register(GitHubCopilotProvider)
