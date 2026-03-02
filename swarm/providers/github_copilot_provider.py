"""GitHub Copilot provider.

This provider prefers the official GitHub Copilot SDK (if installed) which
manages the Copilot CLI lifecycle and authentication. If the SDK is not
available, it falls back to the previous HTTP behaviour that exchanges a
GitHub OAuth token for a short-lived Copilot session token and calls the
Copilot chat API.

Authentication options:
- Preferred: install Copilot CLI + Copilot SDK and authenticate via the CLI
    (or set COPILOT_GITHUB_TOKEN/GH_TOKEN/GITHUB_TOKEN in the environment).
- Fallback: provide a GitHub OAuth token (ghu_/ghp_) in the `GITHUB_COPILOT_TOKEN`
    setting or via the web UI.
"""
from __future__ import annotations

from typing import AsyncIterator
import json
import os
import logging
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

# Module-level session token cache keyed by oauth_token.
# Shared across all provider instances so the exchange is done at most once
# per process (session tokens are valid for ~1 hour).
import time as _time
_SESSION_CACHE: dict[str, tuple[str, float]] = {}  # oauth_token → (session_token, expires_at)


class GitHubCopilotProvider(BaseProvider):
    """
    Uses a GitHub OAuth token (ghu_...) to obtain a short-lived Copilot
    session token, then calls the Copilot chat completions endpoint.
    """
    name = "github_copilot"

    def __init__(self, oauth_token: str | None = None):
        # Resolve token from parameter, settings or common environment variables
        token = oauth_token or settings.github_copilot_token
        token = token or os.environ.get("COPILOT_GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        self._oauth_token = token
        self._session_token: str | None = None

        # Try to detect Copilot SDK availability
        self._use_sdk = False
        self._sdk_client = None
        try:
            import github_copilot_sdk as copilot_sdk  # type: ignore
            # Try common client constructors/creators
            client = None
            for ctor in ("Client", "CopilotClient", "create_client"):
                if hasattr(copilot_sdk, ctor):
                    c = getattr(copilot_sdk, ctor)
                    try:
                        # Prefer passing token via env/constructor if supported
                        client = c(token) if callable(c) else None
                    except TypeError:
                        try:
                            client = c()
                        except Exception:
                            client = None
                    if client:
                        break
            # If still nothing, try top-level module factory names
            if not client and hasattr(copilot_sdk, "connect"):
                try:
                    client = copilot_sdk.connect()
                except Exception:
                    client = None
            if client:
                self._use_sdk = True
                self._sdk_client = client
        except Exception:
            logging.getLogger(__name__).debug("Copilot SDK not available; falling back to HTTP path")

    async def _get_token(self) -> tuple[str, str]:
        """Return (token, auth_scheme) where auth_scheme is 'Bearer' or 'token'.

        Strategy:
        1. Check module-level session token cache (valid for ~1h, shared across instances).
        2. Try the internal exchange endpoint (requires active Copilot subscription).
        3. On success, store in module-level cache with expiry.
        4. If exchange returns 4xx, fall back to OAuth token directly with Bearer auth.
        """
        if not self._oauth_token:
            raise RuntimeError(
                "No GitHub OAuth token configured. Use the device login in the Config page."
            )

        log = logging.getLogger(__name__)

        # Check module-level cache first (shared across all provider instances)
        cached = _SESSION_CACHE.get(self._oauth_token)
        if cached:
            session_tok, expires_at = cached
            if _time.time() < expires_at:
                return session_tok, "Bearer"
            else:
                del _SESSION_CACHE[self._oauth_token]

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    _TOKEN_URL,
                    headers={
                        "Authorization": f"token {self._oauth_token}",
                        "Accept": "application/json",
                        "User-Agent": "GithubCopilot/1.246.0",
                        "Editor-Version": "vscode/1.96.0",
                        "Editor-Plugin-Version": "copilot/1.246.0",
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                session_token = resp.json()["token"]
                # Cache for 55 minutes (tokens are valid ~1h)
                _SESSION_CACHE[self._oauth_token] = (session_token, _time.time() + 55 * 60)
                self._session_token = session_token
                log.debug("Copilot session token obtained via internal exchange (cached)")
                return session_token, "Bearer"
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403, 404):
                log.info(
                    "Internal token exchange returned %s — "
                    "falling back to direct OAuth token on Copilot API",
                    exc.response.status_code,
                )
                return self._oauth_token, "Bearer"
            raise

    async def _get_session_token(self) -> str:
        """Legacy helper kept for compatibility; prefer _get_token()."""
        token, _ = await self._get_token()
        return token

    def _headers(self, token: str, scheme: str = "Bearer") -> dict:
        return {
            "Authorization": f"{scheme} {token}",
            "Content-Type": "application/json",
            "Editor-Version": "vscode/1.96.0",
            "Editor-Plugin-Version": "copilot/1.246.0",
            "Copilot-Integration-Id": "vscode-chat",
            "User-Agent": "GithubCopilot/1.246.0",
        }

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        # If SDK is available, prefer calling it
        if self._use_sdk and self._sdk_client is not None:
            try:
                # Try several possible client method names (best-effort)
                for method_name in ("chat", "complete", "create_completion", "create_chat_completion"):
                    if hasattr(self._sdk_client, method_name):
                        method = getattr(self._sdk_client, method_name)
                        try:
                            # Some SDKs accept messages as list of dicts
                            msgs = [{"role": m.role, "content": m.content} for m in request.messages]
                            resp = method(messages=msgs, model=request.model or None)
                            # If result is awaitable
                            if hasattr(resp, "__await__"):
                                resp = await resp
                            # Try to extract content
                            content = None
                            if isinstance(resp, dict):
                                # common shapes
                                if "choices" in resp:
                                    content = resp["choices"][0].get("message", {}).get("content") or resp["choices"][0].get("text")
                                else:
                                    content = resp.get("content") or resp.get("text")
                            elif hasattr(resp, "content"):
                                content = getattr(resp, "content")
                            if content is None:
                                content = str(resp)
                            return CompletionResponse(content=content, model=getattr(resp, "model", "copilot"))
                        except Exception:
                            continue
            except Exception:
                logging.getLogger(__name__).exception("Copilot SDK call failed; falling back to HTTP")

        # Fallback: HTTP exchange
        token, scheme = await self._get_token()
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
                headers=self._headers(token, scheme),
                timeout=120,
            )
            if resp.status_code == 403:
                body = resp.text or ""
                model_used = payload.get("model", "?")
                raise ValueError(
                    f"Model '{model_used}' is not available on your GitHub Copilot plan (403 Forbidden). "
                    f"Use /api/copilot/models to see available models. Server response: {body!r}"
                )
            if resp.status_code == 404:
                model_used = payload.get("model", "?")
                raise ValueError(
                    f"Model '{model_used}' not found on the Copilot API (404). "
                    "Check the model ID — use the dropdown in Config to select a valid model."
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
        # If SDK client supports streaming, prefer it
        if self._use_sdk and self._sdk_client is not None:
            try:
                for method_name in ("stream", "stream_chat", "stream_complete"):
                    if hasattr(self._sdk_client, method_name):
                        method = getattr(self._sdk_client, method_name)
                        msgs = [{"role": m.role, "content": m.content} for m in request.messages]
                        resp_iter = method(messages=msgs, model=request.model or None, stream=True)
                        if hasattr(resp_iter, "__aiter__"):
                            async for chunk in resp_iter:
                                # try to extract text
                                if isinstance(chunk, str):
                                    yield chunk
                                elif isinstance(chunk, dict):
                                    text = chunk.get("delta") or chunk.get("content") or ""
                                    if text:
                                        yield text
                            return
                        # If sync iterator
                        for chunk in resp_iter:
                            if isinstance(chunk, str):
                                yield chunk
                            elif isinstance(chunk, dict):
                                text = chunk.get("delta") or chunk.get("content") or ""
                                if text:
                                    yield text
                        return
            except Exception:
                logging.getLogger(__name__).exception("Copilot SDK streaming failed; falling back to HTTP stream")

        token, scheme = await self._get_token()
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
                headers=self._headers(token, scheme),
                timeout=120,
            ) as resp:
                if resp.status_code == 403:
                    body = await resp.aread()
                    model_used = payload.get("model", "?")
                    raise ValueError(
                        f"Model '{model_used}' is not available on your GitHub Copilot plan (403). "
                        f"Use the model dropdown in Config to select a valid model. Body: {body.decode()!r}"
                    )
                if resp.status_code == 404:
                    model_used = payload.get("model", "?")
                    raise ValueError(
                        f"Model '{model_used}' not found on Copilot API (404). "
                        "Check the model ID via the dropdown in Config."
                    )
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunk = json.loads(line[6:])
                        choices = chunk.get("choices")
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {}).get("content", "")
                        if delta:
                            yield delta


from swarm.providers.registry import registry  # noqa: E402
registry.register(GitHubCopilotProvider)
