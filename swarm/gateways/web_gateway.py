"""
Web gateway — FastAPI multi-page UI with:
  - /            Home (cards)
  - /chat        Chat page (WebSocket streaming)
  - /config      LLM configuration
  - /memory      Memory browser
  - /scheduler   Job manager

REST API under /api/*
Static files from swarm/web/static/
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from swarm.config import settings
from swarm.gateways.base import BaseGateway

if TYPE_CHECKING:
    from swarm.core.agent import Agent

logger = logging.getLogger(__name__)

_WEB_DIR       = Path(__file__).resolve().parent.parent / "web"
_TEMPLATES_DIR = _WEB_DIR / "templates"
_STATIC_DIR    = _WEB_DIR / "static"

# ---------------------------------------------------------------------------
# Helper — persist _runtime_config to .env so settings survive restarts
# ---------------------------------------------------------------------------
_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"

# Map ConfigPayload field → .env key name
_ENV_KEYS = {
    "provider":             "DEFAULT_PROVIDER",
    "model":                "DEFAULT_MODEL",
    "openai_api_key":       "OPENAI_API_KEY",
    "openai_base_url":      "OPENAI_BASE_URL",
    "anthropic_api_key":    "ANTHROPIC_API_KEY",
    "openrouter_api_key":   "OPENROUTER_API_KEY",
    "openrouter_base_url":  "OPENROUTER_BASE_URL",
    "github_copilot_token": "GITHUB_COPILOT_TOKEN",
    "vllm_base_url":        "VLLM_BASE_URL",
    "context_window":       "CONTEXT_WINDOW",
}

def _persist_env(data: dict) -> None:
    """Write (or update) key=value lines in .env without touching unknown lines."""
    # Read existing lines if file exists
    existing: dict[str, str] = {}
    lines: list[str] = []
    if _ENV_FILE.exists():
        for raw in _ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                existing[k.strip()] = v.strip()
            lines.append(raw)

    # Merge new values
    for payload_key, env_key in _ENV_KEYS.items():
        value = data.get(payload_key)
        if value:  # only persist non-empty values
            existing[env_key] = value

    # Rebuild file: update in-place lines, append new ones
    updated_keys: set[str] = set()
    new_lines: list[str] = []
    for raw in lines:
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, _ = line.partition("=")
            k = k.strip()
            if k in existing:
                new_lines.append(f"{k}={existing[k]}")
                updated_keys.add(k)
                continue
        new_lines.append(raw)

    for env_key, value in existing.items():
        if env_key not in updated_keys:
            new_lines.append(f"{env_key}={value}")

    _ENV_FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    logger.info("Config persisted to %s", _ENV_FILE)


# ---------------------------------------------------------------------------
# Helper — build a live provider instance from a ConfigPayload
# ---------------------------------------------------------------------------
def _provider_from_payload(payload: "ConfigPayload"):
    """Instantiate the right provider with the correct kwargs from the form."""
    from swarm.providers.registry import registry
    provider_cls = registry.get(payload.provider)
    kwargs: dict = {}
    if payload.provider == "openai":
        if payload.openai_api_key:
            kwargs["api_key"] = payload.openai_api_key
        if payload.openai_base_url:
            kwargs["base_url"] = payload.openai_base_url
    elif payload.provider == "anthropic":
        if payload.anthropic_api_key:
            kwargs["api_key"] = payload.anthropic_api_key
    elif payload.provider == "openrouter":
        if payload.openrouter_api_key:
            kwargs["api_key"] = payload.openrouter_api_key
        if payload.openrouter_base_url:
            kwargs["base_url"] = payload.openrouter_base_url
    elif payload.provider == "github_copilot":
        if payload.github_copilot_token:
            kwargs["oauth_token"] = payload.github_copilot_token
    elif payload.provider == "vllm":
        if payload.vllm_base_url:
            kwargs["base_url"] = payload.vllm_base_url
        kwargs["model"] = payload.model  # critical — sets _default_model
    return provider_cls(**kwargs)


# Runtime config (hot-swappable, survives for the process lifetime)
_runtime_config: dict = {
    "provider":             settings.default_provider,
    "model":                settings.default_model,
    "openai_api_key":       settings.openai_api_key,
    "openai_base_url":      settings.openai_base_url,
    "anthropic_api_key":    settings.anthropic_api_key,
    "openrouter_api_key":   settings.openrouter_api_key,
    "openrouter_base_url":  settings.openrouter_base_url,
    "github_copilot_token": settings.github_copilot_token,
    "vllm_base_url":        settings.vllm_base_url,
    "context_window":       settings.context_window,
}


# ---- Pydantic models ----

class ConfigPayload(BaseModel):
    provider: str
    model: str
    openai_api_key: str = ""
    openai_base_url: str = ""
    anthropic_api_key: str = ""
    openrouter_api_key: str = ""
    openrouter_base_url: str = ""
    github_copilot_token: str = ""
    vllm_base_url: str = ""
    context_window: int = 16000

class JobPayload(BaseModel):
    id: str
    cron: str
    task: str

class MemoryPayload(BaseModel):
    content: str


# ---- Gateway ----

class WebGateway(BaseGateway):
    name = "web"

    def __init__(self, agent: "Agent", host: str | None = None, port: int | None = None) -> None:
        super().__init__(agent)
        self._host = host or settings.web_host
        self._port = port or settings.web_port
        self._server: uvicorn.Server | None = None
        self.app = self._build_app()

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="Swarm", version="0.1.0")
        app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

        templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

        agent = self.agent

        # ==============================================================
        # PAGE ROUTES
        # ==============================================================

        @app.get("/", response_class=HTMLResponse)
        async def page_home(request: Request):
            return templates.TemplateResponse("index.html", {"request": request})

        @app.get("/chat", response_class=HTMLResponse)
        async def page_chat(request: Request):
            return templates.TemplateResponse("chat.html", {
                "request": request,
                "model": _runtime_config["model"],
            })

        @app.get("/config", response_class=HTMLResponse)
        async def page_config(request: Request):
            return templates.TemplateResponse("config.html", {
                "request": request,
                "current": _runtime_config,
            })

        @app.get("/memory", response_class=HTMLResponse)
        async def page_memory(request: Request):
            from datetime import date
            lt    = await agent.memory.read_long_term()
            daily = await agent.memory.read_daily()
            return templates.TemplateResponse("memory.html", {
                "request":   request,
                "long_term": lt,
                "daily":     daily,
                "today":     date.today().isoformat(),
            })

        @app.get("/scheduler", response_class=HTMLResponse)
        async def page_scheduler(request: Request):
            from swarm.scheduler import scheduler
            return templates.TemplateResponse("scheduler.html", {
                "request": request,
                "jobs":    scheduler.list_jobs(),
            })

        # ==============================================================
        # API — CONFIG
        # ==============================================================

        @app.get("/api/config")
        async def api_get_config():
            return {k: ("***" if ("key" in k or "token" in k) and v else v)
                    for k, v in _runtime_config.items()}

        @app.post("/api/config")
        async def api_save_config(payload: ConfigPayload):
            data = payload.model_dump()
            _runtime_config.update({k: v for k, v in data.items() if v})
            try:
                agent.provider        = _provider_from_payload(payload)
                agent.model           = payload.model or agent.model
                agent.context_window  = payload.context_window
                _runtime_config["context_window"] = payload.context_window
                _persist_env(data)   # write to .env so config survives restart
                logger.info("Provider switched to %s / %s, context_window=%s",
                            payload.provider, agent.model, payload.context_window)
                return {"ok": True}
            except Exception as exc:
                logger.error("api_save_config error: %s", exc)
                return {"ok": False, "error": str(exc)}

        @app.post("/api/config/test")
        async def api_test_config(payload: ConfigPayload):
            import traceback
            import time
            logs: list[str] = []
            t0 = time.monotonic()

            def log(msg: str):
                elapsed = f"{(time.monotonic()-t0)*1000:.0f}ms"
                entry = f"[{elapsed}] {msg}"
                logs.append(entry)
                logger.info("[config/test] %s", msg)

            try:
                from swarm.providers.registry import registry
                from swarm.providers.base import CompletionRequest, Message

                log(f"Provider: {payload.provider} | Model: {payload.model}")

                if payload.provider == "vllm":
                    url = payload.vllm_base_url or settings.vllm_base_url
                    kwargs = {"base_url": url, "model": payload.model}
                    log(f"vLLM base_url: {url}")
                    log(f"vLLM model: {payload.model!r}")
                    # Direct httpx probe — bypasses OpenAI SDK to isolate SDK vs server issues
                    import httpx
                    try:
                        base = url.rstrip('/')
                        base_no_v1 = base.rsplit('/v1', 1)[0]
                        # Ollama tags endpoint = fast reachability check (no CPU needed)
                        try:
                            async with httpx.AsyncClient() as c:
                                ping = await c.get(base_no_v1 + '/api/tags', timeout=3)
                                log(f"Ollama /api/tags HTTP {ping.status_code}")
                        except Exception as ping_err:
                            log(f"Reachability probe failed: {ping_err}")
                        # Raw chat/completions call — same payload the SDK would send
                        raw_payload = {
                            "model": payload.model,
                            "messages": [{"role": "user", "content": "Reply with only: OK"}],
                            "max_tokens": 5,
                            "stream": False,
                        }
                        log(f"Direct POST {base}/chat/completions  model={payload.model!r}")
                        async with httpx.AsyncClient() as c:
                            r = await c.post(
                                f"{base}/chat/completions",
                                json=raw_payload,
                                headers={"Content-Type": "application/json",
                                         "Authorization": "Bearer EMPTY"},
                                timeout=30,
                            )
                            log(f"Direct HTTP {r.status_code}: {r.text[:300]}")
                            if r.is_success:
                                log("Direct call OK — SDK path proceeding…")
                            else:
                                return {"ok": False,
                                        "error": f"Ollama HTTP {r.status_code}",
                                        "detail": r.text, "logs": logs}
                    except Exception as probe_err:
                        log(f"Direct probe error: {probe_err}")

                log("Instantiating provider...")
                prov = _provider_from_payload(payload)

                log("Sending test completion request...")
                req  = CompletionRequest(
                    messages=[Message(role="user", content="Reply with only: OK")],
                    model=payload.model,
                    max_tokens=5,
                )
                resp = await prov.complete(req)
                log(f"Success! model={resp.model} reply={resp.content[:40]!r}")
                return {"ok": True, "model": resp.model, "reply": resp.content[:80], "logs": logs}
            except Exception as exc:
                tb = traceback.format_exc()
                log(f"ERROR: {exc}")
                logger.error("[config/test] %s\n%s", exc, tb)
                return {"ok": False, "error": str(exc), "detail": tb, "logs": logs}

        # ==============================================================
        # API — CHAT (REST fallback)
        # ==============================================================

        @app.post("/api/chat")
        async def api_chat(request: Request):
            body       = await request.json()
            message    = body.get("message", "")
            session_id = body.get("session_id")
            if session_id:
                sess = agent.sessions.get_or_create(session_id, channel="web")
            else:
                sess = agent.new_session(channel="web")
            reply = await agent.chat(message, sess)
            return {"reply": reply, "session_id": sess.id}

        # ==============================================================
        # API — SESSIONS
        # ==============================================================

        @app.get("/api/sessions")
        async def api_sessions():
            return [{"id": s.id, "channel": s.channel, "messages": len(s.history)}
                    for s in agent.sessions.list()]

        @app.delete("/api/sessions/{session_id}")
        async def api_del_session(session_id: str):
            agent.sessions.delete(session_id)
            return {"ok": True}

        # ==============================================================
        # API — MEMORY
        # ==============================================================

        @app.get("/api/memory")
        async def api_memory():
            return {
                "long_term": await agent.memory.read_long_term(),
                "today":     await agent.memory.read_daily(),
            }

        @app.post("/api/memory/longterm")
        async def api_save_lt(payload: MemoryPayload):
            await agent.memory.write_long_term(payload.content)
            return {"ok": True}

        @app.get("/api/memory/grep")
        async def api_grep(q: str = ""):
            if not q:
                return {}
            return await agent.memory.grep(q)

        # ==============================================================
        # API — JOBS
        # ==============================================================

        @app.get("/api/jobs")
        async def api_jobs():
            from swarm.scheduler import scheduler
            return scheduler.list_jobs()

        @app.post("/api/jobs")
        async def api_add_job(payload: JobPayload):
            from swarm.scheduler import scheduler

            async def _task(task_text: str = payload.task):
                sess = agent.new_session(channel="scheduler")
                await agent.chat(task_text, sess)

            try:
                scheduler.add_cron(_task, payload.cron, job_id=payload.id,
                                   kwargs={"task_text": payload.task})
                return {"ok": True}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        @app.delete("/api/jobs/{job_id}")
        async def api_del_job(job_id: str):
            from swarm.scheduler import scheduler
            try:
                scheduler.remove_job(job_id)
                return {"ok": True}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        # ==============================================================
        # WebSocket — streaming chat
        # ==============================================================

        @app.websocket("/ws/{session_id}")
        async def ws_chat(websocket: WebSocket, session_id: str):
            await websocket.accept()
            # Use new_session so long-term memory is always injected into the system prompt
            existing = agent.sessions.get(session_id)
            if existing:
                sess = existing
            else:
                sess = agent.new_session(channel="web", session_id=session_id)
            try:
                while True:
                    data    = await websocket.receive_json()
                    message = data.get("message", "")
                    if not message:
                        continue
                    await websocket.send_json({"type": "start"})
                    try:
                        async for chunk in agent.stream_chat(message, sess):
                            await websocket.send_json({"type": "chunk", "text": chunk})
                        await websocket.send_json({"type": "end"})
                    except Exception as exc:
                        import traceback
                        tb = traceback.format_exc()
                        logger.error("Stream error: %s\n%s", exc, tb)
                        await websocket.send_json({
                            "type": "error",
                            "message": str(exc),
                            "detail": tb,
                        })
            except WebSocketDisconnect:
                pass

        return app

    async def start(self) -> None:
        config = uvicorn.Config(
            self.app,
            host=self._host,
            port=self._port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        logger.info("Web gateway -> http://%s:%s", self._host, self._port)
        await self._server.serve()

    async def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
