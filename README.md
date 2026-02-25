# project-swarm

> **Orchestration agent deployable on your local PC.**  
> One agent to spawn them all — multi-platform, multi-LLM, tool-equipped, memory-aware.

---

## Architecture

```
project-swarm/
├── swarm/
│   ├── main.py                  # Entry point & CLI flags
│   ├── config/settings.py       # Centralised configuration (pydantic-settings + .env)
│   │
│   ├── core/
│   │   ├── agent.py             # Orchestration agent (tool-calling loop, sub-agents)
│   │   └── session.py           # Persistent conversation sessions
│   │
│   ├── providers/               # 🤖 Multi-LLM support
│   │   ├── registry.py          # ProviderRegistry (add new providers in one line)
│   │   ├── openai_provider.py
│   │   ├── anthropic_provider.py
│   │   ├── openrouter_provider.py
│   │   ├── github_copilot_provider.py
│   │   └── vllm_provider.py
│   │
│   ├── gateways/                # 🌐 Multi-platform
│   │   ├── cli_gateway.py       # Interactive terminal (Rich + prompt_toolkit)
│   │   ├── web_gateway.py       # FastAPI + WebSocket streaming
│   │   └── email_gateway.py     # IMAP polling + SMTP replies
│   │
│   ├── tools/                   # 🧰 Tool system
│   │   ├── registry.py          # ToolRegistry
│   │   ├── file_tools.py        # read_file, write_file, list_directory
│   │   ├── shell_tools.py       # run_shell
│   │   ├── web_search.py        # web_search (Brave)
│   │   └── meta_tools.py        # list_tools, spawn_sub_agent
│   │
│   ├── memory/
│   │   └── manager.py           # 🧠 MEMORY.md (long-term) + YYYY-MM-DD.md (daily)
│   │
│   ├── scheduler/
│   │   └── cron.py              # ⏰ APScheduler wrapper (cron + interval jobs)
│   │
│   └── skills/                  # 🧩 Skills system
│       ├── registry.py          # SkillRegistry
│       ├── loader.py            # ClawHub marketplace client
│       └── builtin/
│           └── standup_skill.py # Example: daily stand-up summary
│
├── memory/                      # Runtime memory files (git-ignored)
├── .env.example                 # Environment template
└── pyproject.toml
```

---

## Quick Start

### 1. Install

```bash
# Clone / open the folder, then:
pip install -e .
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set DEFAULT_PROVIDER and the matching API key
```

### 3. Run

```bash
# Interactive terminal
python -m swarm.main --cli

# Web UI (opens at http://localhost:8080)
python -m swarm.main --web

# All gateways simultaneously
python -m swarm.main --all

# Or if installed:
swarm --all
```

---

## Modules

### Interaction & Conversation

- Persistent `Session` objects store full message history.
- `SessionStore` keeps all active sessions in memory; sessions are keyed by a UUID or a stable key (e.g. `email_user@host.com`).
- Streaming supported via `Agent.stream_chat()` (async generator).

### Multi-platform Gateways

| Gateway | Description |
|---------|-------------|
| `CLIGateway` | Rich terminal with streaming, markdown rendering, slash commands |
| `WebGateway` | FastAPI + WebSocket; minimal chat UI at `/`; REST API at `/api/*` |
| `EmailGateway` | Polls IMAP inbox, maintains per-sender sessions, replies via SMTP |

All gateways share **the same agent** and run concurrently via `asyncio.gather`.

### Multi-LLM Providers

| Provider | Key setting |
|----------|-------------|
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `openrouter` | `OPENROUTER_API_KEY` |
| `github_copilot` | `GITHUB_COPILOT_TOKEN` |
| `vllm` | `VLLM_BASE_URL` |

**Add a new provider:**

```python
from swarm.providers.base import BaseProvider, CompletionRequest, CompletionResponse
from swarm.providers.registry import registry

class MyProvider(BaseProvider):
    name = "myprovider"
    async def complete(self, request): ...
    async def stream(self, request): ...

registry.register(MyProvider)
```

### Tool System

Tools auto-register when the `ToolRegistry.default()` is called.  
**Add a custom tool:**

```python
from swarm.tools.base import BaseTool, ToolSchema
from swarm.tools import ToolRegistry

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something useful."
    def schema(self): return ToolSchema(name=self.name, description=self.description)
    async def execute(self, **kwargs): return "result"

agent.tools.register(MyTool())
```

### Memory

```
memory/
├── MEMORY.md          ← long-term facts (edit manually or via the agent)
└── 2026-02-25.md      ← auto-appended daily log
```

Use `await agent.memory.grep("keyword")` to search across all memory files.

### Scheduler

```python
from swarm.scheduler import scheduler

async def remind():
    print("Time for a break!")

scheduler.add_cron(remind, "0 15 * * 1-5")   # 15:00 every weekday
scheduler.add_interval(remind, minutes=30)     # every 30 min
```

### Skills

```python
# Load a built-in skill
from swarm.skills.builtin.standup_skill import Skill
await agent.skills.load(Skill(), agent)

# Install from ClawHub
from swarm.skills.loader import ClawHub
hub = ClawHub(agent.skills)
await hub.install("code-review", agent)
```

**Write your own skill:**

```python
from swarm.skills.base import BaseSkill, SkillMeta
from swarm.tools.base import BaseTool, ToolSchema

class MyTool(BaseTool):
    name = "my_skill_tool"
    description = "..."
    def schema(self): ...
    async def execute(self, **kw): return "done"

class Skill(BaseSkill):
    meta = SkillMeta(name="my-skill", description="My custom skill")
    async def activate(self, agent):
        agent.tools.register(MyTool())
```

---

## CLI Commands

| Command | Action |
|---------|--------|
| `/help` | Show all commands |
| `/clear` | Clear conversation history |
| `/memory` | Show today's notes |
| `/jobs` | List scheduled jobs |
| `/skills` | List loaded skills |
| `/exit` | Quit |

---

## REST API (Web Gateway)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Chat UI |
| `POST` | `/api/chat` | Send message, get reply |
| `GET` | `/api/sessions` | List active sessions |
| `DELETE` | `/api/sessions/{id}` | Delete a session |
| `GET` | `/api/memory` | Read long-term + daily memory |
| `GET` | `/api/jobs` | List scheduled jobs |
| `WS` | `/ws/{session_id}` | Streaming WebSocket chat |

---

## Requirements

- Python 3.11+
- Internet access for cloud LLM providers
- A `.env` file with at least one provider configured

---

## License

MIT
