"""
project-swarm — main entry point.

Usage:
    python -m swarm.main            # CLI mode
    python -m swarm.main --web      # Web + CLI (parallel)
    python -m swarm.main --all      # CLI + Web + Email (parallel)
    swarm                           # if installed via pip

Flags:
    --cli         Start the CLI gateway (default when no flag given)
    --web         Start the Web gateway (http + websocket)
    --email       Start the Email gateway (IMAP polling)
    --all         Start all gateways simultaneously
    --provider    LLM provider name (overrides .env)
    --model       Model name (overrides .env)
    --port        Web server port (default: 8080)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from swarm.config import settings
from swarm.core.agent import Agent
from swarm.scheduler import scheduler
from swarm.skills.registry import SkillRegistry


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("swarm.main")


async def async_main(args: argparse.Namespace) -> None:
    # ---- Build agent ----
    agent = Agent(
        provider_name=args.provider or settings.default_provider,
        model=args.model or settings.default_model,
    )

    # ---- Skills ----
    skill_registry = SkillRegistry()
    agent.skills = skill_registry  # type: ignore[attr-defined]
    # Auto-load built-in skills
    from swarm.skills.builtin import standup_skill
    await skill_registry.load(standup_skill.Skill(), agent)

    # ---- Bind SpawnSubAgentTool if present ----
    if "spawn_sub_agent" in agent.tools.list():
        agent.tools.get("spawn_sub_agent").bind(agent)  # type: ignore[attr-defined]

    # ---- Scheduler ----
    scheduler.start()

    # ---- Gateways ----
    from swarm.gateways import CLIGateway, WebGateway, EmailGateway

    gateways = []

    use_cli   = args.cli or args.all or not any([args.web, args.email, args.all])
    use_web   = args.web or args.all
    use_email = args.email or args.all

    if use_web:
        wg = WebGateway(agent, port=args.port)
        gateways.append(wg)
    if use_email:
        gateways.append(EmailGateway(agent))
    if use_cli:
        gateways.append(CLIGateway(agent, stream=True))

    if not gateways:
        print("No gateway selected. Use --cli, --web, --email, or --all.")
        return

    if use_web:
        from rich.console import Console
        Console().print(
            f"[bold cyan]Web gateway:[/bold cyan] http://localhost:{args.port}"
        )

    tasks = [asyncio.create_task(g.start()) for g in gateways]
    try:
        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        scheduler.stop()
        for g in gateways:
            await g.stop()


def cli_entry() -> None:
    parser = argparse.ArgumentParser(
        prog="swarm",
        description="project-swarm — local AI orchestration agent",
    )
    parser.add_argument("--cli",   action="store_true", help="Start CLI gateway")
    parser.add_argument("--web",   action="store_true", help="Start web gateway")
    parser.add_argument("--email", action="store_true", help="Start email gateway")
    parser.add_argument("--all",   action="store_true", help="Start all gateways")
    parser.add_argument("--provider", type=str, default=None, help="LLM provider")
    parser.add_argument("--model",    type=str, default=None, help="Model name")
    parser.add_argument("--port",     type=int, default=settings.web_port, help="Web port")

    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    cli_entry()
