"""CLI gateway — interactive terminal chat using Rich + prompt_toolkit."""
from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

from swarm.gateways.base import BaseGateway

if TYPE_CHECKING:
    from swarm.core.agent import Agent

console = Console()

_COMMANDS = {
    "/exit": "Exit the session",
    "/clear": "Clear conversation history",
    "/memory": "Show today's memory",
    "/jobs": "List scheduled jobs",
    "/skills": "List loaded skills",
    "/help": "Show this help",
}


class CLIGateway(BaseGateway):
    name = "cli"
    _running: bool = False

    def __init__(self, agent: "Agent", stream: bool = True) -> None:
        super().__init__(agent)
        self._stream = stream
        self._session_obj = None

    async def start(self) -> None:
        self._running = True
        self._session_obj = self.agent.new_session(channel="cli")

        console.print(
            Panel.fit(
                "[bold cyan]Swarm[/bold cyan] — Orchestration Agent\n"
                "Type [bold]/help[/bold] for commands, [bold]/exit[/bold] to quit.",
                border_style="cyan",
            )
        )

        prompt = PromptSession(history=InMemoryHistory())

        while self._running:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: prompt.prompt("You > ")
                )
            except (EOFError, KeyboardInterrupt):
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            if await self._handle_command(user_input):
                continue

            console.print("[bold green]Swarm >[/bold green]", end=" ")
            if self._stream:
                full = []
                async for chunk in self.agent.stream_chat(user_input, self._session_obj):
                    console.print(chunk, end="", highlight=False)
                    full.append(chunk)
                console.print()
                # Render markdown if it looks like markdown
                content = "".join(full)
                if any(c in content for c in ["#", "```", "**", "__", "-"]):
                    console.print(Markdown(content))
            else:
                reply = await self.agent.chat(user_input, self._session_obj)
                console.print(Markdown(reply))

    async def stop(self) -> None:
        self._running = False

    async def _handle_command(self, cmd: str) -> bool:
        if cmd == "/exit":
            self._running = False
            console.print("[yellow]Goodbye.[/yellow]")
            return True
        if cmd == "/clear":
            if self._session_obj:
                self._session_obj.clear()
            console.print("[yellow]History cleared.[/yellow]")
            return True
        if cmd == "/memory":
            notes = await self.agent.memory.read_daily()
            console.print(Markdown(notes or "_No notes today._"))
            return True
        if cmd == "/jobs":
            from swarm.scheduler import scheduler
            jobs = scheduler.list_jobs()
            console.print(jobs if jobs else "[yellow]No scheduled jobs.[/yellow]")
            return True
        if cmd == "/skills":
            skills = self.agent.skills.list() if hasattr(self.agent, "skills") else []
            console.print(skills if skills else "[yellow]No skills loaded.[/yellow]")
            return True
        if cmd == "/help":
            for c, desc in _COMMANDS.items():
                console.print(f"  [bold cyan]{c}[/bold cyan]  {desc}")
            return True
        return False
