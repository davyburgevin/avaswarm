"""Shell execution tool — runs OS commands asynchronously."""
from __future__ import annotations

import asyncio
from typing import Any

from swarm.tools.base import BaseTool, ToolSchema

# Commands that are unconditionally blocked
_BLOCKED = {"rm -rf /", "format c:", "del /f /s /q"}


class ShellTool(BaseTool):
    name = "run_shell"
    description = (
        "Execute a shell command on the local machine and return stdout/stderr. "
        "Use with care — commands have real effects."
    )

    def __init__(self, timeout: int = 60, working_dir: str | None = None) -> None:
        self._timeout = timeout
        self._cwd = working_dir

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute."},
                    "working_dir": {"type": "string", "description": "Working directory (optional)."},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 60)."},
                },
                "required": ["command"],
            },
        )

    async def execute(self, command: str, working_dir: str | None = None, timeout: int | None = None, **_: Any) -> dict:
        cmd_lower = command.strip().lower()
        for blocked in _BLOCKED:
            if blocked in cmd_lower:
                return {"error": f"Command blocked for safety: {command}"}

        cwd = working_dir or self._cwd
        tmt = timeout or self._timeout

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=tmt)
            return {
                "returncode": proc.returncode,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
            }
        except asyncio.TimeoutError:
            return {"error": f"Command timed out after {tmt}s", "command": command}
        except Exception as exc:
            return {"error": str(exc)}
