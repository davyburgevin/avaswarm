"""File system tools: read, write, list directory."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles

from swarm.tools.base import BaseTool, ToolSchema


def _check_sandboxed_path(path: str, root_dir: Path | None) -> Path:
    """Resolve path and raise if it escapes root_dir sandbox (when set)."""
    resolved = Path(path).resolve()
    if root_dir is not None:
        root = root_dir.resolve()
        try:
            resolved.relative_to(root)  # raises ValueError if outside
        except ValueError:
            raise PermissionError(
                f"Access denied: '{path}' is outside the allowed directory '{root}'."
            )
    return resolved


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read the contents of a file on the local filesystem."
    _root_dir: Path | None = None  # set to sandbox access to a specific directory

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative path to the file."},
                    "start_line": {"type": "integer", "description": "First line to read (1-based, optional)."},
                    "end_line": {"type": "integer", "description": "Last line to read (1-based, optional)."},
                },
                "required": ["path"],
            },
        )

    async def execute(self, path: str, start_line: int | None = None, end_line: int | None = None, **_: Any) -> str:
        safe = _check_sandboxed_path(path, self._root_dir)
        async with aiofiles.open(safe, "r", encoding="utf-8", errors="replace") as f:
            if start_line or end_line:
                lines = await f.readlines()
                sl = (start_line or 1) - 1
                el = end_line or len(lines)
                return "".join(lines[sl:el])
            return await f.read()


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Write or append content to a file on the local filesystem."
    _root_dir: Path | None = None

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                    "content": {"type": "string", "description": "Content to write."},
                    "mode": {"type": "string", "enum": ["write", "append"], "description": "Write mode (default: write)."},
                },
                "required": ["path", "content"],
            },
        )

    async def execute(self, path: str, content: str, mode: str = "write", **_: Any) -> dict:
        safe = _check_sandboxed_path(path, self._root_dir)
        safe.parent.mkdir(parents=True, exist_ok=True)
        file_mode = "w" if mode == "write" else "a"
        async with aiofiles.open(safe, file_mode, encoding="utf-8") as f:
            await f.write(content)
        return {"status": "ok", "path": str(safe), "bytes_written": len(content.encode())}


class ListDirTool(BaseTool):
    name = "list_directory"
    description = "List files and directories at a given path."
    _root_dir: Path | None = None

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list."},
                    "recursive": {"type": "boolean", "description": "List recursively (default: false)."},
                },
                "required": ["path"],
            },
        )

    async def execute(self, path: str, recursive: bool = False, **_: Any) -> list[str]:
        safe = _check_sandboxed_path(path, self._root_dir)
        if recursive:
            return [str(p) for p in safe.rglob("*")]
        return [str(p) for p in safe.iterdir()]
