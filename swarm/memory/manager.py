from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional


class MemoryManager:
    """Minimal MemoryManager used by the app for long-term and daily notes.

    This implementation is intentionally small: it creates the `memory/` folder
    and reads/writes `MEMORY.md` and daily files named YYYY-MM-DD.md.
    """

    def __init__(self, memory_dir: Optional[Path] = None, long_term_file: str = "MEMORY.md") -> None:
        from swarm.config import settings

        self.memory_dir = Path(memory_dir or settings.memory_dir)
        self.long_term_file = long_term_file or settings.long_term_file
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    # --- Long-term memory ---
    async def append_long_term(self, line: str) -> None:
        path = self.memory_dir / self.long_term_file
        def _write():
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"{line}\n")

        await asyncio.to_thread(_write)

    async def read_long_term(self) -> str:
        path = self.memory_dir / self.long_term_file
        if not path.exists():
            return ""
        return await asyncio.to_thread(path.read_text, "utf-8")

    def get_long_term_sync(self) -> str:
        path = self.memory_dir / self.long_term_file
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    async def upsert_profile_entry(self, key: str, value: str) -> bool:
        """Write ``key: value`` to long-term memory only if needed.

        - If the key is absent → append the line and return True.
        - If the key is present with the same value → do nothing, return False.
        - If the key is present with a different value → replace the line and return True.

        This prevents the memory file from growing with duplicate profile entries
        every time the agent starts up.
        """
        path = self.memory_dir / self.long_term_file

        def _upsert() -> bool:
            existing_lines: list[str] = []
            if path.exists():
                # Use utf-8-sig to automatically strip the UTF-8 BOM that
                # Windows tools (e.g. PowerShell Set-Content) may prepend.
                existing_lines = path.read_text(encoding="utf-8-sig").splitlines()

            prefix = f"{key}:"
            matching = [i for i, ln in enumerate(existing_lines) if ln.startswith(prefix)]

            new_line = f"{key}: {value}"

            if not matching:
                # Key absent — append
                with open(path, "a", encoding="utf-8") as f:
                    f.write(f"{new_line}\n")
                return True

            # Key present — check value
            if existing_lines[matching[0]] == new_line and len(matching) == 1:
                # Already correct and no duplicates
                return False

            # Replace all occurrences with a single correct line
            updated = [ln for i, ln in enumerate(existing_lines) if i not in matching]
            updated.insert(matching[0], new_line)
            path.write_text("\n".join(updated) + "\n", encoding="utf-8")
            return True

        return await asyncio.to_thread(_upsert)

    # --- Daily memory ---
    async def append_daily(self, text: str) -> None:
        from datetime import date

        today = date.today().isoformat()
        path = self.memory_dir / f"{today}.md"
        def _write():
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"{text}\n\n")

        await asyncio.to_thread(_write)

    async def read_daily(self, day: Optional["date"] = None) -> str:
        from datetime import date

        day = day or date.today()
        path = self.memory_dir / f"{day.isoformat()}.md"
        if not path.exists():
            return ""
        return await asyncio.to_thread(path.read_text, "utf-8")
