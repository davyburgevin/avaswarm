"""Session — persistent conversation state."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from swarm.providers.base import Message


class Session:
    """Holds the full message history + metadata for one conversation."""

    def __init__(
        self,
        session_id: str | None = None,
        channel: str = "cli",
        system_prompt: str | None = None,
    ) -> None:
        self.id: str = session_id or str(uuid.uuid4())
        self.channel: str = channel
        self.created_at: datetime = datetime.utcnow()
        self.updated_at: datetime = datetime.utcnow()
        self.metadata: dict[str, Any] = {}
        self._history: list[Message] = []

        if system_prompt:
            self._history.append(Message(role="system", content=system_prompt))

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------
    def add(self, role: str, content: str, **kwargs) -> Message:
        msg = Message(role=role, content=content, **kwargs)
        self._history.append(msg)
        self.updated_at = datetime.utcnow()
        return msg

    def add_user(self, content: str) -> Message:
        return self.add("user", content)

    def add_assistant(self, content: str) -> Message:
        return self.add("assistant", content)

    def add_tool(self, content: str, tool_call_id: str, name: str) -> Message:
        return self.add("tool", content, tool_call_id=tool_call_id, name=name)

    @property
    def history(self) -> list[Message]:
        return list(self._history)

    def trim(self, max_messages: int = 40) -> None:
        """Keep system prompt + last N messages."""
        system = [m for m in self._history if m.role == "system"]
        rest = [m for m in self._history if m.role != "system"]
        if len(rest) > max_messages:
            rest = rest[-max_messages:]
        self._history = system + rest

    def clear(self, keep_system: bool = True) -> None:
        if keep_system:
            self._history = [m for m in self._history if m.role == "system"]
        else:
            self._history = []

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "channel": self.channel,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "messages": [
                {"role": m.role, "content": m.content} for m in self._history
            ],
        }


class SessionStore:
    """In-memory store of all active sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(
        self,
        channel: str = "cli",
        system_prompt: str | None = None,
        session_id: str | None = None,
    ) -> Session:
        s = Session(session_id=session_id, channel=channel, system_prompt=system_prompt)
        self._sessions[s.id] = s
        return s

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str, **kwargs) -> Session:
        if session_id not in self._sessions:
            return self.create(session_id=session_id, **kwargs)
        return self._sessions[session_id]

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def list(self) -> list[Session]:
        return list(self._sessions.values())
