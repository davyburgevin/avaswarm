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
    # System prompt — live read/write into _history[0]
    # ------------------------------------------------------------------
    @property
    def system_prompt(self) -> str:
        for m in self._history:
            if m.role == "system":
                return m.content or ""
        return ""

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        for m in self._history:
            if m.role == "system":
                m.content = value
                return
        # No system message yet — prepend one
        self._history.insert(0, Message(role="system", content=value))

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

    def add_assistant_with_tools(self, content: str, tool_calls: list[dict]) -> Message:
        """Add an assistant message that includes tool_calls (required before tool result messages)."""
        return self.add("assistant", content, tool_calls=tool_calls)

    def add_tool(self, content: str, tool_call_id: str, name: str) -> Message:
        return self.add("tool", content, tool_call_id=tool_call_id, name=name)

    @property
    def history(self) -> list[Message]:
        return list(self._history)

    def trim(self, max_messages: int = 40) -> None:
        """Keep system prompt + last N messages (legacy fallback)."""
        system = [m for m in self._history if m.role == "system"]
        rest = [m for m in self._history if m.role != "system"]
        if len(rest) > max_messages:
            rest = rest[-max_messages:]
        self._history = system + rest

    def trim_to_tokens(self, max_tokens: int, model: str = "gpt-4o") -> int:
        """
        Drop oldest non-system messages until history fits within max_tokens.
        Returns the estimated final token count.

        Uses tiktoken when available; falls back to 4 chars ≈ 1 token.
        """
        # ── Token counter ──────────────────────────────────────────────
        try:
            import tiktoken  # type: ignore
            try:
                enc = tiktoken.encoding_for_model(model)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")

            def _count(messages: list) -> int:
                # OpenAI accounting: 3 base + 4 overhead per message
                total = 3
                for m in messages:
                    total += 4
                    content = m.content or ""
                    if isinstance(content, str):
                        total += len(enc.encode(content))
                    elif isinstance(content, list):  # multimodal
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                total += len(enc.encode(part.get("text", "")))
                return total

        except Exception:
            # Graceful fallback: 4 chars ≈ 1 token
            def _count(messages: list) -> int:  # type: ignore[misc]
                return sum(len(m.content or "") // 4 + 4 for m in messages) + 3

        # ── Trim loop ──────────────────────────────────────────────────
        system = [m for m in self._history if m.role == "system"]
        rest   = [m for m in self._history if m.role != "system"]

        system_tokens = _count(system)
        budget = max_tokens - system_tokens - 200  # 200-token safety margin

        if budget <= 0:
            # System prompt alone exceeds budget — keep 1 message minimum
            self._history = system + (rest[-1:] if rest else [])
            return system_tokens

        # Drop oldest messages until we fit within the budget
        while len(rest) > 1 and _count(rest) > budget:
            rest.pop(0)

        self._history = system + rest
        return _count(system + rest)

    def token_stats(self, model: str = "gpt-4o", limit: int = 16000) -> dict:
        """
        Return token usage breakdown for the current history.
        Reuses the same tiktoken logic as trim_to_tokens.
        """
        try:
            import tiktoken  # type: ignore
            try:
                enc = tiktoken.encoding_for_model(model)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")

            def _count(msgs: list) -> int:
                total = 3
                for m in msgs:
                    total += 4
                    content = m.content or ""
                    if isinstance(content, str):
                        total += len(enc.encode(content))
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                total += len(enc.encode(part.get("text", "")))
                return total
        except Exception:
            def _count(msgs: list) -> int:  # type: ignore[misc]
                return sum(len(m.content or "") // 4 + 4 for m in msgs) + 3

        system_msgs = [m for m in self._history if m.role == "system"]
        user_msgs   = [m for m in self._history if m.role == "user"]
        asst_msgs   = [m for m in self._history if m.role == "assistant"]
        tool_msgs   = [m for m in self._history if m.role == "tool"]

        system_tok = _count(system_msgs)
        msg_tok    = _count(user_msgs + asst_msgs)
        tool_tok   = _count(tool_msgs)
        total      = _count(list(self._history))

        return {
            "used":    total,
            "limit":   limit,
            "pct":     min(100, round(total / max(limit, 1) * 100, 1)),
            "system":  system_tok,
            "messages":msg_tok,
            "tools":   tool_tok,
        }

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
