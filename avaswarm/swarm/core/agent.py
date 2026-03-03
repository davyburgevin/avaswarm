"""Orchestration Agent — the central brain of project-swarm."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator, Any

from swarm.config import settings
from swarm.core.session import Session, SessionStore
from swarm.providers.base import BaseProvider, CompletionRequest
from swarm.providers.registry import registry as provider_registry
from swarm.tools.registry import ToolRegistry
from swarm.memory.manager import MemoryManager
from swarm.utils.windows_user import get_windows_display_name, split_name

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM = (
    "You are AvaSwarm, an AI orchestration agent running locally for Avanade. "
    "You can spawn sub-agents, use tools, and maintain persistent memory across sessions. "
    "Be concise and helpful.\n\n"
    "MEMORY RULES — follow strictly:\n"
    "- When a user tells you their name, job, location, preferences, or any personal fact, "
    "IMMEDIATELY call save_memory with a concise statement (e.g. 'The user\'s name is Alice.'). "
    "- When a user asks if you remember something, first call read_memory to check. "
    "- Never say you cannot remember things — use save_memory and read_memory instead.\n\n"
    "AGENT MANAGEMENT:\n"
    "- Use list_agents to see existing sub-agents before creating one. "
    "- Use create_agent(name, instructions, description) to create a new sub-agent. "
    "  The 'name' must be a slug (lowercase, hyphens ok, no spaces). "
    "  The 'instructions' should be a detailed system prompt tailored to the agent's role. "
    "- Use update_agent_instructions(name, instructions, mode) to modify an existing agent. "
    "  mode='replace' overwrites; mode='append' adds to existing instructions."
)


class Agent:
    """
    The main orchestration agent.

    - Manages conversation sessions.
    - Routes completions through the configured LLM provider.
    - Calls tools when the model emits tool_calls.
    - Reads/writes long-term and short-term memory.
    """

    def __init__(
        self,
        provider_name: str | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        tool_registry: ToolRegistry | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        # Provider
        pname = provider_name or settings.default_provider
        provider_cls = provider_registry.get(pname)
        self.provider: BaseProvider = provider_cls()
        self.model = model or settings.default_model

        # Context window (tokens) — controls how many messages are kept per session
        self.context_window: int = settings.context_window

        # Tools
        self.tools: ToolRegistry = tool_registry or ToolRegistry.default()

        # Memory
        self.memory: MemoryManager = memory_manager or MemoryManager()

        # Inject memory manager into memory tools so they can read/write MEMORY.md
        for tool_name in ("save_memory", "read_memory"):
            try:
                tool = self.tools.get(tool_name)
                tool._memory = self.memory
            except KeyError:
                pass

        # Session store
        self.sessions: SessionStore = SessionStore()
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM

        # Schedule user identity save (non-blocking)
        try:
            asyncio.get_event_loop().call_soon_threadsafe(lambda: asyncio.create_task(self._save_user_identity()))
        except Exception:
            # If no running loop, ignore — startup path that runs sync will not save immediately
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def new_session(self, channel: str = "cli", session_id: str | None = None) -> Session:
        """Open a new conversation session (reloads long-term memory each time)."""
        context = self._build_system_context()
        return self.sessions.create(
            channel=channel,
            system_prompt=context,
            session_id=session_id,
        )

    def get_session(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    async def chat(
        self,
        message: str,
        session: Session,
        stream: bool = False,
    ) -> str:
        """
        Send a user message and return the assistant reply.
        Runs the agentic loop: call LLM → execute tools if needed → call LLM again.
        """
        session.add_user(message)

        # Trim context using real token count (tiktoken, falls back to char estimate)
        session.trim_to_tokens(self.context_window, self.model)

        # agentic loop (tool-use)
        for _ in range(10):  # max 10 tool rounds
            request = CompletionRequest(
                messages=session.history,
                model=self.model,
                tools=self.tools.schemas() or None,
            )
            response = await self.provider.complete(request)

            if response.tool_calls:
                # Store raw tool_calls on the assistant message so the history is valid OpenAI format
                raw_tool_calls = [
                    {"id": tc["id"], "type": "function", "function": tc["function"]}
                    for tc in response.tool_calls
                ]
                session.add_assistant_with_tools(response.content or "", raw_tool_calls)
                for tc in response.tool_calls:
                    result = await self._execute_tool(tc)
                    fn_name = tc["function"]["name"]
                    session.add_tool(str(result), tool_call_id=tc["id"], name=fn_name)
            else:
                session.add_assistant(response.content)
                # Persist short-term memory entry
                await self.memory.append_daily(f"User: {message}\nAssistant: {response.content}")
                return response.content

        return "Maximum tool-call depth reached."

    async def chat_and_stream(
        self,
        message: str,
        session: Session,
    ) -> AsyncIterator[dict]:
        """
        Fully-streaming agentic loop:
        - Yields {type:thinking_text, text} for text streamed while deciding/using tools
        - Yields {type:tool_call, name, input} before each tool execution
        - Yields {type:tool_result, name, result, ok} after each tool execution
        - Yields {type:final, had_tools} once the answer is complete
        """
        session.add_user(message)
        session.trim_to_tokens(self.context_window, self.model)

        for iteration in range(10):
            request = CompletionRequest(
                messages=session.history,
                model=self.model,
                tools=self.tools.schemas() or None,
                stream=True,
            )

            accumulated_text: list[str] = []
            tool_calls: list[dict] = []

            async for event in self.provider.stream_with_tools(request):
                if event["type"] == "text":
                    accumulated_text.append(event["text"])
                    # Emit as thinking_text — we don't know yet if tool calls follow
                    yield {"type": "thinking_text", "text": event["text"]}
                elif event["type"] == "tool_calls":
                    tool_calls = event["calls"]

            text_so_far = "".join(accumulated_text)

            if tool_calls:
                raw_tool_calls = [
                    {"id": tc["id"], "type": "function", "function": tc["function"]}
                    for tc in tool_calls
                ]
                session.add_assistant_with_tools(text_so_far, raw_tool_calls)
                for tc in tool_calls:
                    fn_name = tc["function"]["name"]
                    fn_args = tc["function"].get("arguments", "{}")
                    yield {"type": "tool_call", "name": fn_name, "input": fn_args}
                    try:
                        result = await self._execute_tool(tc)
                        result_str = str(result)
                        yield {"type": "tool_result", "name": fn_name,
                               "result": result_str[:200], "ok": True}
                    except Exception as exc:
                        result_str = f"Error: {exc}"
                        yield {"type": "tool_result", "name": fn_name,
                               "result": result_str[:200], "ok": False}
                    session.add_tool(result_str, tool_call_id=tc["id"], name=fn_name)
                continue

            # No tool calls — the text already streamed IS the final answer.
            yield {"type": "final", "had_tools": iteration > 0}
            session.add_assistant(text_so_far)
            await self.memory.append_daily(f"User: {message}\nAssistant: {text_so_far}")
            asyncio.create_task(self._extract_and_save_memory(message, text_so_far))
            return

        yield {"type": "thinking_text", "text": "Maximum tool-call depth reached."}
        yield {"type": "final", "had_tools": True}

    async def stream_chat(
        self,
        message: str,
        session: Session,
    ) -> AsyncIterator[str]:
        """Streaming version of chat (no tool-call loop for simplicity)."""
        session.add_user(message)
        # Trim context using real token count (tiktoken, falls back to char estimate)
        session.trim_to_tokens(self.context_window, self.model)
        request = CompletionRequest(
            messages=session.history,
            model=self.model,
            stream=True,
            tools=self.tools.schemas() or None,
        )
        full_response = []
        async for chunk in self.provider.stream(request):
            full_response.append(chunk)
            yield chunk

        content = "".join(full_response)
        session.add_assistant(content)
        await self.memory.append_daily(f"User: {message}\nAssistant: {content}")

        # Background: extract & save any new facts the user shared
        asyncio.create_task(
            self._extract_and_save_memory(message, content)
        )

    async def spawn_sub_agent(
        self,
        task: str,
        provider_name: str | None = None,
        model: str | None = None,
    ) -> str:
        """
        Create a short-lived sub-agent to handle an isolated task.
        Returns the sub-agent's final response.
        """
        sub = Agent(
            provider_name=provider_name or settings.default_provider,
            model=model or self.model,
            system_prompt=f"You are a sub-agent. Complete this task and reply with the result:\n{task}",
            tool_registry=self.tools,
            memory_manager=self.memory,
        )
        session = sub.new_session(channel="sub_agent")
        result = await sub.chat(task, session)
        logger.info("Sub-agent completed task. Preview: %s", result[:120])
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _extract_and_save_memory(self, user_msg: str, assistant_reply: str) -> None:
        """
        After each exchange, ask the LLM to identify facts worth saving to long-term
        memory (name, preferences, etc.) and persist them to MEMORY.md.
        Runs as a background task — does not block the response.
        Skips the LLM call entirely when no personal-fact keywords are detected.
        """
        # Quick pre-filter: only call the LLM if the message plausibly contains personal facts.
        # This avoids a second API call for every casual exchange (e.g. "Bonjour", "Merci").
        _FACT_MARKERS = (
            "name", "nom", "appelle", "called", "je suis", "i am", "i'm",
            "my ", "mon ", "ma ", "mes ", "notre ", "nos ",
            "job", "rôle", "travaille", "work", "company", "société", "enterprise",
            "live", "habite", "located", "from ", "viens de",
            "prefer", "préfère", "like ", "aime ", "hate", "déteste",
            "birthday", "anniversaire", "age", "âge",
        )
        combined = (user_msg + " " + assistant_reply).lower()
        if not any(marker in combined for marker in _FACT_MARKERS):
            return  # nothing worth extracting — skip the API call

        try:
            existing = await self.memory.read_long_term()
            existing_note = f"\n\nAlready stored:\n{existing}" if existing else ""
            prompt = (
                f"Conversation excerpt:\n"
                f"User: {user_msg}\n"
                f"Assistant: {assistant_reply}\n\n"
                f"Does the user's message reveal any NEW personal facts that should be "
                f"remembered long-term (name, role, company, location, preferences, etc.)?{existing_note}\n\n"
                f"If yes, reply with a bullet list of concise facts (one per line, starting with '- '). "
                f"If nothing new, reply exactly: NONE"
            )
            from swarm.providers.base import Message
            req = CompletionRequest(
                messages=[Message(role="user", content=prompt)],
                model=self.model,
                max_tokens=200,
            )
            resp = await self.provider.complete(req)
            text = resp.content.strip()
            if text and text.upper() != "NONE":
                facts = [line.strip() for line in text.splitlines()
                         if line.strip().startswith("-")]
                for fact in facts:
                    await self.memory.append_long_term(fact)
                    logger.info("[memory] Saved fact: %s", fact)
        except Exception as exc:
            logger.warning("[memory] Extraction failed: %s", exc)

    def _build_system_context(self) -> str:
        """Inject long-term memory into the system prompt."""
        lt_memory = self.memory.get_long_term_sync()
        if lt_memory:
            return f"{self.system_prompt}\n\n## Long-term memory\n{lt_memory}"
        return self.system_prompt

    async def _execute_tool(self, tool_call: dict) -> Any:
        fn_name = tool_call["function"]["name"]
        try:
            args = json.loads(tool_call["function"].get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}
        logger.debug("Executing tool %s with args %s", fn_name, args)
        return await self.tools.call(fn_name, **args)

    async def _save_user_identity(self) -> None:
        """Detect Windows user display name and persist concise facts to long-term memory.

        Uses upsert_profile_entry so that existing correct values are never
        re-written, avoiding the file growing with duplicate entries on every
        agent startup.
        """
        try:
            display = get_windows_display_name()
            if not display:
                return
            first, last = split_name(display)
            changed = await self.memory.upsert_profile_entry("UserDisplayName", display)
            if first:
                changed |= await self.memory.upsert_profile_entry("UserFirstName", first)
            if last:
                changed |= await self.memory.upsert_profile_entry("UserLastName", last)
            if changed:
                logger.info("User identity updated in long-term memory: %s", display)
            else:
                logger.debug("User identity already up-to-date in long-term memory: %s", display)
        except Exception as exc:
            logger.warning("_save_user_identity failed: %s", exc)
