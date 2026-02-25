"""Email gateway — polls IMAP for new messages, replies via SMTP."""
from __future__ import annotations

import asyncio
import email
import imaplib
import logging
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import TYPE_CHECKING

from swarm.config import settings
from swarm.gateways.base import BaseGateway

if TYPE_CHECKING:
    from swarm.core.agent import Agent

logger = logging.getLogger(__name__)


class EmailGateway(BaseGateway):
    """
    Polls an IMAP mailbox for unread messages addressed to the agent.
    Sends replies via SMTP.
    Maintains per-sender sessions.
    """
    name = "email"

    def __init__(self, agent: "Agent") -> None:
        super().__init__(agent)
        self._running = False
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not settings.email_address:
            logger.warning("Email gateway: EMAIL_ADDRESS not configured, skipping.")
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("Email gateway started (polling every %ds)", settings.email_check_interval)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------
    # Polling loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await asyncio.get_event_loop().run_in_executor(None, self._check_inbox)
            except Exception as exc:
                logger.error("Email gateway error: %s", exc)
            await asyncio.sleep(settings.email_check_interval)

    def _check_inbox(self) -> None:
        ctx = ssl.create_default_context()
        with imaplib.IMAP4_SSL(settings.email_host, settings.email_imap_port, ssl_context=ctx) as imap:
            imap.login(settings.email_address, settings.email_password)
            imap.select("INBOX")
            _, msg_ids = imap.search(None, "UNSEEN")
            for mid in msg_ids[0].split():
                _, data = imap.fetch(mid, "(RFC822)")
                raw = data[0][1]
                msg = email.message_from_bytes(raw)
                from_addr = email.utils.parseaddr(msg["From"])[1]
                subject = msg.get("Subject", "(no subject)")
                body = self._extract_body(msg)
                logger.info("Email from %s: %s", from_addr, subject)
                # Run agent in a new event loop (we're in a thread)
                reply = asyncio.run(self._get_reply(from_addr, body))
                self._send_reply(from_addr, subject, reply)
                imap.store(mid, "+FLAGS", "\\Seen")

    def _extract_body(self, msg: email.message.Message) -> str:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    return part.get_payload(decode=True).decode("utf-8", errors="replace")
        else:
            return msg.get_payload(decode=True).decode("utf-8", errors="replace")
        return ""

    async def _get_reply(self, sender: str, message: str) -> str:
        session = self.agent.sessions.get_or_create(
            session_id=f"email_{sender}",
            channel="email",
        )
        return await self.agent.chat(message, session)

    def _send_reply(self, to_addr: str, subject: str, body: str) -> None:
        mime = MIMEMultipart("alternative")
        mime["Subject"] = f"Re: {subject}"
        mime["From"] = settings.email_address
        mime["To"] = to_addr
        mime.attach(MIMEText(body, "plain"))

        ctx = ssl.create_default_context()
        with smtplib.SMTP(settings.email_smtp_host, settings.email_smtp_port) as smtp:
            smtp.starttls(context=ctx)
            smtp.login(settings.email_address, settings.email_password)
            smtp.sendmail(settings.email_address, to_addr, mime.as_string())
        logger.info("Reply sent to %s", to_addr)
