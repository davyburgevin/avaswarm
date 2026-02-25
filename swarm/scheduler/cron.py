"""Scheduler — APScheduler-based cron/interval jobs."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from swarm.config import settings

logger = logging.getLogger(__name__)


class SwarmScheduler:
    """
    Thin wrapper around APScheduler's AsyncIOScheduler.
    Allows scheduling agent tasks via cron expressions or intervals.
    """

    def __init__(self) -> None:
        self._scheduler = AsyncIOScheduler(
            timezone=settings.scheduler_timezone
        )
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._scheduler.start()
            self._started = True
            logger.info("Scheduler started (tz=%s)", settings.scheduler_timezone)

    def stop(self) -> None:
        if self._started:
            self._scheduler.shutdown(wait=False)
            self._started = False

    # ------------------------------------------------------------------
    # Scheduling helpers
    # ------------------------------------------------------------------

    def add_cron(
        self,
        func: Callable[..., Coroutine[Any, Any, Any]],
        cron_expression: str,
        job_id: str | None = None,
        kwargs: dict | None = None,
    ) -> str:
        """
        Schedule a coroutine using a cron expression.
        Expression format: 'minute hour day month day_of_week'
        e.g. '0 9 * * 1-5' = 09:00 every weekday.
        """
        parts = cron_expression.split()
        if len(parts) != 5:
            raise ValueError("Cron expression must have 5 fields: m h d M dow")
        minute, hour, day, month, day_of_week = parts
        trigger = CronTrigger(
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=day_of_week,
            timezone=settings.scheduler_timezone,
        )
        job = self._scheduler.add_job(
            func,
            trigger=trigger,
            id=job_id,
            kwargs=kwargs or {},
            replace_existing=True,
        )
        logger.info("Cron job added: %s  [%s]", job.id, cron_expression)
        return job.id

    def add_interval(
        self,
        func: Callable[..., Coroutine[Any, Any, Any]],
        seconds: int = 0,
        minutes: int = 0,
        hours: int = 0,
        job_id: str | None = None,
        kwargs: dict | None = None,
    ) -> str:
        trigger = IntervalTrigger(
            seconds=seconds,
            minutes=minutes,
            hours=hours,
            timezone=settings.scheduler_timezone,
        )
        job = self._scheduler.add_job(
            func,
            trigger=trigger,
            id=job_id,
            kwargs=kwargs or {},
            replace_existing=True,
        )
        logger.info("Interval job added: %s", job.id)
        return job.id

    def remove_job(self, job_id: str) -> None:
        self._scheduler.remove_job(job_id)
        logger.info("Job removed: %s", job_id)

    def list_jobs(self) -> list[dict]:
        return [
            {"id": j.id, "next_run": str(j.next_run_time), "trigger": str(j.trigger)}
            for j in self._scheduler.get_jobs()
        ]


# Global singleton
scheduler = SwarmScheduler()
