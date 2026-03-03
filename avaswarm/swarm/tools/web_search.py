"""Web search tool — Brave Search API."""
from __future__ import annotations

from typing import Any

import httpx

from swarm.config import settings
from swarm.tools.base import BaseTool, ToolSchema


class BraveSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web using Brave Search and return a list of results."

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "count": {"type": "integer", "description": "Number of results to return (default 5, max 20)."},
                },
                "required": ["query"],
            },
        )

    async def execute(self, query: str, count: int = 5, **_: Any) -> list[dict]:
        if not settings.brave_search_api_key:
            return [{"error": "BRAVE_SEARCH_API_KEY not configured."}]

        count = min(count, 20)
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": count},
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": settings.brave_search_api_key,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("web", {}).get("results", []):
            results.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "description": item.get("description"),
            })
        return results
