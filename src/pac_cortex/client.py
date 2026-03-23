"""BitGN API client stub. Actual API contract TBD until SDK drops."""

from typing import Any

import httpx

from pac_cortex.config import settings


class BitgnClient:
    """Async client for BitGN challenge API."""

    def __init__(self) -> None:
        self._http = httpx.AsyncClient(
            base_url=settings.bitgn_api_url,
            headers={"Authorization": f"Bearer {settings.bitgn_api_key}"},
            timeout=30.0,
        )
        self._call_counts: dict[str, int] = {}

    @property
    def call_count(self) -> int:
        return sum(self._call_counts.values())

    def call_count_for_run(self, run_id: str) -> int:
        return self._call_counts.get(run_id, 0)

    async def get_tasks(self) -> list[dict[str, Any]]:
        resp = await self._http.get("/tasks")
        resp.raise_for_status()
        return resp.json()

    async def start_run(self, task_id: str) -> dict[str, Any]:
        resp = await self._http.post(f"/tasks/{task_id}/runs")
        resp.raise_for_status()
        return resp.json()

    async def call_tool(
        self, run_id: str, tool_name: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        self._call_counts[run_id] = self._call_counts.get(run_id, 0) + 1
        resp = await self._http.post(
            f"/runs/{run_id}/tools/{tool_name}",
            json=args,
        )
        resp.raise_for_status()
        return resp.json()

    async def submit_result(self, run_id: str, result: dict[str, Any]) -> dict[str, Any]:
        resp = await self._http.post(f"/runs/{run_id}/submit", json=result)
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        await self._http.aclose()
