"""Session orchestrator: fetch tasks, run agent, collect results."""

import logging
from typing import Any

from pac_cortex.agent import solve_task
from pac_cortex.client import BitgnClient
from pac_cortex.llm import LLMClient

logger = logging.getLogger(__name__)


async def run_session() -> list[dict[str, Any]]:
    """Execute a full session: fetch all tasks and solve them sequentially."""
    client = BitgnClient()
    llm = LLMClient()
    results: list[dict[str, Any]] = []

    try:
        tasks = await client.get_tasks()
        logger.info("Fetched %d tasks", len(tasks))

        for i, task in enumerate(tasks, 1):
            task_id = task.get("id", "unknown")
            logger.info("Task %d/%d: %s", i, len(tasks), task_id)
            try:
                result = await solve_task(task, client, llm)
                results.append(result)
                logger.info("Task %s finished: %s", task_id, result["status"])
            except Exception:
                logger.exception("Task %s failed", task_id)
                results.append({"run_id": None, "status": "error", "task_id": task_id})
    finally:
        await client.close()

    # Summary
    completed = sum(1 for r in results if r["status"] == "completed")
    logger.info(
        "Session done: %d/%d completed, %d tokens used (prompt=%d, completion=%d)",
        completed,
        len(results),
        llm.total_prompt_tokens + llm.total_completion_tokens,
        llm.total_prompt_tokens,
        llm.total_completion_tokens,
    )
    return results
