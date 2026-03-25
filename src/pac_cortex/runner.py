"""Session orchestrator: fetch tasks, run agent, collect results."""

import logging

import openai
from connectrpc.errors import ConnectError

from pac_cortex.agent import solve_task
from pac_cortex.client import HarnessClient, VmClient
from pac_cortex.config import settings
from pac_cortex.llm import LLMClient

logger = logging.getLogger(__name__)


def run_session(task_filter: list[str] | None = None) -> list[dict]:
    """Execute a full session: fetch all tasks and solve them sequentially."""
    harness = HarnessClient(settings.benchmark_host)
    llm = LLMClient()
    results: list[dict] = []

    tasks = harness.list_tasks(settings.benchmark_id)
    logger.info("Fetched %d tasks", len(tasks))

    for i, task in enumerate(tasks, 1):
        if task_filter and task.task_id not in task_filter:
            continue

        logger.info("Task %d/%d: %s", i, len(tasks), task.task_id)
        trial = harness.start_trial(settings.benchmark_id, task.task_id)
        logger.info("Trial %s | %s", trial.trial_id, trial.instruction[:80])

        try:
            vm = VmClient(trial.harness_url)
            solve_task(trial.instruction, vm, llm)
        except (openai.APIError, ConnectError) as exc:
            logger.exception("Task %s failed (recoverable): %s", task.task_id, type(exc).__name__)

        trial_result = harness.end_trial(trial.trial_id)
        results.append({
            "task_id": task.task_id,
            "trial_id": trial.trial_id,
            "score": trial_result.score,
            "score_detail": trial_result.score_detail,
        })
        status = "OK" if trial_result.score == 1.0 else "FAIL"
        logger.info("Task %s: %s score=%.2f", task.task_id, status, trial_result.score)
        if trial_result.score_detail:
            logger.info("Task %s detail: %s", task.task_id, trial_result.score_detail)

    if results:
        avg = sum(r["score"] for r in results) / len(results) * 100
        logger.info(
            "Session done: %.1f%% avg | tokens prompt=%d completion=%d",
            avg,
            llm.total_prompt_tokens,
            llm.total_completion_tokens,
        )

    return results
