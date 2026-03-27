"""Per-task trace writer — crash-safe, human-readable."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import io

    from pac_cortex.agent import NextStep


class TaskTracer:
    def __init__(
        self,
        task_id: str,
        trial_id: str,
        instruction: str,
        trace_dir: str,
    ) -> None:
        short_trial = trial_id[:8]
        filename = f"{task_id}_{short_trial}.txt"
        path = Path(trace_dir) / filename
        self._file: io.TextIOWrapper = path.open("w", buffering=1, encoding="utf-8")
        self._write(
            f"=== Task: {task_id} | Trial: {trial_id} ===\n"
            f"Instruction: {instruction[:300]}\n"
            f"Started: {datetime.now(UTC).isoformat(timespec='seconds')}\n"
        )

    def _write(self, text: str) -> None:
        self._file.write(text)
        self._file.flush()

    def record_preflight(
        self,
        include_entity_inbox: bool,
        vocab_terms: int,
        notes: str,
        api_calls: int,
    ) -> None:
        self._write(
            f"Pre-flight: entity_inbox={include_entity_inbox} vocab_terms={vocab_terms} "
            f"notes={notes!r} api_calls={api_calls}\n"
        )

    def record_step(
        self,
        step_num: int,
        message_count: int,
        next_step: NextStep,
        injection_warnings: list[str],
    ) -> None:
        fn = next_step.function
        tool_repr = f"{fn.__class__.__name__}({fn.model_dump_json(exclude={'tool'})})"
        plan_repr = str(next_step.plan_remaining_steps_brief)

        text = (
            f"\n--- Step {step_num} ---\n"
            f"Messages in context: {message_count}\n"
            f"NextStep:\n"
            f"  current_state : {next_step.current_state!r}\n"
            f"  confidence    : {next_step.confidence}\n"
            f"  plan          : {plan_repr}\n"
            f"  tool          : {tool_repr}\n"
        )
        if injection_warnings:
            text += f"  [INJECTION WARNING: {injection_warnings}]\n"
        self._write(text)

    def record_tool_result(self, result_str: str) -> None:
        truncated = result_str[:500]
        suffix = f" ... ({len(result_str) - 500} chars truncated)" if len(result_str) > 500 else ""
        self._write(f"\nTool result ({len(result_str)} chars):\n  {truncated}{suffix}\n")

    def record_completion(self, outcome: str, message: str, api_calls: int) -> None:
        self._write(
            f"\n=== COMPLETED ===\n"
            f"Outcome   : {outcome}\n"
            f"Message   : {message[:200]}\n"
            f"API calls : {api_calls}\n"
        )

    def record_error(self, reason: str, api_calls: int) -> None:
        self._write(
            f"\n=== STOPPED: {reason} ===\n"
            f"API calls : {api_calls}\n"
        )

    def close(self) -> None:
        self._file.close()
