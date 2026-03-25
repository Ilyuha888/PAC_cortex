"""CLI entrypoint for PAC_cortex."""

import logging
import sys

from pac_cortex.config import settings


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_smoke() -> None:
    """Test connectivity to BitGN API and LLM."""
    from connectrpc.errors import ConnectError

    from pac_cortex.client import HarnessClient

    print(f"BitGN host:  {settings.benchmark_host}")
    print(f"Benchmark:   {settings.benchmark_id}")
    print(f"LLM model:   {settings.llm_model}")

    try:
        harness = HarnessClient(settings.benchmark_host)
        status = harness.status()
        print(f"BitGN status: {status}")
    except ConnectError as e:
        print(f"BitGN status: {e.code}: {e.message}")
    except Exception as e:
        print(f"BitGN status: {type(e).__name__}: {e}")

    if settings.llm_api_key:
        try:
            from pac_cortex.llm import LLMClient
            llm = LLMClient()
            resp = llm._client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": "ping"}],  # type: ignore[list-item]
                max_tokens=5,
            )
            print(f"LLM status: ok ({resp.choices[0].message.content})")
        except Exception as e:
            print(f"LLM status: {type(e).__name__}: {e}")
    else:
        print("LLM status: no API key configured")


def cmd_run(task_filter: list[str] | None = None) -> None:
    """Execute a full session."""
    from pac_cortex.runner import run_session

    results = run_session(task_filter)
    if not results:
        print("No tasks completed.")
        return
    perfect = sum(1 for r in results if r["score"] == 1.0)
    avg = sum(r["score"] for r in results) / len(results) * 100
    print(f"\nResults: {perfect}/{len(results)} perfect, {avg:.1f}% avg score")
    zeroed = [r["task_id"] for r in results if r["score"] == 0.0]
    if zeroed:
        print(f"{len(zeroed)}/{len(results)} tasks scored 0 — check logs for details")


def main() -> None:
    _setup_logging()

    if len(sys.argv) < 2:
        print("Usage: python -m pac_cortex <command> [task_id ...]")
        print("Commands: run, smoke")
        sys.exit(1)

    command = sys.argv[1]
    if command == "smoke":
        cmd_smoke()
    elif command == "run":
        task_filter = sys.argv[2:] or None
        cmd_run(task_filter)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
