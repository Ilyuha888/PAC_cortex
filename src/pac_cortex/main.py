"""CLI entrypoint for PAC_cortex."""

import asyncio
import logging
import sys

import httpx

from pac_cortex.config import settings


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


async def cmd_smoke() -> None:
    """Test connectivity to BitGN API and LLM."""
    print(f"BitGN API: {settings.bitgn_api_url}")
    print(f"LLM model: {settings.llm_model}")

    # Test BitGN connectivity
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(settings.bitgn_api_url)
            print(f"BitGN status: {resp.status_code}")
    except httpx.ConnectError:
        print("BitGN status: connection failed (expected if sandbox not yet live)")
    except Exception as e:
        print(f"BitGN status: {type(e).__name__}: {e}")

    # Test LLM connectivity
    if settings.llm_api_key:
        try:
            import openai

            client = openai.AsyncOpenAI(api_key=settings.llm_api_key)
            resp = await client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            print(f"LLM status: ok ({resp.choices[0].message.content})")
        except Exception as e:
            print(f"LLM status: {type(e).__name__}: {e}")
    else:
        print("LLM status: no API key configured")


async def cmd_run() -> None:
    """Execute a full session."""
    from pac_cortex.runner import run_session

    results = await run_session()
    completed = sum(1 for r in results if r["status"] == "completed")
    print(f"\nResults: {completed}/{len(results)} tasks completed")


def main() -> None:
    _setup_logging()

    if len(sys.argv) < 2:
        print("Usage: python -m pac_cortex <command>")
        print("Commands: run, smoke")
        sys.exit(1)

    command = sys.argv[1]
    if command == "smoke":
        asyncio.run(cmd_smoke())
    elif command == "run":
        asyncio.run(cmd_run())
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
