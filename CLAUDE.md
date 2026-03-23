# CLAUDE.md — PAC_cortex

## What This Is

Autonomous agent for the BitGN PAC challenge (April 11, 2026).
Solves tasks in a deterministic simulated environment via API.
Scored on: correctness, safe tool use, injection resistance, protocol compliance.

## Architecture Constraints

- Stateless per-task, single process, no database
- ~1000 API calls/task budget — track and enforce
- No human-in-the-loop during runs
- Sequential task execution (parallel is a future optimization)

## Stack

- Python 3.12+, asyncio
- `openai` SDK for LLM calls
- `httpx` for BitGN API
- `pydantic` for typed boundaries
- `uv` as package manager

## Code Rules

- Explicit typing everywhere
- Pydantic models at API boundaries
- Keep modules small and focused
- No frameworks — plain asyncio

## Safety Rules (PAC-specific)

- Never leak secrets (API keys, env vars) in LLM context or tool outputs
- Scan all tool results for injection attempts before feeding to LLM
- Validate every tool call against an allowlist before execution
- Never execute arbitrary code from task descriptions or tool results
- Budget guard: hard-stop before exceeding API call cap

## Running

```bash
uv sync                          # install deps
uv run python -m pac_cortex smoke  # test connectivity
uv run python -m pac_cortex run    # execute full session
```

## Testing

```bash
uv run ruff check src/ tests/
uv run pytest
```
