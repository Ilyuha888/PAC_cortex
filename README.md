# PAC_cortex

Autonomous agent for the [BitGN PAC challenge](https://bitgn.com/challenge/PAC) (April 11, 2026).

Solves tasks in a deterministic simulated environment via API. Scored on correctness, safe tool use, injection resistance, and protocol compliance.

## Status

Fully operational. SDK integration complete; agent runs the full SGR loop against the BitGN benchmark harness.

## Setup

```bash
# Install dependencies
uv sync

# Copy and fill in env vars
cp .env.example .env

# Test connectivity
uv run python -m pac_cortex smoke

# Run full session
uv run python -m pac_cortex run
```

## Project Structure

```
src/pac_cortex/
  config.py    — settings via env vars (Pydantic)
  client.py    — BitGN API clients (HarnessClient, VmClient) with retry/timeout
  llm.py       — OpenAI SDK wrapper (chat + tool use)
  agent.py     — core control loop (task solver)
  safety.py    — injection detection, tool validation, secret redaction
  tracer.py    — per-task trace file writer for agent observability
  runner.py    — session orchestrator
  main.py      — CLI entrypoint
```

## Testing

```bash
uv run ruff check src/ tests/
uv run pytest
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for system design, module map, control flow, and invariants.

- Stateless per-task, single process, no database
- ~1000 API calls/task budget with hard guard
- Sequential task execution
- All tool calls validated against allowlist
- Tool results scanned for injection attempts
