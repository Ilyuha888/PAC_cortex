# PAC_cortex

Autonomous agent for the [BitGN PAC challenge](https://bitgn.com/challenge/PAC) (April 11, 2026).

Solves tasks in a deterministic simulated environment via API. Scored on correctness, safe tool use, injection resistance, and protocol compliance.

## Status

Skeleton — waiting for BitGN sandbox/SDK to go live for integration.

## Setup

```bash
# Install dependencies
uv sync

# Copy and fill in env vars
cp .env.example .env

# Test connectivity (will fail until sandbox is live)
uv run python -m pac_cortex smoke

# Run full session
uv run python -m pac_cortex run
```

## Project Structure

```
src/pac_cortex/
  config.py    — settings via env vars (Pydantic)
  client.py    — BitGN API client stub
  llm.py       — OpenAI SDK wrapper (chat + tool use)
  agent.py     — core control loop (task solver)
  safety.py    — injection detection, tool validation, secret redaction
  runner.py    — session orchestrator
  main.py      — CLI entrypoint
```

## Testing

```bash
uv run ruff check src/ tests/
uv run pytest
```

## Architecture

- Stateless per-task, single process, no database
- ~1000 API calls/task budget with hard guard
- Sequential task execution
- All tool calls validated against allowlist
- Tool results scanned for injection attempts
