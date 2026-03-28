# PAC_cortex

Autonomous agent for the [BitGN PAC challenge](https://bitgn.com/challenge/PAC) (April 11, 2026).

Solves tasks in a deterministic simulated environment via API. Scored on correctness, safe tool use, injection resistance, and protocol compliance.

## Status

**22/22 (100.0%)** on BitGN benchmark. SDK integration complete; agent runs the full SGR loop against the harness.

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
- ~1000 API calls/task budget with hard guard; 50-step agent loop
- Sequential task execution
- **Pre-flight assembler**: separate LLM call analyzes workspace tree + AGENTS.md, produces structured config (vocabulary, constraints, hierarchy conflict detection, task contract)
- **TaskContract dispatch enforcement**: assembler classifies what's allowed, Python code blocks violations before execution (inbox deletion, premature reads, out-of-scope deletes)
- All tool calls validated against allowlist
- Tool results scanned for injection attempts; write content scanned before execution
- Secret redaction on all tool results before LLM context
