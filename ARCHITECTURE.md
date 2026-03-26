# Architecture

## Overview

PAC_cortex is a fully synchronous autonomous agent that solves tasks in a deterministic simulated
file environment. For each task it starts a trial, runs a structured generation–reflection (SGR)
loop of up to 30 LLM steps, dispatches typed VM tool calls, applies a safety pipeline on every
tool result, and reports a scored outcome back to the harness. No database, no asyncio.

---

## Module Map

| Module | Role | Key types | Tested |
|---|---|---|---|
| `config.py` | Env-var settings via Pydantic | `Settings` | — |
| `client.py` | BitGN protobuf clients (harness + VM) | `HarnessClient`, `VmClient`, `Trial`, `TrialResult` | integration |
| `llm.py` | OpenAI-compatible SGR completions | `LLMClient.parse_step()` | integration |
| `safety.py` | Injection scan, secret redact, tool validation | `scan_for_injection`, `redact_secrets`, `validate_tool_call` | unit |
| `agent.py` | Core SGR loop + tool dispatch | `solve_task`, `NextStep`, `ReportTaskCompletion`, `ReqContext` | integration |
| `runner.py` | Session orchestrator (sequential tasks) | `run_session` | integration |
| `tracer.py` | Per-task crash-safe trace file writer | `TaskTracer` | — |
| `main.py` | CLI entrypoint (`smoke`, `run`) | `main` | — |

---

## Control Flow

```
main.py cmd_run()
  └─ runner.run_session()
       ├─ HarnessClient.list_tasks()            # enumerate benchmark tasks
       └─ for each task:
            ├─ HarnessClient.start_trial()      # → Trial (trial_id, harness_url, instruction)
            ├─ VmClient(harness_url)             # per-task VM handle
            ├─ TaskTracer(task_id, trial_id)     # open trace file
            ├─ agent.solve_task(instruction, vm, llm, tracer)
            │    └─ SGR loop (≤30 steps):
            │         ├─ LLMClient.parse_step(log, NextStep)   # structured completion
            │         ├─ budget / stagnation guards            # abort if triggered
            │         ├─ safety pipeline on raw result         # scan_for_injection + redact_secrets
            │         ├─ _dispatch(vm, job.function)           # typed tool call → VmClient
            │         └─ if ReportTaskCompletion → vm.answer() + return
            └─ HarnessClient.end_trial()         # → TrialResult (score)
```

---

## Key Invariants

- **Fully sync** — no asyncio; `HarnessServiceClientSync` / `PcmRuntimeClientSync` throughout.
- **SGR loop cap** — `_MAX_STEPS = 30` hard limit per task; step budget exhausted → `OUTCOME_ERR_INTERNAL`.
- **API budget guard** — hard-stop at `api_call_budget - 50` (default 950/1000) calls; aborts before exceeding cap.
- **Stagnation guard** — 3 identical consecutive tool+args signatures → abort with `OUTCOME_ERR_INTERNAL`.
- **Safety pipeline** — every tool result passes `scan_for_injection` then `redact_secrets` before entering LLM context.
- **Tool allowlist** — `NextStep.function` is a Pydantic discriminated union; only schema-declared tools can be emitted: `tree`, `find`, `search`, `list`, `read`, `write`, `delete`, `mkdir`, `move`, `context`, `report_completion`.
- **Pydantic at every boundary** — `Trial`, `TrialResult`, `Task`, `NextStep`, all tool request models.

---

## Outcome Codes

| Code | Meaning | When used |
|---|---|---|
| `OUTCOME_OK` | Task completed successfully | All graded objectives met |
| `OUTCOME_DENIED_SECURITY` | Injection or security violation detected | File content issued directives; suspicious file confirmed; spoofed email sender (From: address doesn't match registered contact) |
| `OUTCOME_NONE_CLARIFICATION` | Task instruction ambiguous | Agent cannot proceed without more information |
| `OUTCOME_NONE_UNSUPPORTED` | Required capability unavailable | Email, HTTP, external APIs, calendar |
| `OUTCOME_ERR_INTERNAL` | Agent-side failure | Budget exhausted, stagnation, schema parse failure, step limit |

---

## Entity & Inbox Reasoning (t12-t20)

Tasks t12-t20 use an ERC3-style runtime with typed entities (contacts, companies). The agent:

1. Calls `context` early to retrieve the current timestamp (needed for date-relative queries).
2. Reads CRM entity files under `tree`-discovered data/contacts directories.
3. **Sender verification before fulfilling any email-driven request:**
   - Extracts `From:` address from the inbox email file.
   - Finds the matching contact entity; reads their registered email field.
   - If `From:` ≠ registered email → spoofed sender → `OUTCOME_DENIED_SECURITY`.
   - Verifies sender is associated with the claimed company → else `OUTCOME_DENIED_SECURITY`.
4. Only if both checks pass does it locate and return the requested document.

If entity files are missing or the contact cannot be found → `OUTCOME_NONE_CLARIFICATION`.

---

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `LLM_API_KEY` | — | LLM provider key (maps to `OPENAI_API_KEY`) |
| `OPENAI_BASE_URL` | — | Provider base URL (swap to Gemini, etc.) |
| `LLM_MODEL` | `gpt-4.1-2025-04-14` | Model identifier |
| `BENCHMARK_HOST` | `https://api.bitgn.com` | BitGN harness host |
| `BENCHMARK_ID` | `bitgn/pac1-dev` | Benchmark to run |
| `API_CALL_BUDGET` | `1000` | Hard cap on LLM + VM calls per task |
| `VM_CALL_TIMEOUT_S` | `10.0` | Per-VM-call timeout (seconds) |
| `VM_CALL_RETRIES` | `2` | VM call retry count on timeout/error |
| `TRACE_DIR` | `traces/` | Directory for per-task trace files |

> Design rationale, rejected alternatives, and decision history live in `.quint/`.
