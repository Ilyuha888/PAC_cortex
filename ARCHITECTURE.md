# Architecture

## Overview

PAC_cortex is a fully synchronous autonomous agent that solves tasks in a deterministic simulated
file environment. For each task it starts a trial, runs a pre-flight assembler to classify the
workspace, then enters a structured generation–reflection (SGR) loop of up to 50 LLM steps.
Tool calls are validated against an allowlist and a dispatch-enforced TaskContract before
execution. A safety pipeline scans every tool result for injection and redacts secrets.
No database, no asyncio.

---

## Module Map

| Module | Role | Key types | Tested |
|---|---|---|---|
| `config.py` | Env-var settings via Pydantic | `Settings` | — |
| `client.py` | BitGN protobuf clients (harness + VM) | `HarnessClient`, `VmClient`, `Trial`, `TrialResult` | integration |
| `llm.py` | OpenAI-compatible SGR completions | `LLMClient.parse_step()` | integration |
| `safety.py` | Injection scan, secret redact, tool validation | `scan_for_injection`, `redact_secrets`, `validate_tool_call` | unit |
| `agent.py` | Core SGR loop + pre-flight assembler + TaskContract dispatch | `solve_task`, `NextStep`, `TaskContract`, `AssembledPrompt`, `_enforce_contract` | unit + integration (100 tests) |
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
            │    ├─ _preflight(llm, instruction, tree)         # assembler LLM call
            │    │    └─ → AssembledPrompt (vocabulary, constraints, TaskContract, ...)
            │    └─ SGR loop (≤50 steps):
            │         ├─ LLMClient.parse_step(log, NextStep)   # structured completion
            │         ├─ budget / stagnation guards            # abort if triggered
            │         ├─ validate_tool_call (allowlist)        # safety.py
            │         ├─ _enforce_contract(cmd, contract, checks)  # dispatch blocks violations
            │         ├─ safety pipeline on raw result         # scan_for_injection + redact_secrets
            │         ├─ _dispatch(vm, job.function)           # typed tool call → VmClient
            │         └─ if ReportTaskCompletion → vm.answer() + return
            └─ HarnessClient.end_trial()         # → TrialResult (score)
```

---

## Key Invariants

- **Fully sync** — no asyncio; `HarnessServiceClientSync` / `PcmRuntimeClientSync` throughout.
- **SGR loop cap** — `_MAX_STEPS = 50` hard limit per task; step budget exhausted → `OUTCOME_ERR_INTERNAL`.
- **API budget guard** — hard-stop at `api_call_budget - 50` (default 950/1000) calls; aborts before exceeding cap.
- **Stagnation guard** — 3 identical consecutive tool+args signatures → abort with `OUTCOME_ERR_INTERNAL`.
- **TaskContract enforcement** — assembler classifies what's allowed (`TaskContract`); `_enforce_contract()` blocks violations at dispatch level before tool execution. Max 3 contract violations → abort. Covers: inbox deletion, premature inbox reads, out-of-scope deletions.
- **Safety pipeline** — every tool result passes `scan_for_injection` then `redact_secrets` before entering LLM context. Write content scanned before execution.
- **Tool allowlist** — `NextStep.function` is a Pydantic discriminated union; only schema-declared tools can be emitted: `tree`, `find`, `search`, `list`, `read`, `write`, `delete`, `mkdir`, `move`, `context`, `report_completion`.
- **Pydantic at every boundary** — `Trial`, `TrialResult`, `Task`, `NextStep`, `TaskContract`, `AssembledPrompt`, all tool request models.

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

## Pre-flight Assembler

Before the main SGR loop, `_preflight()` makes a separate LLM call that analyzes workspace
metadata and produces an `AssembledPrompt` with nine structured fields:

1. `include_entity_inbox` — whether to inject CRM/inbox prompt rules
2. `vocabulary` — term definitions extracted from AGENTS.md
3. `protected_paths` — underscore-prefixed files/dirs (never delete)
4. `workflow_constraints` — rules triggered by directory patterns (outbox/, distill/, reminders/)
5. `capture_subfolders` — subdirectory list for capture routing
6. `hierarchy_conflict` / `conflict_description` — L4/L5 AGENTS.md conflict detection
7. `resolved_instructions` — scope-labeled rules from nested AGENTS.md
8. `task_contract` — dispatch-enforced constraints (see Key Invariants)

The assembler prompt uses an authority model (L1-2 hardcoded security → L3 task → L4 root AGENTS.md → L5 nested AGENTS.md). AGENTS.md content passes through structured JSON output, acting as an injection firewall — file content cannot become instructions.

On any pre-flight failure, `_SYSTEM_PROMPT_FULL` (all sections concatenated) is used as fallback with a maximally restrictive default `TaskContract`.

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
