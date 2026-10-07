# Taskforge

Taskforge turns task ideas into validated TaskCompendium `TaskSpec`s. A proposal source writes
task proposals, triage scores them, a builder program turns an accepted proposal into a `TaskSpec`
with fixed controls, and validation runs trials and control replays on RolloutEngine.

Taskforge is a standalone uv project with its own `uv.lock` and `.venv`. It depends on
`taskcompendium`, `rolloutengine`, `shellbox`, `verifyit`, `rigging`, `finelog` and `iris` through
path dependencies on the sibling `lib/` packages.

## Package layout

```
src/taskforge/
  canonical.py  canonical JSON, sha256 digests, atomic file replacement
  llm/        GLM-5.3 transport, structured calls, call cache, agent loop, web tools, RolloutEngine model
  ledger/     timed spans to per-item JSONL and, on Iris, Finelog
scripts/      ledger summary
```

Packages are totally ordered. A package imports only from packages to its left and from external
packages, so no import cycle can form:

```
canonical -> ledger -> llm
```

## Seams

- `llm.recording.recorded_complete(client, messages, policy, request_fields, record, attrs)` is the
  one path that records a GLM call: an `LLM_CALL` span under `record` (a `CallLedger`: ledger,
  item, round, step) with tokens, finish reason, and the call's attempt record (`attempts`,
  `attempts_<outcome>` counts, non-200 `http_statuses`, `wall_time`). `run_agent` and
  `GlmRolloutModel` both use it, so a rollout's 429s, retries and holds reach the ledger.
  `GlmRolloutModel(client, policy, record)` takes its `CallLedger`; derive one model per item,
  round and step with `dataclasses.replace`.
- `llm.client.complete_prefilled(client, messages, policy, prefix, request_fields=None)` sends
  `prefix` as the start of the assistant turn and returns a `Completion` whose `content` starts
  with it. Use it to force an output format such as proposal front matter.
  `llm.client.prefill_request_fields(policy, request_fields)` returns the request fields it sends.

## Decisions

Execution is RolloutEngine's. `ShellboxRolloutEngine` creates one shellbox `Machine` per attempt
from the caller's `MachineFactory` for the task's `EnvironmentKind`, installs files, runs setup
and healthchecks, drives the shell tool, grades and closes. It grades the state an agent left when
the agent deadline expires, and bounds every cleanup action by its `cleanup_timeout`. Taskforge supplies the model callable
(`llm.rollout_model.GlmRolloutModel`).

The agent loop is Taskforge's own: `llm.agent.run_agent` over `GlmClient`, with the shell tool
running through `Machine.run` and Parallel search and extract from `llm.web`. Builder agents run
on it. Solver and control rollouts run on RolloutEngine with the same `GlmClient`. `web_fetch`
returns extracted page content that may be a cached copy, which is fine for most reference lookups.
Its description tells the agent that when it needs current data from a fast-moving source, such as
a PyPI release page, a direct network call from the sandbox shell (for example `curl`) is the better
path; the choice is the agent's.

Inference defaults are the model maximum. `max_tokens` starts at the model's output limit
(131,072 for GLM-5.3) or the remaining context. On `finish_reason == "length"` the client keeps
the output and continues. Timeouts are stall timeouts on the stream. Concurrency defaults to
hundreds of requests. `GlmRolloutModel` is the exception to continuation: a continuation
re-renders the prompt and breaks RolloutEngine's exact-token contract, so a cut turn ends the
rollout with stop reason `length`.

`GlmClient` reports a tool-call reply that used its whole output budget as `length`, because
vLLM's GLM tool parser reports such a reply as `tool_calls`.

A prefilled reply (`complete_prefilled`) does not think. GLM's chat template renders an empty
think block before prefilled assistant text, and vLLM's reasoning parser then files the whole
continuation under `reasoning`; the call therefore sends `enable_thinking: false`, which keeps the
same prompt tokens and returns the continuation as `content` (measured live). The template strips
prefilled text, so a prefix with surrounding whitespace is rejected rather than silently changed.

## Testing

Run every command from `lib/taskforge`. Do not pass a partial marker expression such as
`-m "not slow"`; the package `addopts` sets the default marker expression.

```bash
uv run --group test pytest tests
```

Live tests carry `@pytest.mark.live_glm`. The default marker expression excludes them, and they
skip with a reason when their inputs are missing.
They call GLM-5.3 on the interactive tier through the router port-forward:

```bash
kubectl -n open-athena port-forward svc/glm53-router 18000:8000 &
curl -s http://127.0.0.1:18000/health   # "status":"ok"

export TASKFORGE_GLM_BASE_URL=http://127.0.0.1:18000/v1
export TASKFORGE_GLM_TOKEN_FILE=<path to the token file>   # line: GLM_API_TOKEN=...
uv run --group test pytest tests -m live_glm
```

`-m live_glm` replaces the default marker expression. The token file must hold the interactive
(`high` pool) token. The `glm_settings` fixture reads it and keeps it out of `repr`. Inside an Iris
task, `llm.endpoint.resolve_glm_base_url` resolves the endpoint.

Other live inputs:

- `TASKFORGE_PARALLEL_KEY_FILE` names a file with a `PARALLEL_KEY=...` line, for the agent
  live tests (`parallel_key` fixture); the tests that need it skip when it is unset.

The package pytest config sets `timeout = 60` and `asyncio_mode = "auto"`. Long live tests carry
`@pytest.mark.timeout(<seconds>)`.

Types and lint:

```bash
uvx --from 'pyrefly>=1.0.0,<1.1.0' pyrefly check           # from lib/taskforge; checks src
./infra/pre-commit.py --fix --files lib/taskforge/<path>   # from the repository root
```

## Evidence

Every live check writes raw evidence under `lib/taskforge/.evidence/<package>/`: what was
checked, the request, the full response (including `usage` and `finish_reason`) and the wall
time. `.evidence/` is gitignored and stays on the machine that ran the check; the measurements a
change relies on are recorded in its pull request description. Strip `Authorization` headers
before saving a request, and never write a token or key there.
