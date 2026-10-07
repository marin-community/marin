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
  content_hash.py  canonical JSON and its sha256 digests
  atomic_file.py   atomic file replacement
  llm/        GLM-5.3 transport, structured calls, call cache, agent loop, web tools, RolloutEngine model
  ledger/     timed spans to per-item JSONL and, on Iris, Finelog
  spec/       TaskSpec assembly and fixed controls
  sandbox/    MachineFactory selection per EnvironmentKind, up-front task refusals, image builds
  proposal/   the TaskProposal document (model.py), the ProposalSource protocol (source.py), sources/
  triage/     structural checks, the GLM rubric, verdicts
  build/      builder programs: memoized steps, the Build SDK, program authoring, the standard template
  validate/   trials, the failure classifier, control replay, evidence aggregation
scripts/      Iris image builder, cluster probes, ledger summary
```

Packages are totally ordered. A package imports only from packages to its left and from external
packages, so no import cycle can form:

```
content_hash -> atomic_file -> ledger -> spec -> sandbox -> llm -> proposal -> triage -> build -> validate
```

The foundation packages (`content_hash`, `atomic_file`, `ledger`, `spec`, `sandbox`, `llm`) import
only each other and external packages. A later stage reaches an earlier one through its seam
modules, each of which keeps its types beside the code that checks their invariants:
`proposal.model` (`TaskProposal`), `proposal.source` (`ProposalBatch`, `SlotFailure`,
`ProposalSource`), `triage.verdict` (`Verdict`, `TriageDecision`), `build.run` (`TaskDraft`,
`load_draft`, `item_id_for`), `build.author` (`BuildProgram`, `Revision`), `validate.outcome` and
`validate.evidence`.

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
- `spec.controls.Control`: one labeled candidate submission (`kind`, `category`, `concern`,
  `author`, a `Transcript` or `Workspace` payload, an `Expectation`, `stage`). `concern` is a
  required `ControlConcern`: `reference`, `acceptance`, `extraction` or `shortcut`.
  `validate_controls(task, controls)` checks a set against its task; `controls_json` and
  `parse_controls` round-trip it.
- `sandbox.factories.machine_factories(where, controller_url, image_cache)`: the
  `EnvironmentKind -> MachineFactory` mapping for `ShellboxRolloutEngine`. On Iris it takes the
  controller URL and no image cache; on a laptop it takes the directory where the Docker factory
  keeps Skopeo-prepared images and no controller URL. The caller names that directory.
- `proposal.source.ProposalSource[IdeaT]`: `async propose(idea, n) -> ProposalBatch`. A batch holds
  the planning request and completions and one `SlotProposal` or `SlotFailure` per slot, in slot
  order. A failed slot does not drop its siblings.
- `proposal.model.TaskProposal`: YAML front matter (`ProposalHeader`) plus a markdown body with
  required section headings. `parse` and `render` round-trip it; `digest` is the sha256 of the
  canonical form.
- `triage.program.evaluate(proposal, checks, rubric, ctx) -> Verdict`: structural checks run
  first, and a fatal failure or a null proposal rejects without a model call. The rubric scores
  independent samples, and the decision is ACCEPT or REJECT by strict majority, otherwise REPAIR.
- `build.run.run_build(program, proposal, ...)`: runs a builder program of memoized async steps.
  A step's memo key covers its code, arguments, the data globals it reads, `SDK_VERSION`, the
  proposal and the model policy. The verifier must come from a GRADER step and the controls from
  a CONTROLS step, and the controls must pass `spec.controls.validate_controls`. The draft holds
  the task, its `TaskExecution` and its controls. A host failure during the build raises
  `build.infrastructure.BuildInfrastructureFailure` with a `cause` of `no_factory`,
  `scheduling_timeout` or `host_unreachable`, even when the program wrapped it; it is never a
  `BuildFailure`, and the caller does not charge it to the program.
- `validate.trials.run_trials(task, execution, plan, settings, model)`: runs k trials through
  `ShellboxRolloutEngine`. Each trial is `Graded` or `Ungraded` with one typed `Cause`, and
  `validate.classify.classify` is the only failure classifier. `TrialPlan.first_attempt` numbers
  each trial's first attempt file and ledger step, so a re-entered trial continues after the
  attempts on disk.
- `validate.controls.replay(task, execution, controls, plan, settings, tokenize)`: replays each
  control as one `CONTROL` trial named by its id. `ControlPlan.first_attempts` maps a control id
  to its first attempt number (0 when absent), with the same re-entry contract as a trial.

A builder agent's turn and a rollout's model call take the same path to GLM and to the ledger:

```
run_agent (builder turn)          GlmRolloutModel (RolloutEngine rollout)
        |                                     |
        +---------> recorded_complete <-------+
                    (record: CallLedger)
                      |              |
                      v              v
          GlmClient.complete      span(LLM_CALL) -> Ledger.record(LedgerEntry)
          retries, holds,                            |-> JsonlLedger   <root>/<item_id>.jsonl
          context probe                              '-> FinelogLedger taskforge.ledger (under Iris)
                      |
                      v
          GLM-5.3 router, POST /v1/chat/completions (streamed)
```

`CallStore` sits beside this path: it wraps `GlmClient.complete` and `complete_structured` with a
content-addressed cache and does not record to the ledger.

## Decisions

Task specs are upstream's. A built task is a TaskCompendium `TaskSpec` plus the `TaskExecution` it
runs with. TaskCompendium keeps execution settings out of the spec: deadlines, the agent user, and
each stage's working files, setup and healthcheck are a `TaskExecution`. `spec.draft.assemble`
checks the two together, a builder returns both, and validation passes both to RolloutEngine.
#9623 proposes moving the environment, system prompt and concrete tools out of `TaskSpec` into a
`TaskHarnessSpec` and `TaskSequence`, and renaming `TaskExecution` to `TaskExecutionSpec`. Until
that lands, `spec.draft._presentation` alone builds the former and `spec.draft.task_execution` alone
builds the latter, so each move is a single-site change.
Task-specific grading is the task's own scripts, run as a `ShellVerifierSpec` (private files on its
`VerifierSpec`; reward on stdout, by exit code, or in reward files; optionally in a separate
grading environment). Generic verifier types
(math, mcq, judge, pytest, aggregation) belong to `lib/verifyit`, which `TaskSpec` reaches
through its verifier registry. A gap in either is fixed upstream. Interim code for one gap goes
in `taskforge/spec/extensions/<issue>.py`.

Every control names the part of the grader it exercises. Answer extraction is expected to move to
a cheap model, so controls that only pin today's parser carry `concern = extraction` and can be
found and retired together. Each stage needs a `reference` control, an `acceptance` control and a
`shortcut` control, so its required coverage never rests on extraction controls alone. The concern
a control may carry follows its category (`spec.controls.CONCERNS`): `reference` only on
known-correct controls, `shortcut` only on shortcut and reward-hack controls.

Execution is RolloutEngine's. `ShellboxRolloutEngine` creates one shellbox `Machine` per attempt
from the caller's `MachineFactory` for the task's `EnvironmentKind`, installs files, runs setup
and healthchecks, drives the shell tool, grades and closes. It grades the state an agent left when
the agent deadline expires, and bounds every cleanup action by its `cleanup_timeout`. Taskforge supplies the model callable
(`llm.rollout_model.GlmRolloutModel`) and the factories (`sandbox.factories.machine_factories`).
Taskforge reaches a sandbox only through a `Machine` that the engine or a builder step created.

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

A build tells host failures from program failures by where the error was raised, not by its
message. `Build` wraps its machine factories (`build.infrastructure.host_checked_factories`): a
missing factory for the machine kind, a `TimeoutError` from the factory's own wait, and a
connection or controller transport error from creating or driving a machine are infrastructure.
Everything else is the program's, including an image it built or named that fails, a spec the
backend refuses, and a deadline the program set with `startup_timeout`. Shellbox raises a bare
`RuntimeError` for both a failed Docker build and an unreachable Docker daemon, so the daemon case
is charged to the program until shellbox types it.

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

- `TASKFORGE_PARALLEL_KEY_FILE` names a file with a `PARALLEL_KEY=...` line, for the agent and
  build live tests (`parallel_key` fixture); the tests that need it skip when it is unset.

The package pytest config sets `timeout = 60` and `asyncio_mode = "auto"`. Long live tests carry
`@pytest.mark.timeout(<seconds>)`.

Types and lint:

```bash
uvx --from 'pyrefly>=1.0.0,<1.1.0' pyrefly check           # from lib/taskforge; checks src
./infra/pre-commit.py --fix --files lib/taskforge/<path>   # from the repository root
```

The cluster scripts run on Iris and document their submit commands in their docstrings:
`scripts/build_image_job.py` builds a `DockerBuild` context and pushes it to a registry digest
through `scripts/push_image_task.py`, `scripts/iris_machine_probe.py` probes the shellbox Iris
backend, and `scripts/cluster_rollout_probe.py` runs validation trials inside an Iris task.

## Evidence

Every live check writes raw evidence under `<evidence root>/<package>/`: what was checked, the
request, the full response (including `usage` and `finish_reason`) and the wall time. The evidence
root is `$TASKFORGE_EVIDENCE_DIR` when set, else `taskforge-evidence/` in the system temp directory
(`tempfile.gettempdir()`); the `evidence_root` fixture in `tests/conftest.py` resolves it. Evidence
stays outside the checkout on the machine that ran the check; the measurements a change relies on
are recorded in its pull request description. Strip `Authorization` headers before saving a
request, and never write a token or key there.
