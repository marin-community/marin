# Taskforge

Taskforge turns task ideas into TaskCompendium `TaskSpec`s with a model's help. It is the
`marin-taskforge` member of the repository's uv workspace.

## Package layout

```
src/taskforge/
  stage_coordinator/           workpiece_machine.py: Stage, Exit, TRANSITIONS, `transition`, `derive_state`;
                               event_log.py: WorkpieceEventLog, InMemoryEventLog, WorkpieceEventStore,
                               JsonEventStore;
                               stage_functions.py: the stage Protocols and `StageFunctions`;
                               coordinator.py: WorkpieceBudget, StageCoordinator, `run_workpiece`
  agentic/                     agent.py: `generate_tasks`, one managed agent that makes tasks for an idea;
                               tools.py: the agent's Taskforge tools; gate.py: the acceptance rule;
                               config.py, prompts.py, parallel.py (web search), example_build.py
  capability.py                CapabilityIdea, the input of task generation, and its prompt record
  content_hash.py              canonical JSON and its sha256 digests
  ledger/                      timed spans to per-workpiece JSONL, and the Finelog backend
  llm/                         the ModelClient protocol, the GLM client, sampling policy, structured
                               output, ledger recording, agent loop, web tools, the
                               RolloutEngine model
  sandbox/                     MachineFactory selection per host, up-front task refusals, image builds,
                               the host-failure messages build and validation both classify
  spec/                        TaskSpec assembly, lowering, fixed controls, grader verdicts
  stage10_proposal/            model.py: the TaskProposal document and its generation regex;
                               source.py: Portfolio, SlotProposal, SlotFailure, the ProposalSource
                               protocol, `proposal_batch`, `propose_all`
  stage20_triage/              checks.py: structural checks; preflight.py: `preflight`, the default triage;
                               verdict.py: Verdict, TriageDecision and `accept`
  stage30_author/              program.py: build steps and the DAG they form;
                               author.py: Revision, AuthoredProgram, the Author protocol, `static_author`,
                               the program manifest;
                               loader.py: `load_program_source`, the plain loader of a program's source
  stage40_build_with_program/  context.py: RLEnvBuildContext, BuildFailure, BuildInfrastructureFailure;
                               step_store.py: step keys, memos, the memo store and `memoized`;
                               run.py: `run_program`, `run_build`, the draft and `WorkpieceBuildProvenance`
  stage60_review/              decision.py: the Decision types (Accept, Reject, Retry, Modify)
```

Each pipeline stage is a package named `stageNN_<name>`, numbered by tens in pipeline order:
proposal 10, triage 20, author 30, build 40, validate 50, review 60. The gaps leave room to insert a
stage later without renumbering. A stage imports from lower-numbered stages and from the shared
modules (`capability`, `child_process`, `content_hash`, `ledger`, `llm`, `sandbox`, `spec`), and
the shared modules import no stage. The coordinator package, `stage_coordinator`, runs the stages in order and
is unnumbered. It may import any stage and shared module, and `tests/test_imports.py` refuses an
import of it from any other package, along with the other wrong-way imports.
`scripts/ledger_summary.py` totals wall time and tokens per kind and step over JSONL ledgers.
`scripts/build_image_job.py` runs on Iris and documents its submit command in its docstring: it
builds a directory's `DockerBuild` context with `taskforge.sandbox.image_builder` and prints the
pinned registry digest.

## Proposals

A proposal source implements `stage10_proposal.source.ProposalSource[IdeaT, PlanT]` in two phases:

- `async plan(idea, n) -> Portfolio[PlanT]` plans the slots once. A `Portfolio` holds the planning
  request and completions, the source's own plan record (`plan`, which the document prompts quote),
  and a `PlannedSlot` per slot, numbered 1 through `n`, that either proposes or carries the
  `null_reason` the plan gave it. With a dataclass or pydantic plan record it is plain data:
  `content_hash.pretty_json` writes it and pydantic's `TypeAdapter(Portfolio[PlanT])` reads it back.
- `async write(idea, portfolio, slot) -> SlotOutcome` writes one slot's document, a `SlotProposal` or
  a `SlotFailure`, from the idea and the portfolio alone. A slot planned null becomes a proposal with
  `null_reason` set and makes no model call.

Because no slot waits for another, a scheduler can run each slot's `write` as its own unit and start
that slot's workpiece as soon as its proposal exists, and a relaunch with a persisted portfolio writes only
the slots it lacks. `proposal_batch(portfolio, outcomes)` assembles a portfolio and an outcome for
each of its slots, given in any order, into a `ProposalBatch`: the planning request and completions
and one outcome per slot, in slot order, so a failed slot does not drop its siblings.
`propose_all(source, idea, n)` plans, writes every slot concurrently and returns that batch, for a
caller that wants everything at once.

`stage10_proposal.model.TaskProposal` is YAML front matter (`ProposalHeader`, keys in `HEADER_KEYS`
order) plus a markdown body with required section headings. `parse` and `render`
round-trip it and `digest` is the sha256 of the canonical form. `parse` also checks the research
decision: `research_need: needed` lists at least one `research` item and `not_needed` none, and a
non-null body cites every item, and only listed items, as `research[i]`. `solver_machine` pins the
solver's machine: `null` for a container proposal, which runs in the image its build publishes, and
`none`, `shellsim`, `cpython` or `packages` (`SolverMachine`) for a reasoning or shellsim one, where
`none` (no machine: the solver answers in the conversation) is legal only for a reasoning proposal.
The proposer writes it, and the build keeps it. `parse` also refuses a `resources` entry that is
not a relative path inside the build root and a `build` name that repeats.
`document_request_fields(proposal_id, source)` returns the `structured_outputs.regex` request
field that constrains a completion to the canonical layout for that id and source. A reply cut off
by the output limit is a `SlotFailure`, and a reply that fails `parse` gets one repair request
built from `DOCUMENT_REPAIR_PROMPT`.

## Triage

The default triage is `stage20_triage.preflight.preflight(proposal) -> Verdict`, a deterministic
preflight that makes no model call. Header validity is `parse`'s job, which runs before triage, so
the checks in `stage20_triage.checks.CHECKS` read the body and what the header promises. One is
fatal: the required sections have content. Three are advisory: the grader section is consistent with
the declared verification, every research item states a purpose, and the build plan names every
declared resource. A fatal failure rejects the proposal with the failing check as the reason, and a
null proposal is rejected with its `null_reason`. Anything else is accepted and goes to build.
Advisory failures are recorded on the verdict and decide nothing. Whether the host can provide the
declared machine is checked by the build before any model call, and a host that cannot ends the
workpiece as the host's failure.

A consumer could add a model review to triage in one of two designs:

- A panel with repair: several independent model reviews of the proposal, a majority vote that
  accepts, rejects or asks for repair, one model rewrite of a repair proposal within a budget, and
  a repair exit on the stage coordinator that sends the rewrite back to triage.
- A bounded review with obligations: one model review that accepts the proposal and attaches
  obligations its author must honour, carried to build as state on the accept event.

Neither is implemented by default, because experiments found them not to be meaningful predictors
of downstream quality.

## Task specs

A built task is a TaskCompendium `TaskSpec` and the RolloutEngine `TaskRolloutSpec` that says how it
runs. `spec.draft.assemble` builds the semantic task and `spec.draft.lower` alone builds the rollout
spec, so a move of either is a single-site change.

- `spec.draft.assemble(task_id, instruction, answer_type, answer_format, grader, source, *,
  environment, files, output_paths, system, final_tools, tags) -> TaskSpec`. `environment` is
  `requirements(image=, packages=, setup=, workdir=, env=)`, whose commands run as Linux processes
  in the image or with CPython 3.12 and the PyPI `packages`, or, with neither, in ShellSim's
  simulated shell, or `None` for a task without a machine.
  `files` are agent-visible `file("workspace/x", ...)` resources relative to the machine root.
  `grader` is a `GraderPackage` from `answer_grader(spec)` (a verifyit mode graded in process, or
  in a verifier machine given `environment=`), `python_grader(script, config, environment=,
  answer_path=, timeout=)` (`python3 /tests/grade.py` in a verifier machine) or
  `script_grader(argv, reward, environment=, answer_path=, timeout=, ...)`. RolloutEngine grades an
  LLM judge (a verifyit `JudgeSpec`) on the host through the judge service of `TaskRolloutSpec.judge_connection`
  or `VERIFYIT_JUDGE_BASE_URL`, with no verifier machine, so its `RolloutEngine.check` fails on a host
  that has neither.
  `grader_environment(task_image)` is the verifier machine's environment: the task image, or
  `sandbox.images.GRADER_BASE_IMAGE` for a task without one.
- `spec.draft.lower(task, *, host, task_machine, verifier_machine, limits, factories)` runs
  each machine on `host_requirements` for `factories`: with the local backend, the grader base image
  runs there. A package machine always lowers onto the local backend, so on a host without one
  `lower` refuses it: `RolloutEngine.check` finds no factory for it. It
  picks the host's container backend for an image, the local backend for Linux processes without
  one, and ShellSim otherwise, so a lowered spec is bound to its host. A grader with an environment,
  other than a judge, takes a verifier machine; no other grader does. `limits` is a `RolloutLimits`
  from `rollout_limits(...)`, which sets no token budget. The spec names the harness `TOOL_HARNESS`,
  and `lower` rejects what `RolloutEngine.check` rejects (`draft.check`).
- `spec.draft.rollout_engine(factories, *, command_timeout)` is the `RolloutEngine` every Taskforge
  rollout runs on: the factories and RolloutEngine's `ShellHarness(command_timeout=...)` registered as
  `TOOL_HARNESS`. The shell command timeout belongs to the harness registration, not to the spec, and
  each run passes its model: `engine.run_task(spec, model)` returns a `GradedRollout`.
- `spec.controls.Control` is one labeled candidate submission (`kind`, `category`, `concern`,
  `author`, a `Transcript` or `Workspace` payload, an `Expectation`). `concern` is `reference`,
  `acceptance`, `extraction` or `shortcut`, and the concern a control may carry follows its
  category (`spec.controls.CONCERNS`). A task's control set needs a `reference`, an `acceptance`
  and a `shortcut` control, so its coverage never rests on extraction controls, which pin today's
  answer parser. `validate_controls(task, controls)` checks a set against its task.
- `sandbox.factories.machine_factories(where, controller_url)` returns the factories for
  `RolloutEngine`, keyed by shellbox `Backend` value: ShellSim plus Docker on a laptop,
  ShellSim plus gVisor on Iris, plus the local backend for lock-only graders when bubblewrap works
  in the Iris task. Before a trial starts, `task_refusals(spec, factories)` passes each machine of
  a `TaskRolloutSpec` to its factory's `check_environment` and lists a typed refusal, with the factory's
  message, for each machine without a factory or rejected by one. `warm_local_python(factories)`
  builds the local backend's bare CPython environment in a background thread, so the run's driver
  (the ladder that runs Taskforge on Iris), which calls it once at start, spares its first wave of
  local machines the five-second build; the build children share the cache.

Task-specific grading is the task's own code, private in `resources.verifier`, run by a
`ScriptGrader` in a separate verifier machine after the attempt. The verifier machine starts from
the task's image or, for a ShellSim task or a task without a machine, from the grader base
(CPython 3.12, `sh`, `setsid`, root), and receives the captured `output_paths` at their own paths
and the extracted answer at `/app/answer.txt`. A grader program writes its verdict, the reward and
one result per check, as `spec.verdict.verdict_json`. Generic modes that verifyit grades in
process are preferred whenever a task fits one.

## Model calls

`llm.recording` records every model call as an `LLM_CALL` span under a `CallLedger` (ledger, workpiece,
round, step), with tokens, finish reason and the call's attempt record (`attempts`,
`attempts_<outcome>` counts, non-200 `http_statuses`, `wall_time`). `recorded_complete` and
`recorded_structured` wrap one call each; `run_agent` and `GlmRolloutModel` record through
`llm_call_span`, so a rollout's 429s, retries and holds reach the ledger.

```
run_agent (agent turn)            GlmRolloutModel (RolloutEngine rollout)
        |                                     |
        +---------> llm_call_span <-----------+
                    (record: CallLedger)
                      |              |
                      v              v
          GlmClient.complete,     span(LLM_CALL) -> Ledger.record(LedgerEntry)
          retries and holds                          |-> JsonlLedger   <root>/<workpiece_id>.jsonl
                      |                              '-> FinelogLedger taskforge.ledger (under Iris)
                      v
          GLM-5.3 router: POST /v1/chat/completions (streamed)
```

`llm.agent.run_agent(client, policy, messages, tools, max_turns, record, replay)` is Taskforge's
own tool loop; `replay` defaults to `ReasoningReplay.KEEP`, which replays each assistant turn's
`reasoning_content`. `shell_tool` offers RolloutEngine's default shell tool on a shellbox
`Machine`, and `llm.web` adds Parallel search and extract. The client sends the policy's whole
`max_tokens` with every turn, so a prompt plus that budget beyond the context window raises
`ContextExhausted`, which ends the loop with `AgentStop.CONTEXT`; long loops set a smaller
`max_tokens`.

`llm.rollout_model.GlmRolloutModel(client, policy, record)` is the RolloutEngine model callable for
the messages trace. Each turn is one streamed chat-completions call, and the turn reports the reply
and the call's usage with empty token ids; the rollout model requests neither token ids nor logprobs,
and refuses a request that carries them (a `ModelRequest.continuation` or pinned prompt ids).
Under `TraceMode.MESSAGES` the engine accepts such a turn and charges its usage against the attempt's
token budget, which Taskforge leaves unset. Every request of a rollout carries `prompt_cache_key: <workpiece>:<round>:<step>`,
so it shares one node's prefix cache, and `GlmRolloutModel.for_step(step)` derives the model of
one rollout.

## Agentic generation

`agentic.agent.generate_tasks(idea, n, config)` makes up to n tasks for one idea with one managed
agent, in place of the stages. The agent works in a shell with Python, taskcompendium and verifyit,
and no network: a bubblewrap `LocalMachineFactory`, which needs the privileged container profile on
Iris. Its tools do only what the shell cannot:

- `grade`: grade a control answer with the task's grader.
- `rollout`: run solver or adversary rollouts on ShellSim machines.
- `web_search`, `web_fetch`: search and read the web with Parallel.
- `submit`: accept a task when its evidence satisfies the `Gate` and a blind review accepts it.
- `finish`: end the session with a report.

The checks of `submit` are in `agentic/tools.py`. The driver is
`experiments/post_training/capability_driven_envs/generate_tasks_agentic.py`.

## The build DAG

An author writes a build program as `async def` steps decorated with `@rl_env_build_step`, each
returning a value of its annotated type. The final step returns a `TaskSpec` and carries
`@rl_env_build_sink` in place of `@rl_env_build_step`. The author passes the steps, and the sink as
the keyword `sink`, to `rl_env_build_program`. That call orders the steps so each runs after its
inputs, puts the sink last, and refuses a step whose output reaches neither another step nor the
sink. A step receives an `RLEnvBuildContext` (workpiece id, model, sampling policy) as its first
parameter. The decorator reads the step's inputs from the parameters after it, and the runner
passes each one by name: another step's output, or the caller's proposal for a parameter called
`proposal`. `tests/stage40_build_with_program/test_run.py` builds this toy program, abridged here:

```python
@rl_env_build_step
async def numbers(context: RLEnvBuildContext) -> TaskResource:
    return inline_file("workspace/numbers.txt", b"6 7\n")


@rl_env_build_step
async def grader(context: RLEnvBuildContext) -> GraderPackage:
    return verifyit_package(NumericSpec(expected="42", tolerance_abs=0.0, tolerance_rel=0.0))


@rl_env_build_sink
async def task(context: RLEnvBuildContext, numbers: TaskResource, grader: GraderPackage, proposal: dict) -> TaskSpec:
    return TaskSpec(...)  # the prompt from proposal["task"], numbers as a worker resource, grader.grader


PROGRAM = rl_env_build_program(numbers, grader, sink=task)
```

`run_build` runs the program and writes `draft/task.json` and `draft/provenance.json`. The second
file lists every step, the sink last, with the stored memo of its output and whether this build hit
or missed the cache for it. The memo files in the store carry no status, because the store writes
a memo only on a miss.

Each step's output is memoized under a key built from `SDK_VERSION`, a source digest, the content
digest of each input, and a digest of the model and its sampling policy. The decorator computes the
source digest over the step's own source text and everything the step reaches in its own module:
the helpers it calls, the constants it reads, the classes in its return annotation at any depth,
and what those classes reach in turn. Names bound by imports stay out; a library type such as
`TaskSpec` falls under `SDK_VERSION`, so a library change comes with an `SDK_VERSION` bump.
Because inputs enter the key by content, a step that reruns and returns the same output leaves its
consumers cached.

## The stage machine

`run_workpiece` runs a workpiece through the stages with the functions a caller passes in
`StageFunctions`. Each stage returns an event with an exit, and `TRANSITIONS` in
`stage_coordinator/workpiece_machine.py` gives the next stage:

| stage    | event           | exit      | next     |
|----------|-----------------|-----------|----------|
| triage   | `TRIAGED`       | accept    | author   |
| triage   | `TRIAGED`       | reject    | done     |
| author   | `AUTHORED`      | authored  | build    |
| author   | `AUTHOR_FAILED` | revise    | author   |
| author   | `AUTHOR_FAILED` | reject    | done     |
| build    | `BUILT`         | built     | validate |
| build    | `BUILD_FAILED`  | revise    | author   |
| build    | `BUILD_FAILED`  | reject    | done     |
| validate | `VALIDATED`     | validated | review   |
| review   | `DECIDED`       | accept    | done     |
| review   | `DECIDED`       | reject    | done     |
| review   | `DECIDED`       | retry     | validate |
| review   | `DECIDED`       | abandon   | done     |
| review   | `DECIDED`       | modify    | author   |

From `done`, a `TERMINAL` event closes the workpiece as accepted, rejected or abandoned. A stage
raising an exception closes it as failed. `WorkpieceBudget` bounds the build and author failures a
workpiece may revise after and the review retries it may follow. The coordinator spends the budget
when it records the event, choosing revise or reject, retry or abandon. A build error costs a
revision, except `BuildInfrastructureFailure`, `ModelUnavailable` and `OSError`, which fail the
workpiece at the build stage. An author call raising `StructuredOutputError` costs a revision from
the same budget. Its structured reply stayed invalid through the repair turns, as when the output
limit cut it, and the next author call samples again with `Revision(AUTHOR_FAILURE, ...)`. Any other
author error fails the workpiece at the author stage.

The author returns an `AuthoredProgram`, the program with the Python source it came from. The
coordinator writes the source to `authored/<k>/program.py` in the round directory, with `k` the
workpiece's build and author failures so far, and the build stage loads the program from that file
with `StageFunctions.load_program`, so the author runs once per revision. Each author call writes a
new `authored/<k>/`, so a relaunch resuming at build finds the program its `AUTHORED` event records
even when a killed process re-authored the workpiece after the last save. A program defined in
Python, as `static_author` returns, has no source, and the build stage asks the author for it again.

The `OPENED` event records the digests of the proposal and the budget, and the coordinator's model
digest. A relaunch of the workpiece with another proposal, budget or model digest raises
`ValueError`, and so does a build whose program has another digest than the `AUTHORED` event.

A run root holds:

```
workpieces/<workpiece_id>/events.json
workpieces/<workpiece_id>/rounds/<n>/
    authored/<k>/program.json      the manifest of the program authored after k build and author failures
    authored/<k>/program.py        its source, when the author returned it
    draft/                         task.json, provenance.json
    build-failures/<k>.txt         the k-th build or author failure: its step and program line, then its traceback
    validation.json                the summary returned by validate
    decision.json                  the decision returned by review
store/                             the step store: blobs/, workpieces/<workpiece_id>/steps/
```

While a workpiece runs, its events stay in the caller's `WorkpieceEventLog`, and `run_workpiece`
writes `events.json` once, as the last file, when the workpiece reaches a terminal. On the next
`run_workpiece` with an empty log, the coordinator loads `events.json` back into the log and folds
it. An accepted or rejected workpiece then returns its terminal without calling a stage or writing a
byte. A retry with the caller's own log after a failed save of `events.json` writes the file and
returns. An abandoned workpiece goes back to validation with its retries reset, and a failed one
resumes at the stage that raised, with the step store serving the build steps already run. A build
failure in the program's own code spends a revision and sends the workpiece back to the author, and
so does an author call whose structured reply stays invalid after its repairs
(`StructuredOutputError`). The workpiece fails in place instead, with its revisions unspent, when:

- the build host fails (`BuildInfrastructureFailure`), and the relaunch builds the same program
  again;
- the model is unavailable (`ModelUnavailable`), at the author or the build stage;
- storage fails, or a blob or memo is missing or corrupt (`OSError`, with `StoreCorruption` for a
  corrupt one).

The relaunch then resumes at the stage that raised.

A process killed mid-workpiece leaves `events.json` as the workpiece's last terminal wrote it, or
leaves none, so the next run starts again from that terminal or from triage, and the step store
serves the build steps already run. A file-backed `WorkpieceEventLog` saving each event on append is
planned with the ledger.

## Testing

```bash
uv run --package marin-taskforge --frozen --group test pytest -c lib/taskforge/pyproject.toml \
    lib/taskforge/tests experiments/post_training/capability_driven_envs/tests
cd lib/taskforge && uvx --from 'pyrefly>=1.0.0,<1.1.0' pyrefly check
```

`tests/test_pipeline_walkthrough.py` exercises `run_workpiece` with a two-step toy program.

Live tests carry `@pytest.mark.live_glm`. The default marker expression excludes them, and they
skip with a reason when their inputs are missing. They call GLM-5.3's high pool through the router
port-forward:

```bash
KUBECONFIG=<open-athena kubeconfig> kubectl -n open-athena port-forward svc/glm53-router 18000:8000 &
export TASKFORGE_GLM_BASE_URL=http://127.0.0.1:18000/v1
export GLM_HIGH_TOKEN=<high-pool bearer token>
uv run --package marin-taskforge --frozen --group test pytest -c lib/taskforge/pyproject.toml \
    lib/taskforge/tests -m live_glm
```

`-m live_glm` replaces the default marker expression. The agent live tests also take
`TASKFORGE_PARALLEL_KEY_FILE`, a file with a `PARALLEL_KEY=...` line. Live checks write raw evidence
(request without the bearer token, full response, wall time) under `$TASKFORGE_EVIDENCE_DIR`, or
`taskforge-evidence/` in the system temp directory, outside the checkout.
