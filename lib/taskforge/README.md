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
                               run.py: InfrastructureRetry, `run_workpieces`
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
                               source.py: ProposalBatch, SlotFailure, the ProposalSource protocol
  stage20_triage/              checks.py: structural checks; preflight.py: `preflight`, the default triage;
                               verdict.py: Verdict and TriageDecision
  stage30_author/              program.py: build steps and the DAG they form;
                               author.py: Revision, AuthoredProgram, the Author protocol, `static_author`,
                               the program manifest;
                               loader.py: `load_program_source`, the restricted loader of authored programs
  stage40_build_with_program/  agentic_author.py: `author_round`, the agent session that writes, loads,
                               builds and grades a build program;
                               context.py: RLEnvBuildContext, BuildFailure, BuildInfrastructureFailure,
                               BuildExpired;
                               services.py: RLEnvBuildServices, AgentLimits;
                               outputs.py: the step outputs (Sources, Fixtures, Grader, Instructions, ...);
                               sandbox_effects.py: `build_machine`, `grade_candidate`, `publish_image`;
                               sdk.py: the functions, schemas and constants steps call;
                               program_sdk.py: `sdk_exports`, the SDK's names for the loader, and
                               `load_program`;
                               infrastructure.py: host failures told apart from program failures;
                               template/standard.py: the standard six-step program;
                               glm_author.py: GlmAuthor and its prompts;
                               step_store.py: step keys, memos, the memo store and `memoized`;
                               run.py: `run_program`, `run_build`, `build_model_digest`, the draft and
                               `WorkpieceBuildProvenance`
  stage60_review/              decision.py: the Decision types (Accept, Reject, Retry, Modify)
```

Each pipeline stage is a package named `stageNN_<name>`, numbered by tens in pipeline order:
proposal 10, triage 20, author 30, build 40, validate 50, review 60. The gaps leave room to insert a
stage later without renumbering. A stage imports from lower-numbered stages and from the shared
modules (`capability`, `child_process`, `content_hash`, `ledger`, `llm`, `sandbox`, `spec`), and
the shared modules import no stage. The build package is `stage40_build_with_program`: every way Taskforge
builds a task today goes through a build program, so the SDK, the standard template, the build
services, the program runner and the agentic author (an agent session that writes and iterates
that program) all live there. The name leaves room for a `stage40_build_with_agent` package in
which an agent produces the task draft directly, with no build program at all; that is a future
surface, not a home for anything that writes a program, however the writer runs. The coordinator
package, `stage_coordinator`, runs the stages in order and is unnumbered. It may import any stage
and shared module, and `tests/test_imports.py` refuses an import of it from any other package, along
with the other wrong-way imports. `scripts/ledger_summary.py` totals wall time and tokens per kind
and step over JSONL ledgers. `scripts/build_image_job.py` runs on Iris and documents its submit
command in its docstring: it builds a directory's `DockerBuild` context with
`taskforge.sandbox.image_builder` and prints the pinned registry digest.

## Proposals

A proposal source implements `stage10_proposal.source.ProposalSource[IdeaT]`:
`async propose(idea, n) -> ProposalBatch`. A batch holds the planning request and completions and
one `SlotProposal` or `SlotFailure` per slot, in slot order, so a failed slot does not drop its
siblings. `stage10_proposal.model.TaskProposal` is YAML front matter (`ProposalHeader`, keys in
`HEADER_KEYS` order) plus a markdown body with required section headings. `parse` and `render`
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
declared machine is checked by the build (`infrastructure.machine_available`) before any model
call, and a host that cannot ends the workpiece as the host's failure.

A consumer could add a model review to triage in one of two designs:

- A panel with repair: several independent model reviews of the proposal, a majority vote that
  accepts, rejects or asks for repair, and one model rewrite of a repair proposal, sent back to
  triage through the stage coordinator's repair edge.
- A bounded review with obligations: one model review that accepts the proposal and attaches
  obligations its author must honour, carried to the author as the verdict's `author_state`.

Neither is implemented by default, because experiments found them not to be meaningful predictors
of downstream quality. The coordinator keeps both surfaces. Its repair edge is a surface the
preflight or a consumer may use within `max_triage_repairs`: a REPAIR verdict has
`StageFunctions.repair` rewrite the proposal, such as a deterministic rewrite of a fixable proposal
header, and triage judges the rewrite. The edge may later retry a proposal on the findings of the
preflight; the preflight returns only accept or reject today. A verdict's `author_state` reaches
every author call of the workpiece; the preflight attaches none, and nothing in the package writes
one (see the stage machine).

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

## The build DAG

An author writes a build program as `async def` steps decorated with `@rl_env_build_step`, each
returning a value of its annotated type. The final step returns a `TaskSpec` and carries
`@rl_env_build_sink` in place of `@rl_env_build_step`. The author passes the steps, and the sink as
the keyword `sink`, to `rl_env_build_program`. That call orders the steps so each runs after its
inputs, puts the sink last, and refuses a step whose output reaches neither another step nor the
sink. A step receives an `RLEnvBuildContext` (workpiece id, round, model, sampling policy, services,
and the running step's name) as its first parameter. The decorator reads the step's inputs from
the parameters after it, and the runner passes each one by name: another step's output, or the
caller's proposal for a parameter called `proposal`. `tests/stage40_build_with_program/test_run.py`
builds this toy program, abridged here:

```python
@rl_env_build_step
async def numbers(context: RLEnvBuildContext) -> TaskResource:
    return inline_file("workspace/numbers.txt", b"6 7\n")


@rl_env_build_step
async def grader(context: RLEnvBuildContext) -> GraderPackage:
    return verifyit_package(NumericSpec(expected="42", tolerance_abs=0.0, tolerance_rel=0.0))


@rl_env_build_sink
async def task(context: RLEnvBuildContext, numbers: TaskResource, grader: GraderPackage, proposal: TaskProposal) -> TaskSpec:
    return TaskSpec(...)  # the proposal's prompt, numbers as a worker resource, grader.grader


PROGRAM = rl_env_build_program(numbers, grader, sink=task)
```

`run_build` runs the program and writes `draft/task.json` and `draft/provenance.json`. The second
file lists every step, the sink last, with the stored memo of its output and whether this build hit
or missed the cache for it, or reran it because the caller named it in `invalidate`. The memo files
in the store carry no status, because the store writes a memo only on a miss or a rerun.
`step_output(draft, store_root, name, output_type)` reads a step's output back.

Each step's output is memoized under a key built from `SDK_VERSION`, a source digest, the content
digest of each input, and `build_model_digest(model, sampling, services)`. The decorator computes
the source digest over the step's own source text and everything the step reaches in its own
module: the helpers it calls, the constants it reads, the classes in its return annotation at any
depth, and what those classes reach in turn. Names bound by imports stay out; a library type such
as `TaskSpec` falls under `SDK_VERSION`, so a library change comes with an `SDK_VERSION` bump.
The model digest covers the model id, the sampling policy, the agent loops' reasoning replay and
their limits (`AgentLimits`), so a build that drops reasoning, or bounds its agents differently,
reruns every step; `run_build(model_digest=...)` overrides it. The machines, images, web tools and
ledger in `RLEnvBuildServices` stay out of every key and out of every output.
Because inputs enter the key by content, a step that reruns and returns the same output leaves its
consumers cached. The runner memoizes each step through `step_store.memoized`, and a driver
memoizing a helper calls it the same way with a key of its own. Every step records a `STEP` span
with its key digest and cache status, and its model and tool calls carry its name as `step`; a
build without a ledger records nothing.

A step uses the host through `sandbox_effects`: `build_machine` prepares a machine as a task attempt
would, `grade_candidate` grades a candidate reply and workspace through `RolloutEngine.run_task`, and
`publish_image` pushes a task image. `infrastructure.host_checked_factories` wraps the factories
first, and `run_program` raises `BuildInfrastructureFailure` for a host fault anywhere in a step's
cause chain, even one the program wrapped, so the caller retries or abandons the workpiece instead of
asking the author for a revision. A host fault is a machine the host could not create or lost
(classified by the message table in `sandbox.host_failures`, which validation's classifier reads too),
an unreachable controller, or a startup or verifier deadline that expired while a sandbox started,
received files or ran a command (`SANDBOX_DEADLINE`): the grader's own command has its own timeout,
so the deadline is the sandbox's latency. A task image publish fails as the host's
(`IMAGE_JOB_FAILED`) when a build or push job did not schedule, was lost or killed, or the push
failed (`image_builder.ImageJobFailed`), and as the program's `BuildFailure` when the build job ran
and kaniko failed on the Dockerfile (`image_builder.BuildFailed`). `grade_candidate` grades once
more on fresh machines when the grader command timed out, and returns the second grade. A step-store
memo or blob that reads back unreadable or altered raises `step_store.StoreCorruption`, an `OSError`,
which the build child reports as a storage fault.

A build that runs past a clock raises `BuildExpired`, a third outcome beside `BuildFailure` (the
program's) and `BuildInfrastructureFailure` (the host's), carrying the `ExpiredClock` it ran past and
the step: `WALL` when `build_process.run_build_in_process` kills the build at its wall clock, `STEP`
when a step runs past `run.STEP_TIMEOUT`, and `GRADER` when the standard grader step's reference
timed out on both of `grade_candidate`'s gradings. Until someone reads the record nobody knows whether
Taskforge or the program was slow, so the library does not presume: the driver holds the workpiece as
abandoned, spends no revision, does not rebuild it on its own and asks the author for no rewrite.

`stage40_build_with_program/template/standard.py` is the standard program: five steps and the sink `assemble`.

```
sources(proposal) -> fixtures(proposal, sources) -> requirements(proposal, fixtures)
    -> grader(proposal, fixtures, requirements) -> instructions(proposal, fixtures, requirements, grader)
    -> assemble(proposal, fixtures, requirements, grader, instructions) -> TaskSpec
```

`sources` researches each research item of the proposal on the web, and
`run_build(research=ResearchRequirement.ENFORCED)` fails a build whose `Sources` lacks a finding for one of them.
`fixtures` writes the files for the solver machine the proposal declares, listing PyPI packages only for
a package machine and no files for `none`, and `requirements` builds that machine (`draft.NO_MACHINE`
for `none`). `run_build` refuses a task whose machine is not the declared one (`check_solver_machine`)
as a `BuildFailure`, the program's, so no revision can move the solver between no machine, ShellSim,
CPython and a container. Nor can the host: before anything runs,
`run_build` and the agentic author's `author_round` call `infrastructure.machine_available`, which
raises `NO_IMAGE_BUILDER` for a container proposal on a host without an image builder and `NO_FACTORY`
for a package machine on a host without the local backend, so the workpiece ends as the host's, with no
revision spent and no author adapting the task to the host.
`grader` grades its own reference answer and the empty answer with `grade_candidate` before it returns,
sends a failing grader back to the model, and after `sdk.ATTEMPTS` requests fails the build, so a
grader that rejects its own reference, returns a reward outside its score range, or credits the
empty answer above its declared `rejection_ceiling` is a build failure. A reference whose grader
timed out on both of `grade_candidate`'s gradings raises `BuildExpired` (`GRADER`) at once instead of
going back to the model, and the driver holds the workpiece as abandoned. The `Grader` it returns
holds the reference answer, the ceiling (the most a submission wrong in substance can earn under
its partial-credit scheme) and every candidate it graded. The agentic author's `run_build` tool
grades the empty answer in the library after every build, whatever the program did, and fails the
build in the grader step when it is graded above the ceiling; its digest reports that grade. `sdk.checked_structured` treats a reply
the output limit cut off as one failed submission and asks for a shorter one.

`stage40_build_with_program.glm_author.GlmAuthor` asks GLM for a program with the template as its example, and
returns an `AuthoredProgram` holding the program and its source, which
`stage30_author.loader.load_program_source` loads. The build stage comes after the loader's, so the
loader imports none of the SDK's modules: its callers here pass `program_sdk.sdk_exports()`, the
names the SDK modules define, which the loader's refusals suggest from. The loader refuses a module whose top level
holds anything but imports, classes, steps, the sink and a final
`PROGRAM = rl_env_build_program(..., sink=...)`, because a step's key covers its own source and its
output type alone, so an edit to a helper beside the steps would leave every key unchanged. It also
refuses an import outside `ALLOWED_IMPORTS`, a missing `grader` step returning `Grader`, and a
program without a sink made by `@rl_env_build_sink`. Each refusal is a `ValueError` whose message
the author receives.

`stage40_build_with_program.agentic_author.author_round` is the agentic author: one agent session (`run_agent`,
reasoning replay KEEP) per program round, working in a workspace that starts with the template as
`program.py`, the proposal, the SDK reference and the template under the read-only `reference/`, and
an empty `notes.md`. Its tools read, write and edit workspace files (`read_file`, `read_reference`,
`write_file`, `edit_file`, `list_files`), run the loader (`run_loader`), build the program with the
caller's `ProgramBuilder` (`run_build`, which must build as the caller's build stage does, on the same
step store, so a program that built in the session rebuilds there from memo hits alone), grade a
candidate against the last draft the session built (`run_grade_candidate`), and search and fetch the
web when the proposal lists research. Every load of a program (`run_loader`, `run_build`, `submit`)
goes through the session's `ProgramLoader`, by default `stage30_author.load_process` in a child process
bounded by `LOAD_TIMEOUT`, so a module body that never returns is refused rather than holding the
caller's event loop. Every tool result is cut to 6,000 characters, head and tail. A host failure never
reaches the model: `run_build` and `run_grade_candidate` retry a `BuildInfrastructureFailure` up to
`HOST_RETRIES` times with a doubling backoff (at once re-raising a cause in `HOST_REJECTIONS`), then
re-raise it like `ModelUnavailable`, so the round ends without a record and spends no revision; a
loader child the host killed raises it with cause `BUILD_PROCESS`. A `BuildExpired` from `run_build`
never reaches the model either: the tool writes `run.json` with `status: "expired"`, the `clock`, the
`step` and the message as `text`, and re-raises it at once, so the round ends the same way and the
driver holds the workpiece as abandoned. An `OSError` raised outside the program's frames, a storage
fault, is written with `status: "storage"` and re-raised at once, as the coordinator's build stage
re-raises it.
`submit(path, notes)` loads the file and refuses a loader failure or notes over 4,000 characters; a
program that never built in the session is refused once and handed in untested when submitted again;
an accepted submission ends the loop through `run_agent`'s `stop` callback. The session also stops
when a turn's prompt and output plus `turn_max_tokens` exceed `session_token_cap`, at the turn budget,
or after a third reply without a tool call (a reply cut at the output limit is nudged to take a
smaller step instead). It writes, under the workpiece's directory:

```
rounds/<n>/authored/<k>/
  author.json        author_kind, status (accepted, handed_off, author_failed), revision, session_dir,
                     mode, stop_reason, wall; accepted adds program_digest and tested
  session.json       turns, tool calls by tool, tokens, cached share, prompts, stop reason, nudges,
                     submissions, in-session builds
  transcript.json    the opening and user turns, then each assistant turn with its tool calls and results
  submit.json        accepted and every submit call with its outcome
  program.py         accepted only: the submitted file's bytes
  program.json       accepted only: program_manifest
  record.json        accepted only: GlmAuthor's record fields, with author_kind and turns
rounds/<n>/agentic/<k>/
  conversation.json  the messages as sent, read by a same-session revision
  workspace/         program.py, proposal.md, notes.md, reference/
  runs/<j>/          one per in-session build: draft/ and run.json (status built, build_failure,
                     crash, infrastructure, storage, or expired with its clock)
```

Round 0 starts a fresh session. After a build failure, the session that submitted the program
continues with the failure as its next user turn when it ended by `submit` within
`revision_same_session` (its last prompt and its turns used); any other revision starts a fresh session
from a copy of its workspace. A session that ends without a submission hands its workspace to a fresh
session in the same round once (`handed_off`), and the next unsubmitted one ends the round
`author_failed`; a session stopped by the context guard hands off up to four sessions per round. A
relaunch never resumes a session mid-way: a session directory without an `author.json` starts a handoff.

## The stage machine

`run_workpiece` runs a workpiece through the stages with the functions a caller passes in
`StageFunctions`. Each stage returns an event with an exit, and `TRANSITIONS` in
`stage_coordinator/workpiece_machine.py` gives the next stage:

| stage    | event           | exit      | next     |
|----------|-----------------|-----------|----------|
| triage   | `TRIAGED`       | accept    | author   |
| triage   | `TRIAGED`       | repair    | triage   |
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
raising an exception closes it as failed. `WorkpieceBudget` bounds the triage rewrites a workpiece
may take, the build and author failures it may revise after, and the review retries it may
follow. The coordinator spends the budget when it records the event, choosing repair or reject,
revise or reject, retry or abandon.

Triage returns a `Verdict`. The preflight's exits accept or reject, and a reject records the
verdict's reasons. REPAIR is the surface described under Triage, and a `max_triage_repairs` of zero
sends any REPAIR verdict to reject. On a REPAIR verdict within the budget, the coordinator calls
`StageFunctions.repair`, writes the rewrite to `repaired/<k>.md`, and records `TRIAGED` exiting
repair with the rewrite's digest, the triaged proposal's digest, the file and the rewrites spent.
Triage judges the rewrite, and every later stage runs it, read back from its file. A REPAIR verdict
past the budget exits reject, and so does a rewrite that is not a valid proposal; the workpiece
closes as rejected with the reason.

An accept records the verdict's `author_state` on its `TRIAGED` event: a JSON mapping, empty
unless the triage attached state such as findings or obligations for the author. The workpiece
state folds it from the event, so a relaunch reads the same state, and the coordinator passes it
to every author call as the `Author`'s third argument. The preflight attaches none, and the
package's authors ignore it.

A build error costs a revision, except `BuildInfrastructureFailure`, `BuildExpired`,
`ModelUnavailable` and `OSError`, which fail the workpiece at the build stage. An author call
raising `StructuredOutputError` costs a revision from the same budget. Its structured reply stayed
invalid through the repair turns, as when the output limit cut it, and the next author call samples
again with `Revision(AUTHOR_FAILURE, ...)`. Any other author error fails the workpiece at the
author stage.

The author returns an `AuthoredProgram`, the program with the Python source it came from. The
coordinator writes the source to `authored/<k>/program.py` in the round directory, with `k` the
workpiece's build and author failures so far, and the build stage loads the program from that file
with `StageFunctions.load_program`, so the author runs once per revision. Each author call writes a
new `authored/<k>/`, so a relaunch resuming at build finds the program its `AUTHORED` event records
even when a killed process re-authored the workpiece after the last save. A program defined in
Python, as `static_author` returns, has no source, and the build stage asks the author for it again.

`StageCoordinator` holds the build host's `RLEnvBuildServices` and the `ResearchRequirement`, and
passes them to every build with its model, sampling policy and model digest.

The `OPENED` event records the digests of the proposal and the budget, and the coordinator's model
digest and research requirement. A relaunch of the workpiece with another proposal, budget, model
digest or research requirement raises `ValueError`, and so does a build whose program has another digest than the `AUTHORED` event.

A run root holds:

```
workpieces/<workpiece_id>/events.json
workpieces/<workpiece_id>/repaired/<k>.md   the k-th rewrite answering a REPAIR verdict
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
- the build, a step or the grader runs past its clock (`BuildExpired`), which holds the workpiece
  until someone reads the record;
- the model is unavailable (`ModelUnavailable`), at the author or the build stage;
- storage fails, or a blob or memo is missing or corrupt (`OSError`, with `StoreCorruption` for a
  corrupt one).

The relaunch then resumes at the stage that raised.

A process killed mid-workpiece leaves `events.json` as the workpiece's last terminal wrote it, or
leaves none, so the next run starts again from that terminal or from triage, and the step store
serves the build steps already run. A file-backed `WorkpieceEventLog` saving each event on append is
planned with the ledger.

## Running many workpieces

`run_workpieces` in `stage_coordinator/run.py` runs every proposal of an async iterable of
`(workpiece_id, proposal)` pairs through `run_workpiece`, starting each as the iterable yields it,
so a proposal still being written holds back no workpiece. It sets no width: the stages' resource
pools bound how much runs at once. Each workpiece runs in a wrapper that records its terminal and
never raises, so a failed workpiece cancels none of its siblings.

A workpiece stopped by a transient infrastructure failure runs again in-process after a backoff,
resuming at the stage that raised, up to `InfrastructureRetry.max_attempts` attempts; each failed attempt
leaves a `TERMINAL` failed event in its log. Only two failures qualify
(`transient_infrastructure_failure`): `ModelUnavailable`, and a `BuildInfrastructureFailure` whose
cause is not one of `HOST_REJECTIONS` (no factory or no image builder on this host). A
`BuildExpired`, a storage fault, and every task or program failure leave the workpiece failed in
place after one attempt, as `run_workpiece` leaves it.

`run_workpieces` returns each started workpiece's terminal once the iterable is exhausted and every
workpiece has one. A workpiece id yielded a second time raises `ValueError`, and an error reading the
iterable is raised as it is; either way no later proposal starts, and the workpieces already started
first run to their terminals.

## Testing

```bash
uv run --package marin-taskforge --frozen --group test pytest -c lib/taskforge/pyproject.toml \
    lib/taskforge/tests experiments/post_training/capability_driven_envs/tests
cd lib/taskforge && uvx --from 'pyrefly>=1.0.0,<1.1.0' pyrefly check
```

`tests/test_pipeline_walkthrough.py` exercises `run_workpiece` with a three-step toy program.

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
