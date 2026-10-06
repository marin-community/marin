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
  validate/   trials, the failure classifier, control replay, evidence aggregation, solver and
              adversary trials, calibration, attempt files as resumable evidence
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
`load_draft`, `item_id_for`), `build.author` (`BuildProgram`, `Revision`), `validate.outcome`,
`validate.evidence`, `validate.calibration` (`CalibrationSummary`, `Finding`) and `review.decision`
(`Decision`). `tests/test_imports.py` parses every module and fails on an import that points right
in the order, or on a package the order does not name.

## Seams

- `llm.recording.recorded_complete(client, messages, policy, request_fields, record, attrs)` records
  one GLM call as an `LLM_CALL` span under `record` (a `CallLedger`: ledger,
  item, round, step) with tokens, finish reason, and the call's attempt record (`attempts`,
  `attempts_<outcome>` counts, non-200 `http_statuses`, `wall_time`). `run_agent`,
  `GlmRolloutModel` and `BuildLLM.complete` use it, so a rollout's 429s, retries and holds reach
  the ledger. `llm.recording.recorded_structured(client, messages, policy, tool, record, attrs)` is
  its structured-output counterpart: one span over the request and its repair, with summed tokens
  and attempts and a `requests` count. `BuildLLM.structured` and the build author use it, so every
  model call a build makes records its attempts.
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
  a CONTROLS step, and the controls must pass `spec.controls.validate_controls`. The SDK reference
  lists the concerns each control category allows; every stage needs a reference, an acceptance
  and a shortcut control. The draft holds the task, its `TaskExecution` and its controls. A host
  failure during the build raises
  `build.infrastructure.BuildInfrastructureFailure` with a `cause` of `no_factory`,
  `scheduling_timeout` or `host_unreachable`, even when the program wrapped it; it is never a
  `BuildFailure`, and the caller does not charge it to the program.
  `build.infrastructure.HOST_REJECTIONS` names the causes that hold for as long as the host is
  unchanged (`no_factory`).
- `validate.trials.run_trials(task, execution, plan, settings, model)`: runs k trials through
  `ShellboxRolloutEngine`. Each trial is `Graded` or `Ungraded` with one typed `Cause`, and
  `validate.classify.classify` is the only failure classifier. `TrialPlan.first_attempt` numbers
  each trial's first attempt file and ledger step, so a re-entered trial continues after the
  attempts on disk.
- `validate.controls.replay(task, execution, controls, plan, settings, tokenize)`: replays each
  control as one `CONTROL` trial named by its id. `ControlPlan.first_attempts` maps a control id
  to its first attempt number (0 when absent), with the same re-entry contract as a trial.
- `validate.run`, `validate.solver`, `validate.adversary`: one validation round of a `TaskDraft`
  under a `ValidationPolicy` (every field required) at a `ValidationSite(item_id, round,
  evidence_dir, ledger)`. `replay_controls` runs first; only when `controls_passed`, `run_solver`
  (`k` trials) and `run_adversaries` (`adversary_k` trials per `AdversaryRole`) run concurrently.
  Every trial runs under the draft's own convention. The evidence directory holds
  `control/<id>/`, `solver/<index>/` and `adversary/<role>/<index>/` attempt files;
  `load_validation(draft, evidence_dir)` reads a round back as `ValidationEvidence`. `run_solver`
  takes a `solver.ModelFactory` (`Callable[[CallLedger], RolloutModel]`, for GLM
  `partial(GlmRolloutModel, client, sampling)`) and builds each trial's model with
  `site.call_ledger(kind, trial)`, so its `LLM_CALL` spans carry step `solver/<index>`.
  `replay_controls` resumes unsettled controls through `ControlPlan.first_attempts`. Each adversary
  trial is an agent loop (`llm.agent.run_agent`) on its own prepared task machine with a `shell`
  tool and a `submit` tool that grades a candidate (reply plus listed workspace files) through
  `ShellboxRolloutEngine.grade_state` on a fresh machine and returns the grade;
  `ValidationPolicy.adversary_submissions` bounds the verifier calls per attempt. A candidate is
  graded as assistant text, so under a convention that submits through a tool call (`AnswerCall`,
  `FinalAction`) each trial is one refused `SUBMISSION_UNSUPPORTED` attempt
  (`adversary_convention`). The brief
  (`adversary_brief(submissions, context)`) is the system turn, persisted as `adversary.system` in
  the attempt file beside every submission (`attempts.load_adversary_attempt`); `run_adversaries`
  takes the run's `GlmClient` and the consumer's `AdversaryContext` text, and records its model and
  tool calls under step `adversary/<role>/<index>`.
- `validate.calibration.summarize(evidence, policy) -> CalibrationSummary`: pure. It records the
  policy digest and band, the solver's `RewardStats`, the control verdicts, `RoleStats` per role
  and a closed set of `Finding`s (`FindingKind`; `DECISIVE` ones cannot improve by retrying).
  `write_summary`/`load_summary` round-trip it as `calibration.json`. `summarize` tiers every
  graded adversary trial into a `DefectTier` (`REPAIR`, `NOTED`, `NONE`) from its submissions
  (`AdversarySignals`, read against the draft's `TaskFacts`: files supplied, the submission count
  to the claimed exploit against `adversary_repair_submissions`, the parsed `Claim`) and the shell
  transcript's input reads; REPAIR trials are findings that ship the accepted candidate as a
  control (`candidate_control`), NOTED trials are `CalibrationSummary.notes`, every assessment is in
  `CalibrationSummary.assessments`, and `RoleStats` counts passes, submissions, budget use, claims,
  failed audits, budget stops and tiers.
- `review.rules.decide(draft, summary, history) -> Decision`: a pure rule table over a
  `CalibrationSummary`. The `Decision` is `Accept`, `Reject(kind, reasons, summary)` with kind
  `TASK`, `BUDGET` or `HOST`, `Repair(program_digest, brief, invalidate)` or `Retry(cause, count)`,
  and `review.decision.write_decision` and `load_decision` keep it in `decision.json` beside
  `calibration.json`. `review.rules.staged_repair` is the fixed decision for a staged draft.

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
running through `Machine.run` and Parallel search and extract from `llm.web`. Builder agents and
adversary trials run on it. Solver and control rollouts run on RolloutEngine with the same
`GlmClient`. `web_fetch`
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

Attempt files are the validation evidence. `validate.attempts.load_outcome` is the inverse of
`trials.outcome_json`, so a round is rebuilt from its files and there is no second copy of the
outcomes. A trial is settled when its last attempt is graded or ungraded for a cause outside
`RERUNNABLE` (`RETRYABLE` plus `TOKEN_CONTRACT`). A re-entered round loads settled trials and
re-runs the others with `TrialPlan.first_attempt` set to the attempt count on disk, so a crash at
hundreds-wide repeats no settled rollout and no attempt file is overwritten. The evidence
directory is keyed by `task_digest(task, execution, convention)`, so evidence is never read
against a different draft.

Controls run before any sampled trial: a violated control means the grader is wrong, and rollouts
against it would measure the wrong grader. An adversary trial is an agent loop that red-teams the
grader with the verifier as a tool. The first protocol wrapped the solver's model in a role
preamble and graded one final reply: the adversary had no feedback loop and, as one put it, got
"exactly one shot". Over 30 trials of five rounds it passed once, by solving honestly; four
shortcut trials overwrote the input file blind and were graded 0 because the grader compares
against a constant they could not see; the leak role never found anything in 40 trials because
the grader is installed only at grading time. The `submit` tool answers the question each role
was guessing at. It grades a candidate (the final reply plus files the adversary lists) on a fresh
machine through `grade_state`, so the grader stays private and the adversary's shell state is
never graded, and `adversary_submissions` bounds the calls. One role remains: the leak and
ambiguity surfaces are named in the one brief, and the role enum keeps the evidence layout so a
later role is additive. The brief lets the adversary compute what the task entails and probe with
honest answers; what it reports, on its last line (`NO_SHORTCUT` or `SHORTCUT: <why>`), must be an
accepted submission that violates the spirit of the task or skips its intended computation. A
trial is tiered by a coded rule table, never by a model judge. A passing submission that supplied
a task input or grader file, passed with no files on a machine-graded task, came from a session
that never read an input, or (on a single-answer grader) is not the honest answer is a defect to
repair whatever the adversary says. Otherwise the claim decides, against the consumer's
threshold: a claimed shortcut accepted within `adversary_repair_submissions` verifier calls is a
repair, one found later is a note, a pass without a verdict is a note, a claimed shortcut that is
the honest answer is a failed audit, and honest probes reported as `NO_SHORTCUT` are no defect. A
repair ships the accepted candidate as a negative control with concern `shortcut`
(`reward_max = REJECTION_CEILING`) that the revised program must ship, so the next round's control
replay proves the fix. A many-answer grader (`shell`, `judge`, script graders) that accepts text no
honest run produced is a note when the adversary reports no shortcut, because such graders accept
many outputs by design. Grader diagnostics are kept in the record and withheld from the model,
because a grader may print the expected value. Adversary rollouts are evidence, never training
data, and carry no token ids. Notes travel with the summary and never block an accept. Review
consumes findings through rules.

`ValidationPolicy.retry_backoff` is a `RetryBackoff` dataclass (`ExponentialBackoff`'s four
constructor arguments) rather than an `ExponentialBackoff`, so the policy digests and serializes it
without reading rigging's private state; each plan builds its own schedule from it.

The functions that take a `ValidationPolicy` or `ValidationEvidence` outside `validate.run` type
those parameters as protocols (`solver.TrialPolicy`, `adversary.AdversaryPolicy`,
`calibration.SummaryPolicy`, `calibration.RoundEvidence`), because `ValidationPolicy` holds a
`CalibrationBand` and the adversary brief, and the modules defining those take the policy.

Review calls no model. Its rules, in order: decisive findings (a violated control, a shortcut or
leak pass, an ambiguous instruction, a task defect) repair the program even when some trials are
ungraded, because retrying cannot improve them; trials this host cannot run reject the item as
`HOST`, not as a task defect; other ungraded trials retry; a solve rate outside the band gets one
repair per direction, then rejects the task; clean evidence accepts. A repair past the item's
budget rejects as `BUDGET`. A `Repair` carries a brief, not a patch: the loop passes
`brief.failure` to `build.author.author` as `Revision.failure`, so the author is the only model
that writes builder code. The brief renders adversary passes as controls the revised CONTROLS
step must return verbatim, so the next round's control replay checks the fix. `invalidate` names
the steps whose roles the findings condemn, so a model-driven step the author left unchanged is
resampled rather than replayed from the step cache. Staged tasks are not validated; a staged
draft gets one fixed repair asking for separate single-stage tasks.

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
