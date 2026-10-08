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
  sandbox/    MachineFactory selection per host, up-front task refusals, image builds
  proposal/   the TaskProposal document (model.py), the ProposalSource protocol (source.py), sources/
  triage/     structural checks, the GLM rubric, verdicts
  builder/    builder programs: memoized steps, the Build SDK, program authoring, the standard template
  validate/   trials, the failure classifier, control replay, evidence aggregation, solver and
              adversary trials, calibration, attempt files as resumable evidence
  review/     the Decision contract and the rules that derive it from validation evidence
  loop/       the per-item program, its policy, and the event log that item status is derived from
  queue/      the unattended run: GLM endpoint configuration, hundreds-wide scheduling, the Iris job
scripts/      Iris image builder, cluster probes, ledger summary
docker/       grader-base image build context
```

Packages are totally ordered. A package imports only from packages to its left and from external
packages, so no import cycle can form:

```
content_hash -> atomic_file -> ledger -> sandbox -> spec -> llm -> proposal -> triage -> builder -> validate -> review -> loop -> queue
```

The foundation packages (`content_hash`, `atomic_file`, `ledger`, `sandbox`, `spec`, `llm`) import
only each other and external packages. A later stage reaches an earlier one through its seam
modules, each of which keeps its types beside the code that checks their invariants:
`proposal.model` (`TaskProposal`), `proposal.source` (`ProposalBatch`, `SlotFailure`,
`ProposalSource`), `triage.verdict` (`Verdict`, `TriageDecision`), `builder.run` (`TaskDraft`,
`load_draft`, `item_id_for`), `builder.author` (`BuildProgram`, `Revision`), `validate.outcome`,
`validate.evidence`, `validate.calibration` (`CalibrationSummary`, `Finding`), `review.decision`
(`Decision`), `loop.events` (`EventKind`, `ItemState`) and `loop.policy` (`LoopPolicy`).
`tests/test_imports.py` parses every module and fails on an import that points right in the order,
or on a package the order does not name.

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
- `spec.draft.assemble(task_id, instruction, answer_type, answer_format, grader, source, *,
  environment, files, output_paths, system, final_tools, tags) -> TaskSpec`: the semantic task.
  `answer_format` (a TaskCompendium `AnswerFormat` such as `PlainText()` or `AnswerCall()`) says how
  the final answer is requested and read. `environment` is `requirements(image=, setup=, workdir=,
  env=)` or `None` for a task without a machine; `files` are agent-visible `file("workspace/x", ...)`
  resources relative to the machine root; `grader` is a `GraderPackage` from `answer_grader(spec)`
  (a verifyit mode graded in process, or in a verifier machine given `environment=`),
  `python_grader(script, config, environment=, answer_path=, timeout=)` (`python3 /tests/grade.py`
  in a verifier machine) or `script_grader(argv, reward, environment=, answer_path=, timeout=, ...)`.
  `grader_environment(task_image)` is the verifier machine's environment: the task image, or
  `sandbox.images.GRADER_BASE_IMAGE` for a task without one.
- `spec.draft.lower(task, *, host, task_machine, verifier_machine, session, factories) ->
  LoweredTaskSpec`: the only place Taskforge builds a lowered spec. `machine(...)` gives each
  machine's settings and `session(...)` the turn budget and deadlines; `lower` picks ShellSim for a
  machine without an image or packages lock, the local backend for a lock alone, and the host's
  container backend for an image, so a lowered spec is bound to its host. A grader with an
  environment takes a verifier machine; no other grader does.
- `spec.controls.Control`: one labeled candidate submission (`kind`, `category`, `concern`,
  `author`, a `Transcript` or `Workspace` payload, an `Expectation`). `concern` is a required
  `ControlConcern`: `reference`, `acceptance`, `extraction` or `shortcut`. A `Workspace` holds
  files relative to the machine root and needs a task machine. `validate_controls(task, controls)`
  checks a set against its task; `controls_json` and `parse_controls` round-trip it.
- `sandbox.factories.machine_factories(where, controller_url, image_cache)`: the factories for
  `ShellboxRolloutEngine`, keyed by shellbox `Backend` value as `MachineRuntimeSpec.backend` names
  them: ShellSim plus Docker on a laptop, ShellSim plus gVisor on Iris (`container_backend(where)`),
  plus the local backend for lock-only graders when bubblewrap works in the Iris task.
  On Iris it takes the controller URL and no image cache; on a laptop it takes the directory where
  the Docker factory keeps Skopeo-prepared images and no controller URL. The caller names that
  directory. `task_refusals(lowered, factory_capabilities(where))` lists every typed reason a
  lowered task cannot run on them.
- `proposal.source.ProposalSource[IdeaT]`: `async propose(idea, n) -> ProposalBatch`. A batch holds
  the planning request and completions and one `SlotProposal` or `SlotFailure` per slot, in slot
  order. A failed slot does not drop its siblings.
- `proposal.model.TaskProposal`: YAML front matter (`ProposalHeader`) plus a markdown body with
  required section headings. `parse` and `render` round-trip it; `digest` is the sha256 of the
  canonical form.
- `triage.program.evaluate(proposal, checks, rubric, ctx) -> Verdict`: structural checks run
  first, and a fatal failure or a null proposal rejects without a model call. The rubric scores
  independent samples, and the decision is ACCEPT or REJECT by strict majority, otherwise REPAIR.
- `builder.run.run_build(program, proposal, ...)`: runs a builder program of memoized async steps.
  A step's memo key covers its code, arguments, the data globals it reads, `SDK_VERSION`, the
  proposal and the model policy. The grader must come from a GRADER step and the controls from
  a CONTROLS step, and the controls must pass `spec.controls.validate_controls`. The SDK reference
  lists the concerns each control category allows; a task needs a reference, an acceptance and a
  shortcut control. The program lowers its task with `Build.lower` (`spec.lower` for the build's
  host), and `run_build` checks the result with RolloutEngine's `validate_lowered_task`. The draft
  (`TaskDraft(task, lowered, controls, provenance)`; the task carries its answer format) is written
  as `task.json`, `lowered.json`, `controls.json` and, last, `provenance.json`; `load_draft` reads it
  back. A draft is bound to its host: `lowered.json` names that host's machine backends. Images are
  published by digest through `BuildServices.images` (`Build.publish_image`). A host failure during
  the build raises `builder.infrastructure.BuildInfrastructureFailure` with a `cause` of
  `no_factory`, `no_image_builder`, `scheduling_timeout` or `host_unreachable`, even when the
  program wrapped it; it is never a `BuildFailure`, and the caller does not charge it to the
  program. `builder.infrastructure.HOST_REJECTIONS` names the causes that hold for as long as the
  host is unchanged (`no_factory`, `no_image_builder`); the loop abandons such an item without
  retrying.
- `validate.trials.run_trials(lowered, plan, settings, model)`: runs k trials of a
  `LoweredTaskSpec` through `ShellboxRolloutEngine`. `TrialPlan.deadlines` (`total_turn_timeout`,
  `attempt_timeout`) and `EngineSettings` (turn, command, tool-turn, model-turn and cleanup limits)
  replace the builder's session values, and `task_digest(lowered)` hashes what runs, answer format
  included. Each trial is `Graded` or `Ungraded` with one typed `Cause`, and
  `validate.classify.classify(failure, task)` is the only failure classifier. `TrialPlan.first_attempt` numbers
  each trial's first attempt file and ledger step, so a re-entered trial continues after the
  attempts on disk.
- `validate.controls.replay(lowered, controls, plan, settings, tokenize)`: replays each
  control as one `CONTROL` trial named by its id. `ControlPlan.first_attempts` maps a control id
  to its first attempt number (0 when absent), with the same re-entry contract as a trial.
- `validate.run`, `validate.solver`, `validate.adversary`: one validation round of a `TaskDraft`
  under a `ValidationPolicy` (every field required) at a `ValidationSite(item_id, round,
  evidence_dir, ledger)`. `replay_controls` runs first; only when `controls_passed`, `run_solver`
  (`k` trials) and `run_adversaries` (`adversary_k` trials per `AdversaryRole`) run concurrently.
  Every trial presents the task in its own answer format (`TaskSpec.answer_format`). The evidence directory holds
  `control/<id>/`, `solver/<index>/` and `adversary/<role>/<index>/` attempt files;
  `load_validation(draft, evidence_dir)` reads a round back as `ValidationEvidence`. `run_solver`
  takes a `solver.ModelFactory` (`Callable[[CallLedger], RolloutModel]`, for GLM
  `partial(GlmRolloutModel, client, sampling)`) and builds each trial's model with
  `site.call_ledger(kind, trial)`, so its `LLM_CALL` spans carry step `solver/<index>`.
  `replay_controls` resumes unsettled controls through `ControlPlan.first_attempts`. Each adversary
  trial is an agent loop (`llm.agent.run_agent`) on its own prepared task machine with a `shell`
  tool and a `submit` tool that grades a candidate (an answer plus listed workspace files) through
  `ShellboxRolloutEngine.grade_state` on a fresh machine and returns the grade;
  `ValidationPolicy.adversary_submissions` bounds the verifier calls per attempt. The candidate's
  final turn is rendered as the task's answer format submits an answer (`submission_turn`): assistant
  text, an `AnswerCall` answer call, or under `FinalAction` the function calls `submit` lists
  (`calls`), each a final tool of the task. The brief
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
- `review.rules.decide(draft, summary, history, rules: BandRules) -> Decision`: a pure rule table
  over a `CalibrationSummary`. `BandRules` holds one `BandRule(repairs, then: BandChoice)` per band
  kind and `ItemHistory.band_repairs` counts the repairs each kind has had. The `Decision` is
  `Accept(summary, band)`, where `band` (`BandOutcome`) records where the task fell,
  `Reject(kind, reasons, summary)` with kind
  `TASK`, `BUDGET` or `HOST`, `Repair(program_digest, brief, invalidate)` or `Retry(cause, count)`,
  and `review.decision.write_decision` and `load_decision` keep it in `decision.json` beside
  `calibration.json`.
  `RepairBrief` holds the `findings` the author must fix, the `notes` (noted adversary passes from
  `CalibrationSummary.notes`) it shows as information, and the rendered `failure`;
  `review.rules.render_brief(findings, notes)` builds it.
- `loop.policy.LoopPolicy`: every bound of a run (proposals per idea, idea re-proposals, triage
  repairs, build revisions, review repairs, validation retries, build host-failure retries
  (`max_build_retries`) and their shared backoff, the output-token
  budget, the `band_rules`, the `ValidationPolicy`), with no defaults. `band_rules` is a
  `review.rules.BandRules`: per band kind a `BandRule(repairs, then: BandChoice)` that review's
  `decide` applies, so the consumer chooses whether a task still outside the band is accepted or
  rejected. The `ValidationPolicy`'s `adversary_submissions` and `adversary_repair_submissions` reach
  validation and calibration through it. `loop.policy.POLICY` reads and writes it as the
  run's `policy.json`; a missing or unknown key is an error at any depth.
- `loop.events`: `record_event(ledger, item_id, round, kind, seq, input_hash, **attrs)` writes one
  `EntryKind.EVENT` row with `attrs["seq"]` and `attrs["schema"] = EVENT_SCHEMA`.
  `derive_state(entries) -> ItemState` folds a proposal item's events (phase, round, digests,
  counters, `Terminal`); `derive_idea_state` folds an idea's; `item_tokens_out` sums the item's
  `LLM_CALL` output tokens outside validation trials (steps `solver/...` and `adversary/...`;
  `is_trial_step(step)` requires the `/`, so a build step named `solver` still counts). A build the machine host failed is a `BUILD_INFRASTRUCTURE` event
  (`cause`, an `InfrastructureCause`); `build_host_failures(entries) -> Counter[InfrastructureCause]`
  counts an item's over every launch, and an `ABANDONED` terminal carries `causes`
  (`cause:count` pairs) for the retries it spent. `ADVERSARIES_RUN` counts per role the graded
  trials, the passing ones, their verifier submissions and their `SHORTCUT` claims
  (`<role>_graded`, `<role>_passes`, `<role>_submissions`, `<role>_claimed`) and records
  `context_digest`, the sha256 of the consumer's adversary context (`""` without one). `DECIDED`
  names the kinds of the noted adversary passes in `notes`; an accept also carries `band` (a
  `review.decision.BandOutcome`) and the synthesis pass rate (`solved`, `graded`, `solve_rate`), and
  its `TERMINAL` reason is `calibrated` or `accepted outside the band: <band>`. `ItemState.band_repairs`
  counts the review repairs each band kind has triggered.
- `loop.program.run_idea(idea_id, idea, policy, services) -> tuple[TaskProposal, ...]` and
  `run_item(proposal, origin, policy, services) -> Terminal`: one idea's proposals, and one proposal
  carried to `ACCEPTED`, `REJECTED`, `ABANDONED` or `FAILED`. `origin` is a `loop.events.ProposalOrigin`
  (`GENERATED` for a proposal from a `run_idea` batch, `SUPPLIED` for one handed to the loop
  directly), recorded on `OPENED`. `LoopServices[IdeaT]` holds what a run's items share, including
  the `slots` semaphore that bounds model- and sandbox-bound phases across items, `rollout_models`,
  the `validate.solver.ModelFactory` each solver trial's model comes from, `adversary_context` (a
  `validate.adversary.AdversaryContext`: the consumer's section of the adversary brief for an item's
  proposal, `""` for none; adversary trials run their agent loop on `client`), and `describe_idea:
  Callable[[IdeaT], Mapping[str, object]]`, the JSON record `run_idea` writes once to
  `items/idea--<id>/idea.json`. `run_idea` keeps each batch under
  `items/idea--<id>/batches/<reproposal>/`: `plan/{request,completions}.json` and
  `slots/<slot>/{request,completions}.json` with `repair_error.txt` or `failure.txt`, completions in
  the shape of the author's `completions.json`. Both resume from the run root's event logs.
- `queue.run.run_queue(ideas, policy, services, failed) -> RunSummary`: runs every idea and every
  unfinished item of the run rooted at `services.root` in one asyncio process. `services.slots` (the
  run's width) bounds concurrent phases, not requests. An item whose log ends `ACCEPTED` or
  `REJECTED` is skipped, `FAILED` is skipped unless `failed` is `FailedItems.RETRY`, and `ABANDONED`
  re-enters. One item's exception is recorded in `RunSummary.failed` and never cancels a sibling.
  `RunSummary` also counts ungraded trial attempts by cause, `GlmUnavailable` outside trials, and
  build host failures by `InfrastructureCause` over its items' `BUILD_INFRASTRUCTURE` events
  (`build_infrastructure`). `RunSummary` is the run's export of accepted tasks, written as
  `summary.json`: `accepted` maps every `ACCEPTED` item to an `AcceptedTask` (its round, task digest
  and draft directory, the synthesis pass rate of the calibration summary it was accepted on:
  `solved`, `k`, `solve_rate`, and the `review.decision.BandOutcome` of the `Accept` decision), and
  `noted` maps every item whose final decision carries a calibration summary to its `NOTED`-tier
  adversary passes (role, trial, rule, reason).
  Both are read from the item directories, so a relaunch exports items an earlier launch finished.
- `queue.job.run_job(config, inputs, failed)`: the laptop and Iris boundary. `queue.config.RunConfig`
  (`load_run_config`; every field required) names the host and its `image_cache` (a directory on a
  laptop, `null` on Iris), the GLM endpoint as `LaptopGlm` or
  `RelayGlm` with an explicit `Pool`, the builders' Parallel key source, the `LoopPolicy`, the
  `EngineConfig` (the turn, command, tool-turn, model-turn and cleanup limits of `EngineSettings`,
  and the conventions), the width and `restore_from`. The run's builders get the host's factories
  and no `ImageBuilder`, so a build that publishes a task image is abandoned with
  `no_image_builder`. Local paths (a laptop root, `image_cache`, a token
  or key file) are absolute; the config reader expands no `~` and resolves nothing against the
  working directory. `inputs` is an `InputsFactory`: a function of the
  run's `GlmClient` and run root that returns `RunInputs(ideas, source, describe_idea,
  adversary_context, checks, rubric, check_context)`; `adversary_context` is the consumer's section of
  each item's adversary brief (`""` for none). `scripts/run_queue.py` runs it from a config file.

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

Task specs are upstream's. A built task is a TaskCompendium `TaskSpec` and the `LoweredTaskSpec`
that says how it runs. `spec.draft.assemble` builds the semantic task and `spec.draft.lower` alone
builds the lowered record, so a move of either is a single-site change.
Task-specific grading is the task's own code, private in `resources.verifier`, run by a
`ScriptGrader` in a separate verifier machine after the attempt. The verifier machine starts from
the task's image, or for a ShellSim task or a task without a machine from Taskforge's grader base
(`docker/grader-base`: CPython 3.12, `sh`, `setsid`, root), and receives the captured
`output_paths` at their own paths and the extracted answer at `/app/answer.txt`. Generic grader
modes (exact, numeric, mcq, math, ifeval, JSON schema, structured answers, predicted actions and
the rest of verifyit's in-process modes) grade in process without a machine and are preferred
whenever a task fits one; they belong to `lib/verifyit`. A gap in either is fixed upstream.

Every control names the part of the grader it exercises. Answer extraction is expected to move to
a cheap model, so controls that only pin today's parser carry `concern = extraction` and can be
found and retired together. A task's control set needs a `reference` control, an `acceptance`
control and a `shortcut` control, so the required coverage never rests on extraction controls alone. The concern
a control may carry follows its category (`spec.controls.CONCERNS`): `reference` only on
known-correct controls, `shortcut` only on shortcut and reward-hack controls.

Execution is RolloutEngine's. `ShellboxRolloutEngine.run` takes a `LoweredTaskSpec`: a
TaskCompendium `TaskSpec` (schema 0.25, which carries the task's grader and answer format) with the
`TaskRuntimeSpec` and `TaskSessionSpec` that say how it runs. Each attempt creates a shellbox
`Machine` from the caller's `MachineFactory` named by the task machine's `MachineRuntimeSpec.backend`,
installs the task's resources, runs its setup commands, drives the shell tool, grades (in process for
a verifyit grader without an environment, otherwise on the lowered verifier machine) and closes. It
grades the state an agent left when the `total_turn_timeout` expires, and bounds every cleanup
action by the lowered `cleanup_timeout`. Taskforge supplies the model callable
(`llm.rollout_model.GlmRolloutModel`).
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
message. `Build` wraps its machine factories (`builder.infrastructure.host_checked_factories`): a
missing factory for a backend the host should have, a missing image builder, a `TimeoutError`
from the factory's own wait, and a connection or controller transport error from creating or
driving a machine are infrastructure. Everything else is the program's, including an image it
built or named that fails, a spec the backend refuses, and a deadline the program set with
`spec.machine(startup_timeout=...)`. Shellbox raises a bare
`RuntimeError` for both a failed Docker build and an unreachable Docker daemon, so the daemon case
is charged to the program until shellbox types it.

Attempt files are the validation evidence. `validate.attempts.load_outcome` is the inverse of
`trials.outcome_json`, so a round is rebuilt from its files and there is no second copy of the
outcomes. A trial is settled when its last attempt is graded or ungraded for a cause outside
`RERUNNABLE` (`RETRYABLE` plus `TOKEN_CONTRACT`). A re-entered round loads settled trials and
re-runs the others with `TrialPlan.first_attempt` set to the attempt count on disk, so a crash at
hundreds-wide repeats no settled rollout and no attempt file is overwritten. The evidence
directory is keyed by `task_digest(lowered)`, the digest of the lowered task with its answer
format, so evidence is never read against a different draft.

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
replay proves the fix. A many-answer grader (a script grader or a verifyit judge mode) that accepts text no
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

Review calls no model. Its rules, in order: decisive findings (a violated control, an adversary
shortcut tiered as a repair, a task defect) repair the program even when some trials are ungraded,
because retrying cannot improve them; trials this host cannot run reject the item as `HOST`, not as
a task defect; other ungraded trials retry; a solve rate outside the band gets the repairs its
kind's `BandRule` allows, then the consumer's `BandChoice` decides; clean evidence accepts with
band `IN_BAND`. A decisive repair past the item's budget rejects as `BUDGET`. Taskforge's own
policies reject a task still outside the band; the committed capability configuration accepts a
too-easy task and labels it with its synthesis pass rate, the solve rate its summary carries. The
accept choice outranks a spent repair budget, so a consumer that accepts too-easy tasks never loses
one to `BUDGET`. An accept outside the band keeps `calibrated` false in its summary: `Accept.band`
says where the task fell, `calibrated` says the evidence held no finding. A `Repair` carries a
brief, not a patch: the loop passes `brief.failure` to `builder.author.author` as
`Revision.failure`, so the author is the only model that writes builder code. The brief renders
repair-tier shortcuts as controls the revised CONTROLS step must return verbatim, so the next
round's control replay checks the fix, and lists noted adversary results after the findings as
information, not defects. Notes never change a decision:
an accepted summary carries them, and a repair's `invalidate` and a rejection's reasons come from
the findings alone. `invalidate` names
the steps whose roles the findings condemn, so a model-driven step the author left unchanged is
resampled rather than replayed from the step cache.

An item's status is derived from its event log, never stored. Each phase boundary appends one
`EntryKind.EVENT` row through the run's ledger, into the item's own JSONL file beside its spans and,
under Iris, into the Finelog mirror. Large payloads stay in the item directory (proposals,
`verdict.json`, programs, drafts, attempt files, `calibration.json`, `decision.json`) and events name
them by digest. A per-item contiguous `seq` and a schema version are the only guards against a second
writer and against an enum rename silently changing what an old log means; `derive_state` raises on
either. A relaunch re-runs only the sub-phase that lacks its completion event: build steps are
memoized, settled trials load from their attempt files, and review is pure. A log opened under a
different `policy.json` is refused.

The loop keeps a run's bounds separate because their costs differ: a build revision is one author
call, a review repair is a whole validation round. A build failure, or any exception the builder
program raises, goes back to the author as a revision. A
`builder.infrastructure.BuildInfrastructureFailure` (no factory for the machine backend, no image
builder, a scheduling timeout, an unreachable host) is the machine host's failure, not the program's: it spends no
revision and the author never sees it. The loop records `BUILD_INFRASTRUCTURE` with the cause,
waits out the retry backoff without holding a slot and rebuilds the same program; a host failure
after `max_build_retries` rebuilds ends the item `ABANDONED` with its cause counts, and the next
launch rebuilds the same program with a fresh count. A missing factory or image builder is
deterministic on the host (a laptop without Docker stays without Docker), so it abandons the item at once (`retries_used=0`,
`abandon=true`): retrying would spend the backoff ladder for nothing. It is not a `HOST` rejection,
because `REJECTED` is final and a run root moved to a host with the factory must still build the
item; `ABANDONED` is re-entered on the next launch. A host failure never ends an item `REJECTED` or
`FAILED`. `GlmUnavailable` (the endpoint's failure) propagates
and records `FAILED`. A
repair whose rebuild produces the same task digest counts as a failed revision whose failure text is
the brief again, so the author cannot spend the repair budget returning the same program. The
output-token budget sums the item's `LLM_CALL` entries (triage, authoring, build steps) and is
checked before each authoring. Validation trials record their model calls too, under steps
`solver/<index>` and `adversary/<role>/<index>`, but the budget leaves them out
(`loop.events.UNBUDGETED_TRIALS`): `k`, `adversary_k`, the verifier submission budget and the
deadlines bound them instead. The
exclusion keys on the `<kind>/` prefix, since a build step records its calls under its bare function
name, which cannot contain `/`. A `Retry` waits out the backoff without holding a slot; spent
retries end the item `ABANDONED`, never rejected, and the next launch re-enters it at the control
replay with a fresh retry budget. An unhandled exception records `FAILED` and propagates to the queue. A triage verdict is final for its
proposal digest within a run. Review's band choice reaches the loop as `LoopPolicy.band_rules` and
the loop records where an accepted task fell (`DECIDED.band`, the pass rate, and the `TERMINAL`
reason), so a log tells a calibrated acceptance from a consumer's acceptance outside the band
without opening `decision.json`.

An unattended run is one asyncio process. Models and sandboxes are remote, so one process reaches
hundreds-wide without a coordination store. The width semaphore is acquired around phases, so an item
waiting out a retry backoff holds no slot, and there is no request limiter: `GlmClient` pools 512
connections and holds while the router drains. A throttle is added only after `RunSummary` shows a
failure that needs it. Item status is the item's event log, so a relaunch on the same run root, or on
an Iris attempt restored from the previous attempt's archive (`restore_from`), resumes every item.

`summary.json` is the one place a run lists its accepted tasks for consumers; there is no separate
export file. Each accepted record carries the synthesis pass rate (solved of `k` and the solve rate)
from the calibration summary it was accepted on, so a consumer need not open the round's `calibration.json`,
and the band outcome. The band on an accepted record is the decision's: `in_band`,
or `too_easy` / `too_hard` when the policy's `BandRules` chose to accept a task still outside the band
after its revisions. A consumer reads the band rather than treat acceptance as calibration.

The GLM pool is always named. Validation we drive ourselves (live tests, probes, laptop runs) uses
the interactive `high` pool; the committed unattended configuration (`docs/policy.example.json`)
names `bulk`, with a `<relay-job>` placeholder the launcher sets per cluster. A config holds no
secret: `LaptopGlm` names a token file and `RelayGlm` the environment variable that holds the token,
and the Parallel key is a file or a variable name in the same way. Under Iris,
`queue.job.host_secrets` removes those variables and the submitter keys Iris forwards from the
process environment and from `IRIS_JOB_ENV` before any sandbox exists, because Iris copies
`IRIS_JOB_ENV` into every child job. First-run policy values (`k=8`, band `[0.125, 0.875]`,
`adversary_k=2`, `adversary_submissions=10`, `adversary_repair_submissions=3`, `band_rules` accept for
too-easy and reject for too-hard, `max_build_retries=2`, width 256) live in the policy example, not in
code.

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
  builder live tests (`parallel_key` fixture); the tests that need it skip when it is unset.

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
`scripts/run_queue.py` runs a queue from a run config on a laptop or in an Iris task, and
`scripts/cluster_queue_probe.py` is the fail-fast preflight for an unattended run: GLM health for
both pools, machine creation without leaked credentials, a width of concurrent validation rounds,
the Finelog mirror, resume from the event logs, and a registry image pull.

## Evidence

Every live check writes raw evidence under `<evidence root>/<package>/`: what was checked, the
request, the full response (including `usage` and `finish_reason`) and the wall time. The evidence
root is `$TASKFORGE_EVIDENCE_DIR` when set, else `taskforge-evidence/` in the system temp directory
(`tempfile.gettempdir()`); the `evidence_root` fixture in `tests/conftest.py` resolves it. Evidence
stays outside the checkout on the machine that ran the check; the measurements a change relies on
are recorded in its pull request description. Strip `Authorization` headers before saving a
request, and never write a token or key there.
