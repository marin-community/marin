# Taskforge

Taskforge turns task ideas into TaskCompendium `TaskSpec`s with a model's help. It is the
`marin-taskforge` member of the repository's uv workspace.

## Package layout

```
src/taskforge/
  content_hash.py  canonical JSON and its sha256 digests
  ledger/          timed spans to per-item JSONL and, under Iris, Finelog
  llm/             the ModelClient protocol, the GLM client and its token-id route, sampling policy,
                   structured output, call cache, ledger recording, agent loop, web tools,
                   the RolloutEngine model
  sandbox/         MachineFactory selection per host, up-front task refusals, image builds
  spec/            TaskSpec assembly, lowering, fixed controls, grader verdicts
  stage10_proposal/ model.py: the TaskProposal document and its generation regex;
                   source.py: ProposalBatch, SlotFailure, the ProposalSource protocol
  stage20_triage/  checks.py: structural checks; program.py: the GLM rubric and the decision;
                   verdict.py: Verdict, TriageDecision
  stage30_author/  program.py: build steps and the DAG they form;
                   author.py: Revision, the Author protocol, AuthoredProgram, `static_author`, the program manifest;
                   loader.py: `load_program_source`, the restricted loader of authored programs
  stage40_build/   context.py: RLEnvBuildContext, BuildFailure, BuildInfrastructureFailure;
                   services.py: RLEnvBuildServices, AgentLimits;
                   outputs.py: the step outputs (Sources, Fixtures, Grader, Instructions, ...);
                   sandbox_effects.py: `machine`, `try_grader`, `publish_image`;
                   sdk.py: the functions, schemas and constants steps call;
                   infrastructure.py: host failures told apart from program failures;
                   template/standard.py: the standard six-step program;
                   glm_author.py: GlmAuthor and its prompts;
                   step_store.py: step keys, memos, the memo store and `memoized`;
                   run.py: `run_program`, `run_build`, `build_model_digest`, the draft and `ItemBuildProvenance`
  stage50_validate/ trials.py: trials on RolloutEngine; classify.py: the failure classifier;
                   lowering.py: `lower`, a task lowered for the validating host;
                   control_drafts.py: `write_controls`, the controls of a draft round;
                   controls.py: control replay; outcome.py, evidence.py: outcomes and their totals
```

Each pipeline stage is a package named `stageNN_<name>`, numbered by tens in pipeline order:
proposal 10, triage 20, author 30, build 40, validate 50. The gaps leave room to insert a stage later without
renumbering. A stage imports from lower-numbered stages and from the shared modules
(`content_hash`, `ledger`, `llm`, `sandbox`, `spec`), and the shared modules import no stage.
`tests/test_imports.py` checks both directions. `scripts/ledger_summary.py` totals wall time and
tokens per kind and step over JSONL ledgers. The cluster scripts run on Iris and document their
submit commands in their docstrings: `scripts/build_image_job.py` builds a `DockerBuild` context
and pushes it to a registry digest through `scripts/push_image_task.py`, and
`scripts/iris_machine_probe.py` probes the shellbox Iris backend, and
`scripts/cluster_rollout_probe.py` runs validation trials inside an Iris task with GLM resolved
from a relay job. `docker/grader-base` is the build context of the grader base image.

## Proposals

A proposal source implements `stage10_proposal.source.ProposalSource[IdeaT]`:
`async propose(idea, n) -> ProposalBatch`. A batch holds the planning request and completions and
one `SlotProposal` or `SlotFailure` per slot, in slot order, so a failed slot does not drop its
siblings. `stage10_proposal.model.TaskProposal` is YAML front matter (`ProposalHeader`, keys in
`HEADER_KEYS` order) plus a markdown body with required section headings. `parse` and `render`
round-trip it and `digest` is the sha256 of the canonical form. `parse` also checks the research
decision: `research_need: needed` lists at least one `research` item and `not_needed` none, and a
non-null body cites every item, and only listed items, as `research[i]`.
`document_request_fields(proposal_id, source)` returns the `structured_outputs.regex` request
field that constrains a completion to the canonical layout for that id and source. A reply cut off
by the output limit is a `SlotFailure`, and a reply that fails `parse` gets one repair request
built from `DOCUMENT_REPAIR_PROMPT`.

## Triage

`stage20_triage.program.evaluate(proposal, checks, rubric, ctx) -> Verdict` runs the structural
checks first, and a fatal failure or a null proposal rejects without a model call. The rubric
scores independent samples, each a structured call through `CallStore`, and the decision is ACCEPT
or REJECT by strict majority, otherwise REPAIR. A sample whose structured output stays invalid is
drawn once more and then dropped. `GlmRubric.repair` rewrites a REPAIR proposal under the document
regex with continuations off and re-asks once if the rewrite fails `parse`.

## Task specs

A built task is a TaskCompendium `TaskSpec` and the `LoweredTaskSpec` that says how it runs.
`spec.draft.assemble` builds the semantic task and `spec.draft.lower` alone builds the lowered
record, so a move of either is a single-site change.

- `spec.draft.assemble(task_id, instruction, answer_type, answer_format, grader, source, *,
  environment, files, output_paths, system, final_tools, tags) -> TaskSpec`. `environment` is
  `requirements(image=, setup=, workdir=, env=)`, whose commands run as Linux processes in the
  image or, without one, in ShellSim's simulated shell, or `None` for a task without a machine.
  `files` are agent-visible `file("workspace/x", ...)` resources relative to the machine root.
  `grader` is a `GraderPackage` from `answer_grader(spec)` (a verifyit mode graded in process, or
  in a verifier machine given `environment=`), `python_grader(script, config, environment=,
  answer_path=, timeout=)` (`python3 /tests/grade.py` in a verifier machine) or
  `script_grader(argv, reward, environment=, answer_path=, timeout=, ...)`.
  `grader_environment(task_image)` is the verifier machine's environment: the task image, or
  `sandbox.images.GRADER_BASE_IMAGE` for a task without one.
- `spec.draft.lower(task, *, host, task_machine, verifier_machine, session, factories)` picks
  ShellSim for a machine without an image or packages lock, the local backend for a lock alone,
  and the host's container backend for an image, so a lowered spec is bound to its host. A grader
  with an environment takes a verifier machine; no other grader does.
- `spec.controls.Control` is one labeled candidate submission (`kind`, `category`, `concern`,
  `author`, a `Transcript` or `Workspace` payload, an `Expectation`). `concern` is `reference`,
  `acceptance`, `extraction` or `shortcut`, and the concern a control may carry follows its
  category (`spec.controls.CONCERNS`). A task's control set needs a `reference`, an `acceptance`
  and a `shortcut` control, so its coverage never rests on extraction controls, which pin today's
  answer parser. `validate_controls(task, controls)` checks a set against its task.
- `sandbox.factories.machine_factories(where, controller_url)` returns the factories for
  `RolloutEngine`, keyed by shellbox `Backend` value: ShellSim plus Docker on a laptop,
  ShellSim plus gVisor on Iris, plus the local backend for lock-only graders when bubblewrap works
  in the Iris task. Before a trial starts, `task_refusals(lowered, factories)` passes each machine of
  a lowered task to its factory's `check_environment` and lists a typed refusal, with the factory's
  message, for each machine without a factory or rejected by one.

Task-specific grading is the task's own code, private in `resources.verifier`, run by a
`ScriptGrader` in a separate verifier machine after the attempt. The verifier machine starts from
the task's image or, for a ShellSim task or a task without a machine, from the grader base
(CPython 3.12, `sh`, `setsid`, root), and receives the captured `output_paths` at their own paths
and the extracted answer at `/app/answer.txt`. A grader program writes its verdict, the reward and
one result per check, as `spec.verdict.verdict_json`. Generic modes that verifyit grades in
process are preferred whenever a task fits one.

## Model calls

`llm.recording` records every model call as an `LLM_CALL` span under a `CallLedger` (ledger, item,
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
          GlmClient.complete      span(LLM_CALL) -> Ledger.record(LedgerEntry)
          or .generate,                              |-> JsonlLedger   <root>/<item_id>.jsonl
          retries and holds                          '-> FinelogLedger taskforge.ledger (under Iris)
                      |
                      v
          GLM-5.3 router: POST /v1/chat/completions (streamed)
                          or POST /inference/v1/generate (token ids)
```

`llm.agent.run_agent(client, policy, messages, tools, max_turns, record, replay)` is Taskforge's
own tool loop; `replay` defaults to `ReasoningReplay.KEEP`, which replays each assistant turn's
`reasoning_content`. `shell_tool` offers RolloutEngine's default shell tool on a shellbox
`Machine`, and `llm.web` adds Parallel search and extract. The client sends the policy's whole
`max_tokens` with every turn, so a prompt plus that budget beyond the context window raises
`ContextExhausted`, which ends the loop with `AgentStop.CONTEXT`; long loops set a smaller
`max_tokens`.

`llm.rollout_model.GlmRolloutModel(client, policy, transport, record)` is the RolloutEngine model
callable. On `RolloutTransport.CHAT_COMPLETIONS` each turn is one streamed call with
`return_token_ids`, and the served ids come from the chunks. On `RolloutTransport.TOKEN_ROUTE`
every turn after the first renders the conversation once on the server (a one-token probe, its own
`purpose=render` span), keeps the ids the template rendered after the previous assistant turn, and
samples from the rollout's exact prefix plus those ids through `GlmClient.generate`, with header
`x-routing-key: <item>:<round>:<step>`; the chat-completions calls and render probes carry the same
key as `prompt_cache_key`, so every request of a rollout shares one node's prefix cache.
`GlmRolloutModel.for_step(step)` derives the model of one rollout. The route returns each sampled id's bytes
(`return_token_bytes`), and `llm.glm_format.parse_turn` joins them and splits the text into
reasoning, answer and tool calls. A turn that cannot keep the exact prefix raises
`TokenContractBreak` inside its span, which records `cause == "token_contract"`. A response whose
ids, logprobs and bytes do not align raises `TokenEvidenceError`, which fails the run. Both
transports send the same sampling fields, `LLMPolicy.sampling_params()`.

`llm.store.CallStore` caches `complete` and `complete_structured` calls on disk by the digest of
the request and does not record to the ledger.

## The build DAG

An author writes a build program as `async def` steps decorated with `@rl_env_build_step`, each
returning a value of its annotated type. The final step returns a `TaskSpec` and carries
`@rl_env_build_sink` in place of `@rl_env_build_step`. The author passes the steps, and the sink as
the keyword `sink`, to `rl_env_build_program`. That call orders the steps so each runs after its
inputs, puts the sink last, and refuses a step whose output reaches neither another step nor the
sink. A step receives an `RLEnvBuildContext` (item id, round, model, sampling policy, services,
and the running step's name) as its first parameter. The decorator reads the step's inputs from
the parameters after it, and the runner passes each one by name: another step's output, or the
caller's proposal for a parameter called `proposal`. `tests/stage40_build/test_run.py` builds this
toy program, abridged here:

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
the source digest over the step's own source text, followed by the source of its output type when
the program defines that class beside the step. An output type imported from a library, such as
`TaskSpec`, falls under `SDK_VERSION`. A step should therefore read everything it depends on from
its inputs, its own body or an output type defined beside it. Editing a helper defined outside the
body leaves the key unchanged, so a library change of that kind comes with an `SDK_VERSION` bump.
The model digest covers the model id, the sampling policy, the agent loops' reasoning replay and
their limits (`AgentLimits`), so a build that drops reasoning, or bounds its agents differently,
reruns every step; `run_build(model_digest=...)` overrides it. The machines, images, web tools and
ledger in `RLEnvBuildServices` stay out of every key and out of every output.
Because inputs enter the key by content, a step that reruns and returns the same output leaves its
consumers cached. The runner memoizes each step through `step_store.memoized`, and a driver
memoizing a helper calls it the same way with a key of its own. Every step records a `STEP` span
with its key digest and cache status, and its model and tool calls carry its name as `step`; a
build without a ledger records nothing.

A step uses the host through `sandbox_effects`: `machine` prepares a machine as a task attempt
would, `try_grader` grades a candidate reply and workspace through `RolloutEngine.run`, and
`publish_image` pushes a task image. `infrastructure.host_checked_factories` wraps the factories
first, and `run_program` raises `BuildInfrastructureFailure` for a host fault anywhere in a step's
cause chain, even one the program wrapped, so the caller retries or abandons the item instead of
asking the author for a revision.

`stage40_build/template/standard.py` is the standard program: five steps and the sink `assemble`.

```
sources(proposal) -> fixtures(proposal, sources) -> requirements(proposal, fixtures)
    -> grader(proposal, fixtures, requirements) -> instructions(proposal, fixtures, requirements, grader)
    -> assemble(proposal, fixtures, requirements, grader, instructions) -> TaskSpec
```

`sources` researches each research item of the proposal on the web, and
`run_build(research=ResearchRequirement.ENFORCED)` fails a build whose `Sources` lacks a finding for one of them.
`grader` grades its own reference answer and the empty answer with `try_grader` before it returns,
sends a failing grader back to the model, and after `sdk.ATTEMPTS` requests fails the build, so a
grader that rejects its own reference is a build failure. The `Grader` it returns holds the
reference answer and every candidate it graded.

`stage40_build.glm_author.GlmAuthor` asks GLM for a program with the template as its example, and
returns an `AuthoredProgram` holding the program and its source, which
`stage30_author.loader.load_program_source` loads. The loader refuses a module whose top level
holds anything but imports, classes, steps, the sink and a final
`PROGRAM = rl_env_build_program(..., sink=...)`, because a step's key covers its own source and its
output type alone, so an edit to a helper beside the steps would leave every key unchanged. It also
refuses an import outside `ALLOWED_IMPORTS`, a missing `grader` step returning `Grader`, and a
program without a sink made by `@rl_env_build_sink`. Each refusal is a `ValueError` whose message
the author receives.

## Validation

`stage50_validate.lowering.lower(task, host, factories)` lowers a draft's task for the validating
host. `stage50_validate.control_drafts.write_controls(task, grader, proposal, ...)` asks the model
for the round's fixed controls from the proposal's control plan and writes them to
`controls.json`. It reads the reference answer from the draft's `grader` step output and sends
back a control that repeats a candidate the grader step graded, other than the reference and the
empty answer.

`stage50_validate.trials.run_trials(lowered, plan, settings, model)` runs k trials of a
`LoweredTaskSpec` through `RolloutEngine`. Each attempt runs on
`model.for_step("<kind>/<trial>/<attempt>")`, so a `GlmRolloutModel` records and routes every trial
attempt under its own step. `TrialPlan.deadlines` and `EngineSettings`
replace the builder's session limits, and `task_digest(lowered)` hashes what runs, answer format
included. Each trial is `Graded` or `Ungraded` with one typed `Cause`, and
`stage50_validate.classify.classify` is the only failure classifier. A trial is attempted again for
a retryable cause, and for a broken token contract within `TrialPlan.token_contract_retries`;
`TrialPlan.first_attempt` numbers attempts after the ones on disk, so a re-entered trial continues
where it stopped. Solver trials sample on `SOLVER_TRANSPORT`, the token-id route.
`stage50_validate.evidence.Evidence(outcomes, contract_retries).reward_stats(kind)` totals the
graded trials and the attempts a broken contract discarded.
`stage50_validate.controls.replay(lowered, controls, plan, settings, tokenize)` replays each control
as one `CONTROL` trial named by its id, with `llm.rollout_model.ServerRenderer` serving `tokenize`
from the served chat template.

## Testing

```bash
uv run --package marin-taskforge --frozen --group test pytest -c lib/taskforge/pyproject.toml \
    lib/taskforge/tests experiments/post_training/capability_driven_envs/tests
cd lib/taskforge && uvx --from 'pyrefly>=1.0.0,<1.1.0' pyrefly check
```

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
