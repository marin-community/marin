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
  stage30_author/  program.py: build steps and the DAG they form;
                   author.py: Revision, the Author protocol, `static_author`, the program manifest
  stage40_build/   context.py: RLEnvBuildContext, BuildFailure;
                   step_store.py: step keys, memos, the memo store and `memoized`;
                   run.py: `run_program`, `run_build`, the draft and `ItemBuildProvenance`
```

Each pipeline stage is a package named `stageNN_<name>`, numbered by tens in pipeline order:
proposal 10, triage 20, author 30, build 40. The gaps leave room to insert a stage later without
renumbering. A stage imports from lower-numbered stages and from the shared modules
(`content_hash`, `ledger`, `llm`, `sandbox`, `spec`), and the shared modules import no stage.
`tests/test_imports.py` checks both directions. `scripts/ledger_summary.py` totals wall time and
tokens per kind and step over JSONL ledgers. The cluster scripts run on Iris and document their
submit commands in their docstrings: `scripts/build_image_job.py` builds a `DockerBuild` context
and pushes it to a registry digest through `scripts/push_image_task.py`, and
`scripts/iris_machine_probe.py` probes the shellbox Iris backend. `docker/grader-base` is the
build context of the grader base image.

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
  in the Iris task. `task_refusals(lowered, factory_capabilities(where))` lists every typed reason
  a lowered task cannot run on them, before a trial starts.

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
`x-routing-key: <item>:<round>:<step>`. The route returns each sampled id's bytes
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
sink. A step receives an `RLEnvBuildContext` (item id, model, sampling policy) as its first
parameter. The decorator reads the step's inputs from the parameters after it, and the runner
passes each one by name: another step's output, or the caller's proposal for a parameter called
`proposal`. `tests/stage40_build/test_run.py` builds this toy program, abridged here:

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
