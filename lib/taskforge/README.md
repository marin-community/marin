# Taskforge

Taskforge turns task ideas into validated TaskCompendium `TaskSpec`s. A proposal source writes
task proposals, triage scores them, a builder program turns an accepted proposal into a `TaskSpec`
with fixed controls, and validation runs trials and control replays on RolloutEngine. It
reimplements the `experiments/post_training/capability_env_gen` pipeline as a library.

Taskforge is a standalone uv project with its own `uv.lock` and `.venv`. It depends on
`taskcompendium`, `rolloutengine`, `shellbox`, `verifyit`, `rigging`, `finelog` and `iris` through
path dependencies on the sibling `lib/` packages.

## Package layout

```
src/taskforge/
  canonical.py  canonical JSON, sha256 digests, atomic file replacement
  llm/        GLM-5.3 transport, structured calls, call cache, agent loop, web tools, RolloutEngine model
  ledger/     timed spans to per-item JSONL and, on Iris, Finelog
  spec/       TaskSpec assembly and fixed controls
  sandbox/    MachineFactory selection per EnvironmentKind, up-front task refusals, image builds
  proposal/   the TaskProposal document (model.py), the ProposalSource protocol (source.py), sources/
  triage/     structural checks, the GLM rubric, verdicts
  validate/   trials, the failure classifier, control replay, evidence aggregation
  build/      builder programs: memoized steps, the Build SDK, program authoring, the standard template
  review/  loop/  queue/   empty; reserved for review decisions, the item loop and the work queue
scripts/      Iris image builder, cluster probes, ledger summary
```

Imports point down: `build`, `validate`, `triage` and `proposal` use `llm`, `sandbox`, `spec` and
`ledger`, which use only `canonical` and external packages. Stage packages do not import each
other, with one exception: `proposal.model` defines the seam type `TaskProposal`, and `triage` and
`build` import it.

## Seams

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
  the task, its `TaskExecution` and its controls.
- `validate.trials.run_trials(task, execution, plan, settings, model)`: runs k trials through
  `ShellboxRolloutEngine`. Each trial is `Graded` or `Ungraded` with one typed `Cause`, and
  `validate.classify.classify` is the only failure classifier.

## Decisions

Task specs are upstream's. A built task is a TaskCompendium `TaskSpec` plus the `TaskExecution` it
runs with. TaskCompendium keeps execution settings out of the spec: deadlines, the agent user, and
each stage's working files, setup and healthcheck are a `TaskExecution`. `spec.draft.assemble`
checks the two together, a builder returns both, and validation passes both to RolloutEngine.
Task-specific grading is the task's own scripts, run as a `ShellVerifierSpec` (private files on its
`VerifierSpec`; reward on stdout, by exit code, or in reward files; optionally in a separate
grading environment). Generic verifier types
(math, mcq, judge, pytest, aggregation) belong to `lib/verifyit`, which `TaskSpec` reaches
through its verifier registry. A gap in either is fixed upstream. Interim code for one gap goes
in `taskforge/spec/extensions/<issue>.py`.

Execution is RolloutEngine's. `ShellboxRolloutEngine` creates one shellbox `Machine` per attempt
from the caller's `MachineFactory` for the task's `EnvironmentKind`, installs files, runs setup
and healthchecks, drives the shell tool, grades and closes. It grades the state an agent left when
the agent deadline expires, and bounds every cleanup action by its `cleanup_timeout`. Taskforge supplies the model callable
(`llm.rollout_model.GlmRolloutModel`) and the factories (`sandbox.factories.machine_factories`).
Taskforge reaches a sandbox only through a `Machine` that the engine or a builder step created.

The agent loop is Taskforge's own: `llm.agent.run_agent` over `GlmClient`, with the shell tool
running through `Machine.run` and Parallel search and extract from `llm.web`. Builder agents run
on it. Solver and control rollouts run on RolloutEngine with the same `GlmClient`.

Inference defaults are the model maximum. `max_tokens` starts at the model's output limit
(131,072 for GLM-5.3) or the remaining context. On `finish_reason == "length"` the client keeps
the output and continues. Timeouts are stall timeouts on the stream. Concurrency defaults to
hundreds of requests. `GlmRolloutModel` is the exception to continuation: a continuation
re-renders the prompt and breaks RolloutEngine's exact-token contract, so a cut turn ends the
rollout with stop reason `length`.

`GlmClient` reports a tool-call reply that used its whole output budget as `length`, because
vLLM's GLM tool parser reports such a reply as `tool_calls`.

## Testing

Run every command from `lib/taskforge`. Do not pass a partial marker expression such as
`-m "not slow"`; the package `addopts` sets the default marker expression.

```bash
uv run --group test pytest tests
```

Live tests carry `@pytest.mark.live_glm` and skip with a reason when their inputs are missing.
They call GLM-5.3 on the interactive tier through the router port-forward:

```bash
KUBECONFIG=~/.kube/open-athena kubectl -n open-athena port-forward svc/glm53-router 18000:8000 &
curl -s http://127.0.0.1:18000/health   # "status":"ok"

export TASKFORGE_GLM_BASE_URL=http://127.0.0.1:18000/v1
export TASKFORGE_GLM_TOKEN_FILE=~/openathena/glm-infer/glm_api_token.txt   # line: GLM_API_TOKEN=...
uv run --group test pytest tests -m live_glm
```

`-m live_glm` replaces the default marker expression. The token file must hold the interactive
(`high` pool) token. The `glm_settings` fixture reads it and keeps it out of `repr`. Inside an Iris
task, `llm.endpoint.resolve_glm_base_url` resolves the endpoint.

Other live inputs:

- `TASKFORGE_PARALLEL_KEY_FILE` names a file with a `PARALLEL_KEY=...` line, for the agent and
  build live tests (`parallel_key` fixture, default `~/openathena/build_envs/.parallel_key`).
- `TASKFORGE_CAPABILITY_CATALOG` names the capability catalog (`new_catalog.json`), which is not
  checked in, for the capability proposal and triage live tests (`capability_catalog` fixture).

The package pytest config sets `timeout = 60` and `asyncio_mode = "auto"`. Long live tests carry
`@pytest.mark.timeout(<seconds>)`.

Types and lint:

```bash
uvx --from 'pyrefly>=1.0.0,<1.1.0' pyrefly check           # from lib/taskforge; checks src
./infra/pre-commit.py --fix --files lib/taskforge/<path>   # from the repository root
```

The cluster scripts run on Iris and document their submit commands in their docstrings:
`scripts/build_image_job.py` builds a `DockerBuild` context and pushes it to a registry digest,
`scripts/iris_machine_probe.py` probes the shellbox Iris backend, and
`scripts/cluster_rollout_probe.py` runs validation trials inside an Iris task.

## Evidence

Every live check writes raw evidence under `lib/taskforge/.evidence/<package>/`: what was
checked, the request, the full response (including `usage` and `finish_reason`) and the wall
time. `.evidence/` is gitignored, so evidence stays on the machine that ran the check. Strip
`Authorization` headers before saving a request, and never write a token or key there.
