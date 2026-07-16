# Running Evaluations with Marin

This guide shows the post-hoc evaluation API in Marin: how to evaluate an existing checkpoint by
combining composable eval steps. For a high-level overview of the evaluation stack, see
[Evaluation Overview](../explanations/evaluation.md).

## Prerequisites

- A trained model checkpoint. Evals take an `ArtifactStep[LevanterCheckpoint]` handle — either the
  return value of a training run, or a pre-existing checkpoint wrapped with `ArtifactStep.adopt`.
- Access to the TPU or GPU resources required by the backend you choose.

## Core APIs

The composable helpers live in `experiments/evals/evals.py`:

```python
from experiments.evals.evals import (
    Backend,
    EvalGroup,
    eval_step,
    eval_steps,
    eval_report,
    core_evals,
    key_evals,
    base_model_evals,
)
```

- An `EvalGroup` is one backend run over a set of tasks. It is the composable unit: each group
  becomes one artifact, addressed by `evaluation/{backend}/{model}/{group_id}`.
- `eval_step(model, group)` builds one eval artifact; `eval_steps(model, groups)` builds one per group.
- `eval_report(results, name=...)` aggregates a suite's results into one `EvalReport` artifact.
- `core_evals` / `key_evals` / `base_model_evals` are named menus — lists of `EvalGroup` drawn from
  `experiments/evals/task_configs.py`, the same task menu the in-loop `EvalSuite` on `train_lm` uses.

Each helper returns an `ArtifactStep`. Build them with `run(...)` (which returns the loaded artifacts)
or with `StepRunner().run([lower(x) for x in steps])`.

## 1. Run a named suite

```python
from fray.cluster import ResourceConfig
from marin.execution.lazy import ArtifactStep, run
from marin.training.training import LevanterCheckpoint

from experiments.evals.evals import eval_report, eval_steps, key_evals

# Adopt a pre-existing checkpoint as a typed handle (no copy, no recompute). A relative source
# resolves against the local bucket (MARIN_PREFIX, set by iris); pass an absolute gs:// path to pin.
model = ArtifactStep.adopt(
    "perplexity-models/llama-200m",
    "2026.06.30",
    "gcsfuse_mount/perplexity-models/llama-200m",
    kind=LevanterCheckpoint,
)

results = eval_steps(model, key_evals(resources=ResourceConfig.with_tpu("v6e-8")))
report = eval_report(results, name=f"{model.name}/key")

if __name__ == "__main__":
    run(report)
```

`key_evals` returns two groups: a generation group over `KEY_GENERATION_TASKS` (the `LM_EVAL` backend)
and a multiple-choice group over `KEY_MULTIPLE_CHOICE_TASKS` (the `LEVANTER` backend). `eval_report`
depends on both and materializes the merged per-task metrics.

`core_evals` and `base_model_evals` follow the same shape. `base_model_evals` runs CORE plus each MMLU
cut as its own group, so every cut is evaluated (each group has a distinct identity).

## 2. Compose your own groups

An `EvalGroup` states its tasks, its backend, and its resources explicitly:

```python
from fray.cluster import ResourceConfig
from marin.execution.lazy import run

from experiments.evals.evals import Backend, EvalGroup, eval_report, eval_steps
from experiments.evals.task_configs import CORE_TASKS, KEY_GENERATION_TASKS

groups = [
    EvalGroup(tasks=CORE_TASKS, backend=Backend.LEVANTER,
              resources=ResourceConfig.with_tpu("v4-8"), id="core"),
    EvalGroup(tasks=KEY_GENERATION_TASKS, backend=Backend.LM_EVAL,
              resources=ResourceConfig.with_tpu("v6e-8"), id="generation",
              engine_kwargs={"max_model_len": 4096, "max_gen_toks": 4096}),
]

report = eval_report(eval_steps(model, groups), name=f"{model.name}/custom")

if __name__ == "__main__":
    run(report)
```

`id` is the task-group segment of the artifact name. Choose a stable id per group; two groups on one
model must have distinct ids so their outputs do not share an address.

## 3. Reading results back

`resolve` (or `run`) returns the typed artifacts. An `EvalReport` carries the merged metrics:

```python
from marin.execution.lazy import resolve

report_artifact = resolve(report)
print(report_artifact.task_metrics)  # {task: {metric: value}}
print(report_artifact.averages)      # backend-recorded cross-task averages
```

Each individual result is an `EvalResult` with the same accessors: `LevanterEvalResult.averages()` and
`task_metrics()` for the Levanter backend, `LmEvalHarnessResult.task_metrics()` for lm-eval.

## 4. Run the repository example scripts

The checked-in examples track real usage and are the safest starting points. They are
deferred-version CLIs (see `marin.experiment.cli`): `--version` supplies the run-wide version, the
plan prints without `--run`, and `--run` builds it.

```bash
# print the plan (no build)
uv run python -m experiments.evals.run_key_evals --version dev
# build it; --limit caps examples per task for a fast cluster smoke
uv run python -m experiments.evals.run_key_evals --version dev --run --limit 5
uv run python -m experiments.evals.run_base_model_evals --version 2026.07.16 --run
```

They adopt a checkpoint, build the suite, compile a report, and log results to W&B. `--version dev`
resolves the eval artifacts to a mutable version that rebuilds every run — pass a calendar version to
pin a run.

## Parameter reference

### `EvalGroup`

- `tasks`: the `EvalTaskConfig` entries this group evaluates together in one backend job.
- `backend`: `Backend.LEVANTER` (MCQ / logprob) or `Backend.LM_EVAL` (generation via vLLM).
- `resources`: hardware for this group's job.
- `id`: the group's identity segment in the artifact name.
- `engine_kwargs`: optional vLLM engine overrides (`LM_EVAL` groups).
- `apply_chat_template`: whether to apply the model chat template before evaluation.
- `max_eval_instances`: optional cap on evaluated examples.
- `discover_latest_checkpoint`: whether to resolve the latest checkpoint under the model path.
- `env_vars`: extra env vars for the child worker (e.g. `HF_ALLOW_CODE_EVAL=1` for humaneval).

### `eval_report`

- `results`: the `EvalResult` steps to aggregate.
- `name`: the report's identity segment (`evaluation/report/{name}`).

For deeper dives, see:

- [Evaluation Overview](../explanations/evaluation.md)
- `experiments/evals/task_configs.py`
- `experiments/evals/evals.py`
