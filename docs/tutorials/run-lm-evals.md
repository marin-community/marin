# Running Evaluations with Marin

This guide shows the post-hoc evaluation API in Marin: how to evaluate an existing checkpoint by
combining composable eval steps. For a high-level overview of the evaluation stack, see
[Evaluation Overview](../explanations/evaluation.md).

## Prerequisites

- A trained model checkpoint. Evals take an `ArtifactStep[LevanterCheckpoint]` handle — either the
  return value of a training run, or a pre-existing checkpoint wrapped with `ArtifactStep.adopt`.
- Access to the TPU or GPU resources the serving backend needs.

Each eval serves the model once as an OpenAI-compatible endpoint (marin-serve: vLLM or Levanter),
evaluates the group's tasks against that URL with the evalchemy fork, and tears the server down.
Multiple-choice and generation tasks run the same way (see [Evaluation Overview](../explanations/evaluation.md)).

## Core APIs

The composable helpers live in `experiments/evals/evals.py`:

```python
from experiments.evals.evalchemy.serve_and_eval import ServeSpec
from experiments.evals.evals import (
    EvalGroup,
    eval_step,
    eval_steps,
    eval_report,
    core_evals,
    key_evals,
    base_model_evals,
)
```

- An `EvalGroup` is one set of tasks evaluated against one served model. It is the composable unit:
  each group becomes one artifact, addressed by `evaluation/evalchemy/{model}/{group_id}`.
- `eval_step(model, group)` builds one eval artifact; `eval_steps(model, groups)` builds one per group.
- `eval_report(results, name=...)` aggregates a suite's results into one `EvalReport` artifact.
- `core_evals` / `key_evals` / `base_model_evals` are named menus — lists of `EvalGroup` drawn from
  `experiments/evals/task_configs.py`, the same task menu the in-loop `EvalSuite` on `train_lm` uses.

Each helper returns an `ArtifactStep`. Build them with `run(...)` (which returns the loaded artifacts)
or with `StepRunner().run([lower(x) for x in steps])`.

## 1. Run a named suite

```python
from marin.execution.lazy import ArtifactStep, run
from marin.training.training import LevanterCheckpoint

from experiments.evals.evalchemy.serve_and_eval import ServeSpec
from experiments.evals.evals import eval_report, eval_steps, key_evals

# Adopt a pre-existing checkpoint as a typed handle (no copy, no recompute). A relative source
# resolves against the local bucket (MARIN_PREFIX, set by iris); pass an absolute gs:// path to pin.
model = ArtifactStep.adopt(
    "perplexity-models/llama-200m",
    "2026.06.30",
    "gcsfuse_mount/perplexity-models/llama-200m",
    kind=LevanterCheckpoint,
)

results = eval_steps(model, key_evals(serve=ServeSpec(tpu_type="v6e-8")))
report = eval_report(results, name=f"{model.name}/key")

if __name__ == "__main__":
    run(report)
```

`key_evals` returns two groups: a generation group over `KEY_GENERATION_TASKS` and a multiple-choice
group over `KEY_MULTIPLE_CHOICE_TASKS`. `eval_report` depends on both and materializes the merged
per-task metrics.

`core_evals` and `base_model_evals` follow the same shape. `base_model_evals` runs CORE plus each MMLU
cut as its own group, so every cut is evaluated (each group has a distinct identity).

## 2. Compose your own groups

An `EvalGroup` states its tasks, its serving backend, and its id explicitly:

```python
from marin.execution.lazy import run

from experiments.evals.evalchemy.serve_and_eval import ServeSpec
from experiments.evals.evals import EvalGroup, eval_report, eval_steps
from experiments.evals.task_configs import CORE_TASKS, KEY_GENERATION_TASKS

groups = [
    EvalGroup(tasks=CORE_TASKS, id="core", serve=ServeSpec(tpu_type="v4-8")),
    EvalGroup(tasks=KEY_GENERATION_TASKS, id="generation",
              serve=ServeSpec(tpu_type="v6e-8"), max_gen_toks=4096),
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
print(report_artifact.averages)      # suite-level rollups
```

Each individual result is an `EvalchemyResult`; `task_metrics()` reads the per-task scores from the
evalchemy output tree.

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

- `tasks`: the `EvalTaskConfig` entries this group evaluates together against one served model.
- `id`: the group's identity segment in the artifact name.
- `serve`: a `ServeSpec` — the serving backend (`vllm` or `levanter`) and slice.
- `tokenizer`: HF tokenizer id the eval client loads; defaults to the served checkpoint. Set it to a
  base-model HF id when serving a `gs://` path the eval image cannot load a tokenizer from.
- `apply_chat_template`: use the chat OpenAI route (`local-chat-completions`) instead of completions.
- `max_gen_toks`: generation length cap for generation tasks.
- `max_eval_instances`: optional cap on evaluated examples (a small value gives a fast smoke).
- `num_concurrent`: parallel in-flight requests the eval client sends the endpoint.
- `discover_latest_checkpoint`: whether to resolve the latest HF checkpoint under the model path.

### `eval_report`

- `results`: the `EvalResult` steps to aggregate.
- `name`: the report's identity segment (`evaluation/report/{name}`).

For deeper dives, see:

- [Evaluation Overview](../explanations/evaluation.md)
- `experiments/evals/task_configs.py`
- `experiments/evals/evals.py`
