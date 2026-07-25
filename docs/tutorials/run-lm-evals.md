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

Reusable step builders live in `marin.experiment.evaluation`; experiment task menus remain under
`experiments/evals`:

```python
from marin.evaluation.hardware import AcceleratorChoice
from marin.evaluation.model_config import ServeConfig
from marin.experiment.evaluation import (
    EvalGroup,
    eval_step,
    eval_steps,
    eval_report,
)
from experiments.evals.evals import (
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

from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.experiment.evaluation import eval_report, eval_steps
from experiments.evals.evals import key_evals

# Adopt a pre-existing checkpoint as a typed handle (no copy, no recompute). A relative source
# resolves against the local bucket (MARIN_PREFIX, set by iris); pass an absolute gs:// path to pin.
model = ArtifactStep.adopt(
    "perplexity-models/llama-200m",
    "2026.06.30",
    "gcsfuse_mount/perplexity-models/llama-200m",
    kind=LevanterCheckpoint,
)

accelerator = AcceleratorChoice(platform=Platform.TPU, tpu_type="v6e-8")
results = eval_steps(model, key_evals(accelerator=accelerator))
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

from marin.evaluation.evalchemy.runner import EvalchemyRunConfig
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ServeConfig
from marin.experiment.evaluation import EvalGroup, eval_report, eval_steps
from experiments.evals.task_configs import CORE_TASKS, KEY_GENERATION_TASKS

accelerator = AcceleratorChoice(platform=Platform.TPU, tpu_type="v6e-8")
groups = [
    EvalGroup(
        config=EvalchemyRunConfig(name="core", tasks=CORE_TASKS),
        accelerator=accelerator,
    ),
    EvalGroup(
        config=EvalchemyRunConfig(name="generation", tasks=KEY_GENERATION_TASKS, max_gen_toks=4096),
        serve=ServeConfig(),
        accelerator=accelerator,
    ),
]

report = eval_report(eval_steps(model, groups), name=f"{model.name}/custom")

if __name__ == "__main__":
    run(report)
```

`EvalchemyRunConfig.name` is the task-group segment of the artifact name. Choose a stable name per
group.

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

## Serving on GPU

`AcceleratorChoice` selects TPU or GPU placement. `ServeConfig` contains server behavior and is
independent of placement. The GPU path runs the Marin CUDA vLLM fork in an isolated `uvx`
environment, which runs without `nvcc`. Set `vllm_extra_args` for models that would otherwise
JIT-compile a kernel at warmup.

Qwen gated-delta-net models (`qwen_gdn_linear_attn`: `Qwen/Qwen3.5-35B-A3B`, `Qwen/Qwen3-Next-80B-A3B`)
are the current case. Their default FlashInfer GDN prefill kernel is JIT-compiled, so without a compiler
the serve child dies at warmup (`Could not find nvcc`) and never registers its endpoint. Pass
`--gdn-prefill-backend triton` to use the compiler-free triton backend:

```python
EvalGroup(
    config=EvalchemyRunConfig(name="core", tasks=CORE_TASKS),
    serve=ServeConfig(
        tensor_parallel_size=8,
        vllm_extra_args=("--gdn-prefill-backend", "triton"),
    ),
    accelerator=AcceleratorChoice(
        platform=Platform.GPU,
        gpu_type="H100",
        gpu_count=8,
    ),
)
```

Other GPU-served models (DeepSeek-V2-Lite, Qwen3-30B-A3B, …) need no extra flags. See
`marin.evaluation.model_config.ServeConfig` and
`marin.evaluation.hardware.AcceleratorChoice` for the two contracts.

## Parameter reference

### `EvalGroup`

- `config`: the `EvalchemyRunConfig`, including tasks, generation settings, concurrency, and limits.
- `serve`: the model-server behavior (`vllm` or `levanter` and its server options).
- `resource_hint`: optional CPU, memory, and disk overrides for the inference worker.
- `accelerator`: the exact TPU or GPU slice used for inference.
- `tokenizer`: HF tokenizer id the eval client loads; defaults to the served checkpoint. Set it to a
  base-model HF id when serving a `gs://` path the eval image cannot load a tokenizer from.
- `discover_latest_checkpoint`: whether to resolve the latest HF checkpoint under the model path.

### `eval_report`

- `results`: the `EvalResult` steps to aggregate.
- `name`: the report's identity segment (`evaluation/report/{name}`).

For deeper dives, see:

- [Evaluation Overview](../explanations/evaluation.md)
- `experiments/evals/task_configs.py`
- `experiments/evals/evals.py`
