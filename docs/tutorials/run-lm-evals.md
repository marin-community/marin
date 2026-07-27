# Running Evaluations with Marin

The shared evaluation launcher runs Evalchemy and Harbor evaluations against a registered model.
It starts one OpenAI-compatible model server, runs every selected evaluation against that endpoint,
writes one durable record per evaluation, and tears the server down.

Use the launcher for routine post-hoc evaluations. Use the composable APIs later in this page when
an evaluation must be part of a Marin pipeline or consume an `ArtifactStep[LevanterCheckpoint]`.
See [Evaluation Overview](../explanations/evaluation.md) for the evaluator and serving architecture.

## Prerequisites

- Run commands from the Marin repository with `uv`.
- Use a model key registered in `experiments/evaluation/models.py` or
  `experiments/evaluation/serve/models/`.
- Have access to the selected Iris cluster and its TPU or GPU serving resources.
- For Harbor evaluations, use an environment with access to the Daytona credential described in
  [Harbor credentials](#harbor-credentials).

## Launch registered evaluations

The command accepts one model key and either a named suite or comma-separated evaluation keys:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model <model-key> \
  --evals <suite-or-eval-keys>
```

Run the resolved plan before submitting an unfamiliar model or suite. `--dry-run` prints the model
location, serving backend, accelerator, target cluster or region, task names, and records prefix:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-0.6b \
  --evals smoke \
  --dry-run
```

### Command recipes

Run a capped Evalchemy smoke on the smallest registered Qwen model:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-0.6b \
  --evals smoke
```

Run the 14-task NLP suite on a base model. This suite works with base and instruct models:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model llama-3.1-8b-base \
  --evals nlp \
  --version 2026.07.27 \
  --description "Llama 3.1 8B base NLP baseline"
```

Run the chat-template benchmarks on an instruct model:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-8b \
  --evals chat
```

Run only MMLU and GSM8K, cap each evaluation at 32 instances, and return after submission:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model llama3.1-8b-instruct \
  --evals mmlu,gsm8k \
  --limit 32 \
  --no-wait
```

Run a two-task Terminal-Bench 2 Harbor check on the validated Qwen3-32B GPU shape:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-32b \
  --evals tb2-lite
```

Run one capped Evalchemy task and one capped Harbor benchmark against the same model server:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-32b \
  --evals mmlu,tb2-lite \
  --limit 2
```

Run the full Terminal-Bench 2 preset or all standard Harbor presets:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-32b \
  --evals tb2 \
  --no-wait

uv run python -m experiments.evaluation.cli launch \
  --model qwen3-32b \
  --evals agentic \
  --no-wait
```

`agentic` runs seven full Harbor datasets and can consume substantial Daytona and inference
capacity. Use `tb2-lite`, `swebench-lite`, `aime-smoke`, or `--limit N` for validation runs.

Run one Grug/OpenCode trial with the model and agent policy registered for that benchmark:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model grug-agentic-s3-step1903 \
  --evals grug-opencode-id \
  --limit 1
```

The `qwen3-32b` / `tb2-lite` path follows the H100x2 acceptance run recorded in
[issue #6503](https://github.com/marin-community/marin/issues/6503). Harbor trial restore and
persistence use the selected GCS or S3 records store after
[PR #7625](https://github.com/marin-community/marin/pull/7625).

### Object-store model loading

CUDA vLLM serves `s3://` checkpoints with RunAI model streamer 0.16.1. The launcher gives the
streamer's low-speed check 10 seconds and sends its warning and error logs to the Iris job log. The
Grug H100x8 entry also enables distributed loading: each data-parallel rank reads its assigned
weights and broadcasts them locally instead of all eight ranks reading the full 124.9 GiB export.

The following variables may prefix an evaluation command and are forwarded to the remote inference
child:

- `RUNAI_STREAMER_S3_REQUEST_TIMEOUT_MS` overrides the 10,000 ms low-speed window.
- `RUNAI_STREAMER_CONCURRENCY` controls the number of object-store reader workers. Try `4` if an S3
  endpoint fails under the default concurrency of `8`; this may increase model-load time.
- `RUNAI_STREAMER_S3_MAX_INFLIGHT_MIB` caps in-flight data per S3 client on RunAI 0.16.1. Leave the
  derived default in place unless concurrency reduction is insufficient.
- `RUNAI_STREAMER_CHUNK_BYTESIZE` changes the default 8 MiB object-store chunk. Benchmark changes;
  larger chunks reduce request count and increase memory per request.
- `RUNAI_STREAMER_LOG_TO_STDERR`, `RUNAI_STREAMER_LOG_LEVEL`, and `RUNAI_STREAMER_S3_TRACE` control
  diagnostics. S3 trace logging writes a file and is intended for a bounded reproduction.

For example, retry a model-load failure with half the default reader concurrency:

```bash
RUNAI_STREAMER_CONCURRENCY=4 \
uv run python -m experiments.evaluation.cli launch \
  --model grug-agentic-s3-step1903 \
  --evals grug-opencode-id \
  --limit 1
```

Do not set `AWS_RETRY_MODE` or `AWS_MAX_ATTEMPTS` to increase retries for this loader. The AWS CRT
client bundled with RunAI already retries each request five times, and that build does not propagate
`AWS_MAX_ATTEMPTS` to the CRT retry count. Marin separately retries vLLM startup up to three times
when RunAI reports a transient read failure.

### Named suites

| Suite | Evaluator | Contents and constraints |
| --- | --- | --- |
| `smoke` | Evalchemy | Capped MMLU abstract algebra and GSM8K; use before a larger run. |
| `core` | Evalchemy | Eleven OpenLLM-style, generation, code, and MATH500 evaluations. Use a model with a chat template. |
| `nlp` | Evalchemy | Fourteen log-likelihood and greedy NLP evaluations; supports base and instruct models. |
| `chat` | Evalchemy | MATH500, AIME24, and OlympiadBench; requires a chat-template model. |
| `math` | Evalchemy | MATH500, AIME24, and zero-shot GSM8K; requires a chat-template model. |
| `agentic` | Harbor | Terminal-Bench 2, SWE-bench, GAIA, BFCL, Aider, MedAgentBench, and FinanceAgent in Daytona. |

The `code` suite is registered but is not runnable with the current pinned evaluation image because
its HumanEvalPlus and MBPPPlus dependencies are absent. Individual evaluation keys are defined in
`experiments/evaluation/evals.py`. Useful Harbor keys outside `agentic` include `tb2-lite`,
`swebench-lite`, `swebench-full`, `aime-smoke`, `aime-harbor`, and `grug-opencode-id`.

### Choose a model

`--model` takes a registry key, not an arbitrary Hugging Face repository:

| Model key | Model type and default placement | Typical selection |
| --- | --- | --- |
| `qwen3-0.6b` | Instruct, TPU | `smoke` |
| `llama-3.1-8b-base` | Base, TPU | `nlp` |
| `llama3.1-8b-instruct` | Instruct, TPU | `core`, `nlp`, or `chat` |
| `qwen3-32b` | Instruct, CoreWeave H100x2 | `tb2-lite`, `tb2`, or `agentic` |
| `grug-agentic-s3-step1903` | Instruct export, CoreWeave H100x8 | `grug-opencode-id` |

The complete catalog is the union of `experiments/evaluation/models.py` and YAML files under
`experiments/evaluation/serve/models/`. The
[model catalog schema](https://github.com/marin-community/marin/blob/main/experiments/evaluation/serve/models/README.md)
documents how to register a Hugging Face model or object-store checkpoint.

### Launch controls

- `--evals smoke` selects a named suite. `--evals mmlu,gsm8k` selects explicit keys. One invocation
  may mix Evalchemy and Harbor keys.
- `--limit N` overrides the configured instance cap for every selected evaluation.
- `--no-wait` returns after Iris submission. Without it, the command waits for terminal records and
  prints their metrics.
- `--platform tpu|gpu` overrides the model's default platform when the model resource hint supports
  that platform.
- `--accelerator v6e-8` or `--accelerator H100x8` requests an exact compatible slice.
- `--version` and `--description` attach run metadata.
- `--cluster` changes the submitting Iris cluster. `--records-prefix` changes the result store.

Each invocation serves the model once and evaluates the selected keys in order. An evaluation
failure gets its own terminal record and does not skip later evaluations. An inference failure
records the current and remaining evaluations as infrastructure failures.

### Harbor credentials

Daytona-backed definitions first use `DAYTONA_API_KEY` from the launch environment, then use the
approved `DAYTONA_EVAL_API_KEY` version in Google Secret Manager. Do not put resolved credentials
in model YAML, evaluation definitions, or command arguments. Evalchemy-only runs do not require a
Daytona credential. See [Harbor evaluation](../harbor-integration.md) for dataset, agent, and
endpoint details.

### Results

TPU-routed runs default to `gs://marin-eval-metadata/runs`. CoreWeave GPU runs default to
`s3://marin-us-east-02a/marin/eval-metadata/runs`. `--dry-run` prints the effective prefix.

Every selected evaluation writes `{records_prefix}/{run_id}/record.json` plus its mechanism-specific
results and normalized sample parquet. Harbor also persists trial directories and trajectories in
the same GCS or S3 results tree. A Harbor trial with `exception_info` marks the evaluation failed
after its artifacts are saved; a verifier-scored zero without an exception remains a completed
evaluation with a zero score.

[Evaldash](https://evaldash.oa.dev) indexes records from both default stores. The record is the
source of truth for model, evaluation identity, status, metrics, hardware, provenance, and Iris job
paths.

## Use the shared launcher in a pipeline

`experiments.evaluation.pipeline.eval_step` wraps the same registered model and evaluation
selection in a lazy artifact. Save the pipeline as `eval_pipeline.py`:

```python
from experiments.evaluation.pipeline import eval_step
from marin.execution.step_runner import StepRunner

step = eval_step(
    "qwen3-1.7b",
    "smoke",
    version="2026.07.27",
    limit=32,
)
StepRunner().run([step.lower()])
```

Run the pipeline itself inside an Iris job so `eval_step` can submit its
orchestrator job:

```bash
uv run iris --cluster marin job run -- python eval_pipeline.py
```

The step uses the same serving, executor, record, and result paths as the CLI. Its artifact identity
includes the model, evaluation selection, limit, and version.

## Compose Evalchemy steps from Levanter checkpoints

The lower-level post-hoc API accepts an `ArtifactStep[LevanterCheckpoint]`. Use the return value of
a training run, or adopt an existing checkpoint with `ArtifactStep.adopt`. These APIs compose
Evalchemy task groups directly; Harbor evaluations use the shared launcher above.

### Core APIs

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

### Run a named suite

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

### Compose your own groups

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

### Read results

`resolve` (or `run`) returns the typed artifacts. An `EvalReport` carries the merged metrics:

```python
from marin.execution.lazy import resolve

report_artifact = resolve(report)
print(report_artifact.task_metrics)  # {task: {metric: value}}
print(report_artifact.averages)      # suite-level rollups
```

Each individual result is an `EvalchemyResult`; `task_metrics()` reads the per-task scores from the
evalchemy output tree.

### Run the repository example scripts

The checked-in examples use the lower-level API for adopted Levanter checkpoints. They are
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

### Serve a composed group on GPU

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

### Parameter reference

#### `EvalGroup`

- `config`: the `EvalchemyRunConfig`, including tasks, generation settings, concurrency, and limits.
- `serve`: the model-server behavior (`vllm` or `levanter` and its server options).
- `resource_hint`: optional CPU, memory, and disk overrides for the inference worker.
- `accelerator`: the exact TPU or GPU slice used for inference.
- `tokenizer`: HF tokenizer id the eval client loads; defaults to the served checkpoint. Set it to a
  base-model HF id when serving a `gs://` path the Evalchemy client cannot load a tokenizer from.
- `discover_latest_checkpoint`: whether to resolve the latest HF checkpoint under the model path.

#### `eval_report`

- `results`: the `EvalResult` steps to aggregate.
- `name`: the report's identity segment (`evaluation/report/{name}`).

For deeper dives, see:

- [Evaluation Overview](../explanations/evaluation.md)
- `experiments/evaluation/README.md`
- `experiments/evaluation/evals.py`
- `experiments/evaluation/models.py`
- `experiments/evals/task_configs.py`
- `experiments/evals/evals.py`
