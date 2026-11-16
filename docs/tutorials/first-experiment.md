# First Experiment: Train a Tiny Model on TinyStories

In this tutorial, you will run your first Marin experiment: training a tiny language model
on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.
The goal is not to train a good model — it is to run something from start to finish.

We assume you have already gone through the [installation](installation.md) tutorial.

## What you will do

1. Write a minimal experiment script using Marin's lazy artifact API.
2. Run it locally on CPU.
3. Inspect the artifacts it produces.

## The structure of a Marin experiment

A Marin experiment script constructs lazy artifact handles, lowers them to a step graph,
and hands the graph to `StepRunner`. Nothing runs at import time; all I/O happens in
`StepRunner.run`.

```python
from marin.execution.lazy import lower
from marin.execution.step_runner import StepRunner

def build():
    ...  # return a lazy artifact handle

if __name__ == "__main__":
    StepRunner().run([lower(build())])
```

## Step 1: Tokenize the dataset

`tokenized` from `marin.experiment.data` returns an `ArtifactStep[TokenizedCache]` handle — a
lazy reference to a Levanter tokenized cache. The tokenization step runs before training, and
its output is cached for future runs.

```python
from marin.experiment.data import tokenized
from experiments.marin_tokenizer import marin_tokenizer

tinystories_tokenized = tokenized(
    name="tokenized/tinystories",
    source="roneneldan/TinyStories",  # HuggingFace dataset id
    tokenizer=marin_tokenizer,
    version="2026.06.28",
    sample_count=1000,  # cap at 1 000 samples per shard to keep the tutorial fast
)
```

`tinystories_tokenized` is an `ArtifactStep[TokenizedCache]` handle. Constructing it does not download or
tokenize anything. The actual work happens when `StepRunner` encounters this step in the
dependency graph.

## Step 2: Choose a model configuration

```python
from levanter.models.llama import LlamaConfig

# A tiny Llama for fast local testing.
llama_nano = LlamaConfig(
    max_seq_len=512,
    hidden_dim=32,
    intermediate_dim=128,
    num_heads=2,
    num_kv_heads=2,
    num_layers=2,
)
```

The `llama_nano` configuration from `experiments/llama.py` defines a model with the same
shape, pre-tuned for CPU runs. You can import it directly:

```python
from experiments.llama import llama_nano
```

## Step 3: Assemble the training run

`train_lm` from `marin.experiment.train` takes every experiment decision as an explicit
argument and returns an `ArtifactStep[LevanterCheckpoint]` handle. It handles the mechanical
plumbing — the mesh, the checkpointer, the Fray dispatch — while you supply the policy.

```python
from fray.cluster import ResourceConfig
from levanter.optim.config import AdamConfig
from marin.execution.lazy import ArtifactStep
from marin.experiment.train import train_lm
from marin.training.training import LevanterCheckpoint

BATCH_SIZE = 4
SEQ_LEN = llama_nano.max_seq_len
NUM_TRAIN_STEPS = 100


def build() -> ArtifactStep[LevanterCheckpoint]:
    return train_lm(
        name="checkpoints/marin-nano-tinystories",
        version="2026.06.28",
        model=llama_nano,
        optimizer=AdamConfig(learning_rate=6e-4, weight_decay=0.1),
        datasets={tinystories_tokenized: 1.0},
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=NUM_TRAIN_STEPS,
        z_loss_weight=None,
        evals=None,  # skip harness evals for this tiny tutorial run
        resources=ResourceConfig.with_cpu(),
    )
```

Key arguments:

- `name` and `version` form the output path `{prefix}/{name}/{version}`.
- `datasets` is a dict of `ArtifactStep[TokenizedCache]` handles to weights; `train_lm`
  assembles the Levanter data mixture and resolves each dataset to its path at run time. Dataset
  dependencies are inferred automatically — no separate `deps` list is needed.
- `resources=ResourceConfig.with_cpu()` keeps the run local (no TPU or GPU needed).

## Step 4: Wire the main block

```python
from marin.execution.lazy import lower
from marin.execution.step_runner import StepRunner

if __name__ == "__main__":
    StepRunner().run([lower(build())])
```

`lower(build())` traverses the dependency graph from `build()` and converts each handle
into a `StepSpec`. `StepRunner.run` checks the cache for each step and runs any that are
missing.

## Running the experiment

```bash
MARIN_PREFIX=local_store uv run python my_experiment.py
```

`MARIN_PREFIX` sets the root directory for all outputs. It can be a local path or anything
[fsspec](https://filesystem-spec.readthedocs.io/en/latest/) supports (e.g. `gs://`). A
relative path such as `local_store` resolves against the directory you run from. If
you already exported `MARIN_PREFIX` in your shell, just run `uv run python my_experiment.py`.
See [Understanding `MARIN_PREFIX`](../explanations/marin-prefix.md).

Training reports metrics to Weights & Biases. Without an explicit `run_id`, `train_lm` names
the W&B run ID after the last segment of the output path, which here is the version
`2026.06.28`. A second person following this tutorial against the same project reuses that
ID and can get a `403` when W&B attempts to resume the existing run. Either
keep the run local with `WANDB_MODE=offline`, or pass `run_id="<your-name>-nano-tinystories"`
to `train_lm` so the run is yours:

```bash
MARIN_PREFIX=local_store WANDB_MODE=offline uv run python my_experiment.py
```

This takes a few minutes on a CPU. The output ends with something like:

```
INFO step_runner.py -- All steps complete.
```

## Inspecting the artifacts

After the run, your prefix directory contains:

```
local_store/
  tokenized/tinystories/2026.06.28/    # the tokenized dataset cache
  checkpoints/marin-nano-tinystories/2026.06.28/  # the model checkpoint
```

Each artifact is at a stable, human-readable path determined by its `name` and `version`.
Rerunning the same script skips steps whose outputs already exist.

### Rerunning a failed step

`StepRunner` skips steps that succeeded and reruns steps that failed, so a failed step
retries on the next invocation. To make a previous failure raise instead of retrying,
pass `force_run_failed=False`:

```python
StepRunner().run([lower(build())], force_run_failed=False)
```

### Rerunning a succeeded step

Remove the artifact directory, then rerun the script. Alternatively, bump the `version`
in `train_lm` to produce a new artifact at a new path without touching the old one.

### Checked-in smoke launcher

The repository also includes a small launcher for checking the standard training path.
It prints its plan by default; pass `--run` to execute it. Use `--version dev` while
iterating, or a calendar version to pin outputs:

```bash
# Print the CPU/TinyStories plan.
uv run python -m experiments.tutorials.train_tiny_model \
  --device cpu --dataset tinystories --version dev

uv run python -m experiments.tutorials.train_tiny_model \
  --device cpu --dataset tinystories --version dev --run
```

If the launcher appears to do nothing, check whether `--run` was omitted. TinyStories and
WikiText tokenize a 1,000-document sample; `fineweb-edu` downloads a prebuilt tokenized cache
from Hugging Face and trains `llama_150m` from `experiments/llama.py`.

### Running the launcher on a cluster

The same launcher runs on an accelerator when you submit it as an Iris job. `--device`
accepts `h100x1`, `h100x8`, `gb200x1`, `gb200x4`, `v5litepod-16`, and `v6e-4`. Run this
from the checkout root after `iris --cluster=marin login`:

```bash
uv run iris --cluster=marin job run --cpu=1 --memory=2G --extra=cpu --region us-east5 \
  -- python -m experiments.tutorials.train_tiny_model \
     --device v6e-4 --dataset wikitext --version dev --run
```

`--region` pins the coordinator, and the accelerator job inherits its region so training stays
next to the data. The selected region must offer the requested device. Omit `--region` to let
Iris select a compatible location.

The job you submit is a one-CPU coordinator that runs the launcher; it has no accelerator
of its own. Inside it, `train_lm` requests the accelerator through Fray, Marin's job-submission
layer, and Iris schedules that request as a second job, a child of the coordinator. The
`--cpu` and `--memory` flags size the coordinator only. `--extra=cpu` installs the CPU
dependency set (the `cpu` extra in `lib/marin/pyproject.toml`) on the coordinator; the child
installs the `tpu` or `gpu` extra for its device. Do not set `MARIN_PREFIX`: the cluster
provides one, and the `dev` version places the checkpoint under `users/<you>/checkpoints/`
so it cannot collide with another person's run. Your `WANDB_API_KEY` is copied from your shell; see
[Local runs versus submitted jobs](installation.md#local-runs-versus-submitted-jobs) for
what else reaches a job.

The H100 and GB200 devices live on CoreWeave clusters. Replace the GCP `--region` selector
with a CoreWeave target and use the matching device:

```bash
uv run iris --cluster=marin job run --cpu=1 --memory=2G --extra=cpu \
  --target-cluster cw-rno2a \
  -- python -m experiments.tutorials.train_tiny_model \
     --device h100x8 --dataset wikitext --version dev --run
```

For GB200, use `--target-cluster cw-us-east-08a` with `--device gb200x1` or `gb200x4`.
`iris cluster list` shows every configured cluster. [Training on Cloud GPUs](cloud-gpu.md)
covers the storage and credential differences.

`job run` prints the job id on submission, in the form `/<username>/<job-name>`, and streams
logs until the job ends. From another shell:

```bash
uv run iris --cluster=marin job logs -f /<username>/<job-name>
uv run iris --cluster=marin job describe /<username>/<job-name>
```

## Next steps

- Train a full [1B parameter model](train-an-lm.md) using the DCLM mixture.
- Learn how lazy artifacts work in [Lazy artifacts](../explanations/lazy-artifacts.md).
- Read about the full [language modeling pipeline](../explanations/lm-pipeline.md).
