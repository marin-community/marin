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

`tokenized` from `marin.experiment.data` returns a `Dataset` handle — a lazy reference to
a Levanter tokenized cache. The tokenization step runs before training, and its output is
cached for future runs.

```python
from marin.experiment.data import tokenized
from experiments.marin_tokenizer import marin_tokenizer

tinystories_tokenized = tokenized(
    name="tokenized/tinystories",
    source="roneneldan/TinyStories",  # HuggingFace dataset id
    tokenizer=marin_tokenizer,
    sample_count=1000,  # cap at 1 000 samples per shard to keep the tutorial fast
)
```

`tinystories_tokenized` is a `Dataset` handle. Constructing it does not download or
tokenize anything. The actual work happens when `StepRunner` encounters this step in the
dependency graph.

## Step 2: Choose a model configuration

```python
from levanter.models.llama import LlamaConfig

# A tiny Llama for fast local testing.
llama_nano = LlamaConfig(
    max_seq_len=2048,
    hidden_dim=128,
    intermediate_dim=512,
    num_heads=4,
    num_kv_heads=4,
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
argument and returns a lazy `Checkpoint` handle. It handles the mechanical plumbing —
the mesh, the checkpointer, the Fray dispatch — while you supply the policy.

```python
from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig
from marin.execution.lazy import Checkpoint
from marin.experiment.data import mixture
from marin.experiment.train import train_lm

BATCH_SIZE = 4
SEQ_LEN = 2048
NUM_TRAIN_STEPS = 100


def build() -> Checkpoint:
    return train_lm(
        name="checkpoints/marin-nano-tinystories",
        version="v1",
        model=llama_nano,
        optimizer=AdamConfig(learning_rate=6e-4, weight_decay=0.1),
        data=lambda ctx: mixture(ctx, {tinystories_tokenized: 1.0}),
        deps=[tinystories_tokenized],
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
- `data` is a builder called with the run's `RunContext`; `mixture` resolves each dataset
  handle to its GCS path at run time.
- `deps` lists every `Dataset` handle the `data` builder uses, so `StepRunner` ensures
  they are materialized first.
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
uv run python my_experiment.py --prefix local_store
```

The `--prefix` argument sets the root directory for all outputs. It can be a local path or
anything [fsspec](https://filesystem-spec.readthedocs.io/en/latest/) supports (e.g.
`gs://`). You can also set `MARIN_PREFIX` in the environment instead. See
[Understanding `MARIN_PREFIX` and `--prefix`](../explanations/marin-prefix.md).

This takes a few minutes on a CPU. The output ends with something like:

```
INFO step_runner.py -- All steps complete.
```

## Inspecting the artifacts

After the run, your prefix directory contains:

```
local_store/
  tokenized/tinystories/v1/    # the tokenized dataset cache
  checkpoints/marin-nano-tinystories/v1/  # the model checkpoint
```

Each artifact is at a stable, human-readable path determined by its `name` and `version`.
Rerunning the same script skips steps whose outputs already exist.

### Rerunning a failed step

`StepRunner` skips steps that succeeded. To force a failed step to rerun:

```bash
uv run python my_experiment.py --prefix local_store --force_run_failed true
```

### Rerunning a succeeded step

Remove the artifact directory, then rerun the script. Alternatively, bump the `version`
in `train_lm` to produce a new artifact at a new path without touching the old one.

## Next steps

- Train a full [1B parameter model](train-an-lm.md) using the DCLM mixture.
- Learn how lazy artifacts work in [Lazy artifacts](../explanations/lazy-artifacts.md).
- Read about the full [language modeling pipeline](../explanations/lm-pipeline.md).
