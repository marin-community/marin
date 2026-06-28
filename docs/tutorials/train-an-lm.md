# Training a Language Model

This tutorial walks through training a language model with Marin's lazy artifact API,
using the DCLM 1B/1x baseline as a concrete example.

## Prerequisites

- Basic [installation](installation.md)
- Access to an Iris cluster or a local GPU (see [`lib/iris/OPS.md`](https://github.com/marin-community/marin/blob/main/lib/iris/OPS.md))

## How a training script is structured

A Marin training script builds a lazy `Checkpoint` handle, lowers it to a runnable step
graph, and passes it to `StepRunner`. Nothing executes at import time.

```python
from marin.execution.lazy import lower
from marin.execution.step_runner import StepRunner
from marin.experiment.train import train_lm

def build() -> Checkpoint:
    ...  # assemble the checkpoint handle

if __name__ == "__main__":
    StepRunner().run([lower(build())])
```

`StepRunner.run` walks the dependency graph, checks the cache for each step, and runs any
that are missing or explicitly forced. Dataset tokenization runs before training; a step
that already succeeded is skipped.

## Defining the model

Choose a Levanter model configuration. The full architecture is stated as literals so it
enters the artifact's fingerprint:

```python
from levanter.models.llama import LlamaConfig

SEQ_LEN = 2048
BATCH_SIZE = 256
NUM_TRAIN_TOKENS = 28.8e9  # 1B-1x, Chinchilla-optimal for ~1.4B parameters
NUM_TRAIN_STEPS = int(NUM_TRAIN_TOKENS) // (BATCH_SIZE * SEQ_LEN)

llama_1_4b = LlamaConfig(
    max_seq_len=SEQ_LEN,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=16,
    num_kv_heads=16,
    num_layers=24,
)
```

Pre-defined configurations for common sizes live in `experiments/llama.py`.

## Building the data mixture

Training data is expressed as a dict of `Dataset` handles to weights. Each handle is a
lazy reference to a tokenized cache on GCS; `mixture` assembles a Levanter `LmDataConfig`
from them at run time:

```python
from marin.experiment.data import mixture
from experiments.pretraining_datasets.dclm import DCLM_MIXTURE_WEIGHTS, dclm_datasets
from experiments.llama import llama3_tokenizer

train = dclm_datasets(tokenizer=llama3_tokenizer)
weighted = {train[name]: DCLM_MIXTURE_WEIGHTS[name] for name in train}
```

`dclm_datasets` returns a dict of pre-tokenized `Dataset` handles, one per DCLM component.
To tokenize a custom dataset instead, use `tokenized` from `marin.experiment.data`:

```python
from marin.experiment.data import tokenized
from experiments.marin_tokenizer import marin_tokenizer

my_data = tokenized(
    name="tokenized/my-dataset",
    source="org/my-hf-dataset",  # HuggingFace dataset id
    tokenizer=marin_tokenizer,
)
```

## Calling `train_lm`

`train_lm` takes every identity-bearing decision as a required argument and defaults none
of them. The complete `build()` function for DCLM 1B/1x:

```python
from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig
from marin.execution.lazy import Checkpoint, lower
from marin.execution.step_runner import StepRunner
from marin.experiment.data import mixture
from marin.experiment.train import train_lm
from experiments.evals.uncheatable import uncheatable_validation
from experiments.llama import llama3_tokenizer
from experiments.paloma import paloma_validation
from experiments.pretraining_datasets.dclm import DCLM_MIXTURE_WEIGHTS, dclm_datasets
from experiments.recipes import core_tasks

TRAIN_RESOURCES = ResourceConfig.with_tpu("v4-128")

def build(*, version: str = "v1") -> Checkpoint:
    train = dclm_datasets(tokenizer=llama3_tokenizer)
    validation = [
        *paloma_validation(tokenizer=llama3_tokenizer),
        *uncheatable_validation(tokenizer=llama3_tokenizer),
    ]
    weighted = {train[name]: DCLM_MIXTURE_WEIGHTS[name] for name in train}

    return train_lm(
        name="checkpoints/dclm_1b_1x",
        version=version,
        model=llama_1_4b,
        optimizer=AdamConfig(
            learning_rate=3e-3,
            weight_decay=0.033,
            warmup=5000,
            min_lr_ratio=0.1,
        ),
        data=lambda ctx: mixture(ctx, weighted, validation=validation),
        deps=(*weighted, *validation),
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=NUM_TRAIN_STEPS,
        z_loss_weight=1e-4,
        evals=core_tasks(every=10000),
        resources=TRAIN_RESOURCES,
        tags=["DCLM_1B_1X"],
    )

if __name__ == "__main__":
    StepRunner().run([lower(build())])
```

### Required arguments

| Argument | Purpose |
|---|---|
| `name` | Artifact name; forms the output path `{prefix}/{name}/{version}` |
| `version` | Artifact version; bump this to produce a new run without overwriting the old one |
| `model` | Levanter `LmConfig` (architecture and hyperparameters) |
| `optimizer` | Levanter `OptimizerConfig` (learning rate, schedule, weight decay) |
| `data` | A builder `(ctx) -> LmDataConfig`; typically a `mixture(...)` call |
| `deps` | Dataset handles the `data` builder depends on — must list all of them |
| `batch_size` | Training batch size in sequences |
| `seq_len` | Sequence length |
| `num_train_steps` | Total number of optimizer steps |
| `z_loss_weight` | Auxiliary z-loss coefficient; pass `None` to omit |
| `evals` | `EvalSuite` or `None`; pass `None` to opt out of harness evals |
| `resources` | The hardware to dispatch training onto (a run-arg, excluded from fingerprint) |

`train_lm` owns the mechanical plumbing that is identical across runs: the data-parallel
mesh, the rolling resumption checkpointer, W&B metric replication, and the Fray dispatch of
the training job. None of those are experiment decisions.

### GPU variant

Swap `ResourceConfig.with_tpu(...)` for `ResourceConfig.with_gpu("H100", count=8)` (or
any other GPU spec). Everything else stays the same.

## Running the experiment

Submit the script as a CPU-only Iris job. `StepRunner` inside the script dispatches the
TPU or GPU training sub-job via Fray:

```bash
uv run iris --cluster=marin job run \
  --cpu=1 --memory=2G --extra=cpu \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python -m experiments.tutorials.dclm_1b_1x_inline
```

See [`lib/iris/OPS.md`](https://github.com/marin-community/marin/blob/main/lib/iris/OPS.md)
for the full `iris job run` reference, including `--no-wait` for detached submission and
`iris job logs -f` for log streaming.

## Monitoring training

W&B receives metrics throughout training. The run name defaults to the `run_id` argument
(when omitted, `train_lm` derives one from the artifact name). Checkpoints are written to
`{prefix}/{name}/{version}/checkpoints/`.

## Memory pressure

If training OOMs, see [Making Things Fit in HBM](../references/hbm-optimization.md) for a
practical tuning checklist covering gradient checkpointing, activation offloading, and
tensor parallelism.

## Reference implementation

`experiments/tutorials/dclm_1b_1x_inline.py` is the canonical training script: every
decision — model, data mixture, optimizer, token budget, z-loss, evals — is stated inline
and visible without opening another file.
