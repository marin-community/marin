# Training a Language Model

This tutorial walks through training a language model with Marin's lazy artifact API,
using the DCLM 1B/1x baseline as a concrete example.

## Prerequisites

- Basic [installation](installation.md)
- Access to an Iris cluster or a local GPU (see [`lib/iris/OPS.md`](https://github.com/marin-community/marin/blob/main/lib/iris/OPS.md))

## How a training script is structured

A Marin training script builds a lazy `ArtifactStep[LevanterCheckpoint]` handle, lowers it to
a runnable step graph, and passes it to `StepRunner`. Nothing executes at import time.

```python
from marin.execution.lazy import ArtifactStep, lower
from marin.execution.step_runner import StepRunner
from marin.experiment.train import train_lm
from marin.training.training import LevanterCheckpoint

def build() -> ArtifactStep[LevanterCheckpoint]:
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

Training data is expressed as a dict of `ArtifactStep[TokenizedCache]` handles to weights. Pass
this dict as `datasets=` to `train_lm`; it assembles the Levanter data mixture internally:

```python
from experiments.datasets.dclm import DCLM_MIXTURE_WEIGHTS, dclm_datasets
from experiments.llama import llama3_tokenizer

train = dclm_datasets(tokenizer=llama3_tokenizer)
weighted = {train[name]: DCLM_MIXTURE_WEIGHTS[name] for name in train}
```

`dclm_datasets` returns a dict of pre-tokenized `ArtifactStep[TokenizedCache]` handles, one per
DCLM component. To tokenize a custom dataset instead, use `tokenized` from `marin.experiment.data`:

```python
from marin.experiment.data import tokenized
from experiments.marin_tokenizer import marin_tokenizer

my_data = tokenized(
    name="tokenized/my-dataset",
    source="org/my-hf-dataset",  # HuggingFace dataset id
    tokenizer=marin_tokenizer,
    version="2026.06.28",
)
```

## Calling `train_lm`

`train_lm` takes every identity-bearing decision as a required argument and defaults none
of them. The complete `build()` function for DCLM 1B/1x:

```python
from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig
from marin.execution.lazy import ArtifactStep, lower
from marin.execution.step_runner import StepRunner
from marin.experiment.train import train_lm
from marin.training.training import LevanterCheckpoint
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.llama import llama3_tokenizer
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.dclm import DCLM_MIXTURE_WEIGHTS, dclm_datasets
from experiments.recipes import core_tasks

TRAIN_RESOURCES = ResourceConfig.with_tpu("v4-128")

def build(*, version: str = "2026.06.28") -> ArtifactStep[LevanterCheckpoint]:
    train = dclm_datasets(tokenizer=llama3_tokenizer)
    validation = [
        *paloma_datasets(tokenizer=llama3_tokenizer).values(),
        *uncheatable_datasets(tokenizer=llama3_tokenizer).values(),
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
        datasets=weighted,
        validation=validation,
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
| `datasets` | Dict of `ArtifactStep[TokenizedCache]` handles to weights; `train_lm` assembles the data mixture |
| `validation` | Sequence of `ArtifactStep[TokenizedCache]` handles for held-out loss tracking (optional) |
| `batch_size` | Training batch size in sequences |
| `seq_len` | Sequence length |
| `num_train_steps` | Total number of optimizer steps |
| `z_loss_weight` | Auxiliary z-loss coefficient; pass `None` to omit |
| `evals` | `EvalSuite` or `None`; pass `None` to opt out of harness evals |
| `resources` | The hardware to dispatch training onto (a runtime arg, excluded from fingerprint) |

`train_lm` owns the mechanical plumbing that is identical across runs: the data-parallel
mesh, the rolling resumption checkpointer, W&B metric replication, and the Fray dispatch of
the training job. None of those are experiment decisions.

### GPU variant

Swap `ResourceConfig.with_tpu(...)` for
`ResourceConfig.with_gpu("H100", count=8, regions=[ANY_REGION])` (or any other GPU spec).
Marin's H100s live in a federated CoreWeave cluster, so a GPU run also needs an `s3://`
storage prefix and a `--target-cluster` pin on the submission — see [Training on Cloud
GPUs](cloud-gpu.md). Everything else about the script stays the same.

## Running the experiment

Submit the script as a CPU-only Iris job. `StepRunner` inside the script dispatches the
TPU training sub-job via Fray:

```bash
uv run iris --cluster=marin job run \
  --cpu=1 --memory=2G --extra=cpu \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python -m experiments.tutorials.exp1078_reproduce_dclm_7b1x
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

`experiments/tutorials/exp1078_reproduce_dclm_7b1x.py` is the canonical training script: every
decision — model, data mixture, optimizer, token budget, z-loss, evals — is stated inline
and visible without opening another file. It reproduces the DCLM 7B/1x baseline; the same
structure scales down to smaller models by changing the `LlamaConfig` and token budget.
