# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tutorial: Training a small model on TinyStories using TPU (lazy-artifact style).

Every training decision is stated inline: the model, the data, the optimizer, the token
budget. The library's :func:`~marin.experiment.train.train_lm` handles only the TPU
plumbing (the mesh, the resumption checkpointer, the Fray dispatch).

For CPU training, see train_tiny_model_cpu.py.
For GPU training, see train_tiny_model_gpu.py.
"""

from fray import ResourceConfig
from levanter.optim import AdamConfig
from marin.execution.lazy import Checkpoint, lower
from marin.execution.step_runner import StepRunner
from marin.experiment.data import mixture, tokenized
from marin.experiment.train import train_lm

from experiments.llama import llama_30m
from experiments.marin_tokenizer import marin_tokenizer

RESOURCES = ResourceConfig.with_tpu(
    "v5litepod-16",
    slice_count=1,
    cpu=32,
    ram="128g",
    disk="50g",
)

# 1. Choose a dataset and tokenize it inline.
# sample_count=1000 caps documents per shard; it bears identity (a sampled cache
# differs from the full one).
tok = tokenized(
    "roneneldan/TinyStories",
    tokenizer=marin_tokenizer,
    source="roneneldan/TinyStories",
    sample_count=1000,
)


def build(*, version: str = "v1") -> Checkpoint:
    """A 30M-parameter Llama model trained on TinyStories (TPU), every decision stated inline."""
    return train_lm(
        name="checkpoints/marin-tinystories-30m",
        version=version,
        model=llama_30m,
        optimizer=AdamConfig(learning_rate=6e-4, weight_decay=0.1),
        # 2. Single-component mixture: all training tokens from TinyStories.
        data=lambda ctx: mixture(ctx, {tok: 1.0}),
        deps=(tok,),
        batch_size=128,
        seq_len=llama_30m.max_seq_len,  # 1024
        num_train_steps=10000,
        z_loss_weight=None,
        evals=None,  # no point running evals on such a small model
        resources=RESOURCES,
        tags=["llama", "30m", "tinystories", "tutorial"],
    )


if __name__ == "__main__":
    # Lower the checkpoint to a StepSpec graph and run it: TinyStories tokenizes
    # (cached), then one TPU training job runs.
    StepRunner().run([lower(build())])
