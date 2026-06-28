# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""How To: Replicating DCLM 1B/1x in Marin (lazy-artifact style).

Every training *decision* is built here, in plain config: the model, the data mixture,
the optimizer, the token budget, the z-loss, and the eval cadence are all visible in
this file. The library's :func:`~marin.experiment.train.train_lm` assembles them into a
lazy :class:`~marin.execution.lazy.Checkpoint` — no executor, no import-time step graph,
no content-addressing — handling only the mechanical marin-on-TPU plumbing (the mesh,
the resumption checkpointer, the eval-harness wiring, the Fray dispatch). It bakes in no
optimizer, mixture, or eval suite of its own.

A golden test (``tests/experiment/test_train_lm_golden.py``) pins this to the
``default_train`` recipe's resolved decisions so the readable inline code and the
executed config cannot drift.
"""

from fray.cluster import ResourceConfig
from levanter.models.llama import LlamaConfig
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

SEQ_LEN = 2048
BATCH_SIZE = 256
NUM_TRAIN_TOKENS = 28.8e9  # 1B-1x DCLM competition scale (Chinchilla-optimal for 1.4B)
NUM_TRAIN_STEPS = int(NUM_TRAIN_TOKENS) // (BATCH_SIZE * SEQ_LEN)

# The TPU each training job is dispatched onto. A run-arg, not part of the checkpoint's
# identity: re-running on different hardware is the same artifact.
TRAIN_RESOURCES = ResourceConfig.with_tpu("v4-128")

# 1.4B-parameter Llama, DCLM 1B-1x competition scale.
llama_1_4b_dclm = LlamaConfig(
    max_seq_len=SEQ_LEN,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=16,
    num_kv_heads=16,
    num_layers=24,
)


def build(*, version: str = "v1") -> Checkpoint:
    """The DCLM 1B/1x training run as a lazy checkpoint, every decision stated inline."""
    train = dclm_datasets(tokenizer=llama3_tokenizer)
    validation = [*paloma_validation(tokenizer=llama3_tokenizer), *uncheatable_validation(tokenizer=llama3_tokenizer)]
    weighted = {train[name]: DCLM_MIXTURE_WEIGHTS[name] for name in train}

    return train_lm(
        name="checkpoints/dclm_1b_1x_how_to",
        version=version,
        model=llama_1_4b_dclm,
        optimizer=AdamConfig(learning_rate=3e-3, weight_decay=0.033, warmup=5000, min_lr_ratio=0.1),
        data=lambda ctx: mixture(ctx, weighted, validation=validation),
        deps=(*weighted, *validation),
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=NUM_TRAIN_STEPS,
        z_loss_weight=1e-4,
        evals=core_tasks(every=10000),
        resources=TRAIN_RESOURCES,
        run_id="dclm_1b_1x_how_to",
        tags=["HOWTOS", "DCLM_1B_1X"],
    )


if __name__ == "__main__":
    # Lower the checkpoint to a StepSpec graph and run it: the data tokenizes (cached),
    # then one TPU training job runs. No executor, no import-time graph.
    StepRunner().run([lower(build())])
