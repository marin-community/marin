# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""How To: Replicating DCLM 7B/1x in Marin (lazy-artifact style).

Link: https://arxiv.org/pdf/2406.11794

Every training decision is stated inline: the model, the data mixture, the optimizer,
the token budget, the z-loss, and the eval cadence. The library's
:func:`~marin.experiment.train.train_lm` assembles them into a lazy
:class:`~marin.execution.lazy.Checkpoint`, handling only the mechanical marin-on-TPU
plumbing (the mesh, the resumption checkpointer, the eval-harness wiring, the Fray
dispatch). It bakes in no optimizer, mixture, or eval suite of its own.

Reproduces the DCLM baseline for the 7B/1x competition pool (Chinchilla-optimal for 7B
parameters).
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
BATCH_SIZE = 2048
NUM_TRAIN_TOKENS = 140e9  # 140 billion tokens, Chinchilla-optimal for 7B parameters
NUM_TRAIN_STEPS = int(NUM_TRAIN_TOKENS) // (BATCH_SIZE * SEQ_LEN)

# The TPU pod each training job is dispatched onto. A run-arg, not part of the
# checkpoint's identity: re-running on different hardware is the same artifact.
TRAIN_RESOURCES = ResourceConfig.with_tpu("v4-128", slice_count=4)

# 7B-parameter Llama, DCLM 7B-1x competition scale.
# [Reference: DCLM paper, Table 1 - Competition Scales]
# https://github.com/mlfoundations/dclm/blob/main/training/configs/7b_1x_fast_2e-3_lr_5e-6_zloss.json
llama_7b_dclm = LlamaConfig(
    max_seq_len=SEQ_LEN,
    hidden_dim=4096,
    intermediate_dim=11008,
    num_heads=32,
    num_kv_heads=32,
    num_layers=32,
)


def build(*, version: str = "v1") -> Checkpoint:
    """The DCLM 7B/1x training run as a lazy checkpoint, every decision stated inline."""
    train = dclm_datasets(tokenizer=llama3_tokenizer)
    validation = [*paloma_validation(tokenizer=llama3_tokenizer), *uncheatable_validation(tokenizer=llama3_tokenizer)]
    weighted = {train[name]: DCLM_MIXTURE_WEIGHTS[name] for name in train}

    return train_lm(
        name="checkpoints/dclm_7b_1x_how_to",
        version=version,
        model=llama_7b_dclm,
        optimizer=AdamConfig(learning_rate=2e-3, weight_decay=0.05, warmup=5000, min_lr_ratio=0.1),
        data=lambda ctx: mixture(ctx, weighted, validation=validation),
        deps=(*weighted, *validation),
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=NUM_TRAIN_STEPS,
        z_loss_weight=5e-6,
        evals=core_tasks(every=10000),
        resources=TRAIN_RESOURCES,
        run_id="dclm_7b_1x_how_to",
        tags=["HOWTOS", "DCLM_7B_1X"],
    )


if __name__ == "__main__":
    # Lower the checkpoint to a StepSpec graph and run it: the data tokenizes (cached),
    # then one TPU training job runs. No executor, no import-time graph.
    StepRunner().run([lower(build())])
