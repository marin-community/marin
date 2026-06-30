# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""How To: Replicate the DCLM 7B/1x baseline in Marin (https://arxiv.org/pdf/2406.11794).

The DCLM baseline for the 7B/1x competition pool, Chinchilla-optimal for 7B parameters.
"""

import draccus
from fray.cluster import ResourceConfig
from levanter.models.llama import LlamaConfig
from levanter.optim import AdamConfig
from marin.execution.lazy import ArtifactStep
from marin.experiment.train import EvalSuite, train_lm
from marin.training.training import LevanterCheckpoint

from experiments.evals.task_configs import CORE_TASKS
from experiments.evals.uncheatable import uncheatable_validation
from experiments.launch import LaunchConfig, launch, run_steps
from experiments.llama import llama3_tokenizer
from experiments.paloma import paloma_validation
from experiments.pretraining_datasets.dclm import DCLM_MIXTURE_WEIGHTS, dclm_datasets

SEQ_LEN = 2048
BATCH_SIZE = 2048
NUM_TRAIN_TOKENS = 140e9  # 140 billion tokens, Chinchilla-optimal for 7B parameters
NUM_TRAIN_STEPS = int(NUM_TRAIN_TOKENS) // (BATCH_SIZE * SEQ_LEN)

# The TPU pod each training job is dispatched onto. A runtime arg, not part of the
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


def build(*, version: str = "2026.06.28") -> ArtifactStep[LevanterCheckpoint]:
    """The DCLM 7B/1x training run as a lazy checkpoint, every decision stated inline."""
    train = dclm_datasets(tokenizer=llama3_tokenizer)
    validation = [*paloma_validation(tokenizer=llama3_tokenizer), *uncheatable_validation(tokenizer=llama3_tokenizer)]
    weighted = {train[name]: DCLM_MIXTURE_WEIGHTS[name] for name in train}

    return train_lm(
        name="checkpoints/dclm_7b_1x_how_to",
        version=version,
        model=llama_7b_dclm,
        optimizer=AdamConfig(learning_rate=2e-3, weight_decay=0.05, warmup=5000, min_lr_ratio=0.1),
        datasets=weighted,
        validation=validation,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=NUM_TRAIN_STEPS,
        z_loss_weight=5e-6,
        evals=EvalSuite(CORE_TASKS, every=10000),
        resources=TRAIN_RESOURCES,
        run_id="dclm_7b_1x_how_to",
        tags=["HOWTOS", "DCLM_7B_1X"],
    )


def train_dclm_7b(config: LaunchConfig) -> None:
    """Run the DCLM 7B/1x reproduction, in-process or on ``--cluster``."""
    run_steps(config, build())


if __name__ == "__main__":
    launch(draccus.parse(LaunchConfig), train_dclm_7b)
