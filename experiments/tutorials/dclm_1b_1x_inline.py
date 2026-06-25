# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""How To: Replicating DCLM 1B/1x in Marin (inline-protocol style).

Same run as ``exp1077_reproduce_dclm_1b1x`` but written with the
``marin.experiment`` helpers so every decision is visible in this file rather
than hidden inside ``default_train``. A golden test
(``tests/experiment/test_train_lm_golden.py``) asserts this resolves to the same
executor config as the ``default_train`` version.
"""

from fray.cluster import ResourceConfig
from levanter.models.llama import LlamaConfig
from marin.execution.executor import executor_main
from marin.experiment import Parallelism, adam, mixture, train_lm, wandb

from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets.dclm import DCLM_MIXTURE_WEIGHTS, dclm_components_llama3
from experiments.recipes import core_tasks, marin_validation

SEQ_LEN = 2048
BATCH_SIZE = 256
NUM_TRAIN_TOKENS = 28.8e9  # 1B-1x DCLM competition scale (Chinchilla-optimal for 1.4B)

# 1.4B-parameter Llama, DCLM 1B-1x competition scale.
llama_1_4b_dclm = LlamaConfig(
    max_seq_len=SEQ_LEN,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=16,
    num_kv_heads=16,
    num_layers=24,
)


def build():
    data = mixture(
        dclm_components_llama3,
        DCLM_MIXTURE_WEIGHTS,
        validation=marin_validation(llama3_tokenizer),
    )
    return train_lm(
        name="dclm_1b_1x_how_to",
        model=llama_1_4b_dclm,
        data=data,
        train_batch_size=BATCH_SIZE,
        train_seq_len=SEQ_LEN,
        tokens=NUM_TRAIN_TOKENS,
        resources=ResourceConfig.with_tpu("v4-128"),
        optimizer=adam(lr=3e-3, weight_decay=0.033, warmup=5000, min_lr_ratio=0.1),
        z_loss=1e-4,
        precision="bf16",
        parallelism=Parallelism(tensor=1),
        evals=core_tasks(every=10000),
        eval_every=1000,
        checkpoint_every="10min",
        tracker=wandb(project="marin", tags=["HOWTOS", "DCLM_1B_1X"]),
    )


if __name__ == "__main__":
    executor_main(
        steps=[build()],
        description="DCLM 1B/1X baseline, written as an inline protocol.",
    )
