# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax

from levanter.grug.main import GrugTrainingConfig
from levanter.grug.model import GrugModelConfig
from levanter.grug.main import run_training


def test_synthetic_training_step_runs():
    # On TPU, Grug uses Splash attention which requires KV sequence length to be a multiple of 128.
    # `run_training` trains on tokens[:, :-1], so pick max_seq_len=129 -> token length 128.
    max_seq_len = 129 if jax.default_backend() == "tpu" else 16
    cfg = GrugTrainingConfig(
        model=GrugModelConfig(
            vocab_size=257,
            hidden_dim=64,
            intermediate_dim=256,
            num_layers=1,
            num_heads=4,
            num_kv_heads=4,
            max_seq_len=max_seq_len,
        ),
        learning_rate=1e-3,
        weight_decay=0.01,
        steps=1,
        global_batch_size=2,
        seed=0,
    )

    run_training(cfg, cache_dir=None)
