# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure recipe construction for the July row-norm comparison."""

import dataclasses

from experiments.grug.moe.heuristic import MoeHeuristic as BaselineMoeHeuristic
from experiments.grug.moe.model import GrugModelConfig as BaselineModelConfig
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe_row_norm.heuristic import MoeHeuristic
from experiments.grug.moe_row_norm.model import GrugModelConfig
from experiments.grug.moe_row_norm.optimizer import GrugMoeRowNormConfig

SEQ_LEN: int = 8192
HIDDEN_DIM: int = 512
BATCH_SIZE: int = 16
NUM_STEPS: int = 10_980


def row_norm_recipe() -> tuple[GrugModelConfig, GrugMoeRowNormConfig]:
    """Return the factorized recipe at the July d512 compute-optimal cell."""
    heuristic = MoeHeuristic()
    model = dataclasses.replace(
        heuristic.build_model_config(HIDDEN_DIM, seq_len=SEQ_LEN),
        disable_pko=True,
        disable_long_rope=True,
    )
    tokens = float(NUM_STEPS * BATCH_SIZE * SEQ_LEN)
    optimizer = heuristic.build_optimizer_config(BATCH_SIZE, tokens, HIDDEN_DIM, seq_len=SEQ_LEN)
    return model, optimizer


def baseline_recipe() -> tuple[BaselineModelConfig, GrugMoeMuonHConfig]:
    """Return the unmodified July recipe at its d512 compute-optimal cell."""
    heuristic = BaselineMoeHeuristic()
    model = dataclasses.replace(
        heuristic.build_model_config(HIDDEN_DIM, seq_len=SEQ_LEN),
        disable_pko=True,
        disable_long_rope=True,
    )
    tokens = float(NUM_STEPS * BATCH_SIZE * SEQ_LEN)
    optimizer = heuristic.build_optimizer_config(BATCH_SIZE, tokens, HIDDEN_DIM, seq_len=SEQ_LEN)
    return model, optimizer
