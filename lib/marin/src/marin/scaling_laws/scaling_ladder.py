# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scaling ladder: compute-optimal training runs based on IsoFLOP analysis.

This module provides functions and configs for training models with compute-optimal
configurations derived from IsoFLOP analysis.
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import timedelta

import fsspec
import jmp
from fray.cluster import ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LMMixtureDatasetConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

from marin.scaling_laws.isoflop_analysis import (
    ScalingFit,
    ScalingRecipe,
    predict_optimal_config,
)
from marin.scaling_laws.tpu_utils import pick_v5p_type
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScalingLadderRungConfig:
    """Configuration for one rung of the scaling ladder (one compute-optimal training run).

    This config references an IsoFLOP analysis output and specifies
    the target compute budget. At runtime, the optimal config is loaded
    from the analysis output.

    The ScalingRecipe handles all model-specific decisions (architecture, optimizer)
    and owns the vocab_size (derived from the tokenizer choice).
    """

    analysis_output_path: str
    """Path to the IsoFLOP analysis output directory."""

    target_budget: float
    """Target compute budget in FLOPs."""

    label: str
    """Dataset label to use for scaling fit (e.g., 'nemo', 'comma', 'dclm')."""

    tokenized: LMMixtureDatasetConfig
    """Tokenized dataset for training (with validation sets already added)."""

    output_path: str
    """Where to write training outputs."""

    recipe: ScalingRecipe
    """Scaling recipe that handles model/optimizer config building (includes vocab_size)."""

    seq_len: int = 4096
    """Sequence length for training."""


def run_scaling_ladder_rung(config: ScalingLadderRungConfig) -> None:
    """Run one rung of the scaling ladder (one compute-optimal training run).

    The candidate contains all training configuration:
    - Model config: `candidate.model_config`
    - Optimizer config: `candidate.optimizer_config`
    - Training schedule: `candidate.batch_size`, `candidate.train_steps`
    """
    result_path = os.path.join(config.analysis_output_path, "isoflop_analysis_result.json")
    fs, _, _ = fsspec.get_fs_token_paths(result_path)

    with fs.open(result_path, "r") as f:
        analysis_result = json.load(f)

    scaling_fits: dict[str, ScalingFit] = {}
    for key, value in analysis_result["scaling_fits"].items():
        if len(value) != 2:
            raise ValueError(f"Expected 2 scaling fit values for '{key}', got {len(value)}")
        scaling_fits[key] = ScalingFit(float(value[0]), float(value[1]))

    candidate = predict_optimal_config(
        scaling_fits=scaling_fits,
        target_flops=config.target_budget,
        label=config.label,
        recipe=config.recipe,
        seq_len=config.seq_len,
    )

    if candidate is None:
        raise RuntimeError(
            f"Could not find optimal config for budget {config.target_budget:.2e} and label '{config.label}'"
        )

    params = candidate.model_config.total_trainable_params(config.recipe.vocab_size)
    logger.info(
        f"Training with optimal config for {config.target_budget:.2e} FLOPs:\n"
        f"  params={params:.2e}\n"
        f"  tokens={candidate.tokens:.2e}"
    )

    model_cfg = candidate.model_config
    optimizer_cfg = candidate.optimizer_config
    batch_size = candidate.batch_size
    train_steps = candidate.train_steps
    tpu_type = pick_v5p_type(candidate, config.seq_len, config.recipe)

    train_config = TrainLmConfig(
        data=config.tokenized,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="marin",
                tags=[
                    "optimal-training",
                    f"FLOPs={config.target_budget:.1e}",
                    f"label={config.label}",
                    f"N={params:.1e}",
                ],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=batch_size,
            num_train_steps=train_steps,
            steps_per_eval=1000,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[dict(every=5000)],
            ),
            mesh=MeshConfig(
                # Special axes for MoEs
                # TODO: this is actually bad and we should remove, but keeping for now
                compute_mapping={
                    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                }
            ),
            allow_nondivisible_batch_size=True,
        ),
        train_seq_len=config.seq_len,
        model=model_cfg,
        optimizer=optimizer_cfg,
    )

    full_config = TrainLmOnPodConfig(
        train_config=train_config,
        resources=ResourceConfig.with_tpu(tpu_type),
        output_path=config.output_path,
    )

    run_levanter_train_lm(full_config)
