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
configurations derived from IsoFLOP analysis. Experiments create ExecutorSteps
directly using the provided functions.

Example usage in experiments:

    from marin.execution.executor import ExecutorStep, output_path_of
    from marin.scaling_laws import (
        ScalingLadderRungConfig,
        run_scaling_ladder_rung,
    )

    # Create optimal training step that depends on analysis output
    optimal_step = ExecutorStep(
        name="optimal-1e21",
        fn=run_scaling_ladder_rung,
        config=ScalingLadderRungConfig(
            analysis_output_path=output_path_of(analysis_step),
            target_budget=1e21,
            label="nemo",
            tokenized=my_tokenized_dataset,
            output_path="checkpoints/optimal-1e21",
        ),
    )
"""

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta

import fsspec
import jmp
from fray.cluster import ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LMMixtureDatasetConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.lm_model import LmConfig
from levanter.models.qwen import Qwen3Config
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

from marin.processing.tokenize import get_vocab_size_for_tokenizer
from marin.processing.tokenize.data_configs import add_validation_sets_to_mixture, lm_data_config
from marin.scaling_laws.isoflop_analysis import (
    CandidateConfig,
    ScalingFit,
    predict_optimal_config,
)
from marin.scaling_laws.tpu_utils import pick_v5p_type
from marin.scaling_laws.recipe import ScalingRecipe
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

logger = logging.getLogger(__name__)

# Type alias for model builder callbacks
# Takes (candidate, seq_len) and returns a model config
ModelBuilder = Callable[[CandidateConfig, int], LmConfig]


def default_model_builder(candidate: CandidateConfig, seq_len: int) -> Qwen3Config:
    """Default model builder that creates Qwen3Config.

    This is provided as a convenience for the common case. Users can pass
    their own model_builder function to use different model types.
    """
    return Qwen3Config(
        hidden_dim=candidate.hidden_size,
        intermediate_dim=candidate.intermediate_dim,
        num_layers=candidate.num_layers,
        num_heads=candidate.num_heads,
        num_kv_heads=candidate.num_kv_heads,
        max_seq_len=seq_len,
        rope=Llama3RotaryEmbeddingsConfig(),
    )


def _prepare_data_config(
    tokenized: str | LMMixtureDatasetConfig,
    validation_sets: dict | None = None,
) -> LMMixtureDatasetConfig:
    """Prepare a tokenized dataset for training.

    This is a local helper that prepares data configs without depending on
    experiment-specific validation sets. Callers should pass validation sets
    explicitly if needed.

    Args:
        tokenized: The tokenized dataset - can be a path string or an
            already-configured LMMixtureDatasetConfig.
        validation_sets: Optional dict of validation sets to add. If None,
            no validation sets are added.

    Returns:
        LMMixtureDatasetConfig ready for training.
    """
    if isinstance(tokenized, LMMixtureDatasetConfig):
        pretraining_data = tokenized
        if validation_sets:
            pretraining_data = add_validation_sets_to_mixture(pretraining_data, validation_sets)
    else:
        # String path
        pretraining_data = lm_data_config(
            training_set=tokenized,
            validation_sets=validation_sets,
            permutation_type="feistel",
        )
    return pretraining_data


@dataclass(frozen=True)
class ScalingLadderRungConfig:
    """Configuration for one rung of the scaling ladder (one compute-optimal training run).

    This config references an IsoFLOP analysis output and specifies
    the target compute budget. At runtime, the optimal config is loaded
    from the analysis output.
    """

    analysis_output_path: str
    """Path to the IsoFLOP analysis output directory."""

    target_budget: float
    """Target compute budget in FLOPs."""

    label: str
    """Dataset label to use for scaling fit (e.g., 'nemo', 'comma', 'dclm')."""

    tokenized: str | LMMixtureDatasetConfig
    """Tokenized dataset for training. Can be a path string or LMMixtureDatasetConfig."""

    output_path: str
    """Where to write training outputs."""

    recipe: ScalingRecipe
    """Scaling recipe with hyperparameters."""

    model_builder: ModelBuilder | None = None
    """Function to build model config from CandidateConfig. If None, uses default_model_builder (Qwen3)."""

    tokenizer: str = "stanford-crfm/marin-tokenizer"
    """Tokenizer to use."""

    seq_len: int = 4096
    """Sequence length for training."""

    validation_sets: dict | None = None
    """Optional validation sets to add for eval loss tracking."""


def run_scaling_ladder_rung(config: ScalingLadderRungConfig) -> None:
    """Run one rung of the scaling ladder (one compute-optimal training run)."""
    result_path = os.path.join(config.analysis_output_path, "isoflop_analysis_result.json")
    fs, _, _ = fsspec.get_fs_token_paths(result_path)

    with fs.open(result_path, "r") as f:
        analysis_result = json.load(f)

    scaling_fits: dict[str, ScalingFit] = {}
    for key, value in analysis_result["scaling_fits"].items():
        if len(value) != 2:
            raise ValueError(f"Expected 2 scaling fit values for '{key}', got {len(value)}")
        scaling_fits[key] = ScalingFit(float(value[0]), float(value[1]))

    vocab_size = get_vocab_size_for_tokenizer(config.tokenizer)

    candidate = predict_optimal_config(
        scaling_fits=scaling_fits,
        target_flops=config.target_budget,
        label=config.label,
        vocab_size=vocab_size,
        recipe=config.recipe,
        seq_len=config.seq_len,
    )

    if candidate is None:
        raise RuntimeError(
            f"Could not find optimal config for budget {config.target_budget:.2e} and label '{config.label}'"
        )

    logger.info(
        f"Training with optimal config for {config.target_budget:.2e} FLOPs:\n"
        f"  hidden_size={candidate.hidden_size}, num_layers={candidate.num_layers}\n"
        f"  batch_size={candidate.batch_size}, train_steps={candidate.train_steps}\n"
        f"  learning_rate={candidate.learning_rate:.6f}, tokens={candidate.tokens:.2e}"
    )

    # Use provided model builder or default to Qwen3
    model_builder = config.model_builder or default_model_builder
    model_cfg = model_builder(candidate, config.seq_len)

    tpu_type = pick_v5p_type(candidate, vocab_size, config.seq_len)

    optimizer_cfg = config.recipe.build_optimizer_config(candidate.learning_rate, candidate.beta2)

    pretraining_data = _prepare_data_config(config.tokenized, config.validation_sets)

    train_config = TrainLmConfig(
        data=pretraining_data,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="marin",
                tags=[
                    "optimal-training",
                    f"FLOPs={config.target_budget:.1e}",
                    f"label={config.label}",
                ],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=candidate.batch_size,
            num_train_steps=candidate.train_steps,
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
