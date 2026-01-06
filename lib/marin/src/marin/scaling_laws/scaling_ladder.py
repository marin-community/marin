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

This module provides ExecutorSteps for training models with compute-optimal
configurations derived from IsoFLOP analysis.

Usage:
    from marin.scaling_laws import isoflop_analysis_step, scaling_ladder_rung_step

    # First, run IsoFLOP analysis
    analysis = isoflop_analysis_step(
        name="scaling-analysis",
        training_runs=isoflop_training_steps,
    )

    # Then create optimal training steps (ladder rungs) that depend on the analysis
    rung_1e21 = scaling_ladder_rung_step(
        name="optimal-1e21",
        analysis_step=analysis,
        target_budget=1e21,
        label="nemo",
        tokenized=my_tokenized_dataset,
    )
"""

import json
import logging
import os
from collections.abc import Sequence
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

from marin.execution.executor import ExecutorStep, InputName, output_path_of, this_output_path
from marin.processing.tokenize import get_vocab_size_for_tokenizer
from marin.processing.tokenize.data_configs import add_validation_sets_to_mixture, lm_data_config
from marin.processing.tokenize.tokenize import TokenizeConfig
from marin.scaling_laws.isoflop_analysis import (
    IsoFlopSweepConfig,
    ScalingFit,
    build_model_config,
    build_optimizer_config,
    isoflop_analysis_step,
    pick_v5p_type,
    predict_optimal_config,
)
from marin.scaling_laws.recipe import MARIN_2025_RECIPE, ScalingRecipe
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

logger = logging.getLogger(__name__)

# Type alias for tokenizer steps
TokenizerStep = ExecutorStep[TokenizeConfig]


def _prepare_data_config(
    tokenized: InputName | str | LMMixtureDatasetConfig,
    validation_sets: dict[str, TokenizerStep] | None = None,
) -> LMMixtureDatasetConfig:
    """Prepare a tokenized dataset for training.

    This is a local helper that prepares data configs without depending on
    experiment-specific validation sets. Callers should pass validation sets
    explicitly if needed.

    Args:
        tokenized: The tokenized dataset - can be an InputName, path string,
            or an already-configured LMMixtureDatasetConfig.
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
        # InputName or string path
        pretraining_data = lm_data_config(
            training_set=tokenized,
            validation_sets=validation_sets,
            permutation_type="feistel",
        )
    return pretraining_data


@dataclass(frozen=True)
class ScalingLadderRungConfig:
    """Configuration for one rung of the scaling ladder (one compute-optimal training run).

    This config references an IsoFLOP analysis step and specifies
    the target compute budget. At runtime, the optimal config is loaded
    from the analysis output.
    """

    analysis_output_path: str
    """Path to the IsoFLOP analysis output directory."""

    target_budget: float
    """Target compute budget in FLOPs."""

    label: str
    """Dataset label to use for scaling fit (e.g., 'nemo', 'comma', 'dclm')."""

    tokenized: InputName | str | LMMixtureDatasetConfig
    """Tokenized dataset for training. Can be a path, InputName, or LMMixtureDatasetConfig."""

    output_path: str
    """Where to write training outputs."""

    recipe: ScalingRecipe = MARIN_2025_RECIPE
    """Scaling recipe with hyperparameters."""

    tokenizer: str = "stanford-crfm/marin-tokenizer"
    """Tokenizer to use."""

    seq_len: int = 4096
    """Sequence length for training."""

    sweep_config: IsoFlopSweepConfig | None = None
    """Optional sweep config for predict_optimal_config. Uses defaults if None."""

    validation_sets: dict[str, TokenizerStep] | None = None
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
        sweep_config=config.sweep_config,
        vocab_size=vocab_size,
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

    model_cfg = build_model_config(candidate, config.seq_len)

    param_count = model_cfg.total_trainable_params(vocab_size)
    tpu_type = pick_v5p_type(
        param_count,
        candidate.hidden_size,
        candidate.num_layers,
        candidate.batch_size,
        config.seq_len,
        vocab_size,
    )

    optimizer_cfg = build_optimizer_config(candidate, config.recipe)

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


def scaling_ladder_rung_step(
    name: str,
    analysis_step: ExecutorStep,
    target_budget: float,
    label: str,
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    recipe: ScalingRecipe = MARIN_2025_RECIPE,
    tokenizer: str = "stanford-crfm/marin-tokenizer",
    seq_len: int = 4096,
    override_output_path: str | None = None,
    validation_sets: dict[str, TokenizerStep] | None = None,
) -> ExecutorStep:
    """Create an ExecutorStep for one rung of the scaling ladder.

    This step depends on an IsoFLOP analysis step and will train a model
    using the optimal configuration predicted from the scaling fits.

    Args:
        name: Name for this executor step
        analysis_step: The IsoFLOP analysis step to read fits from
        target_budget: Target compute budget in FLOPs
        label: Dataset label to use for scaling fit (e.g., 'nemo', 'comma')
        tokenized: Tokenized dataset to train on. Can be an ExecutorStep, InputName,
            or LMMixtureDatasetConfig.
        recipe: ScalingRecipe with hyperparameters
        tokenizer: Tokenizer to use
        seq_len: Sequence length for training
        override_output_path: Optional override for the output path
        validation_sets: Optional validation sets for eval loss tracking

    Returns:
        ExecutorStep configured to run one optimal training run
    """
    if isinstance(tokenized, ExecutorStep):
        resolved_tokenized: InputName | str | LMMixtureDatasetConfig = output_path_of(tokenized)
    elif isinstance(tokenized, LMMixtureDatasetConfig):
        resolved_tokenized = tokenized
    else:
        resolved_tokenized = tokenized

    output_path = override_output_path if override_output_path is not None else this_output_path()

    config = ScalingLadderRungConfig(
        analysis_output_path=output_path_of(analysis_step),
        target_budget=target_budget,
        label=label,
        tokenized=resolved_tokenized,
        output_path=output_path,
        recipe=recipe,
        tokenizer=tokenizer,
        seq_len=seq_len,
        validation_sets=validation_sets,
    )

    step = ExecutorStep(
        name=os.path.join("checkpoints", name),
        fn=run_scaling_ladder_rung,
        config=config,
        description=f"Scaling ladder rung: optimal training for {target_budget:.1e} FLOPs based on IsoFLOP analysis",
    )

    if override_output_path is not None:
        step = step.with_output_path(override_output_path)

    return step


# ---------------- Scaling Ladder Suite ----------------


@dataclass
class ScalingLadderSuite:
    """A suite containing IsoFLOP analysis and scaling ladder rungs (optimal training steps).

    This is returned by `scaling_ladder_suite()` and contains all the steps
    needed for end-to-end scaling ladder: IsoFLOP analysis + optimal training runs.
    """

    analysis: ExecutorStep
    """The IsoFLOP analysis step."""

    optimal_runs: list[ExecutorStep]
    """Scaling ladder rungs: training steps for each target budget, using predicted optimal configs."""

    @property
    def all_steps(self) -> list[ExecutorStep]:
        """All steps in the suite (analysis + optimal runs)."""
        return [self.analysis, *self.optimal_runs]


def scaling_ladder_suite(
    name: str,
    training_runs: Sequence[ExecutorStep | InputName],
    target_budgets: Sequence[float],
    label: str,
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    recipe: ScalingRecipe = MARIN_2025_RECIPE,
    tokenizer: str = "stanford-crfm/marin-tokenizer",
    seq_len: int = 4096,
    metric_key: str = "eval/paloma/c4_en/bpb",
    label_map: dict[str, str] | None = None,
    validation_sets: dict[str, TokenizerStep] | None = None,
) -> ScalingLadderSuite:
    """Create a complete scaling ladder: IsoFLOP analysis + optimal training runs.

    This is the full pipeline interface that creates:
    1. An IsoFLOP analysis step that fits scaling laws
    2. Scaling ladder rungs (optimal training steps) for each target budget

    The optimal training steps depend on the analysis step and will train
    models using compute-optimal configurations predicted from the scaling fits.

    Args:
        name: Base name for the steps
        training_runs: IsoFLOP training run ExecutorSteps to analyze
        target_budgets: Target compute budgets (in FLOPs) for optimal training
        label: Dataset label to use for scaling fit (e.g., 'nemo', 'comma')
        tokenized: Tokenized dataset for optimal training runs. Can be an ExecutorStep,
            InputName, or LMMixtureDatasetConfig.
        recipe: ScalingRecipe with hyperparameters
        tokenizer: Tokenizer to use
        seq_len: Sequence length for training
        metric_key: Which metric to use for loss
        label_map: Optional mapping from experiment_name -> display label
        validation_sets: Optional validation sets for eval loss tracking

    Returns:
        ScalingLadderSuite containing the analysis step and optimal training steps

    Example:
        >>> suite = scaling_ladder_suite(
        ...     name="nemo-scaling",
        ...     training_runs=isoflop_training_steps,
        ...     target_budgets=[1e21, 3e21, 1e22],
        ...     label="nemo",
        ...     tokenized=nemotron_tokenized,
        ... )
        >>> all_steps = [*isoflop_training_steps, *suite.all_steps]
    """
    analysis = isoflop_analysis_step(
        name=f"{name}-analysis",
        training_runs=training_runs,
        metric_key=metric_key,
        label_map=label_map,
        recipe=recipe,
    )

    optimal_runs = []
    for budget in target_budgets:
        run_step = scaling_ladder_rung_step(
            name=f"{name}-optimal-{budget:.0e}",
            analysis_step=analysis,
            target_budget=budget,
            label=label,
            tokenized=tokenized,
            recipe=recipe,
            tokenizer=tokenizer,
            seq_len=seq_len,
            validation_sets=validation_sets,
        )
        optimal_runs.append(run_step)

    return ScalingLadderSuite(
        analysis=analysis,
        optimal_runs=optimal_runs,
    )
