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

"""Exp2166: Scaling Ladder Analysis for Common Pile (comma-mix).

This experiment runs scaling ladder analysis on the isoflop training sweeps
for the Common Pile (comma-mix) dataset.

The scaling ladder:
1. Fits scaling laws from IsoFLOP sweep data to find compute-optimal configurations
2. Generates visualization plots (isoflop curves and scaling fit plots)
3. Optionally trains compute-optimal models at larger target budgets
"""

import json
import logging
import os
from dataclasses import dataclass, replace
from datetime import timedelta

import fsspec
import jmp
from fray.cluster import ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LMDatasetSourceConfig, LMMixtureDatasetConfig
from levanter.main import train_lm
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

from experiments.common_pile.tokenize_common_pile import comma_main_mixture
from experiments.defaults import default_validation_sets
from experiments.isoflop_sweep import (
    IsoFlopAnalysisConfig,
    MARIN_2025_RECIPE,
    MARIN_SCALING_SUITES,
    run_isoflop_analysis_step,
)
from experiments.llama import llama3_tokenizer
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.processing.tokenize import step_to_lm_mixture_component
from marin.scaling_laws import ScalingFit, predict_optimal_config
from marin.scaling_laws.tpu_utils import pick_v5p_type, HBM_PER_CHIP_GIB
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get training steps from the isoflop sweep
comma_training, _ = MARIN_SCALING_SUITES["common_pile"]

# --- Configuration ---
TARGET_BUDGETS: list[float] = [1e18, 3e18, 6e18, 1e19, 3e19, 6e19, 1e20, 1e21]
EXPERIMENT_NAME = "exp2166-scaling-ladder-comma-validation"
LABEL = "comma-mix"
SEQ_LEN = 4096
MAX_TPU_TYPE = "v5p-64"  # Cap TPU size; use gradient accumulation for larger models


@dataclass(frozen=True)
class OptimalTrainingConfig:
    """Config for training a compute-optimal model based on scaling law analysis."""

    analysis_output_path: str
    """Path to the analysis output containing scaling fits."""

    target_budget: float
    """Target compute budget in FLOPs."""

    label: str
    """Dataset/experiment label to use for scaling fit lookup."""

    output_path: str
    """Output path for checkpoints and logs."""

    tokenized: LMMixtureDatasetConfig
    """Tokenized dataset for training. Executor will resolve InputName and unwrap VersionedValue."""

    validation_configs: dict[str, LMDatasetSourceConfig] | None = None
    """Validation set configs. Passed through config so executor resolves InputName paths."""


def run_optimal_training(config: OptimalTrainingConfig) -> None:
    """Run compute-optimal training at the given budget.

    Reads scaling fits from analysis output, predicts optimal config,
    builds training config, and runs training directly.
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
        recipe=MARIN_2025_RECIPE,
        seq_len=SEQ_LEN,
    )

    if candidate is None:
        raise RuntimeError(
            f"Could not find optimal config for budget {config.target_budget:.2e} and label '{config.label}'"
        )

    params = candidate.model_config.total_trainable_params(MARIN_2025_RECIPE.vocab_size)
    estimated_memory = MARIN_2025_RECIPE.estimate_memory_bytes(candidate)

    # Compute TPU type and gradient accumulation settings
    max_cores = int(MAX_TPU_TYPE.split("-")[1])
    num_chips = max_cores // 2
    max_memory = num_chips * HBM_PER_CHIP_GIB * 1024**3

    per_device_parallelism: int | None = None
    if estimated_memory <= max_memory:
        # Fits without gradient accumulation
        tpu_type = pick_v5p_type(estimated_memory)
    else:
        # Need gradient accumulation to fit in MAX_TPU_TYPE
        tpu_type = MAX_TPU_TYPE
        microbatch_size = candidate.batch_size
        while (microbatch_size / candidate.batch_size) * estimated_memory > max_memory:
            microbatch_size //= 2
        if microbatch_size < num_chips:
            raise ValueError(
                f"Cannot fit model in {MAX_TPU_TYPE}: need microbatch >= {num_chips}, got {microbatch_size}"
            )
        per_device_parallelism = microbatch_size // num_chips

    print(
        f"Optimal config for {config.target_budget:.2e} FLOPs:\n"
        f"  hidden_dim={candidate.model_config.hidden_dim}, layers={candidate.model_config.num_layers}\n"
        f"  params={params:.2e}, tokens={candidate.tokens:.2e}\n"
        f"  batch_size={candidate.batch_size}, train_steps={candidate.train_steps}\n"
        f"  estimated_memory={estimated_memory / 1e9:.2f} GB -> {tpu_type}\n"
        f"  per_device_parallelism={per_device_parallelism or 'None (no grad accum)'}"
    )

    # For very large models, use aggressive gradient checkpointing to reduce memory
    # Following exp1295_32b.py pattern: offload only carries, not inputs
    model_config = candidate.model_config
    if config.target_budget >= 1e21:
        from haliax import ScanCheckpointPolicy

        model_config = replace(model_config, gradient_checkpointing=ScanCheckpointPolicy(save_carries="offload"))
        logger.info("Using offload carries gradient checkpointing for large model")

    # Build TrainLmConfig directly (like old run_scaling_ladder_rung)
    # config.tokenized is already processed by executor's instantiate_config
    data = config.tokenized
    if config.validation_configs:
        # Merge validation configs into the data mixture with weight 0
        new_configs = {
            **data.configs,
            **{k: v for k, v in config.validation_configs.items() if k not in data.configs},
        }
        if isinstance(data.train_weights, dict):
            new_weights = {
                **data.train_weights,
                **{name: 0.0 for name in config.validation_configs if name not in data.train_weights},
            }
        else:
            # Varying weights case
            new_weights = [
                (step_idx, {**weights, **{name: 0.0 for name in config.validation_configs if name not in weights}})
                for step_idx, weights in data.train_weights
            ]
        data = replace(data, configs=new_configs, train_weights=new_weights)

    inner_config = train_lm.TrainLmConfig(
        data=data,
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
            train_batch_size=candidate.batch_size,
            per_device_parallelism=per_device_parallelism if per_device_parallelism else -1,
            num_train_steps=candidate.train_steps,
            steps_per_eval=1000,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[dict(every=5000)],
            ),
            mesh=MeshConfig(
                compute_mapping={
                    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                }
            ),
            allow_nondivisible_batch_size=True,
        ),
        train_seq_len=SEQ_LEN,
        model=model_config,
        optimizer=candidate.optimizer_config,
    )

    pod_config = TrainLmOnPodConfig(
        train_config=inner_config,
        resources=ResourceConfig.with_tpu(tpu_type),
        output_path=config.output_path,
    )

    logger.info(f"Launching training with resources: {pod_config.resources}")
    run_levanter_train_lm(pod_config)


# --- Step 1: IsoFLOP Analysis ---
# Creates scaling law fits from the training runs
analysis_step = ExecutorStep(
    name=f"{EXPERIMENT_NAME}-analysis",
    fn=run_isoflop_analysis_step,
    config=IsoFlopAnalysisConfig(
        training_runs=[r.as_input_name() for r in comma_training],
        output_path=this_output_path(),
        recipe=MARIN_2025_RECIPE,
    ),
)

# --- Create validation configs ---
# Convert validation TokenizerSteps to LMDatasetSourceConfig at module import time.
# This way instantiate_config resolves InputName paths before run_optimal_training runs.
validation_steps = default_validation_sets(tokenizer=llama3_tokenizer)
validation_configs = {
    name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
}

# --- Step 2: Optimal Training Runs ---
# Train compute-optimal models at each target budget
optimal_runs: list[ExecutorStep] = []
for budget in TARGET_BUDGETS:
    step = ExecutorStep(
        name=f"{EXPERIMENT_NAME}-optimal-{budget:.0e}",
        fn=run_optimal_training,
        config=OptimalTrainingConfig(
            analysis_output_path=analysis_step.as_input_name(),
            target_budget=budget,
            label=LABEL,
            output_path=this_output_path(),
            tokenized=comma_main_mixture(permutation_type="linear"),
            validation_configs=validation_configs,
        ),
    )
    optimal_runs.append(step)

# All steps for this experiment
all_steps = [analysis_step, *optimal_runs]

if __name__ == "__main__":
    executor_main(steps=all_steps)
