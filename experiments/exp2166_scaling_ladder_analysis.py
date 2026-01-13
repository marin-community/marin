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

"""Exp2166: Scaling Ladder Analysis for Nemotron.

This experiment runs scaling ladder analysis on the isoflop training sweeps
for the Nemotron (nemo-wider-depth-adapt) dataset.

The scaling ladder:
1. Fits scaling laws from IsoFLOP sweep data to find compute-optimal configurations
2. Generates visualization plots (isoflop curves and scaling fit plots)
3. Optionally trains compute-optimal models at larger target budgets
"""

import json
import logging
import os

import fsspec
from fray.cluster import ResourceConfig

from experiments.defaults import default_train, default_validation_sets
from experiments.isoflop_sweep import (
    IsoFlopAnalysisConfig,
    MARIN_2025_RECIPE,
    MARIN_SCALING_SUITES,
    nemotron_mix,
    run_isoflop_analysis_step,
)
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.processing.tokenize import add_validation_sets_to_mixture
from marin.scaling_laws import ScalingFit, predict_optimal_config
from marin.scaling_laws.tpu_utils import pick_v5p_type

logger = logging.getLogger(__name__)

# Get training steps from the isoflop sweep
nemotron_training, _ = MARIN_SCALING_SUITES["nemotron"]

# --- Configuration ---
TARGET_BUDGETS: list[float] = [1e18, 3e18, 6e18, 1e19, 3e19, 6e19, 1e20]
EXPERIMENT_NAME = "exp2166-scaling-ladder-nemotron-validation"
LABEL = "nemo-wider-depth-adapt"
TOKENIZER = "stanford-crfm/marin-tokenizer"
SEQ_LEN = 4096

# Add validation sets to the training mixture
nemotron_mix_with_validation = add_validation_sets_to_mixture(nemotron_mix, default_validation_sets(tokenizer=TOKENIZER))


def run_optimal_training(
    analysis_output_path: str,
    target_budget: float,
    label: str,
) -> ExecutorStep:
    """Create an ExecutorStep for compute-optimal training at the given budget.

    Loads scaling fits from the analysis output, predicts the optimal config,
    and returns an ExecutorStep using default_train.
    """
    result_path = os.path.join(analysis_output_path, "isoflop_analysis_result.json")
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
        target_flops=target_budget,
        label=label,
        recipe=MARIN_2025_RECIPE,
        seq_len=SEQ_LEN,
    )

    if candidate is None:
        raise RuntimeError(f"Could not find optimal config for budget {target_budget:.2e} and label '{label}'")

    params = candidate.model_config.total_trainable_params(MARIN_2025_RECIPE.vocab_size)
    logger.info(
        f"Training with optimal config for {target_budget:.2e} FLOPs:\n"
        f"  params={params:.2e}\n"
        f"  tokens={candidate.tokens:.2e}"
    )

    estimated_memory = MARIN_2025_RECIPE.estimate_memory_bytes(candidate, SEQ_LEN)
    tpu_type = pick_v5p_type(estimated_memory)

    train_config = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu(tpu_type),
        train_batch_size=candidate.batch_size,
        num_train_steps=candidate.train_steps,
        learning_rate=candidate.optimizer_config.learning_rate,
        optimizer_config=candidate.optimizer_config,
        train_seq_len=SEQ_LEN,
    )

    return default_train(
        name=f"{EXPERIMENT_NAME}-optimal-{target_budget:.0e}",
        tokenized=nemotron_mix_with_validation,
        model_config=candidate.model_config,
        train_config=train_config,
        tags=[
            "optimal-training",
            f"FLOPs={target_budget:.1e}",
            f"label={label}",
            f"N={params:.1e}",
        ],
        use_default_validation=False,  # Already added above
    )


# --- Step 1: IsoFLOP Analysis ---
# Creates scaling law fits from the training runs
analysis_step = ExecutorStep(
    name=f"{EXPERIMENT_NAME}-analysis",
    fn=run_isoflop_analysis_step,
    config=IsoFlopAnalysisConfig(
        training_runs=[r.as_input_name() for r in nemotron_training],
        output_path=this_output_path(),
        recipe=MARIN_2025_RECIPE,
    ),
)

# --- Step 2: Optimal Training Runs ---
# Train compute-optimal models at each target budget
optimal_runs: list[ExecutorStep] = []
for budget in TARGET_BUDGETS:
    step = ExecutorStep(
        name=f"{EXPERIMENT_NAME}-optimal-{budget:.0e}",
        fn=lambda b=budget: run_optimal_training(
            analysis_output_path=analysis_step.as_input_name(),
            target_budget=b,
            label=LABEL,
        ),
        config=None,
    )
    optimal_runs.append(step)

# All steps for this experiment
all_steps = [analysis_step, *optimal_runs]

if __name__ == "__main__":
    executor_main(steps=all_steps)
