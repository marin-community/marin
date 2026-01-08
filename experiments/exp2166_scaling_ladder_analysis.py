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

from experiments.defaults import default_validation_sets
from experiments.isoflop_sweep import MARIN_2025_RECIPE, MARIN_SCALING_SUITES, nemotron_mix
from marin.execution.executor import ExecutorStep, executor_main, output_path_of
from marin.processing.tokenize import add_validation_sets_to_mixture
from marin.scaling_laws import (
    IsoFlopAnalysisConfig,
    ScalingLadderRungConfig,
    run_isoflop_analysis_step,
    run_scaling_ladder_rung,
)

# Get training steps from the isoflop sweep
nemotron_training, _ = MARIN_SCALING_SUITES["nemotron"]

# --- Configuration ---
TARGET_BUDGETS: list[float] = [1e18, 3e18, 6e18, 1e19, 3e19, 6e19, 1e20]
EXPERIMENT_NAME = "exp2166-scaling-ladder-nemotron-validation"
LABEL = "nemo-wider-depth-adapt"
TOKENIZER = "stanford-crfm/marin-tokenizer"

# Add validation sets to the training mixture
nemotron_mix_with_validation = add_validation_sets_to_mixture(nemotron_mix, default_validation_sets(tokenizer=TOKENIZER))

# --- Step 1: IsoFLOP Analysis ---
# Creates scaling law fits from the training runs
analysis_step = ExecutorStep(
    name=f"{EXPERIMENT_NAME}-analysis",
    fn=run_isoflop_analysis_step,
    config=IsoFlopAnalysisConfig(
        training_runs=[output_path_of(r) for r in nemotron_training],
        output_path=f"analysis/{EXPERIMENT_NAME}",
        recipe=MARIN_2025_RECIPE,
    ),
)

# --- Step 2: Optimal Training Runs ---
# Train compute-optimal models at each target budget
optimal_runs: list[ExecutorStep] = []
for budget in TARGET_BUDGETS:
    step = ExecutorStep(
        name=f"{EXPERIMENT_NAME}-optimal-{budget:.0e}",
        fn=run_scaling_ladder_rung,
        config=ScalingLadderRungConfig(
            analysis_output_path=output_path_of(analysis_step),
            target_budget=budget,
            label=LABEL,
            tokenized=nemotron_mix_with_validation,
            output_path=f"checkpoints/{EXPERIMENT_NAME}-optimal-{budget:.0e}",
            recipe=MARIN_2025_RECIPE,
        ),
    )
    optimal_runs.append(step)

# All steps for this experiment
all_steps = [analysis_step, *optimal_runs]

if __name__ == "__main__":
    executor_main(steps=all_steps)
