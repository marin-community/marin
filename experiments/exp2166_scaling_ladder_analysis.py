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

The analysis steps depend on completed isoflop training runs from isoflop_sweep.py.
Once complete, results are saved to the output path and uploaded to WandB.
"""

from experiments.isoflop_sweep import MARIN_SCALING_SUITES, nemotron_mix
from marin.execution.executor import executor_main
from marin.scaling_laws import scaling_ladder_suite

# Get training steps and datasets for each suite
nemotron_training, _ = MARIN_SCALING_SUITES["nemotron"]

# --- Scaling Ladder Suites ---
# These analyze completed isoflop training runs and optionally train compute-optimal models

# Target budgets for compute-optimal training runs (beyond the isoflop sweep)
# Set to empty list to only run analysis without training
TARGET_BUDGETS: list[float] = [1e18, 3e18, 6e18, 1e19, 3e19, 6e19, 1e20]


nemotron_suite = scaling_ladder_suite(
    name="exp2166-scaling-ladder-nemotron",
    training_runs=nemotron_training,
    target_budgets=TARGET_BUDGETS,
    label="nemo-wider-depth-adapt",
    tokenized=nemotron_mix,
    wandb_project="marin-analysis",
)

all_steps = [*nemotron_suite.all_steps]

if __name__ == "__main__":
    executor_main(steps=all_steps)
