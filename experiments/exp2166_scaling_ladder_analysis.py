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

"""Exp2166: IsoFLOP Analysis and Scaling Ladders for Nemotron, Comma, and Dolma3.

This experiment runs IsoFLOP analysis on the isoflop training sweeps
for three datasets:
- Nemotron (nemo-wider-depth-adapt)
- Common Pile / Comma (comma-mix)
- Dolma3 (dolma3-mix-150b-1025)

The IsoFLOP analysis fits scaling laws to find compute-optimal configurations and
generates visualization plots. It also demonstrates scaling ladder runs (compute-optimal
training runs) that use the predicted configurations.
"""

from experiments.isoflop_sweep import MARIN_SCALING_SUITES, nemotron_mix, dolma3_mix
from marin.execution.executor import executor_main
from marin.scaling_laws import isoflop_analysis_step, scaling_ladder_suite

# Get training steps for each dataset (eval_tasks=None by default, so only training steps)
nemotron_training, _ = MARIN_SCALING_SUITES["nemotron"]
comma_training, _ = MARIN_SCALING_SUITES["common_pile"]
dolma3_training, _ = MARIN_SCALING_SUITES["dolma3_mix_150b"]


# --- IsoFLOP analysis-only steps (no scaling ladder rungs) ---

nemotron_analysis = isoflop_analysis_step(
    name="exp2166-isoflop-analysis-nemotron",
    training_runs=nemotron_training,
    wandb_run_name="exp2166-isoflop-analysis-nemotron",
)


dolma3_analysis = isoflop_analysis_step(
    name="exp2166-isoflop-analysis-dolma3",
    training_runs=dolma3_training,
    wandb_run_name="exp2166-isoflop-analysis-dolma3",
)


# --- Full scaling ladder suites ---
# These create IsoFLOP analysis + scaling ladder rungs (optimal training runs) for target budgets

# Nemotron suite: analyze isoflop runs, then train optimal models at larger budgets
nemotron_suite = scaling_ladder_suite(
    name="exp2166-nemo",
    training_runs=nemotron_training,
    target_budgets=[1e21, 3e21],
    label="nemo",
    dataset=nemotron_mix,
)


# Dolma3 suite
dolma3_suite = scaling_ladder_suite(
    name="exp2166-dolma3",
    training_runs=dolma3_training,
    target_budgets=[1e21, 3e21],
    label="dolma3",
    dataset=dolma3_mix,
)


all_steps = [nemotron_analysis, dolma3_analysis, *nemotron_suite.all_steps, *dolma3_suite.all_steps]

if __name__ == "__main__":
    executor_main(steps=all_steps)
