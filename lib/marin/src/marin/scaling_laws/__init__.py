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

from marin.scaling_laws.isoflop_analysis import (
    DEFAULT_BUDGETS,
    DEFAULT_EVAL_METRIC_KEY,
    DEFAULT_FLOP_TOLERANCE,
    DEFAULT_SEQ_LEN,
    DEFAULT_STEPS_PER_RUN,
    CandidateConfig,
    FitScalingLawsResult,
    IsoFlopAnalysisConfig,
    IsoFlopAnalysisResult,
    IsoFlopTrainArgs,
    MinimaRecord,
    QuadraticFitCoeffs,
    ScalingFit,
    candidate_configs,
    compute_training_flops,
    compute_transformer_params,
    fit_scaling_laws,
    generate_isoflop_train_args,
    predict_optimal_config,
    predict_optimal_configs_for_budgets,
    run_isoflop_analysis,
    run_isoflop_analysis_step,
    solve_for_batch_size,
    solve_for_train_steps,
)
from marin.scaling_laws.tpu_utils import (
    estimate_memory_bytes,
    pick_v5p_type,
)
from marin.scaling_laws.recipe import (
    ScalingRecipe,
)
from marin.scaling_laws.scaling_ladder import (
    ScalingLadderRungConfig,
    run_scaling_ladder_rung,
)
from marin.scaling_laws.scaling_plots import (
    create_isoflop_plot,
    create_scaling_plot,
    save_plots,
    upload_plots_to_wandb,
)

__all__ = [
    # Constants
    "DEFAULT_BUDGETS",
    "DEFAULT_EVAL_METRIC_KEY",
    "DEFAULT_FLOP_TOLERANCE",
    "DEFAULT_SEQ_LEN",
    "DEFAULT_STEPS_PER_RUN",
    # Data classes
    "CandidateConfig",
    "FitScalingLawsResult",
    "IsoFlopAnalysisConfig",
    "IsoFlopAnalysisResult",
    "IsoFlopTrainArgs",
    "MinimaRecord",
    "QuadraticFitCoeffs",
    "ScalingFit",
    "ScalingLadderRungConfig",
    "ScalingRecipe",
    # Functions
    "candidate_configs",
    "compute_training_flops",
    "compute_transformer_params",
    "create_isoflop_plot",
    "create_scaling_plot",
    "estimate_memory_bytes",
    "fit_scaling_laws",
    "generate_isoflop_train_args",
    "pick_v5p_type",
    "predict_optimal_config",
    "predict_optimal_configs_for_budgets",
    "run_isoflop_analysis",
    "run_isoflop_analysis_step",
    "run_scaling_ladder_rung",
    "save_plots",
    "solve_for_batch_size",
    "solve_for_train_steps",
    "upload_plots_to_wandb",
]
