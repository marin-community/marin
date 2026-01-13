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
    DEFAULT_SEQ_LEN,
    CandidateConfig,
    FitScalingLawsResult,
    IsoFlopRecord,
    MinimaRecord,
    ModelConfiguration,
    QuadraticFitCoeffs,
    ScalingFit,
    ScalingRecipe,
    fit_scaling_laws,
    generate_training_configs,
    predict_optimal_config,
    round_flops_to_bucket,
)
from marin.scaling_laws.tpu_utils import (
    pick_v5p_type,
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
    "DEFAULT_SEQ_LEN",
    # Data classes and Protocols
    "CandidateConfig",
    "FitScalingLawsResult",
    "IsoFlopRecord",
    "MinimaRecord",
    "ModelConfiguration",
    "QuadraticFitCoeffs",
    "ScalingFit",
    "ScalingRecipe",
    # Functions
    "create_isoflop_plot",
    "create_scaling_plot",
    "fit_scaling_laws",
    "generate_training_configs",
    "pick_v5p_type",
    "predict_optimal_config",
    "round_flops_to_bucket",
    "save_plots",
    "upload_plots_to_wandb",
]
