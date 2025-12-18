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
    CandidateConfig,
    IsoFlopAnalysisConfig,
    IsoFlopAnalysisResult,
    IsoFlopSweepConfig,
    candidate_configs,
    isoflop_analysis_step,
    pick_v5p_type,
    predict_optimal_config,
    predict_optimal_configs_for_budgets,
    run_isoflop_analysis,
)
from marin.scaling_laws.scaling_ladder import (
    ScalingLadderRungConfig,
    ScalingLadderSuite,
    scaling_ladder_rung_step,
    scaling_ladder_suite,
)

# Plotting functions are imported separately to avoid plotly dependency in core module
# from marin.scaling_laws.scaling_plots import create_isoflop_plot, create_scaling_plot, save_plots

__all__ = [
    # Primary interface (ExecutorStep factories)
    "isoflop_analysis_step",
    "scaling_ladder_suite",
    "scaling_ladder_rung_step",
    # Programmatic interface
    "run_isoflop_analysis",
    # Dataclasses
    "CandidateConfig",
    "IsoFlopAnalysisConfig",
    "IsoFlopAnalysisResult",
    "IsoFlopSweepConfig",
    "ScalingLadderSuite",
    "ScalingLadderRungConfig",
    # Constants
    "DEFAULT_BUDGETS",
    # Utilities
    "candidate_configs",
    "pick_v5p_type",
    "predict_optimal_config",
    "predict_optimal_configs_for_budgets",
]
