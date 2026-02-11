# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
    "pick_v5p_type",
    "predict_optimal_config",
    "round_flops_to_bucket",
    "save_plots",
    "upload_plots_to_wandb",
]
