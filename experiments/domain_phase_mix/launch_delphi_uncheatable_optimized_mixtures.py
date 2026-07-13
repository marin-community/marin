# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train selected optimized mixtures on the Delphi scaling ladder.

This launcher validates and submits selected two-phase Dolma3/Dolmino top-level
mixtures:

- OLMix delta=0.01, KL=0.05, aggregate cap=4.
- DSP effective-exposure, KL=0.1.
- Canonical DSP, KL=0.1.
- OLMix Table-9 macro delta=0.01, KL=0.05, aggregate cap=4.
- DSP effective-exposure Table-9 macro, KL=0.025.
- DSP effective-exposure Table-9 macro, trust-region KL sweep values
  0.05, 0.1, 0.2, and 0.5.
- DSP effective-exposure Table-9 macro, follow-up trust-region KL values
  0.25, 0.3, and 0.4.
- Per-component effective-exposure DSP Table-9 macro, trust-region KL sweep
  values 0.025, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, and 0.5.
- Adaptive-shrinkage Table-9 macro probes at the 3e18 validation rung.
- One-phase OLMix uncheatable BPB delta=0.01, KL=0.05, aggregate cap=4.
- One-phase DSP effective-exposure uncheatable BPB, KL=0.1.
- One-phase OLMix Table-9 macro delta=0.01, KL=0.05, aggregate cap=4.
- One-phase DSP effective-exposure Table-9 macro, trust-region KL sweep values
  0.05, 0.1, 0.2, 0.25, and 0.3.
- Repeat controls for the one-phase DSP Table-9 KL=0.1 tied-phase candidate
  and the two-phase split-saturation/penalty Table-9 KL=0.3 candidate.
- Gamma-capped asymmetric-bowl/effective-exposure DSP probes at the 3e18
  validation rung.

Unlike ``launch_delphi_baseline_mixtures.py``, this script intentionally accepts
phase-asymmetric mixtures and uses the historical 80/20 Dolma3/Dolmino
two-phase schedule.  The simulated-epoch target budget is fixed across Delphi
scales; each rung only changes the realized training token budget.  One-phase
mixtures are represented as equal phase-0 and phase-1 weights; this preserves
the launcher schedule while making the effective mixture constant over training.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

import fsspec
import jmp
from fray.cluster import ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import DatasetComponent
from levanter.main import train_lm
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.evaluation.olmo_base_eval.run import olmo_base_eval_step
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.execution.types import ExecutorStep, InputName, this_output_path
from marin.processing.tokenize import step_to_lm_mixture_component
from marin.rl.placement import marin_prefix_for_region
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

from experiments.defaults import default_validation_sets
from experiments.domain_phase_mix.config import PhaseSchedule, WeightConfig
from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import (
    TOP_LEVEL_DOMAIN_TOKEN_COUNTS,
    TOP_LEVEL_TOTAL_AVAILABLE_TOKENS,
)
from experiments.domain_phase_mix.experiment import MixtureExperiment
from experiments.domain_phase_mix.launch_delphi_baseline_mixtures import (
    DEFAULT_TPU_REGION,
    DEFAULT_TPU_ZONE,
    LABEL,
    SEQ_LEN_DELPHI,
    SIMULATED_EPOCH_TARGET_BUDGET,
    TARGET_BUDGETS,
    _add_validation_components,
    _candidate_for_budget,
    _read_scaling_fits,
    _slug,
    _tensor_parallel_size,
)
from experiments.domain_phase_mix.qsplit240_replay import SKIP_EVAL_HARNESS_ENV_VAR
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    DEFAULT_RUNTIME_CACHE_REGION,
    DOMAIN_NAMES,
    PHASE_BOUNDARIES,
    PHASE_NAMES,
    build_top_level_domains,
)
from experiments.llama import llama3_tokenizer
from experiments.scaling_law_sweeps.completed_adamh import completed_adamh_heuristic

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUT_DIR = SCRIPT_DIR / "exploratory" / "two_phase_many" / "reference_outputs"
LOCAL_ARTIFACT_DIR = REFERENCE_OUTPUT_DIR / "delphi_uncheatable_optimized_mixtures_20260625"
MIXTURE_ASSET_DIR = SCRIPT_DIR / "assets" / "delphi_optimized_mixtures"
KL_SWEEP_MIXTURE_GCS_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_table9_dsp_kl_sweep_3e18_20260627/mixtures"
)
TABLE9_DSP_VALIDATION_MIXTURE_GCS_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_table9_dsp_validation_mixtures_20260628/mixtures"
)
TABLE9_ADAPTIVE_SHRINKAGE_VALIDATION_MIXTURE_GCS_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "delphi_table9_adaptive_shrinkage_validation_mixtures_20260628/mixtures"
)
TABLE9_PHASE_SPLIT_DSP_VALIDATION_MIXTURE_GCS_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "delphi_table9_phase_split_dsp_validation_mixtures_20260630/mixtures"
)
DSP_EXPOSURE_REPAIR_VALIDATION_MIXTURE_GCS_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_dsp_exposure_repair_validation_mixtures_20260702/mixtures"
)
DSP_SUPPORT_AWARE_VALIDATION_MIXTURE_GCS_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_dsp_support_aware_validation_mixtures_20260703/mixtures"
)
DSP_CANONICAL_BOWL_VALIDATION_MIXTURE_GCS_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_dsp_canonical_bowl_validation_mixtures_20260703/mixtures"
)
DSP_GAMMA_CAPPED_BOWL_VALIDATION_MIXTURE_GCS_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "delphi_dsp_gamma_capped_bowl_validation_mixtures_20260704/mixtures"
)
DSP_SUFFICIENCY_FLOORED_MIXTURE_GCS_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_sufficiency_floored_mixtures_20260705/mixtures"
)
DSP_WINNER_NEIGHBORHOOD_MIXTURE_GCS_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_winner_neighborhood_mixtures_20260705/mixtures"
)
DSP_AUGMENTED_PROFILE_MIXTURE_GCS_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_augmented_profile_mixtures_20260705/mixtures"
)
DSP_TABLE9_CONTROLLED_TILT_MIXTURE_GCS_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_table9_controlled_tilt_mixtures_20260705/mixtures"
)
DSP_TABLE9_FRESH_ANNEAL_MIXTURE_GCS_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_table9_fresh_anneal_mixtures_20260705/mixtures"
)
DELPHI_BASELINE_NOISE_VALIDATION_MIXTURE_GCS_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_baseline_noise_validation_mixtures_20260703/mixtures"
)
ONE_PHASE_TABLE9_VALIDATION_MIXTURE_GCS_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_one_phase_table9_validation_mixtures_20260628/mixtures"
)
ONE_PHASE_UNCHEATABLE_VALIDATION_MIXTURE_GCS_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "delphi_one_phase_uncheatable_validation_mixtures_20260629/mixtures"
)
ONE_PHASE_OLMIX_KL_SWEEP_MIXTURE_GCS_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_one_phase_olmix_kl_sweep_3e18_20260705/mixtures"
)
SEP_LF_KL_SWEEP_MIXTURE_GCS_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_sep_lf_kl_sweep_mixtures_20260706/mixtures"
)
BEST_PHASE_MODEL_VALIDATION_MIXTURE_GCS_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_best_phase_model_validation_mixtures_20260710/mixtures"
)
SEP_FRONTIER_TIED_MIXTURE_GCS_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_sep_frontier_tied_mixtures_20260710/mixtures"
)
CENTERED_RECENCY_REORDER_MIXTURE_GCS_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "delphi_centered_recency_sepheads_reorder_mixtures_20260710/mixtures"
)
GENERALIZED_POWER_REORDER_MIXTURE_GCS_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "delphi_generalized_power_sepheads_reorder_mixtures_20260710/mixtures"
)
SYMMETRIC_SEPHEADS_GEOMETRY_FRONTIER_MIXTURE_GCS_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "delphi_symmetric_sepheads_geometry_frontier_mixtures_20260711/mixtures"
)
ORIGINAL_STYLE_MATCHED_SEPHEADS_MIXTURE_GCS_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "delphi_original_style_matched_sepheads_ablation_20260712/mixtures"
)
ORIGINAL_STYLE_MATCHED_SEPHEADS_DATA_SEED = 690300
TABLE9_REQUEST_SET_DIR = InputName.hardcoded("raw/eval-datasets/olmo_base_eval_table9/v2")
TABLE9_EVAL_RESOURCES = ResourceConfig.with_tpu("v6e-8", regions=["us-east5"], zone="us-east5-b", disk="80g")
TABLE9_TARGET_METRIC = "olmo_base_easy/table9_51_component_macro_bpb"

EXPERIMENT_NAME: str = "pinlin_calvin_xu/data_mixture/delphi_uncheatable_optimized_mixtures_20260625"
DEFAULT_ANALYSIS_OUTPUT_PATH = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_baseline_mixtures_issue6607_20260623/analysis-af9355"
)
DEFAULT_MAX_CONCURRENT = 4
RUN_ID_BASE = 662_000
PHASE_SCHEDULE = PhaseSchedule.from_boundaries(boundaries=PHASE_BOUNDARIES, names=list(PHASE_NAMES))
PHASE_FRACTIONS = {phase.name: phase.end_fraction - phase.start_fraction for phase in PHASE_SCHEDULE.phases}


class DelphiValidationMixture(StrEnum):
    """Selected mixtures for Delphi scaling validation."""

    OLMIX_D001_KL005_CAP4 = "olmix_d001_kl005_cap4"
    DSP_EFFECTIVE_EXPOSURE_KL01 = "dsp_effexp_kl01"
    DSP_CANONICAL_KL01 = "dsp_canon_kl01"
    OLMIX_TABLE9_D001_KL005_CAP4 = "olmix_table9_d001_kl005_cap4"
    DSP_EFFECTIVE_EXPOSURE_TABLE9_KL0025 = "dsp_effexp_table9_kl0025"
    DSP_EFFECTIVE_EXPOSURE_TABLE9_KL0P05 = "dsp_effexp_table9_kl0p05"
    DSP_EFFECTIVE_EXPOSURE_TABLE9_KL0P1 = "dsp_effexp_table9_kl0p1"
    DSP_EFFECTIVE_EXPOSURE_TABLE9_KL0P2 = "dsp_effexp_table9_kl0p2"
    DSP_EFFECTIVE_EXPOSURE_TABLE9_KL0P25 = "dsp_effexp_table9_kl0p25"
    DSP_EFFECTIVE_EXPOSURE_TABLE9_KL0P3 = "dsp_effexp_table9_kl0p3"
    DSP_EFFECTIVE_EXPOSURE_TABLE9_KL0P4 = "dsp_effexp_table9_kl0p4"
    DSP_EFFECTIVE_EXPOSURE_TABLE9_KL0P5 = "dsp_effexp_table9_kl0p5"
    DSP_PER_COMPONENT_TABLE9_KL0P025 = "dsp_percomp_table9_kl0p025"
    DSP_PER_COMPONENT_TABLE9_KL0P05 = "dsp_percomp_table9_kl0p05"
    DSP_PER_COMPONENT_TABLE9_KL0P1 = "dsp_percomp_table9_kl0p1"
    DSP_PER_COMPONENT_TABLE9_KL0P2 = "dsp_percomp_table9_kl0p2"
    DSP_PER_COMPONENT_TABLE9_KL0P25 = "dsp_percomp_table9_kl0p25"
    DSP_PER_COMPONENT_TABLE9_KL0P3 = "dsp_percomp_table9_kl0p3"
    DSP_PER_COMPONENT_TABLE9_KL0P4 = "dsp_percomp_table9_kl0p4"
    DSP_PER_COMPONENT_TABLE9_KL0P5 = "dsp_percomp_table9_kl0p5"
    DSP_SHRINK_FIXED_SPEARMAN_KL0P2 = "dsp_shrink_fixed_spearman_kl0p2"
    DSP_SHRINK_FIXED_R2HARM_KL0P2 = "dsp_shrink_fixed_r2harm_kl0p2"
    DSP_SHRINK_TV_SPEARMAN_B0P5_KL0P2 = "dsp_shrink_tv_spearman_b0p5_kl0p2"
    DSP_SHRINK_TV_R2HARM_B1_KL0P2 = "dsp_shrink_tv_r2harm_b1_kl0p2"
    DSP_SHRINK_DELTA_SPEARMAN_B0P5_KL0P2 = "dsp_shrink_delta_spearman_b0p5_kl0p2"
    DSP_SHRINK_DELTA_R2HARM_B1_KL0P2 = "dsp_shrink_delta_r2harm_b1_kl0p2"
    DSP_SHRINK_UNC_SPEARMAN_G0P25_KL0P2 = "dsp_shrink_unc_spearman_g0p25_kl0p2"
    DSP_SHRINK_UNC_R2HARM_G0P5_KL0P2 = "dsp_shrink_unc_r2harm_g0p5_kl0p2"
    QSPLIT_RUN00018_TABLE9_ANCHOR = "qsplit_run00018_table9_anchor"
    DSP_SPLIT_TABLE9_L2_0P01_KL0P3 = "dsp_split_table9_l2_0p01_kl0p3"
    DSP_SPLIT_TABLE9_L2_0P01_KL0P4 = "dsp_split_table9_l2_0p01_kl0p4"
    DSP_SPLIT_TABLE9_L2_0P01_KL0P4_REPEAT = "dsp_split_table9_l2_0p01_kl0p4_repeat"
    DSP_EFFECTIVE_EXPOSURE_TABLE9_L2_0P01_KL0P5 = "dsp_effexp_table9_l2_0p01_kl0p5"
    OLMIX_ONE_PHASE_UNCHEATABLE_D001_KL005_CAP4 = "olmix_onephase_uncheatable_d001_kl005_cap4"
    OLMIX_ONE_PHASE_UNCHEATABLE_D001_KL0_CAP4 = "olmix_onephase_uncheatable_d001_kl0_cap4"
    OLMIX_ONE_PHASE_UNCHEATABLE_D001_KL0P005_CAP4 = "olmix_onephase_uncheatable_d001_kl0p005_cap4"
    OLMIX_ONE_PHASE_UNCHEATABLE_D001_KL0P01_CAP4 = "olmix_onephase_uncheatable_d001_kl0p01_cap4"
    OLMIX_ONE_PHASE_UNCHEATABLE_D001_KL0P025_CAP4 = "olmix_onephase_uncheatable_d001_kl0p025_cap4"
    OLMIX_ONE_PHASE_UNCHEATABLE_D001_KL0P075_CAP4 = "olmix_onephase_uncheatable_d001_kl0p075_cap4"
    OLMIX_ONE_PHASE_UNCHEATABLE_D001_KL0P1_CAP4 = "olmix_onephase_uncheatable_d001_kl0p1_cap4"
    OLMIX_ONE_PHASE_UNCHEATABLE_D001_KL0P2_CAP4 = "olmix_onephase_uncheatable_d001_kl0p2_cap4"
    OLMIX_ONE_PHASE_UNCHEATABLE_D001_KL0P5_CAP4 = "olmix_onephase_uncheatable_d001_kl0p5_cap4"
    DSP_ONE_PHASE_EFFECTIVE_EXPOSURE_UNCHEATABLE_KL0P1 = "dsp_onephase_effexp_uncheatable_kl0p1"
    OLMIX_ONE_PHASE_TABLE9_D001_KL005_CAP4 = "olmix_onephase_table9_d001_kl005_cap4"
    OLMIX_ONE_PHASE_TABLE9_D001_KL0_CAP4 = "olmix_onephase_table9_d001_kl0_cap4"
    OLMIX_ONE_PHASE_TABLE9_D001_KL0P005_CAP4 = "olmix_onephase_table9_d001_kl0p005_cap4"
    OLMIX_ONE_PHASE_TABLE9_D001_KL0P01_CAP4 = "olmix_onephase_table9_d001_kl0p01_cap4"
    OLMIX_ONE_PHASE_TABLE9_D001_KL0P025_CAP4 = "olmix_onephase_table9_d001_kl0p025_cap4"
    OLMIX_ONE_PHASE_TABLE9_D001_KL0P075_CAP4 = "olmix_onephase_table9_d001_kl0p075_cap4"
    OLMIX_ONE_PHASE_TABLE9_D001_KL0P1_CAP4 = "olmix_onephase_table9_d001_kl0p1_cap4"
    OLMIX_ONE_PHASE_TABLE9_D001_KL0P2_CAP4 = "olmix_onephase_table9_d001_kl0p2_cap4"
    OLMIX_ONE_PHASE_TABLE9_D001_KL0P5_CAP4 = "olmix_onephase_table9_d001_kl0p5_cap4"
    DSP_ONE_PHASE_EFFECTIVE_EXPOSURE_TABLE9_KL0P05 = "dsp_onephase_effexp_table9_kl0p05"
    DSP_ONE_PHASE_EFFECTIVE_EXPOSURE_TABLE9_KL0P1 = "dsp_onephase_effexp_table9_kl0p1"
    DSP_ONE_PHASE_EFFECTIVE_EXPOSURE_TABLE9_KL0P2 = "dsp_onephase_effexp_table9_kl0p2"
    DSP_ONE_PHASE_EFFECTIVE_EXPOSURE_TABLE9_KL0P25 = "dsp_onephase_effexp_table9_kl0p25"
    DSP_ONE_PHASE_EFFECTIVE_EXPOSURE_TABLE9_KL0P3 = "dsp_onephase_effexp_table9_kl0p3"
    DSP_ONE_PHASE_EFFECTIVE_EXPOSURE_TABLE9_KL0P1_REPEAT_A = "dsp_onephase_effexp_table9_kl0p1_repeat_a"
    DSP_ONE_PHASE_EFFECTIVE_EXPOSURE_TABLE9_KL0P1_REPEAT_B = "dsp_onephase_effexp_table9_kl0p1_repeat_b"
    DSP_SPLIT_TABLE9_L2_0P01_KL0P3_REPEAT_A = "dsp_split_table9_l2_0p01_kl0p3_repeat_a"
    DSP_SPLIT_TABLE9_L2_0P01_KL0P3_REPEAT_B = "dsp_split_table9_l2_0p01_kl0p3_repeat_b"
    DSP_UNCHEATABLE_EXPOSURE_TARGETED = "dsp_uncheatable_exposure_targeted"
    DSP_UNCHEATABLE_EXPOSURE_ALL_DEFICITS = "dsp_uncheatable_exposure_all_deficits"
    DSP_UNCHEATABLE_SUPPORT_AWARE_RAW_OPTIMUM = "dsp_uncheatable_support_aware_raw_optimum"
    DSP_TABLE9_EXPOSURE_TARGETED = "dsp_table9_exposure_targeted"
    DSP_TABLE9_EXPOSURE_ALL_DEFICITS = "dsp_table9_exposure_all_deficits"
    DSP_CANON_TABLE9_KL0P2 = "dsp_canon_table9_kl0p2"
    DSP_CANON_TABLE9_KL0P5 = "dsp_canon_table9_kl0p5"
    DSP_ABOWL_TABLE9_KL0P05 = "dsp_abowl_table9_kl0p05"
    DSP_ABOWL_TABLE9_KL0P1 = "dsp_abowl_table9_kl0p1"
    DSP_ABOWL_TABLE9_KL0P2 = "dsp_abowl_table9_kl0p2"
    DSP_CANON_UNCHEATABLE_KL0P2 = "dsp_canon_uncheatable_kl0p2"
    DSP_CANON_UNCHEATABLE_KL0P5 = "dsp_canon_uncheatable_kl0p5"
    DSP_ABOWL_UNCHEATABLE_KL0P05 = "dsp_abowl_uncheatable_kl0p05"
    DSP_ABOWL_UNCHEATABLE_KL0P1 = "dsp_abowl_uncheatable_kl0p1"
    DSP_ABOWL_UNCHEATABLE_KL0P2 = "dsp_abowl_uncheatable_kl0p2"
    DSP_GAMMA_BOWL_TABLE9_G1_KL0P2_TWOPHASE = "dsp_gamma_bowl_table9_g1_kl0p2_twophase"
    DSP_GAMMA_BOWL_TABLE9_G8_KL0P2_TWOPHASE = "dsp_gamma_bowl_table9_g8_kl0p2_twophase"
    DSP_GAMMA_BOWL_TABLE9_G10_KL0P2_TWOPHASE = "dsp_gamma_bowl_table9_g10_kl0p2_twophase"
    DSP_GAMMA_BOWL_TABLE9_G12_KL0P2_TWOPHASE = "dsp_gamma_bowl_table9_g12_kl0p2_twophase"
    DSP_GAMMA_BOWL_TABLE9_G16_KL0P2_TWOPHASE = "dsp_gamma_bowl_table9_g16_kl0p2_twophase"
    DSP_GAMMA_BOWL_TABLE9_G10_KL0P1_TWOPHASE = "dsp_gamma_bowl_table9_g10_kl0p1_twophase"
    DSP_GAMMA_BOWL_TABLE9_G10_KL0P3_TWOPHASE = "dsp_gamma_bowl_table9_g10_kl0p3_twophase"
    DSP_GAMMA_BOWL_TABLE9_G10_KL0P1_ONEPHASE = "dsp_gamma_bowl_table9_g10_kl0p1_onephase"
    DSP_GAMMA_BOWL_TABLE9_G10_KL0P2_ONEPHASE = "dsp_gamma_bowl_table9_g10_kl0p2_onephase"
    DSP_GAMMA_EFFEXP_TABLE9_G10_KL0P2_TWOPHASE = "dsp_gamma_effexp_table9_g10_kl0p2_twophase"
    DSP_GAMMA_BOWL_UNCHEATABLE_G10_KL0P2_TWOPHASE = "dsp_gamma_bowl_uncheatable_g10_kl0p2_twophase"
    DSP_GAMMA_BOWL_UNCHEATABLE_G10_KL0P2_ONEPHASE = "dsp_gamma_bowl_uncheatable_g10_kl0p2_onephase"
    SUFF_UNCHEAT_FLOOR_A0 = "suff_uncheat_floor_a0"
    SUFF_UNCHEAT_FLOOR_A0P7 = "suff_uncheat_floor_a0p7"
    SUFF_UNCHEAT_FLOOR_A1P0 = "suff_uncheat_floor_a1p0"
    SUFF_UNCHEAT_FLOOR_A1P2 = "suff_uncheat_floor_a1p2"
    SUFF_UNCHEAT_FLOOR_A1P0_G6 = "suff_uncheat_floor_a1p0_g6"
    SUFF_UNCHEAT_FLOOR_A1P0_EFFEXP = "suff_uncheat_floor_a1p0_effexp"
    SUFF_UNCHEAT_ONEPHASE = "suff_uncheat_onephase"
    SUFF_UNCHEAT_EVALREL = "suff_uncheat_evalrel"
    SUFF_TABLE9_FLOOR_A0 = "suff_table9_floor_a0"
    SUFF_TABLE9_FLOOR_A1P0 = "suff_table9_floor_a1p0"
    SUFF_TABLE9_ONEPHASE = "suff_table9_onephase"
    WNBR_UNCHEAT_WINNER = "wnbr_uncheat_winner"
    WNBR_UNCHEAT_TILT_K0 = "wnbr_uncheat_tilt_k0"
    WNBR_UNCHEAT_TILT_K0P5 = "wnbr_uncheat_tilt_k0p5"
    WNBR_UNCHEAT_TILT_K1P5 = "wnbr_uncheat_tilt_k1p5"
    WNBR_UNCHEAT_OVERWEIGHT_0P7 = "wnbr_uncheat_overweight_0p7"
    WNBR_UNCHEAT_OVERWEIGHT_1P3 = "wnbr_uncheat_overweight_1p3"
    WNBR_TABLE9_WINNER = "wnbr_table9_winner"
    WNBR_TABLE9_TILT_K1P5 = "wnbr_table9_tilt_k1p5"
    WNBR_TABLE9_OVERWEIGHT_1P3 = "wnbr_table9_overweight_1p3"
    AUGP_UNCHEAT_PROFILE = "augp_uncheat_profile"
    AUGP_UNCHEAT_PROFILE_1PHASE = "augp_uncheat_profile_1phase"
    AUGP_UNCHEAT_BLEND = "augp_uncheat_blend"
    AUGP_TABLE9_PROFILE = "augp_table9_profile"
    T9TILT_K0 = "t9_tilt_k0_onephase"
    T9TILT_KMOD = "t9_tilt_kmod"
    T9TILT_KHIGH = "t9_tilt_khigh"
    T9TILT_KMAX = "t9_tilt_kmax"
    T9ANNEAL_K0 = "t9anneal_k0_onephase"
    T9ANNEAL_KMOD = "t9anneal_kmod"
    T9ANNEAL_KHIGH = "t9anneal_khigh"
    T9ANNEAL_KMAX = "t9anneal_kmax"
    T9AN2_K0 = "t9an2_k0_onephase"
    T9AN2_K0R = "t9an2_k0_repeat"
    T9AN2_E10 = "t9an2_expanded_k0p10"
    T9AN2_E18 = "t9an2_expanded_k0p18"
    T9AN2_E28 = "t9an2_expanded_k0p28"
    T9AN2_N15 = "t9an2_narrow_k0p15"
    T9REP_NARROW_S0 = "t9rep_narrow_s0"
    T9REP_NARROW_S1 = "t9rep_narrow_s1"
    T9REP_NARROW_S2 = "t9rep_narrow_s2"
    T9REP_ONEPH_S0 = "t9rep_oneph_s0"
    T9REP_ONEPH_S1 = "t9rep_oneph_s1"
    T9REP_ONEPH_S2 = "t9rep_oneph_s2"
    T9VR_K0 = "t9vr_k0"
    T9VR_KMOD = "t9vr_kmod"
    T9VR_KHIGH = "t9vr_khigh"
    T9VRR_KMOD_S0 = "t9vrr_kmod_s0"
    T9VRR_KMOD_S1 = "t9vrr_kmod_s1"
    T9VRR_KMOD_S2 = "t9vrr_kmod_s2"
    T9VRR_KMOD_S3 = "t9vrr_kmod_s3"
    T9VRR_K0_S0 = "t9vrr_k0_s0"
    T9VRR_K0_S1 = "t9vrr_k0_s1"
    T9VRR_K0_S2 = "t9vrr_k0_s2"
    T9VRR_K0_S3 = "t9vrr_k0_s3"
    SEPLF_UNCH_SEP_KL0 = "seplf_unch_sep_kl0"
    SEPLF_UNCH_SEP_KL0P1 = "seplf_unch_sep_kl0p1"
    SEPLF_UNCH_SEP_KL0P2 = "seplf_unch_sep_kl0p2"
    SEPLF_UNCH_SEP_KL0P3 = "seplf_unch_sep_kl0p3"
    SEPLF_UNCH_SEP_KL0P4 = "seplf_unch_sep_kl0p4"
    SEPLF_UNCH_LF2P_KL0P2 = "seplf_unch_lf2p_kl0p2"
    SEPLF_T9_SEP_KL0 = "seplf_t9_sep_kl0"
    SEPLF_T9_SEP_KL0P1 = "seplf_t9_sep_kl0p1"
    SEPLF_T9_SEP_KL0P2 = "seplf_t9_sep_kl0p2"
    SEPLF_T9_SEP_KL0P3 = "seplf_t9_sep_kl0p3"
    SEPLF_T9_SEP_KL0P4 = "seplf_t9_sep_kl0p4"
    SEPLF_T9_LF2P_KL0P2 = "seplf_t9_lf2p_kl0p2"
    BESTPHASE_UNCH_2P_KL5 = "bestphase_uncheatable_2p_kl5"
    BESTPHASE_UNCH_TIED_KL5 = "bestphase_uncheatable_tied_kl5"
    BESTPHASE_UNCH_1P_KL5 = "bestphase_uncheatable_1p_kl5"
    BESTPHASE_UNCH_2P_KL10 = "bestphase_uncheatable_2p_kl10"
    BESTPHASE_UNCH_TIED_KL10 = "bestphase_uncheatable_tied_kl10"
    BESTPHASE_UNCH_1P_KL10 = "bestphase_uncheatable_1p_kl10"
    BESTPHASE_T9_2P_KL5 = "bestphase_table9_2p_kl5"
    BESTPHASE_T9_TIED_KL5 = "bestphase_table9_tied_kl5"
    BESTPHASE_T9_1P_KL5 = "bestphase_table9_1p_kl5"
    BESTPHASE_T9_2P_KL10 = "bestphase_table9_2p_kl10"
    BESTPHASE_T9_TIED_KL10 = "bestphase_table9_tied_kl10"
    BESTPHASE_T9_1P_KL10 = "bestphase_table9_1p_kl10"
    SEPFRONT_UNCH_2P_S0 = "sepfront_unch_2p_s0"
    SEPFRONT_UNCH_TIED_S0 = "sepfront_unch_tied_s0"
    SEPFRONT_UNCH_2P_S1 = "sepfront_unch_2p_s1"
    SEPFRONT_UNCH_TIED_S1 = "sepfront_unch_tied_s1"
    SEPFRONT_UNCH_2P_S2 = "sepfront_unch_2p_s2"
    SEPFRONT_UNCH_TIED_S2 = "sepfront_unch_tied_s2"
    SEPFRONT_T9_2P_S0 = "sepfront_t9_2p_s0"
    SEPFRONT_T9_TIED_S0 = "sepfront_t9_tied_s0"
    SEPFRONT_T9_2P_S1 = "sepfront_t9_2p_s1"
    SEPFRONT_T9_TIED_S1 = "sepfront_t9_tied_s1"
    SEPFRONT_T9_2P_S2 = "sepfront_t9_2p_s2"
    SEPFRONT_T9_TIED_S2 = "sepfront_t9_tied_s2"
    CENTREC_SEP_UNCH_OKL1 = "centrec_sep_uncheatable_okl1"
    CENTREC_SEP_UNCH_OKL3 = "centrec_sep_uncheatable_okl3"
    CENTREC_SEP_T9_OKL1 = "centrec_sep_table9_okl1"
    CENTREC_SEP_T9_OKL3 = "centrec_sep_table9_okl3"
    GENPOW_SEP_UNCH_OKL0P3 = "genpow_sep_unch_okl0p3"
    GENPOW_SEP_UNCH_OKL1 = "genpow_sep_unch_okl1"
    GENPOW_SEP_T9_OKL0P3 = "genpow_sep_t9_okl0p3"
    GENPOW_SEP_T9_OKL1 = "genpow_sep_t9_okl1"
    SYMSEP_UNCH_1P_KL0P05 = "symsep_unch_1p_kl0p05"
    SYMSEP_UNCH_1P_KL0P1 = "symsep_unch_1p_kl0p1"
    SYMSEP_UNCH_1P_KL0P2 = "symsep_unch_1p_kl0p2"
    SYMSEP_UNCH_2P_KL0P05 = "symsep_unch_2p_kl0p05"
    SYMSEP_UNCH_2P_KL0P1 = "symsep_unch_2p_kl0p1"
    SYMSEP_UNCH_2P_KL0P2 = "symsep_unch_2p_kl0p2"
    SYMSEP_T9_1P_KL0P05 = "symsep_t9_1p_kl0p05"
    SYMSEP_T9_1P_KL0P1 = "symsep_t9_1p_kl0p1"
    SYMSEP_T9_1P_KL0P2 = "symsep_t9_1p_kl0p2"
    SYMSEP_T9_2P_KL0P05 = "symsep_t9_2p_kl0p05"
    SYMSEP_T9_2P_KL0P1 = "symsep_t9_2p_kl0p1"
    SYMSEP_T9_2P_KL0P2 = "symsep_t9_2p_kl0p2"
    GEOMFRONT_UNCH_1P_KL0P2 = "geomfront_unch_1p_kl0p2"
    GEOMFRONT_UNCH_1P_KL0P3 = "geomfront_unch_1p_kl0p3"
    GEOMFRONT_UNCH_1P_KL0P5 = "geomfront_unch_1p_kl0p5"
    GEOMFRONT_UNCH_2P_KL0P2 = "geomfront_unch_2p_kl0p2"
    GEOMFRONT_UNCH_2P_KL0P3 = "geomfront_unch_2p_kl0p3"
    GEOMFRONT_UNCH_2P_KL0P5 = "geomfront_unch_2p_kl0p5"
    GEOMFRONT_UNCH_TIED_KL0P2 = "geomfront_unch_tied_kl0p2"
    GEOMFRONT_UNCH_TIED_KL0P3 = "geomfront_unch_tied_kl0p3"
    GEOMFRONT_UNCH_TIED_KL0P5 = "geomfront_unch_tied_kl0p5"
    GEOMFRONT_T9_1P_KL0P15 = "geomfront_t9_1p_kl0p15"
    GEOMFRONT_T9_1P_KL0P2 = "geomfront_t9_1p_kl0p2"
    GEOMFRONT_T9_1P_KL0P3 = "geomfront_t9_1p_kl0p3"
    GEOMFRONT_T9_2P_KL0P15 = "geomfront_t9_2p_kl0p15"
    GEOMFRONT_T9_2P_KL0P2 = "geomfront_t9_2p_kl0p2"
    GEOMFRONT_T9_2P_KL0P3 = "geomfront_t9_2p_kl0p3"
    GEOMFRONT_T9_TIED_KL0P15 = "geomfront_t9_tied_kl0p15"
    GEOMFRONT_T9_TIED_KL0P2 = "geomfront_t9_tied_kl0p2"
    GEOMFRONT_T9_TIED_KL0P3 = "geomfront_t9_tied_kl0p3"
    ORIGSTYLE_SEP_UNCH_1P_KL0P05 = "origstyle_sep_unch_1p_kl0p05"
    ORIGSTYLE_SEP_UNCH_1P_KL0P075 = "origstyle_sep_unch_1p_kl0p075"
    ORIGSTYLE_SEP_UNCH_1P_KL0P1 = "origstyle_sep_unch_1p_kl0p1"
    ORIGSTYLE_SEP_UNCH_1P_KL0P15 = "origstyle_sep_unch_1p_kl0p15"
    ORIGSTYLE_SEP_UNCH_1P_KL0P2 = "origstyle_sep_unch_1p_kl0p2"
    ORIGSTYLE_SEP_UNCH_1P_KL0P3 = "origstyle_sep_unch_1p_kl0p3"
    ORIGSTYLE_SEP_UNCH_2P_KL0P05 = "origstyle_sep_unch_2p_kl0p05"
    ORIGSTYLE_SEP_UNCH_2P_KL0P075 = "origstyle_sep_unch_2p_kl0p075"
    ORIGSTYLE_SEP_UNCH_2P_KL0P1 = "origstyle_sep_unch_2p_kl0p1"
    ORIGSTYLE_SEP_UNCH_2P_KL0P15 = "origstyle_sep_unch_2p_kl0p15"
    ORIGSTYLE_SEP_UNCH_2P_KL0P2 = "origstyle_sep_unch_2p_kl0p2"
    ORIGSTYLE_SEP_UNCH_2P_KL0P3 = "origstyle_sep_unch_2p_kl0p3"
    ORIGSTYLE_SEP_T9_1P_KL0P05 = "origstyle_sep_t9_1p_kl0p05"
    ORIGSTYLE_SEP_T9_1P_KL0P075 = "origstyle_sep_t9_1p_kl0p075"
    ORIGSTYLE_SEP_T9_1P_KL0P1 = "origstyle_sep_t9_1p_kl0p1"
    ORIGSTYLE_SEP_T9_1P_KL0P15 = "origstyle_sep_t9_1p_kl0p15"
    ORIGSTYLE_SEP_T9_1P_KL0P2 = "origstyle_sep_t9_1p_kl0p2"
    ORIGSTYLE_SEP_T9_1P_KL0P3 = "origstyle_sep_t9_1p_kl0p3"
    ORIGSTYLE_SEP_T9_2P_KL0P05 = "origstyle_sep_t9_2p_kl0p05"
    ORIGSTYLE_SEP_T9_2P_KL0P075 = "origstyle_sep_t9_2p_kl0p075"
    ORIGSTYLE_SEP_T9_2P_KL0P1 = "origstyle_sep_t9_2p_kl0p1"
    ORIGSTYLE_SEP_T9_2P_KL0P15 = "origstyle_sep_t9_2p_kl0p15"
    ORIGSTYLE_SEP_T9_2P_KL0P2 = "origstyle_sep_t9_2p_kl0p2"
    ORIGSTYLE_SEP_T9_2P_KL0P3 = "origstyle_sep_t9_2p_kl0p3"
    PROPORTIONAL_NOISE_3E18_A = "proportional_noise_3e18_a"
    PROPORTIONAL_NOISE_3E18_B = "proportional_noise_3e18_b"
    PROPORTIONAL_NOISE_3E18_C = "proportional_noise_3e18_c"
    PROPORTIONAL_NOISE_3E18_D = "proportional_noise_3e18_d"
    PROPORTIONAL_NOISE_3E18_E = "proportional_noise_3e18_e"
    PROPORTIONAL_NOISE_3E18_F = "proportional_noise_3e18_f"
    PROPORTIONAL_NOISE_3E18_G = "proportional_noise_3e18_g"
    PROPORTIONAL_NOISE_3E18_H = "proportional_noise_3e18_h"
    PROPORTIONAL_NOISE_3E18_I = "proportional_noise_3e18_i"
    PROPORTIONAL_NOISE_3E18_J = "proportional_noise_3e18_j"


@dataclass(frozen=True)
class MixtureSource:
    """Source metadata for one selected mixture."""

    key: DelphiValidationMixture
    display_name: str
    source_csv: str
    github_issue: int
    target_metric: str
    method: str
    wandb_series_tag: str
    expected_max_simulated_epoch: float | None = None
    data_seed_override: int | None = None


def _table9_dsp_validation_source(
    *,
    key: DelphiValidationMixture,
    display_name: str,
    method: str,
    wandb_series_tag: str,
    expected_max_simulated_epoch: float,
) -> MixtureSource:
    return MixtureSource(
        key=key,
        display_name=display_name,
        source_csv=f"{TABLE9_DSP_VALIDATION_MIXTURE_GCS_DIR}/{key.value}.csv",
        github_issue=6611,
        target_metric=TABLE9_TARGET_METRIC,
        method=method,
        wandb_series_tag=wandb_series_tag,
        expected_max_simulated_epoch=expected_max_simulated_epoch,
    )


def _one_phase_table9_validation_source(
    *,
    key: DelphiValidationMixture,
    display_name: str,
    method: str,
    expected_max_simulated_epoch: float,
) -> MixtureSource:
    return MixtureSource(
        key=key,
        display_name=display_name,
        source_csv=f"{ONE_PHASE_TABLE9_VALIDATION_MIXTURE_GCS_DIR}/{key.value}.csv",
        github_issue=6609,
        target_metric=TABLE9_TARGET_METRIC,
        method=method,
        wandb_series_tag="delphi-one-phase-table9-validation",
        expected_max_simulated_epoch=expected_max_simulated_epoch,
    )


def _one_phase_uncheatable_validation_source(
    *,
    key: DelphiValidationMixture,
    display_name: str,
    method: str,
    expected_max_simulated_epoch: float,
) -> MixtureSource:
    return MixtureSource(
        key=key,
        display_name=display_name,
        source_csv=f"{ONE_PHASE_UNCHEATABLE_VALIDATION_MIXTURE_GCS_DIR}/{key.value}.csv",
        github_issue=6609,
        target_metric="eval/uncheatable_eval/bpb",
        method=method,
        wandb_series_tag="delphi-one-phase-uncheatable-validation",
        expected_max_simulated_epoch=expected_max_simulated_epoch,
    )


def _one_phase_olmix_kl_sweep_source(
    *,
    key: DelphiValidationMixture,
    display_name: str,
    github_issue: int,
    target_metric: str,
    method: str,
    expected_max_simulated_epoch: float = 4.00001,
) -> MixtureSource:
    return MixtureSource(
        key=key,
        display_name=display_name,
        source_csv=f"{ONE_PHASE_OLMIX_KL_SWEEP_MIXTURE_GCS_DIR}/{key.value}.csv",
        github_issue=github_issue,
        target_metric=target_metric,
        method=method,
        wandb_series_tag="delphi-one-phase-olmix-kl-sweep",
        expected_max_simulated_epoch=expected_max_simulated_epoch,
    )


def _table9_adaptive_shrinkage_source(
    *,
    key: DelphiValidationMixture,
    display_name: str,
    method: str,
    expected_max_simulated_epoch: float,
) -> MixtureSource:
    return MixtureSource(
        key=key,
        display_name=display_name,
        source_csv=f"{TABLE9_ADAPTIVE_SHRINKAGE_VALIDATION_MIXTURE_GCS_DIR}/{key.value}.csv",
        github_issue=6611,
        target_metric=TABLE9_TARGET_METRIC,
        method=method,
        wandb_series_tag="delphi-table9-adaptive-shrinkage",
        expected_max_simulated_epoch=expected_max_simulated_epoch,
    )


def _table9_phase_split_dsp_source(
    *,
    key: DelphiValidationMixture,
    display_name: str,
    method: str,
    expected_max_simulated_epoch: float,
) -> MixtureSource:
    return MixtureSource(
        key=key,
        display_name=display_name,
        source_csv=f"{TABLE9_PHASE_SPLIT_DSP_VALIDATION_MIXTURE_GCS_DIR}/{key.value}.csv",
        github_issue=6611,
        target_metric=TABLE9_TARGET_METRIC,
        method=method,
        wandb_series_tag="delphi-table9-phase-dsp-validation",
        expected_max_simulated_epoch=expected_max_simulated_epoch,
    )


def _dsp_exposure_repair_source(
    *,
    key: DelphiValidationMixture,
    display_name: str,
    target_metric: str,
    method: str,
    expected_max_simulated_epoch: float,
) -> MixtureSource:
    return MixtureSource(
        key=key,
        display_name=display_name,
        source_csv=f"{DSP_EXPOSURE_REPAIR_VALIDATION_MIXTURE_GCS_DIR}/{key.value}.csv",
        github_issue=6611,
        target_metric=target_metric,
        method=method,
        wandb_series_tag="delphi-dsp-exposure-repair-validation",
        expected_max_simulated_epoch=expected_max_simulated_epoch,
    )


def _dsp_canonical_bowl_source(
    *,
    key: DelphiValidationMixture,
    display_name: str,
    target_metric: str,
    method: str,
    expected_max_simulated_epoch: float,
) -> MixtureSource:
    return MixtureSource(
        key=key,
        display_name=display_name,
        source_csv=f"{DSP_CANONICAL_BOWL_VALIDATION_MIXTURE_GCS_DIR}/{key.value}.csv",
        github_issue=6611,
        target_metric=target_metric,
        method=method,
        wandb_series_tag="delphi-dsp-canonical-bowl-validation",
        expected_max_simulated_epoch=expected_max_simulated_epoch,
    )


def _dsp_gamma_capped_bowl_source(
    *,
    key: DelphiValidationMixture,
    display_name: str,
    target_metric: str,
    method: str,
    expected_max_simulated_epoch: float,
) -> MixtureSource:
    return MixtureSource(
        key=key,
        display_name=display_name,
        source_csv=f"{DSP_GAMMA_CAPPED_BOWL_VALIDATION_MIXTURE_GCS_DIR}/{key.value}.csv",
        github_issue=6611,
        target_metric=target_metric,
        method=method,
        wandb_series_tag="delphi-dsp-gamma-capped-bowl-validation",
        expected_max_simulated_epoch=expected_max_simulated_epoch,
    )


def _sufficiency_floored_source(
    *,
    key: DelphiValidationMixture,
    display_name: str,
    target_metric: str,
    method: str,
    expected_max_simulated_epoch: float,
) -> MixtureSource:
    return MixtureSource(
        key=key,
        display_name=display_name,
        source_csv=f"{DSP_SUFFICIENCY_FLOORED_MIXTURE_GCS_DIR}/{key.value}.csv",
        github_issue=6611,
        target_metric=target_metric,
        method=method,
        wandb_series_tag="delphi-sufficiency-floored-validation",
        expected_max_simulated_epoch=expected_max_simulated_epoch,
    )


def _winner_neighborhood_source(
    *,
    key: DelphiValidationMixture,
    display_name: str,
    target_metric: str,
    method: str,
    expected_max_simulated_epoch: float,
) -> MixtureSource:
    return MixtureSource(
        key=key,
        display_name=display_name,
        source_csv=f"{DSP_WINNER_NEIGHBORHOOD_MIXTURE_GCS_DIR}/{key.value}.csv",
        github_issue=6611,
        target_metric=target_metric,
        method=method,
        wandb_series_tag="delphi-winner-neighborhood-validation",
        expected_max_simulated_epoch=expected_max_simulated_epoch,
    )


def _augmented_profile_source(
    *,
    key: DelphiValidationMixture,
    display_name: str,
    target_metric: str,
    method: str,
    expected_max_simulated_epoch: float,
) -> MixtureSource:
    return MixtureSource(
        key=key,
        display_name=display_name,
        source_csv=f"{DSP_AUGMENTED_PROFILE_MIXTURE_GCS_DIR}/{key.value}.csv",
        github_issue=6611,
        target_metric=target_metric,
        method=method,
        wandb_series_tag="delphi-augmented-profile-validation",
        expected_max_simulated_epoch=expected_max_simulated_epoch,
    )


def _table9_controlled_tilt_source(
    *,
    key: DelphiValidationMixture,
    display_name: str,
    method: str,
    expected_max_simulated_epoch: float,
) -> MixtureSource:
    return MixtureSource(
        key=key,
        display_name=display_name,
        source_csv=f"{DSP_TABLE9_CONTROLLED_TILT_MIXTURE_GCS_DIR}/{key.value}.csv",
        github_issue=6611,
        target_metric=TABLE9_TARGET_METRIC,
        method=method,
        wandb_series_tag="delphi-table9-controlled-tilt-validation",
        expected_max_simulated_epoch=expected_max_simulated_epoch,
    )


def _table9_fresh_anneal_source(
    *,
    key: DelphiValidationMixture,
    display_name: str,
    method: str,
    expected_max_simulated_epoch: float,
) -> MixtureSource:
    return MixtureSource(
        key=key,
        display_name=display_name,
        source_csv=f"{DSP_TABLE9_FRESH_ANNEAL_MIXTURE_GCS_DIR}/{key.value}.csv",
        github_issue=6611,
        target_metric=TABLE9_TARGET_METRIC,
        method=method,
        wandb_series_tag="delphi-table9-fresh-anneal-validation",
        expected_max_simulated_epoch=expected_max_simulated_epoch,
    )


def _proportional_noise_source(*, key: DelphiValidationMixture, label: str) -> MixtureSource:
    return MixtureSource(
        key=key,
        display_name=f"Proportional 3e18 noise repeat {label}",
        source_csv=f"{DELPHI_BASELINE_NOISE_VALIDATION_MIXTURE_GCS_DIR}/proportional.csv",
        github_issue=6611,
        target_metric="noise_floor/proportional_3e18",
        method=f"proportional_3e18_noise_repeat_{label.lower()}",
        wandb_series_tag="delphi-3e18-baseline-noise-panel",
        expected_max_simulated_epoch=None,
    )


def _sep_lf_kl_sweep_source(
    *,
    key: DelphiValidationMixture,
    display_name: str,
    method: str,
    target_metric: str,
    wandb_series_tag: str,
    expected_max_simulated_epoch: float,
) -> MixtureSource:
    return MixtureSource(
        key=key,
        display_name=display_name,
        source_csv=f"{SEP_LF_KL_SWEEP_MIXTURE_GCS_DIR}/{key.value}.csv",
        github_issue=6611,
        target_metric=target_metric,
        method=method,
        wandb_series_tag=wandb_series_tag,
        expected_max_simulated_epoch=expected_max_simulated_epoch,
    )


def _best_phase_model_source(
    *,
    key: DelphiValidationMixture,
    display_name: str,
    target_metric: str,
    expected_max_simulated_epoch: float,
) -> MixtureSource:
    series_target = "table9" if target_metric == TABLE9_TARGET_METRIC else "uncheatable"
    return MixtureSource(
        key=key,
        display_name=display_name,
        source_csv=f"{BEST_PHASE_MODEL_VALIDATION_MIXTURE_GCS_DIR}/{key.value}.csv",
        github_issue=6611,
        target_metric=target_metric,
        method=key.value,
        wandb_series_tag=f"delphi-best-phase-model-{series_target}",
        expected_max_simulated_epoch=expected_max_simulated_epoch,
    )


def _sep_frontier_tied_source(
    *,
    key: DelphiValidationMixture,
    objective: str,
    policy: str,
    repeat: int,
    data_seed: int,
    expected_max_simulated_epoch: float,
) -> MixtureSource:
    target_metric = TABLE9_TARGET_METRIC if objective == "table9" else "eval/uncheatable_eval/bpb"
    objective_tag = "t9" if objective == "table9" else "unch"
    return MixtureSource(
        key=key,
        display_name=f"separate-heads {objective} KL=0.1 {policy} seed-pair {repeat}",
        source_csv=f"{SEP_FRONTIER_TIED_MIXTURE_GCS_DIR}/sepfront_{objective_tag}_{policy}.csv",
        github_issue=6611,
        target_metric=target_metric,
        method=f"separate_heads_frontier_{objective}_{policy}_repeat_{repeat}",
        wandb_series_tag="delphi-separate-heads-frontier-tied-pairs",
        expected_max_simulated_epoch=expected_max_simulated_epoch,
        data_seed_override=data_seed,
    )


def _centered_recency_reorder_source(
    *,
    key: DelphiValidationMixture,
    objective: str,
    order_kl: float,
    data_seed: int,
    expected_max_simulated_epoch: float,
) -> MixtureSource:
    target_metric = TABLE9_TARGET_METRIC if objective == "table9" else "eval/uncheatable_eval/bpb"
    return MixtureSource(
        key=key,
        display_name=f"centered-recency reorder of separate-heads {objective}, order KL={order_kl:g}",
        source_csv=f"{CENTERED_RECENCY_REORDER_MIXTURE_GCS_DIR}/{key.value}.csv",
        github_issue=6611,
        target_metric=target_metric,
        method=key.value,
        wandb_series_tag="delphi-centered-recency-sepheads-reorder",
        expected_max_simulated_epoch=expected_max_simulated_epoch,
        data_seed_override=data_seed,
    )


def _generalized_power_reorder_source(
    *,
    key: DelphiValidationMixture,
    objective: str,
    order_kl: float,
    data_seed: int,
    expected_max_simulated_epoch: float,
) -> MixtureSource:
    target_metric = TABLE9_TARGET_METRIC if objective == "table9" else "eval/uncheatable_eval/bpb"
    return MixtureSource(
        key=key,
        display_name=f"generalized-power reorder of separate-heads {objective}, order KL={order_kl:g}",
        source_csv=f"{GENERALIZED_POWER_REORDER_MIXTURE_GCS_DIR}/{key.value}.csv",
        github_issue=6611,
        target_metric=target_metric,
        method=key.value,
        wandb_series_tag="delphi-generalized-power-sepheads-reorder",
        expected_max_simulated_epoch=expected_max_simulated_epoch,
        data_seed_override=data_seed,
    )


def _symmetric_frontier_source(
    *,
    key: DelphiValidationMixture,
    family: str,
    objective: str,
    policy: str,
    kl_reg: float,
    expected_max_simulated_epoch: float,
) -> MixtureSource:
    target_metric = TABLE9_TARGET_METRIC if objective == "table9" else "eval/uncheatable_eval/bpb"
    return MixtureSource(
        key=key,
        display_name=f"{family} {objective} {policy} KL={kl_reg:g}",
        source_csv=f"{SYMMETRIC_SEPHEADS_GEOMETRY_FRONTIER_MIXTURE_GCS_DIR}/{key.value}.csv",
        github_issue=6611,
        target_metric=target_metric,
        method=key.value,
        wandb_series_tag="delphi-symmetric-sepheads-geometry-frontier",
        expected_max_simulated_epoch=expected_max_simulated_epoch,
    )


def _original_style_matched_sepheads_source(
    *,
    key: DelphiValidationMixture,
    objective: str,
    policy: str,
    kl_reg: float,
    expected_max_simulated_epoch: float,
) -> MixtureSource:
    target_metric = TABLE9_TARGET_METRIC if objective == "table9" else "eval/uncheatable_eval/bpb"
    return MixtureSource(
        key=key,
        display_name=f"original-style matched separate-heads {objective} {policy} KL={kl_reg:g}",
        source_csv=f"{ORIGINAL_STYLE_MATCHED_SEPHEADS_MIXTURE_GCS_DIR}/{key.value}.csv",
        github_issue=6611,
        target_metric=target_metric,
        method=key.value,
        wandb_series_tag="delphi-original-style-matched-sepheads-ablation",
        expected_max_simulated_epoch=expected_max_simulated_epoch,
        data_seed_override=ORIGINAL_STYLE_MATCHED_SEPHEADS_DATA_SEED,
    )


MIXTURE_SOURCES: dict[DelphiValidationMixture, MixtureSource] = {
    DelphiValidationMixture.OLMIX_D001_KL005_CAP4: MixtureSource(
        key=DelphiValidationMixture.OLMIX_D001_KL005_CAP4,
        display_name="OLMix delta=0.01 KL=0.05 cap=4",
        source_csv=str(
            REFERENCE_OUTPUT_DIR
            / "olmix_huber_delta_sweep_300m_20260625"
            / "delta_0p01"
            / "uncheatable_eval_bpb_rep_cap4"
            / "proposed_mixture_weights.csv"
        ),
        github_issue=6608,
        target_metric="eval/uncheatable_eval/bpb",
        method="olmix_delta0p01_kl0p05_cap4",
        wandb_series_tag="delphi-uncheatable-optimized-mixtures",
        expected_max_simulated_epoch=4.0,
    ),
    DelphiValidationMixture.DSP_EFFECTIVE_EXPOSURE_KL01: MixtureSource(
        key=DelphiValidationMixture.DSP_EFFECTIVE_EXPOSURE_KL01,
        display_name="DSP effective-exposure KL=0.1",
        source_csv=str(
            REFERENCE_OUTPUT_DIR
            / "dsp_effective_exposure_l2_kl_sweep_deletion_augmented_300m_20260625"
            / "dsp_effective_exposure_l2_0.01_kl_only_0.1"
            / "proposed_mixture_weights.csv"
        ),
        github_issue=6602,
        target_metric="eval/uncheatable_eval/bpb",
        method="dsp_effective_exposure_l2_0p01_kl0p1",
        wandb_series_tag="delphi-uncheatable-optimized-mixtures",
    ),
    DelphiValidationMixture.DSP_CANONICAL_KL01: MixtureSource(
        key=DelphiValidationMixture.DSP_CANONICAL_KL01,
        display_name="DSP canonical KL=0.1",
        source_csv=str(
            REFERENCE_OUTPUT_DIR
            / "dsp_l2_kl_sweep_deletion_augmented_300m_20260625"
            / "dsp_l2_0.0001_kl_only_0.1"
            / "proposed_mixture_weights.csv"
        ),
        github_issue=6602,
        target_metric="eval/uncheatable_eval/bpb",
        method="dsp_canonical_l2_1e-4_kl0p1",
        wandb_series_tag="delphi-uncheatable-optimized-mixtures",
    ),
    DelphiValidationMixture.OLMIX_TABLE9_D001_KL005_CAP4: MixtureSource(
        key=DelphiValidationMixture.OLMIX_TABLE9_D001_KL005_CAP4,
        display_name="OLMix Table-9 macro delta=0.01 KL=0.05 cap=4",
        source_csv=str(MIXTURE_ASSET_DIR / "olmix_table9_delta0p01_kl0p05_cap4.csv"),
        github_issue=6611,
        target_metric=TABLE9_TARGET_METRIC,
        method="olmix_table9_delta0p01_kl0p05_cap4",
        wandb_series_tag="delphi-table9-optimized-mixtures",
        expected_max_simulated_epoch=4.0,
    ),
    DelphiValidationMixture.DSP_EFFECTIVE_EXPOSURE_TABLE9_KL0025: MixtureSource(
        key=DelphiValidationMixture.DSP_EFFECTIVE_EXPOSURE_TABLE9_KL0025,
        display_name="DSP effective-exposure Table-9 macro KL=0.025",
        source_csv=str(MIXTURE_ASSET_DIR / "dsp_effexp_table9_kl0p025.csv"),
        github_issue=6611,
        target_metric=TABLE9_TARGET_METRIC,
        method="dsp_effective_exposure_table9_kl0p025",
        wandb_series_tag="delphi-table9-optimized-mixtures",
        expected_max_simulated_epoch=8.530735,
    ),
    DelphiValidationMixture.DSP_EFFECTIVE_EXPOSURE_TABLE9_KL0P05: MixtureSource(
        key=DelphiValidationMixture.DSP_EFFECTIVE_EXPOSURE_TABLE9_KL0P05,
        display_name="DSP effective-exposure Table-9 macro KL=0.05",
        source_csv=f"{KL_SWEEP_MIXTURE_GCS_DIR}/dsp_effexp_table9_kl0p05.csv",
        github_issue=6611,
        target_metric=TABLE9_TARGET_METRIC,
        method="dsp_effective_exposure_table9_kl0p05",
        wandb_series_tag="delphi-table9-dsp-kl-sweep",
        expected_max_simulated_epoch=7.612337,
    ),
    DelphiValidationMixture.DSP_EFFECTIVE_EXPOSURE_TABLE9_KL0P1: MixtureSource(
        key=DelphiValidationMixture.DSP_EFFECTIVE_EXPOSURE_TABLE9_KL0P1,
        display_name="DSP effective-exposure Table-9 macro KL=0.1",
        source_csv=f"{KL_SWEEP_MIXTURE_GCS_DIR}/dsp_effexp_table9_kl0p1.csv",
        github_issue=6611,
        target_metric=TABLE9_TARGET_METRIC,
        method="dsp_effective_exposure_table9_kl0p1",
        wandb_series_tag="delphi-table9-dsp-kl-sweep",
        expected_max_simulated_epoch=6.93664,
    ),
    DelphiValidationMixture.DSP_EFFECTIVE_EXPOSURE_TABLE9_KL0P2: MixtureSource(
        key=DelphiValidationMixture.DSP_EFFECTIVE_EXPOSURE_TABLE9_KL0P2,
        display_name="DSP effective-exposure Table-9 macro KL=0.2",
        source_csv=f"{KL_SWEEP_MIXTURE_GCS_DIR}/dsp_effexp_table9_kl0p2.csv",
        github_issue=6611,
        target_metric=TABLE9_TARGET_METRIC,
        method="dsp_effective_exposure_table9_kl0p2",
        wandb_series_tag="delphi-table9-dsp-kl-sweep",
        expected_max_simulated_epoch=6.078404,
    ),
    DelphiValidationMixture.DSP_EFFECTIVE_EXPOSURE_TABLE9_KL0P25: _table9_dsp_validation_source(
        key=DelphiValidationMixture.DSP_EFFECTIVE_EXPOSURE_TABLE9_KL0P25,
        display_name="DSP effective-exposure Table-9 macro KL=0.25",
        method="dsp_effective_exposure_table9_kl0p25",
        wandb_series_tag="delphi-table9-dsp-kl-sweep",
        expected_max_simulated_epoch=5.630997,
    ),
    DelphiValidationMixture.DSP_EFFECTIVE_EXPOSURE_TABLE9_KL0P3: _table9_dsp_validation_source(
        key=DelphiValidationMixture.DSP_EFFECTIVE_EXPOSURE_TABLE9_KL0P3,
        display_name="DSP effective-exposure Table-9 macro KL=0.3",
        method="dsp_effective_exposure_table9_kl0p3",
        wandb_series_tag="delphi-table9-dsp-kl-sweep",
        expected_max_simulated_epoch=5.050459,
    ),
    DelphiValidationMixture.DSP_EFFECTIVE_EXPOSURE_TABLE9_KL0P4: _table9_dsp_validation_source(
        key=DelphiValidationMixture.DSP_EFFECTIVE_EXPOSURE_TABLE9_KL0P4,
        display_name="DSP effective-exposure Table-9 macro KL=0.4",
        method="dsp_effective_exposure_table9_kl0p4",
        wandb_series_tag="delphi-table9-dsp-kl-sweep",
        expected_max_simulated_epoch=4.327265,
    ),
    DelphiValidationMixture.DSP_EFFECTIVE_EXPOSURE_TABLE9_KL0P5: MixtureSource(
        key=DelphiValidationMixture.DSP_EFFECTIVE_EXPOSURE_TABLE9_KL0P5,
        display_name="DSP effective-exposure Table-9 macro KL=0.5",
        source_csv=f"{KL_SWEEP_MIXTURE_GCS_DIR}/dsp_effexp_table9_kl0p5.csv",
        github_issue=6611,
        target_metric=TABLE9_TARGET_METRIC,
        method="dsp_effective_exposure_table9_kl0p5",
        wandb_series_tag="delphi-table9-dsp-kl-sweep",
        expected_max_simulated_epoch=3.863495,
    ),
    DelphiValidationMixture.DSP_PER_COMPONENT_TABLE9_KL0P025: _table9_dsp_validation_source(
        key=DelphiValidationMixture.DSP_PER_COMPONENT_TABLE9_KL0P025,
        display_name="DSP per-component effective-exposure Table-9 KL=0.025",
        method="dsp_per_component_effective_exposure_table9_kl0p025",
        wandb_series_tag="delphi-table9-per-component-dsp-kl-sweep",
        expected_max_simulated_epoch=7.415576,
    ),
    DelphiValidationMixture.DSP_PER_COMPONENT_TABLE9_KL0P05: _table9_dsp_validation_source(
        key=DelphiValidationMixture.DSP_PER_COMPONENT_TABLE9_KL0P05,
        display_name="DSP per-component effective-exposure Table-9 KL=0.05",
        method="dsp_per_component_effective_exposure_table9_kl0p05",
        wandb_series_tag="delphi-table9-per-component-dsp-kl-sweep",
        expected_max_simulated_epoch=7.052982,
    ),
    DelphiValidationMixture.DSP_PER_COMPONENT_TABLE9_KL0P1: _table9_dsp_validation_source(
        key=DelphiValidationMixture.DSP_PER_COMPONENT_TABLE9_KL0P1,
        display_name="DSP per-component effective-exposure Table-9 KL=0.1",
        method="dsp_per_component_effective_exposure_table9_kl0p1",
        wandb_series_tag="delphi-table9-per-component-dsp-kl-sweep",
        expected_max_simulated_epoch=6.800308,
    ),
    DelphiValidationMixture.DSP_PER_COMPONENT_TABLE9_KL0P2: _table9_dsp_validation_source(
        key=DelphiValidationMixture.DSP_PER_COMPONENT_TABLE9_KL0P2,
        display_name="DSP per-component effective-exposure Table-9 KL=0.2",
        method="dsp_per_component_effective_exposure_table9_kl0p2",
        wandb_series_tag="delphi-table9-per-component-dsp-kl-sweep",
        expected_max_simulated_epoch=7.050150,
    ),
    DelphiValidationMixture.DSP_PER_COMPONENT_TABLE9_KL0P25: _table9_dsp_validation_source(
        key=DelphiValidationMixture.DSP_PER_COMPONENT_TABLE9_KL0P25,
        display_name="DSP per-component effective-exposure Table-9 KL=0.25",
        method="dsp_per_component_effective_exposure_table9_kl0p25",
        wandb_series_tag="delphi-table9-per-component-dsp-kl-sweep",
        expected_max_simulated_epoch=6.824361,
    ),
    DelphiValidationMixture.DSP_PER_COMPONENT_TABLE9_KL0P3: _table9_dsp_validation_source(
        key=DelphiValidationMixture.DSP_PER_COMPONENT_TABLE9_KL0P3,
        display_name="DSP per-component effective-exposure Table-9 KL=0.3",
        method="dsp_per_component_effective_exposure_table9_kl0p3",
        wandb_series_tag="delphi-table9-per-component-dsp-kl-sweep",
        expected_max_simulated_epoch=6.405132,
    ),
    DelphiValidationMixture.DSP_PER_COMPONENT_TABLE9_KL0P4: _table9_dsp_validation_source(
        key=DelphiValidationMixture.DSP_PER_COMPONENT_TABLE9_KL0P4,
        display_name="DSP per-component effective-exposure Table-9 KL=0.4",
        method="dsp_per_component_effective_exposure_table9_kl0p4",
        wandb_series_tag="delphi-table9-per-component-dsp-kl-sweep",
        expected_max_simulated_epoch=5.451996,
    ),
    DelphiValidationMixture.DSP_PER_COMPONENT_TABLE9_KL0P5: _table9_dsp_validation_source(
        key=DelphiValidationMixture.DSP_PER_COMPONENT_TABLE9_KL0P5,
        display_name="DSP per-component effective-exposure Table-9 KL=0.5",
        method="dsp_per_component_effective_exposure_table9_kl0p5",
        wandb_series_tag="delphi-table9-per-component-dsp-kl-sweep",
        expected_max_simulated_epoch=4.610282,
    ),
    DelphiValidationMixture.DSP_SHRINK_FIXED_SPEARMAN_KL0P2: _table9_adaptive_shrinkage_source(
        key=DelphiValidationMixture.DSP_SHRINK_FIXED_SPEARMAN_KL0P2,
        display_name="DSP Table-9 fixed shrinkage, OOF Spearman, KL=0.2",
        method="dsp_table9_fixed_shrinkage_oof_spearman_kl0p2",
        expected_max_simulated_epoch=6.280670,
    ),
    DelphiValidationMixture.DSP_SHRINK_FIXED_R2HARM_KL0P2: _table9_adaptive_shrinkage_source(
        key=DelphiValidationMixture.DSP_SHRINK_FIXED_R2HARM_KL0P2,
        display_name="DSP Table-9 fixed shrinkage, OOF R2 x harm-t, KL=0.2",
        method="dsp_table9_fixed_shrinkage_oof_r2_x_harm_t_kl0p2",
        expected_max_simulated_epoch=4.244335,
    ),
    DelphiValidationMixture.DSP_SHRINK_TV_SPEARMAN_B0P5_KL0P2: _table9_adaptive_shrinkage_source(
        key=DelphiValidationMixture.DSP_SHRINK_TV_SPEARMAN_B0P5_KL0P2,
        display_name="DSP Table-9 TV-adaptive shrinkage, OOF Spearman, beta=0.5, KL=0.2",
        method="dsp_table9_tv_adaptive_shrinkage_oof_spearman_beta0p5_kl0p2",
        expected_max_simulated_epoch=5.665925,
    ),
    DelphiValidationMixture.DSP_SHRINK_TV_R2HARM_B1_KL0P2: _table9_adaptive_shrinkage_source(
        key=DelphiValidationMixture.DSP_SHRINK_TV_R2HARM_B1_KL0P2,
        display_name="DSP Table-9 TV-adaptive shrinkage, OOF R2 x harm-t, beta=1, KL=0.2",
        method="dsp_table9_tv_adaptive_shrinkage_oof_r2_x_harm_t_beta1_kl0p2",
        expected_max_simulated_epoch=3.190545,
    ),
    DelphiValidationMixture.DSP_SHRINK_DELTA_SPEARMAN_B0P5_KL0P2: _table9_adaptive_shrinkage_source(
        key=DelphiValidationMixture.DSP_SHRINK_DELTA_SPEARMAN_B0P5_KL0P2,
        display_name="DSP Table-9 delta-adaptive shrinkage, OOF Spearman, beta=0.5, KL=0.2",
        method="dsp_table9_delta_adaptive_shrinkage_oof_spearman_beta0p5_kl0p2",
        expected_max_simulated_epoch=3.399065,
    ),
    DelphiValidationMixture.DSP_SHRINK_DELTA_R2HARM_B1_KL0P2: _table9_adaptive_shrinkage_source(
        key=DelphiValidationMixture.DSP_SHRINK_DELTA_R2HARM_B1_KL0P2,
        display_name="DSP Table-9 delta-adaptive shrinkage, OOF R2 x harm-t, beta=1, KL=0.2",
        method="dsp_table9_delta_adaptive_shrinkage_oof_r2_x_harm_t_beta1_kl0p2",
        expected_max_simulated_epoch=1.957739,
    ),
    DelphiValidationMixture.DSP_SHRINK_UNC_SPEARMAN_G0P25_KL0P2: _table9_adaptive_shrinkage_source(
        key=DelphiValidationMixture.DSP_SHRINK_UNC_SPEARMAN_G0P25_KL0P2,
        display_name="DSP Table-9 uncertainty penalty, OOF Spearman, gamma=0.25, KL=0.2",
        method="dsp_table9_uncertainty_penalty_oof_spearman_gamma0p25_kl0p2",
        expected_max_simulated_epoch=6.279232,
    ),
    DelphiValidationMixture.DSP_SHRINK_UNC_R2HARM_G0P5_KL0P2: _table9_adaptive_shrinkage_source(
        key=DelphiValidationMixture.DSP_SHRINK_UNC_R2HARM_G0P5_KL0P2,
        display_name="DSP Table-9 uncertainty penalty, OOF R2 x harm-t, gamma=0.5, KL=0.2",
        method="dsp_table9_uncertainty_penalty_oof_r2_x_harm_t_gamma0p5_kl0p2",
        expected_max_simulated_epoch=4.216520,
    ),
    DelphiValidationMixture.QSPLIT_RUN00018_TABLE9_ANCHOR: _table9_phase_split_dsp_source(
        key=DelphiValidationMixture.QSPLIT_RUN00018_TABLE9_ANCHOR,
        display_name="QSplit run00018 Table-9 anchor",
        method="qsplit_run00018_table9_anchor",
        expected_max_simulated_epoch=61.809205,
    ),
    DelphiValidationMixture.DSP_SPLIT_TABLE9_L2_0P01_KL0P3: _table9_phase_split_dsp_source(
        key=DelphiValidationMixture.DSP_SPLIT_TABLE9_L2_0P01_KL0P3,
        display_name="DSP split saturation/penalty Table-9 L2=0.01 KL=0.3",
        method="dsp_split_saturation_penalty_table9_l2_0p01_kl0p3",
        expected_max_simulated_epoch=4.293921,
    ),
    DelphiValidationMixture.DSP_SPLIT_TABLE9_L2_0P01_KL0P4: _table9_phase_split_dsp_source(
        key=DelphiValidationMixture.DSP_SPLIT_TABLE9_L2_0P01_KL0P4,
        display_name="DSP split saturation/penalty Table-9 L2=0.01 KL=0.4",
        method="dsp_split_saturation_penalty_table9_l2_0p01_kl0p4",
        expected_max_simulated_epoch=3.667958,
    ),
    DelphiValidationMixture.DSP_SPLIT_TABLE9_L2_0P01_KL0P4_REPEAT: _table9_phase_split_dsp_source(
        key=DelphiValidationMixture.DSP_SPLIT_TABLE9_L2_0P01_KL0P4_REPEAT,
        display_name="DSP split saturation/penalty Table-9 L2=0.01 KL=0.4 repeat",
        method="dsp_split_saturation_penalty_table9_l2_0p01_kl0p4_repeat",
        expected_max_simulated_epoch=3.667958,
    ),
    DelphiValidationMixture.DSP_EFFECTIVE_EXPOSURE_TABLE9_L2_0P01_KL0P5: _table9_phase_split_dsp_source(
        key=DelphiValidationMixture.DSP_EFFECTIVE_EXPOSURE_TABLE9_L2_0P01_KL0P5,
        display_name="DSP effective-exposure Table-9 L2=0.01 KL=0.5",
        method="dsp_effective_exposure_table9_l2_0p01_kl0p5",
        expected_max_simulated_epoch=4.805505,
    ),
    DelphiValidationMixture.OLMIX_ONE_PHASE_UNCHEATABLE_D001_KL005_CAP4: _one_phase_uncheatable_validation_source(
        key=DelphiValidationMixture.OLMIX_ONE_PHASE_UNCHEATABLE_D001_KL005_CAP4,
        display_name="One-phase OLMix uncheatable BPB delta=0.01 KL=0.05 cap=4",
        method="one_phase_olmix_uncheatable_delta0p01_kl0p05_cap4",
        expected_max_simulated_epoch=4.000002,
    ),
    DelphiValidationMixture.OLMIX_ONE_PHASE_UNCHEATABLE_D001_KL0_CAP4: _one_phase_olmix_kl_sweep_source(
        key=DelphiValidationMixture.OLMIX_ONE_PHASE_UNCHEATABLE_D001_KL0_CAP4,
        display_name="One-phase OLMix uncheatable BPB delta=0.01 KL=0 cap=4",
        github_issue=6608,
        target_metric="eval/uncheatable_eval/bpb",
        method="one_phase_olmix_uncheatable_delta0p01_kl0_cap4",
    ),
    DelphiValidationMixture.OLMIX_ONE_PHASE_UNCHEATABLE_D001_KL0P005_CAP4: _one_phase_olmix_kl_sweep_source(
        key=DelphiValidationMixture.OLMIX_ONE_PHASE_UNCHEATABLE_D001_KL0P005_CAP4,
        display_name="One-phase OLMix uncheatable BPB delta=0.01 KL=0.005 cap=4",
        github_issue=6608,
        target_metric="eval/uncheatable_eval/bpb",
        method="one_phase_olmix_uncheatable_delta0p01_kl0p005_cap4",
    ),
    DelphiValidationMixture.OLMIX_ONE_PHASE_UNCHEATABLE_D001_KL0P01_CAP4: _one_phase_olmix_kl_sweep_source(
        key=DelphiValidationMixture.OLMIX_ONE_PHASE_UNCHEATABLE_D001_KL0P01_CAP4,
        display_name="One-phase OLMix uncheatable BPB delta=0.01 KL=0.01 cap=4",
        github_issue=6608,
        target_metric="eval/uncheatable_eval/bpb",
        method="one_phase_olmix_uncheatable_delta0p01_kl0p01_cap4",
    ),
    DelphiValidationMixture.OLMIX_ONE_PHASE_UNCHEATABLE_D001_KL0P025_CAP4: _one_phase_olmix_kl_sweep_source(
        key=DelphiValidationMixture.OLMIX_ONE_PHASE_UNCHEATABLE_D001_KL0P025_CAP4,
        display_name="One-phase OLMix uncheatable BPB delta=0.01 KL=0.025 cap=4",
        github_issue=6608,
        target_metric="eval/uncheatable_eval/bpb",
        method="one_phase_olmix_uncheatable_delta0p01_kl0p025_cap4",
    ),
    DelphiValidationMixture.OLMIX_ONE_PHASE_UNCHEATABLE_D001_KL0P075_CAP4: _one_phase_olmix_kl_sweep_source(
        key=DelphiValidationMixture.OLMIX_ONE_PHASE_UNCHEATABLE_D001_KL0P075_CAP4,
        display_name="One-phase OLMix uncheatable BPB delta=0.01 KL=0.075 cap=4",
        github_issue=6608,
        target_metric="eval/uncheatable_eval/bpb",
        method="one_phase_olmix_uncheatable_delta0p01_kl0p075_cap4",
    ),
    DelphiValidationMixture.OLMIX_ONE_PHASE_UNCHEATABLE_D001_KL0P1_CAP4: _one_phase_olmix_kl_sweep_source(
        key=DelphiValidationMixture.OLMIX_ONE_PHASE_UNCHEATABLE_D001_KL0P1_CAP4,
        display_name="One-phase OLMix uncheatable BPB delta=0.01 KL=0.1 cap=4",
        github_issue=6608,
        target_metric="eval/uncheatable_eval/bpb",
        method="one_phase_olmix_uncheatable_delta0p01_kl0p1_cap4",
    ),
    DelphiValidationMixture.OLMIX_ONE_PHASE_UNCHEATABLE_D001_KL0P2_CAP4: _one_phase_olmix_kl_sweep_source(
        key=DelphiValidationMixture.OLMIX_ONE_PHASE_UNCHEATABLE_D001_KL0P2_CAP4,
        display_name="One-phase OLMix uncheatable BPB delta=0.01 KL=0.2 cap=4",
        github_issue=6608,
        target_metric="eval/uncheatable_eval/bpb",
        method="one_phase_olmix_uncheatable_delta0p01_kl0p2_cap4",
    ),
    DelphiValidationMixture.OLMIX_ONE_PHASE_UNCHEATABLE_D001_KL0P5_CAP4: _one_phase_olmix_kl_sweep_source(
        key=DelphiValidationMixture.OLMIX_ONE_PHASE_UNCHEATABLE_D001_KL0P5_CAP4,
        display_name="One-phase OLMix uncheatable BPB delta=0.01 KL=0.5 cap=4",
        github_issue=6608,
        target_metric="eval/uncheatable_eval/bpb",
        method="one_phase_olmix_uncheatable_delta0p01_kl0p5_cap4",
    ),
    DelphiValidationMixture.DSP_ONE_PHASE_EFFECTIVE_EXPOSURE_UNCHEATABLE_KL0P1: _one_phase_uncheatable_validation_source(
        key=DelphiValidationMixture.DSP_ONE_PHASE_EFFECTIVE_EXPOSURE_UNCHEATABLE_KL0P1,
        display_name="One-phase DSP effective-exposure uncheatable BPB KL=0.1",
        method="one_phase_dsp_effective_exposure_uncheatable_l2_0p01_kl0p1",
        expected_max_simulated_epoch=8.128131,
    ),
    DelphiValidationMixture.OLMIX_ONE_PHASE_TABLE9_D001_KL005_CAP4: _one_phase_table9_validation_source(
        key=DelphiValidationMixture.OLMIX_ONE_PHASE_TABLE9_D001_KL005_CAP4,
        display_name="One-phase OLMix Table-9 macro delta=0.01 KL=0.05 cap=4",
        method="one_phase_olmix_table9_delta0p01_kl0p05_cap4",
        expected_max_simulated_epoch=4.0,
    ),
    DelphiValidationMixture.OLMIX_ONE_PHASE_TABLE9_D001_KL0_CAP4: _one_phase_olmix_kl_sweep_source(
        key=DelphiValidationMixture.OLMIX_ONE_PHASE_TABLE9_D001_KL0_CAP4,
        display_name="One-phase OLMix Table-9 macro delta=0.01 KL=0 cap=4",
        github_issue=6611,
        target_metric=TABLE9_TARGET_METRIC,
        method="one_phase_olmix_table9_delta0p01_kl0_cap4",
    ),
    DelphiValidationMixture.OLMIX_ONE_PHASE_TABLE9_D001_KL0P005_CAP4: _one_phase_olmix_kl_sweep_source(
        key=DelphiValidationMixture.OLMIX_ONE_PHASE_TABLE9_D001_KL0P005_CAP4,
        display_name="One-phase OLMix Table-9 macro delta=0.01 KL=0.005 cap=4",
        github_issue=6611,
        target_metric=TABLE9_TARGET_METRIC,
        method="one_phase_olmix_table9_delta0p01_kl0p005_cap4",
    ),
    DelphiValidationMixture.OLMIX_ONE_PHASE_TABLE9_D001_KL0P01_CAP4: _one_phase_olmix_kl_sweep_source(
        key=DelphiValidationMixture.OLMIX_ONE_PHASE_TABLE9_D001_KL0P01_CAP4,
        display_name="One-phase OLMix Table-9 macro delta=0.01 KL=0.01 cap=4",
        github_issue=6611,
        target_metric=TABLE9_TARGET_METRIC,
        method="one_phase_olmix_table9_delta0p01_kl0p01_cap4",
    ),
    DelphiValidationMixture.OLMIX_ONE_PHASE_TABLE9_D001_KL0P025_CAP4: _one_phase_olmix_kl_sweep_source(
        key=DelphiValidationMixture.OLMIX_ONE_PHASE_TABLE9_D001_KL0P025_CAP4,
        display_name="One-phase OLMix Table-9 macro delta=0.01 KL=0.025 cap=4",
        github_issue=6611,
        target_metric=TABLE9_TARGET_METRIC,
        method="one_phase_olmix_table9_delta0p01_kl0p025_cap4",
    ),
    DelphiValidationMixture.OLMIX_ONE_PHASE_TABLE9_D001_KL0P075_CAP4: _one_phase_olmix_kl_sweep_source(
        key=DelphiValidationMixture.OLMIX_ONE_PHASE_TABLE9_D001_KL0P075_CAP4,
        display_name="One-phase OLMix Table-9 macro delta=0.01 KL=0.075 cap=4",
        github_issue=6611,
        target_metric=TABLE9_TARGET_METRIC,
        method="one_phase_olmix_table9_delta0p01_kl0p075_cap4",
    ),
    DelphiValidationMixture.OLMIX_ONE_PHASE_TABLE9_D001_KL0P1_CAP4: _one_phase_olmix_kl_sweep_source(
        key=DelphiValidationMixture.OLMIX_ONE_PHASE_TABLE9_D001_KL0P1_CAP4,
        display_name="One-phase OLMix Table-9 macro delta=0.01 KL=0.1 cap=4",
        github_issue=6611,
        target_metric=TABLE9_TARGET_METRIC,
        method="one_phase_olmix_table9_delta0p01_kl0p1_cap4",
    ),
    DelphiValidationMixture.OLMIX_ONE_PHASE_TABLE9_D001_KL0P2_CAP4: _one_phase_olmix_kl_sweep_source(
        key=DelphiValidationMixture.OLMIX_ONE_PHASE_TABLE9_D001_KL0P2_CAP4,
        display_name="One-phase OLMix Table-9 macro delta=0.01 KL=0.2 cap=4",
        github_issue=6611,
        target_metric=TABLE9_TARGET_METRIC,
        method="one_phase_olmix_table9_delta0p01_kl0p2_cap4",
    ),
    DelphiValidationMixture.OLMIX_ONE_PHASE_TABLE9_D001_KL0P5_CAP4: _one_phase_olmix_kl_sweep_source(
        key=DelphiValidationMixture.OLMIX_ONE_PHASE_TABLE9_D001_KL0P5_CAP4,
        display_name="One-phase OLMix Table-9 macro delta=0.01 KL=0.5 cap=4",
        github_issue=6611,
        target_metric=TABLE9_TARGET_METRIC,
        method="one_phase_olmix_table9_delta0p01_kl0p5_cap4",
    ),
    DelphiValidationMixture.DSP_ONE_PHASE_EFFECTIVE_EXPOSURE_TABLE9_KL0P05: _one_phase_table9_validation_source(
        key=DelphiValidationMixture.DSP_ONE_PHASE_EFFECTIVE_EXPOSURE_TABLE9_KL0P05,
        display_name="One-phase DSP effective-exposure Table-9 macro KL=0.05",
        method="one_phase_dsp_effective_exposure_table9_kl0p05",
        expected_max_simulated_epoch=20.648222,
    ),
    DelphiValidationMixture.DSP_ONE_PHASE_EFFECTIVE_EXPOSURE_TABLE9_KL0P1: _one_phase_table9_validation_source(
        key=DelphiValidationMixture.DSP_ONE_PHASE_EFFECTIVE_EXPOSURE_TABLE9_KL0P1,
        display_name="One-phase DSP effective-exposure Table-9 macro KL=0.1",
        method="one_phase_dsp_effective_exposure_table9_kl0p1",
        expected_max_simulated_epoch=16.541119,
    ),
    DelphiValidationMixture.DSP_ONE_PHASE_EFFECTIVE_EXPOSURE_TABLE9_KL0P2: _one_phase_table9_validation_source(
        key=DelphiValidationMixture.DSP_ONE_PHASE_EFFECTIVE_EXPOSURE_TABLE9_KL0P2,
        display_name="One-phase DSP effective-exposure Table-9 macro KL=0.2",
        method="one_phase_dsp_effective_exposure_table9_kl0p2",
        expected_max_simulated_epoch=13.673005,
    ),
    DelphiValidationMixture.DSP_ONE_PHASE_EFFECTIVE_EXPOSURE_TABLE9_KL0P25: _one_phase_table9_validation_source(
        key=DelphiValidationMixture.DSP_ONE_PHASE_EFFECTIVE_EXPOSURE_TABLE9_KL0P25,
        display_name="One-phase DSP effective-exposure Table-9 macro KL=0.25",
        method="one_phase_dsp_effective_exposure_table9_kl0p25",
        expected_max_simulated_epoch=12.559517,
    ),
    DelphiValidationMixture.DSP_ONE_PHASE_EFFECTIVE_EXPOSURE_TABLE9_KL0P3: _one_phase_table9_validation_source(
        key=DelphiValidationMixture.DSP_ONE_PHASE_EFFECTIVE_EXPOSURE_TABLE9_KL0P3,
        display_name="One-phase DSP effective-exposure Table-9 macro KL=0.3",
        method="one_phase_dsp_effective_exposure_table9_kl0p3",
        expected_max_simulated_epoch=11.590955,
    ),
    DelphiValidationMixture.DSP_ONE_PHASE_EFFECTIVE_EXPOSURE_TABLE9_KL0P1_REPEAT_A: MixtureSource(
        key=DelphiValidationMixture.DSP_ONE_PHASE_EFFECTIVE_EXPOSURE_TABLE9_KL0P1_REPEAT_A,
        display_name="One-phase DSP effective-exposure Table-9 macro KL=0.1 repeat A",
        source_csv=f"{ONE_PHASE_TABLE9_VALIDATION_MIXTURE_GCS_DIR}/dsp_onephase_effexp_table9_kl0p1.csv",
        github_issue=6611,
        target_metric=TABLE9_TARGET_METRIC,
        method="one_phase_dsp_effective_exposure_table9_kl0p1_repeat_a",
        wandb_series_tag="delphi-table9-phase-diagnostic-repeats",
        expected_max_simulated_epoch=16.541119,
    ),
    DelphiValidationMixture.DSP_ONE_PHASE_EFFECTIVE_EXPOSURE_TABLE9_KL0P1_REPEAT_B: MixtureSource(
        key=DelphiValidationMixture.DSP_ONE_PHASE_EFFECTIVE_EXPOSURE_TABLE9_KL0P1_REPEAT_B,
        display_name="One-phase DSP effective-exposure Table-9 macro KL=0.1 repeat B",
        source_csv=f"{ONE_PHASE_TABLE9_VALIDATION_MIXTURE_GCS_DIR}/dsp_onephase_effexp_table9_kl0p1.csv",
        github_issue=6611,
        target_metric=TABLE9_TARGET_METRIC,
        method="one_phase_dsp_effective_exposure_table9_kl0p1_repeat_b",
        wandb_series_tag="delphi-table9-phase-diagnostic-repeats",
        expected_max_simulated_epoch=16.541119,
    ),
    DelphiValidationMixture.DSP_SPLIT_TABLE9_L2_0P01_KL0P3_REPEAT_A: MixtureSource(
        key=DelphiValidationMixture.DSP_SPLIT_TABLE9_L2_0P01_KL0P3_REPEAT_A,
        display_name="DSP split saturation/penalty Table-9 L2=0.01 KL=0.3 repeat A",
        source_csv=f"{TABLE9_PHASE_SPLIT_DSP_VALIDATION_MIXTURE_GCS_DIR}/dsp_split_table9_l2_0p01_kl0p3.csv",
        github_issue=6611,
        target_metric=TABLE9_TARGET_METRIC,
        method="dsp_split_saturation_penalty_table9_l2_0p01_kl0p3_repeat_a",
        wandb_series_tag="delphi-table9-phase-diagnostic-repeats",
        expected_max_simulated_epoch=4.293921,
    ),
    DelphiValidationMixture.DSP_SPLIT_TABLE9_L2_0P01_KL0P3_REPEAT_B: MixtureSource(
        key=DelphiValidationMixture.DSP_SPLIT_TABLE9_L2_0P01_KL0P3_REPEAT_B,
        display_name="DSP split saturation/penalty Table-9 L2=0.01 KL=0.3 repeat B",
        source_csv=f"{TABLE9_PHASE_SPLIT_DSP_VALIDATION_MIXTURE_GCS_DIR}/dsp_split_table9_l2_0p01_kl0p3.csv",
        github_issue=6611,
        target_metric=TABLE9_TARGET_METRIC,
        method="dsp_split_saturation_penalty_table9_l2_0p01_kl0p3_repeat_b",
        wandb_series_tag="delphi-table9-phase-diagnostic-repeats",
        expected_max_simulated_epoch=4.293921,
    ),
    DelphiValidationMixture.DSP_UNCHEATABLE_EXPOSURE_TARGETED: _dsp_exposure_repair_source(
        key=DelphiValidationMixture.DSP_UNCHEATABLE_EXPOSURE_TARGETED,
        display_name="DSP uncheatable aggregate-exposure targeted repair",
        target_metric="eval/uncheatable_eval/bpb",
        method="dsp_uncheatable_exposure_targeted_repair",
        expected_max_simulated_epoch=8.128129,
    ),
    DelphiValidationMixture.DSP_UNCHEATABLE_EXPOSURE_ALL_DEFICITS: _dsp_exposure_repair_source(
        key=DelphiValidationMixture.DSP_UNCHEATABLE_EXPOSURE_ALL_DEFICITS,
        display_name="DSP uncheatable aggregate-exposure all-deficits repair",
        target_metric="eval/uncheatable_eval/bpb",
        method="dsp_uncheatable_exposure_all_deficits_repair",
        expected_max_simulated_epoch=8.128129,
    ),
    DelphiValidationMixture.DSP_UNCHEATABLE_SUPPORT_AWARE_RAW_OPTIMUM: MixtureSource(
        key=DelphiValidationMixture.DSP_UNCHEATABLE_SUPPORT_AWARE_RAW_OPTIMUM,
        display_name="DSP uncheatable support-aware raw optimum",
        source_csv=(f"{DSP_SUPPORT_AWARE_VALIDATION_MIXTURE_GCS_DIR}/dsp_uncheatable_support_aware_raw_optimum.csv"),
        github_issue=6602,
        target_metric="eval/uncheatable_eval/bpb",
        method="dsp_uncheatable_support_aware_effexp_floor_raw_optimum",
        wandb_series_tag="delphi-dsp-support-aware-validation",
        expected_max_simulated_epoch=7.832061,
    ),
    DelphiValidationMixture.DSP_TABLE9_EXPOSURE_TARGETED: _dsp_exposure_repair_source(
        key=DelphiValidationMixture.DSP_TABLE9_EXPOSURE_TARGETED,
        display_name="DSP Table-9 aggregate-exposure targeted repair",
        target_metric=TABLE9_TARGET_METRIC,
        method="dsp_table9_exposure_targeted_repair",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.DSP_TABLE9_EXPOSURE_ALL_DEFICITS: _dsp_exposure_repair_source(
        key=DelphiValidationMixture.DSP_TABLE9_EXPOSURE_ALL_DEFICITS,
        display_name="DSP Table-9 aggregate-exposure all-deficits repair",
        target_metric=TABLE9_TARGET_METRIC,
        method="dsp_table9_exposure_all_deficits_repair",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.DSP_CANON_TABLE9_KL0P2: _dsp_canonical_bowl_source(
        key=DelphiValidationMixture.DSP_CANON_TABLE9_KL0P2,
        display_name="Canonical DSP Table-9 macro KL=0.2",
        target_metric=TABLE9_TARGET_METRIC,
        method="dsp_canonical_table9_kl0p2",
        expected_max_simulated_epoch=11.087,
    ),
    DelphiValidationMixture.DSP_CANON_TABLE9_KL0P5: _dsp_canonical_bowl_source(
        key=DelphiValidationMixture.DSP_CANON_TABLE9_KL0P5,
        display_name="Canonical DSP Table-9 macro KL=0.5",
        target_metric=TABLE9_TARGET_METRIC,
        method="dsp_canonical_table9_kl0p5",
        expected_max_simulated_epoch=2.312,
    ),
    DelphiValidationMixture.DSP_ABOWL_TABLE9_KL0P05: _dsp_canonical_bowl_source(
        key=DelphiValidationMixture.DSP_ABOWL_TABLE9_KL0P05,
        display_name="Asymmetric-bowl DSP Table-9 macro KL=0.05",
        target_metric=TABLE9_TARGET_METRIC,
        method="dsp_asymmetric_bowl_table9_kl0p05",
        expected_max_simulated_epoch=8.164,
    ),
    DelphiValidationMixture.DSP_ABOWL_TABLE9_KL0P1: _dsp_canonical_bowl_source(
        key=DelphiValidationMixture.DSP_ABOWL_TABLE9_KL0P1,
        display_name="Asymmetric-bowl DSP Table-9 macro KL=0.1",
        target_metric=TABLE9_TARGET_METRIC,
        method="dsp_asymmetric_bowl_table9_kl0p1",
        expected_max_simulated_epoch=6.454,
    ),
    DelphiValidationMixture.DSP_ABOWL_TABLE9_KL0P2: _dsp_canonical_bowl_source(
        key=DelphiValidationMixture.DSP_ABOWL_TABLE9_KL0P2,
        display_name="Asymmetric-bowl DSP Table-9 macro KL=0.2",
        target_metric=TABLE9_TARGET_METRIC,
        method="dsp_asymmetric_bowl_table9_kl0p2",
        expected_max_simulated_epoch=4.767,
    ),
    DelphiValidationMixture.DSP_CANON_UNCHEATABLE_KL0P2: _dsp_canonical_bowl_source(
        key=DelphiValidationMixture.DSP_CANON_UNCHEATABLE_KL0P2,
        display_name="Canonical DSP uncheatable BPB KL=0.2",
        target_metric="eval/uncheatable_eval/bpb",
        method="dsp_canonical_uncheatable_kl0p2",
        expected_max_simulated_epoch=4.814,
    ),
    DelphiValidationMixture.DSP_CANON_UNCHEATABLE_KL0P5: _dsp_canonical_bowl_source(
        key=DelphiValidationMixture.DSP_CANON_UNCHEATABLE_KL0P5,
        display_name="Canonical DSP uncheatable BPB KL=0.5",
        target_metric="eval/uncheatable_eval/bpb",
        method="dsp_canonical_uncheatable_kl0p5",
        expected_max_simulated_epoch=3.092,
    ),
    DelphiValidationMixture.DSP_ABOWL_UNCHEATABLE_KL0P05: _dsp_canonical_bowl_source(
        key=DelphiValidationMixture.DSP_ABOWL_UNCHEATABLE_KL0P05,
        display_name="Asymmetric-bowl DSP uncheatable BPB KL=0.05",
        target_metric="eval/uncheatable_eval/bpb",
        method="dsp_asymmetric_bowl_uncheatable_kl0p05",
        expected_max_simulated_epoch=4.592,
    ),
    DelphiValidationMixture.DSP_ABOWL_UNCHEATABLE_KL0P1: _dsp_canonical_bowl_source(
        key=DelphiValidationMixture.DSP_ABOWL_UNCHEATABLE_KL0P1,
        display_name="Asymmetric-bowl DSP uncheatable BPB KL=0.1",
        target_metric="eval/uncheatable_eval/bpb",
        method="dsp_asymmetric_bowl_uncheatable_kl0p1",
        expected_max_simulated_epoch=3.958,
    ),
    DelphiValidationMixture.DSP_ABOWL_UNCHEATABLE_KL0P2: _dsp_canonical_bowl_source(
        key=DelphiValidationMixture.DSP_ABOWL_UNCHEATABLE_KL0P2,
        display_name="Asymmetric-bowl DSP uncheatable BPB KL=0.2",
        target_metric="eval/uncheatable_eval/bpb",
        method="dsp_asymmetric_bowl_uncheatable_kl0p2",
        expected_max_simulated_epoch=3.210,
    ),
    DelphiValidationMixture.DSP_GAMMA_BOWL_TABLE9_G1_KL0P2_TWOPHASE: _dsp_gamma_capped_bowl_source(
        key=DelphiValidationMixture.DSP_GAMMA_BOWL_TABLE9_G1_KL0P2_TWOPHASE,
        display_name="Gamma-capped bowl Table-9 gamma=1 KL=0.2 two-phase",
        target_metric=TABLE9_TARGET_METRIC,
        method="dsp_gamma_capped_bowl_table9_g1_kl0p2_twophase",
        expected_max_simulated_epoch=7.528739,
    ),
    DelphiValidationMixture.DSP_GAMMA_BOWL_TABLE9_G8_KL0P2_TWOPHASE: _dsp_gamma_capped_bowl_source(
        key=DelphiValidationMixture.DSP_GAMMA_BOWL_TABLE9_G8_KL0P2_TWOPHASE,
        display_name="Gamma-capped bowl Table-9 gamma=8 KL=0.2 two-phase",
        target_metric=TABLE9_TARGET_METRIC,
        method="dsp_gamma_capped_bowl_table9_g8_kl0p2_twophase",
        expected_max_simulated_epoch=4.878790,
    ),
    DelphiValidationMixture.DSP_GAMMA_BOWL_TABLE9_G10_KL0P2_TWOPHASE: _dsp_gamma_capped_bowl_source(
        key=DelphiValidationMixture.DSP_GAMMA_BOWL_TABLE9_G10_KL0P2_TWOPHASE,
        display_name="Gamma-capped bowl Table-9 gamma=10 KL=0.2 two-phase",
        target_metric=TABLE9_TARGET_METRIC,
        method="dsp_gamma_capped_bowl_table9_g10_kl0p2_twophase",
        expected_max_simulated_epoch=4.933405,
    ),
    DelphiValidationMixture.DSP_GAMMA_BOWL_TABLE9_G12_KL0P2_TWOPHASE: _dsp_gamma_capped_bowl_source(
        key=DelphiValidationMixture.DSP_GAMMA_BOWL_TABLE9_G12_KL0P2_TWOPHASE,
        display_name="Gamma-capped bowl Table-9 gamma=12 KL=0.2 two-phase",
        target_metric=TABLE9_TARGET_METRIC,
        method="dsp_gamma_capped_bowl_table9_g12_kl0p2_twophase",
        expected_max_simulated_epoch=4.720611,
    ),
    DelphiValidationMixture.DSP_GAMMA_BOWL_TABLE9_G16_KL0P2_TWOPHASE: _dsp_gamma_capped_bowl_source(
        key=DelphiValidationMixture.DSP_GAMMA_BOWL_TABLE9_G16_KL0P2_TWOPHASE,
        display_name="Gamma-capped bowl Table-9 gamma=16 KL=0.2 two-phase",
        target_metric=TABLE9_TARGET_METRIC,
        method="dsp_gamma_capped_bowl_table9_g16_kl0p2_twophase",
        expected_max_simulated_epoch=4.298583,
    ),
    DelphiValidationMixture.DSP_GAMMA_BOWL_TABLE9_G10_KL0P1_TWOPHASE: _dsp_gamma_capped_bowl_source(
        key=DelphiValidationMixture.DSP_GAMMA_BOWL_TABLE9_G10_KL0P1_TWOPHASE,
        display_name="Gamma-capped bowl Table-9 gamma=10 KL=0.1 two-phase",
        target_metric=TABLE9_TARGET_METRIC,
        method="dsp_gamma_capped_bowl_table9_g10_kl0p1_twophase",
        expected_max_simulated_epoch=6.627094,
    ),
    DelphiValidationMixture.DSP_GAMMA_BOWL_TABLE9_G10_KL0P3_TWOPHASE: _dsp_gamma_capped_bowl_source(
        key=DelphiValidationMixture.DSP_GAMMA_BOWL_TABLE9_G10_KL0P3_TWOPHASE,
        display_name="Gamma-capped bowl Table-9 gamma=10 KL=0.3 two-phase",
        target_metric=TABLE9_TARGET_METRIC,
        method="dsp_gamma_capped_bowl_table9_g10_kl0p3_twophase",
        expected_max_simulated_epoch=4.097390,
    ),
    DelphiValidationMixture.DSP_GAMMA_BOWL_TABLE9_G10_KL0P1_ONEPHASE: _dsp_gamma_capped_bowl_source(
        key=DelphiValidationMixture.DSP_GAMMA_BOWL_TABLE9_G10_KL0P1_ONEPHASE,
        display_name="Gamma-capped bowl Table-9 gamma=10 KL=0.1 one-phase",
        target_metric=TABLE9_TARGET_METRIC,
        method="dsp_gamma_capped_bowl_table9_g10_kl0p1_onephase",
        expected_max_simulated_epoch=11.296483,
    ),
    DelphiValidationMixture.DSP_GAMMA_BOWL_TABLE9_G10_KL0P2_ONEPHASE: _dsp_gamma_capped_bowl_source(
        key=DelphiValidationMixture.DSP_GAMMA_BOWL_TABLE9_G10_KL0P2_ONEPHASE,
        display_name="Gamma-capped bowl Table-9 gamma=10 KL=0.2 one-phase",
        target_metric=TABLE9_TARGET_METRIC,
        method="dsp_gamma_capped_bowl_table9_g10_kl0p2_onephase",
        expected_max_simulated_epoch=7.575692,
    ),
    DelphiValidationMixture.DSP_GAMMA_EFFEXP_TABLE9_G10_KL0P2_TWOPHASE: _dsp_gamma_capped_bowl_source(
        key=DelphiValidationMixture.DSP_GAMMA_EFFEXP_TABLE9_G10_KL0P2_TWOPHASE,
        display_name="Gamma-capped effective-exposure Table-9 gamma=10 KL=0.2 two-phase",
        target_metric=TABLE9_TARGET_METRIC,
        method="dsp_gamma_capped_effexp_table9_g10_kl0p2_twophase",
        expected_max_simulated_epoch=8.549688,
    ),
    DelphiValidationMixture.DSP_GAMMA_BOWL_UNCHEATABLE_G10_KL0P2_TWOPHASE: _dsp_gamma_capped_bowl_source(
        key=DelphiValidationMixture.DSP_GAMMA_BOWL_UNCHEATABLE_G10_KL0P2_TWOPHASE,
        display_name="Gamma-capped bowl uncheatable BPB gamma=10 KL=0.2 two-phase",
        target_metric="eval/uncheatable_eval/bpb",
        method="dsp_gamma_capped_bowl_uncheatable_g10_kl0p2_twophase",
        expected_max_simulated_epoch=3.162718,
    ),
    DelphiValidationMixture.DSP_GAMMA_BOWL_UNCHEATABLE_G10_KL0P2_ONEPHASE: _dsp_gamma_capped_bowl_source(
        key=DelphiValidationMixture.DSP_GAMMA_BOWL_UNCHEATABLE_G10_KL0P2_ONEPHASE,
        display_name="Gamma-capped bowl uncheatable BPB gamma=10 KL=0.2 one-phase",
        target_metric="eval/uncheatable_eval/bpb",
        method="dsp_gamma_capped_bowl_uncheatable_g10_kl0p2_onephase",
        expected_max_simulated_epoch=4.956585,
    ),
    DelphiValidationMixture.SUFF_UNCHEAT_FLOOR_A0: _sufficiency_floored_source(
        key=DelphiValidationMixture.SUFF_UNCHEAT_FLOOR_A0,
        display_name="Sufficiency-floored bowl uncheatable alpha=0 (base) two-phase",
        target_metric="eval/uncheatable_eval/bpb",
        method="suff_uncheat_floor_a0",
        expected_max_simulated_epoch=3.162718,
    ),
    DelphiValidationMixture.SUFF_UNCHEAT_FLOOR_A0P7: _sufficiency_floored_source(
        key=DelphiValidationMixture.SUFF_UNCHEAT_FLOOR_A0P7,
        display_name="Sufficiency-floored bowl uncheatable alpha=0.7 two-phase",
        target_metric="eval/uncheatable_eval/bpb",
        method="suff_uncheat_floor_a0p7",
        expected_max_simulated_epoch=3.435798,
    ),
    DelphiValidationMixture.SUFF_UNCHEAT_FLOOR_A1P0: _sufficiency_floored_source(
        key=DelphiValidationMixture.SUFF_UNCHEAT_FLOOR_A1P0,
        display_name="Sufficiency-floored bowl uncheatable alpha=1.0 two-phase",
        target_metric="eval/uncheatable_eval/bpb",
        method="suff_uncheat_floor_a1p0",
        expected_max_simulated_epoch=4.773820,
    ),
    DelphiValidationMixture.SUFF_UNCHEAT_FLOOR_A1P2: _sufficiency_floored_source(
        key=DelphiValidationMixture.SUFF_UNCHEAT_FLOOR_A1P2,
        display_name="Sufficiency-floored bowl uncheatable alpha=1.2 over-repair two-phase",
        target_metric="eval/uncheatable_eval/bpb",
        method="suff_uncheat_floor_a1p2",
        expected_max_simulated_epoch=3.958145,
    ),
    DelphiValidationMixture.SUFF_UNCHEAT_FLOOR_A1P0_G6: _sufficiency_floored_source(
        key=DelphiValidationMixture.SUFF_UNCHEAT_FLOOR_A1P0_G6,
        display_name="Sufficiency-floored bowl uncheatable alpha=1.0 gamma=6 two-phase",
        target_metric="eval/uncheatable_eval/bpb",
        method="suff_uncheat_floor_a1p0_g6",
        expected_max_simulated_epoch=5.045360,
    ),
    DelphiValidationMixture.SUFF_UNCHEAT_FLOOR_A1P0_EFFEXP: _sufficiency_floored_source(
        key=DelphiValidationMixture.SUFF_UNCHEAT_FLOOR_A1P0_EFFEXP,
        display_name="Sufficiency-floored effexp uncheatable alpha=1.0 two-phase",
        target_metric="eval/uncheatable_eval/bpb",
        method="suff_uncheat_floor_a1p0_effexp",
        expected_max_simulated_epoch=6.309046,
    ),
    DelphiValidationMixture.SUFF_UNCHEAT_ONEPHASE: _sufficiency_floored_source(
        key=DelphiValidationMixture.SUFF_UNCHEAT_ONEPHASE,
        display_name="Sufficiency panel bowl uncheatable one-phase control",
        target_metric="eval/uncheatable_eval/bpb",
        method="suff_uncheat_onephase",
        expected_max_simulated_epoch=4.956234,
    ),
    DelphiValidationMixture.SUFF_UNCHEAT_EVALREL: _sufficiency_floored_source(
        key=DelphiValidationMixture.SUFF_UNCHEAT_EVALREL,
        display_name="Eval-relevant late heuristic uncheatable two-phase",
        target_metric="eval/uncheatable_eval/bpb",
        method="suff_uncheat_evalrel",
        expected_max_simulated_epoch=7.210945,
    ),
    DelphiValidationMixture.SUFF_TABLE9_FLOOR_A0: _sufficiency_floored_source(
        key=DelphiValidationMixture.SUFF_TABLE9_FLOOR_A0,
        display_name="Sufficiency-floored bowl Table-9 alpha=0 (base) two-phase",
        target_metric=TABLE9_TARGET_METRIC,
        method="suff_table9_floor_a0",
        expected_max_simulated_epoch=4.878790,
    ),
    DelphiValidationMixture.SUFF_TABLE9_FLOOR_A1P0: _sufficiency_floored_source(
        key=DelphiValidationMixture.SUFF_TABLE9_FLOOR_A1P0,
        display_name="Sufficiency-floored bowl Table-9 alpha=1.0 two-phase",
        target_metric=TABLE9_TARGET_METRIC,
        method="suff_table9_floor_a1p0",
        expected_max_simulated_epoch=7.056531,
    ),
    DelphiValidationMixture.SUFF_TABLE9_ONEPHASE: _sufficiency_floored_source(
        key=DelphiValidationMixture.SUFF_TABLE9_ONEPHASE,
        display_name="Sufficiency panel bowl Table-9 one-phase control",
        target_metric=TABLE9_TARGET_METRIC,
        method="suff_table9_onephase",
        expected_max_simulated_epoch=7.221461,
    ),
    DelphiValidationMixture.WNBR_UNCHEAT_WINNER: _winner_neighborhood_source(
        key=DelphiValidationMixture.WNBR_UNCHEAT_WINNER,
        display_name="Winner-neighborhood uncheatable winner-exact (0.985974 re-anchor)",
        target_metric="eval/uncheatable_eval/bpb",
        method="wnbr_uncheat_winner",
        expected_max_simulated_epoch=8.128129,
    ),
    DelphiValidationMixture.WNBR_UNCHEAT_TILT_K0: _winner_neighborhood_source(
        key=DelphiValidationMixture.WNBR_UNCHEAT_TILT_K0,
        display_name="Winner-neighborhood uncheatable winner-aggregate no-tilt (one-phase)",
        target_metric="eval/uncheatable_eval/bpb",
        method="wnbr_uncheat_tilt_k0",
        expected_max_simulated_epoch=8.128129,
    ),
    DelphiValidationMixture.WNBR_UNCHEAT_TILT_K0P5: _winner_neighborhood_source(
        key=DelphiValidationMixture.WNBR_UNCHEAT_TILT_K0P5,
        display_name="Winner-neighborhood uncheatable less-tilt k=0.5",
        target_metric="eval/uncheatable_eval/bpb",
        method="wnbr_uncheat_tilt_k0p5",
        expected_max_simulated_epoch=8.128129,
    ),
    DelphiValidationMixture.WNBR_UNCHEAT_TILT_K1P5: _winner_neighborhood_source(
        key=DelphiValidationMixture.WNBR_UNCHEAT_TILT_K1P5,
        display_name="Winner-neighborhood uncheatable more-tilt k=1.5",
        target_metric="eval/uncheatable_eval/bpb",
        method="wnbr_uncheat_tilt_k1p5",
        expected_max_simulated_epoch=8.124668,
    ),
    DelphiValidationMixture.WNBR_UNCHEAT_OVERWEIGHT_0P7: _winner_neighborhood_source(
        key=DelphiValidationMixture.WNBR_UNCHEAT_OVERWEIGHT_0P7,
        display_name="Winner-neighborhood uncheatable less-overweight 0.7",
        target_metric="eval/uncheatable_eval/bpb",
        method="wnbr_uncheat_overweight_0p7",
        expected_max_simulated_epoch=7.199548,
    ),
    DelphiValidationMixture.WNBR_UNCHEAT_OVERWEIGHT_1P3: _winner_neighborhood_source(
        key=DelphiValidationMixture.WNBR_UNCHEAT_OVERWEIGHT_1P3,
        display_name="Winner-neighborhood uncheatable more-overweight 1.3",
        target_metric="eval/uncheatable_eval/bpb",
        method="wnbr_uncheat_overweight_1p3",
        expected_max_simulated_epoch=9.056778,
    ),
    DelphiValidationMixture.WNBR_TABLE9_WINNER: _winner_neighborhood_source(
        key=DelphiValidationMixture.WNBR_TABLE9_WINNER,
        display_name="Winner-neighborhood Table-9 winner-exact re-anchor",
        target_metric=TABLE9_TARGET_METRIC,
        method="wnbr_table9_winner",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.WNBR_TABLE9_TILT_K1P5: _winner_neighborhood_source(
        key=DelphiValidationMixture.WNBR_TABLE9_TILT_K1P5,
        display_name="Winner-neighborhood Table-9 more-tilt k=1.5",
        target_metric=TABLE9_TARGET_METRIC,
        method="wnbr_table9_tilt_k1p5",
        expected_max_simulated_epoch=16.541057,
    ),
    DelphiValidationMixture.WNBR_TABLE9_OVERWEIGHT_1P3: _winner_neighborhood_source(
        key=DelphiValidationMixture.WNBR_TABLE9_OVERWEIGHT_1P3,
        display_name="Winner-neighborhood Table-9 more-overweight 1.3",
        target_metric=TABLE9_TARGET_METRIC,
        method="wnbr_table9_overweight_1p3",
        expected_max_simulated_epoch=19.848261,
    ),
    DelphiValidationMixture.AUGP_UNCHEAT_PROFILE: _augmented_profile_source(
        key=DelphiValidationMixture.AUGP_UNCHEAT_PROFILE,
        display_name="Augmented-profile uncheatable two-phase (aug aggregate + winner tilt)",
        target_metric="eval/uncheatable_eval/bpb",
        method="augp_uncheat_profile",
        expected_max_simulated_epoch=8.196884,
    ),
    DelphiValidationMixture.AUGP_UNCHEAT_PROFILE_1PHASE: _augmented_profile_source(
        key=DelphiValidationMixture.AUGP_UNCHEAT_PROFILE_1PHASE,
        display_name="Augmented-profile uncheatable one-phase control",
        target_metric="eval/uncheatable_eval/bpb",
        method="augp_uncheat_profile_1phase",
        expected_max_simulated_epoch=8.045398,
    ),
    DelphiValidationMixture.AUGP_UNCHEAT_BLEND: _augmented_profile_source(
        key=DelphiValidationMixture.AUGP_UNCHEAT_BLEND,
        display_name="Augmented+winner 50/50 blend uncheatable two-phase",
        target_metric="eval/uncheatable_eval/bpb",
        method="augp_uncheat_blend",
        expected_max_simulated_epoch=7.612009,
    ),
    DelphiValidationMixture.AUGP_TABLE9_PROFILE: _augmented_profile_source(
        key=DelphiValidationMixture.AUGP_TABLE9_PROFILE,
        display_name="Augmented-profile Table-9 two-phase",
        target_metric=TABLE9_TARGET_METRIC,
        method="augp_table9_profile",
        expected_max_simulated_epoch=22.929399,
    ),
    DelphiValidationMixture.T9TILT_K0: _table9_controlled_tilt_source(
        key=DelphiValidationMixture.T9TILT_K0,
        display_name="Table-9 controlled-tilt k=0 (one-phase ablation of eff-exp kl0.1)",
        method="t9_tilt_k0_onephase",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.T9TILT_KMOD: _table9_controlled_tilt_source(
        key=DelphiValidationMixture.T9TILT_KMOD,
        display_name="Table-9 controlled-tilt k=0.10 (moderate late tilt)",
        method="t9_tilt_kmod",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.T9TILT_KHIGH: _table9_controlled_tilt_source(
        key=DelphiValidationMixture.T9TILT_KHIGH,
        display_name="Table-9 controlled-tilt k=0.20 (higher late tilt)",
        method="t9_tilt_khigh",
        expected_max_simulated_epoch=15.379907,
    ),
    DelphiValidationMixture.T9TILT_KMAX: _table9_controlled_tilt_source(
        key=DelphiValidationMixture.T9TILT_KMAX,
        display_name="Table-9 controlled-tilt k=0.32 (aggressive late tilt)",
        method="t9_tilt_kmax",
        expected_max_simulated_epoch=12.036290,
    ),
    DelphiValidationMixture.T9ANNEAL_K0: _table9_fresh_anneal_source(
        key=DelphiValidationMixture.T9ANNEAL_K0,
        display_name="Table-9 fresh-anneal k=0 (one-phase ablation)",
        method="t9anneal_k0_onephase",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.T9ANNEAL_KMOD: _table9_fresh_anneal_source(
        key=DelphiValidationMixture.T9ANNEAL_KMOD,
        display_name="Table-9 fresh-HQ anneal k=0.15",
        method="t9anneal_kmod",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.T9ANNEAL_KHIGH: _table9_fresh_anneal_source(
        key=DelphiValidationMixture.T9ANNEAL_KHIGH,
        display_name="Table-9 fresh-HQ anneal k=0.35",
        method="t9anneal_khigh",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.T9ANNEAL_KMAX: _table9_fresh_anneal_source(
        key=DelphiValidationMixture.T9ANNEAL_KMAX,
        display_name="Table-9 fresh-HQ anneal k=0.6",
        method="t9anneal_kmax",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.T9AN2_K0: _table9_fresh_anneal_source(
        key=DelphiValidationMixture.T9AN2_K0,
        display_name="Table-9 anneal-v2 k=0 (one-phase ablation)",
        method="t9an2_k0_onephase",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.T9AN2_K0R: _table9_fresh_anneal_source(
        key=DelphiValidationMixture.T9AN2_K0R,
        display_name="Table-9 anneal-v2 k=0 repeat (noise floor)",
        method="t9an2_k0_repeat",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.T9AN2_E10: _table9_fresh_anneal_source(
        key=DelphiValidationMixture.T9AN2_E10,
        display_name="Table-9 anneal-v2 expanded k=0.10",
        method="t9an2_expanded_k0p10",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.T9AN2_E18: _table9_fresh_anneal_source(
        key=DelphiValidationMixture.T9AN2_E18,
        display_name="Table-9 anneal-v2 expanded k=0.18",
        method="t9an2_expanded_k0p18",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.T9AN2_E28: _table9_fresh_anneal_source(
        key=DelphiValidationMixture.T9AN2_E28,
        display_name="Table-9 anneal-v2 expanded k=0.28",
        method="t9an2_expanded_k0p28",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.T9AN2_N15: _table9_fresh_anneal_source(
        key=DelphiValidationMixture.T9AN2_N15,
        display_name="Table-9 anneal-v2 narrow k=0.15 (v1 repro)",
        method="t9an2_narrow_k0p15",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.T9REP_NARROW_S0: _table9_fresh_anneal_source(
        key=DelphiValidationMixture.T9REP_NARROW_S0,
        display_name="Table-9 anneal repeat narrow s0",
        method="t9rep_narrow_s0",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.T9REP_NARROW_S1: _table9_fresh_anneal_source(
        key=DelphiValidationMixture.T9REP_NARROW_S1,
        display_name="Table-9 anneal repeat narrow s1",
        method="t9rep_narrow_s1",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.T9REP_NARROW_S2: _table9_fresh_anneal_source(
        key=DelphiValidationMixture.T9REP_NARROW_S2,
        display_name="Table-9 anneal repeat narrow s2",
        method="t9rep_narrow_s2",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.T9REP_ONEPH_S0: _table9_fresh_anneal_source(
        key=DelphiValidationMixture.T9REP_ONEPH_S0,
        display_name="Table-9 anneal repeat oneph s0",
        method="t9rep_oneph_s0",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.T9REP_ONEPH_S1: _table9_fresh_anneal_source(
        key=DelphiValidationMixture.T9REP_ONEPH_S1,
        display_name="Table-9 anneal repeat oneph s1",
        method="t9rep_oneph_s1",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.T9REP_ONEPH_S2: _table9_fresh_anneal_source(
        key=DelphiValidationMixture.T9REP_ONEPH_S2,
        display_name="Table-9 anneal repeat oneph s2",
        method="t9rep_oneph_s2",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.T9VR_K0: _table9_fresh_anneal_source(
        key=DelphiValidationMixture.T9VR_K0,
        display_name="Table-9 value-room k=0 (one-phase ablation)",
        method="t9vr_k0",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.T9VR_KMOD: _table9_fresh_anneal_source(
        key=DelphiValidationMixture.T9VR_KMOD,
        display_name="Table-9 value-room tilt k=0.15 (surrogate-derived anneal)",
        method="t9vr_kmod",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.T9VR_KHIGH: _table9_fresh_anneal_source(
        key=DelphiValidationMixture.T9VR_KHIGH,
        display_name="Table-9 value-room tilt k=0.30",
        method="t9vr_khigh",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.T9VRR_KMOD_S0: _table9_fresh_anneal_source(
        key=DelphiValidationMixture.T9VRR_KMOD_S0,
        display_name="Table-9 value-room kmod repeat s0",
        method="t9vrr_kmod_s0",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.T9VRR_KMOD_S1: _table9_fresh_anneal_source(
        key=DelphiValidationMixture.T9VRR_KMOD_S1,
        display_name="Table-9 value-room kmod repeat s1",
        method="t9vrr_kmod_s1",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.T9VRR_KMOD_S2: _table9_fresh_anneal_source(
        key=DelphiValidationMixture.T9VRR_KMOD_S2,
        display_name="Table-9 value-room kmod repeat s2",
        method="t9vrr_kmod_s2",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.T9VRR_KMOD_S3: _table9_fresh_anneal_source(
        key=DelphiValidationMixture.T9VRR_KMOD_S3,
        display_name="Table-9 value-room kmod repeat s3",
        method="t9vrr_kmod_s3",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.T9VRR_K0_S0: _table9_fresh_anneal_source(
        key=DelphiValidationMixture.T9VRR_K0_S0,
        display_name="Table-9 value-room k0 repeat s0",
        method="t9vrr_k0_s0",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.T9VRR_K0_S1: _table9_fresh_anneal_source(
        key=DelphiValidationMixture.T9VRR_K0_S1,
        display_name="Table-9 value-room k0 repeat s1",
        method="t9vrr_k0_s1",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.T9VRR_K0_S2: _table9_fresh_anneal_source(
        key=DelphiValidationMixture.T9VRR_K0_S2,
        display_name="Table-9 value-room k0 repeat s2",
        method="t9vrr_k0_s2",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.T9VRR_K0_S3: _table9_fresh_anneal_source(
        key=DelphiValidationMixture.T9VRR_K0_S3,
        display_name="Table-9 value-room k0 repeat s3",
        method="t9vrr_k0_s3",
        expected_max_simulated_epoch=16.541118,
    ),
    DelphiValidationMixture.PROPORTIONAL_NOISE_3E18_A: _proportional_noise_source(
        key=DelphiValidationMixture.PROPORTIONAL_NOISE_3E18_A,
        label="A",
    ),
    DelphiValidationMixture.PROPORTIONAL_NOISE_3E18_B: _proportional_noise_source(
        key=DelphiValidationMixture.PROPORTIONAL_NOISE_3E18_B,
        label="B",
    ),
    DelphiValidationMixture.PROPORTIONAL_NOISE_3E18_C: _proportional_noise_source(
        key=DelphiValidationMixture.PROPORTIONAL_NOISE_3E18_C,
        label="C",
    ),
    DelphiValidationMixture.PROPORTIONAL_NOISE_3E18_D: _proportional_noise_source(
        key=DelphiValidationMixture.PROPORTIONAL_NOISE_3E18_D,
        label="D",
    ),
    DelphiValidationMixture.PROPORTIONAL_NOISE_3E18_E: _proportional_noise_source(
        key=DelphiValidationMixture.PROPORTIONAL_NOISE_3E18_E,
        label="E",
    ),
    DelphiValidationMixture.PROPORTIONAL_NOISE_3E18_F: _proportional_noise_source(
        key=DelphiValidationMixture.PROPORTIONAL_NOISE_3E18_F,
        label="F",
    ),
    DelphiValidationMixture.PROPORTIONAL_NOISE_3E18_G: _proportional_noise_source(
        key=DelphiValidationMixture.PROPORTIONAL_NOISE_3E18_G,
        label="G",
    ),
    DelphiValidationMixture.PROPORTIONAL_NOISE_3E18_H: _proportional_noise_source(
        key=DelphiValidationMixture.PROPORTIONAL_NOISE_3E18_H,
        label="H",
    ),
    DelphiValidationMixture.PROPORTIONAL_NOISE_3E18_I: _proportional_noise_source(
        key=DelphiValidationMixture.PROPORTIONAL_NOISE_3E18_I,
        label="I",
    ),
    DelphiValidationMixture.PROPORTIONAL_NOISE_3E18_J: _proportional_noise_source(
        key=DelphiValidationMixture.PROPORTIONAL_NOISE_3E18_J,
        label="J",
    ),
    DelphiValidationMixture.SEPLF_UNCH_SEP_KL0: _sep_lf_kl_sweep_source(
        key=DelphiValidationMixture.SEPLF_UNCH_SEP_KL0,
        display_name="separate-heads uncheatable raw optimum (KL=0)",
        method="seplf_unch_sep_kl0",
        target_metric="eval/uncheatable_eval/bpb",
        wandb_series_tag="delphi-sep-lf-kl-sweep-uncheatable",
        expected_max_simulated_epoch=16.722468,
    ),
    DelphiValidationMixture.SEPLF_UNCH_SEP_KL0P1: _sep_lf_kl_sweep_source(
        key=DelphiValidationMixture.SEPLF_UNCH_SEP_KL0P1,
        display_name="separate-heads uncheatable KL=0.1",
        method="seplf_unch_sep_kl0p1",
        target_metric="eval/uncheatable_eval/bpb",
        wandb_series_tag="delphi-sep-lf-kl-sweep-uncheatable",
        expected_max_simulated_epoch=6.070029,
    ),
    DelphiValidationMixture.SEPLF_UNCH_SEP_KL0P2: _sep_lf_kl_sweep_source(
        key=DelphiValidationMixture.SEPLF_UNCH_SEP_KL0P2,
        display_name="separate-heads uncheatable KL=0.2",
        method="seplf_unch_sep_kl0p2",
        target_metric="eval/uncheatable_eval/bpb",
        wandb_series_tag="delphi-sep-lf-kl-sweep-uncheatable",
        expected_max_simulated_epoch=4.610551,
    ),
    DelphiValidationMixture.SEPLF_UNCH_SEP_KL0P3: _sep_lf_kl_sweep_source(
        key=DelphiValidationMixture.SEPLF_UNCH_SEP_KL0P3,
        display_name="separate-heads uncheatable KL=0.3",
        method="seplf_unch_sep_kl0p3",
        target_metric="eval/uncheatable_eval/bpb",
        wandb_series_tag="delphi-sep-lf-kl-sweep-uncheatable",
        expected_max_simulated_epoch=3.872798,
    ),
    DelphiValidationMixture.SEPLF_UNCH_SEP_KL0P4: _sep_lf_kl_sweep_source(
        key=DelphiValidationMixture.SEPLF_UNCH_SEP_KL0P4,
        display_name="separate-heads uncheatable KL=0.4",
        method="seplf_unch_sep_kl0p4",
        target_metric="eval/uncheatable_eval/bpb",
        wandb_series_tag="delphi-sep-lf-kl-sweep-uncheatable",
        expected_max_simulated_epoch=3.408462,
    ),
    DelphiValidationMixture.SEPLF_UNCH_LF2P_KL0P2: _sep_lf_kl_sweep_source(
        key=DelphiValidationMixture.SEPLF_UNCH_LF2P_KL0P2,
        display_name="LEARN-FORGET uncheatable KL=0.2 (confirming ~eff-exp)",
        method="seplf_unch_lf2p_kl0p2",
        target_metric="eval/uncheatable_eval/bpb",
        wandb_series_tag="delphi-sep-lf-kl-sweep-uncheatable",
        expected_max_simulated_epoch=3.300175,
    ),
    DelphiValidationMixture.SEPLF_T9_SEP_KL0: _sep_lf_kl_sweep_source(
        key=DelphiValidationMixture.SEPLF_T9_SEP_KL0,
        display_name="separate-heads Table-9 raw optimum (KL=0)",
        method="seplf_t9_sep_kl0",
        target_metric=TABLE9_TARGET_METRIC,
        wandb_series_tag="delphi-sep-lf-kl-sweep-table9",
        expected_max_simulated_epoch=14.970762,
    ),
    DelphiValidationMixture.SEPLF_T9_SEP_KL0P1: _sep_lf_kl_sweep_source(
        key=DelphiValidationMixture.SEPLF_T9_SEP_KL0P1,
        display_name="separate-heads Table-9 KL=0.1",
        method="seplf_t9_sep_kl0p1",
        target_metric=TABLE9_TARGET_METRIC,
        wandb_series_tag="delphi-sep-lf-kl-sweep-table9",
        expected_max_simulated_epoch=6.448497,
    ),
    DelphiValidationMixture.SEPLF_T9_SEP_KL0P2: _sep_lf_kl_sweep_source(
        key=DelphiValidationMixture.SEPLF_T9_SEP_KL0P2,
        display_name="separate-heads Table-9 KL=0.2",
        method="seplf_t9_sep_kl0p2",
        target_metric=TABLE9_TARGET_METRIC,
        wandb_series_tag="delphi-sep-lf-kl-sweep-table9",
        expected_max_simulated_epoch=5.060286,
    ),
    DelphiValidationMixture.SEPLF_T9_SEP_KL0P3: _sep_lf_kl_sweep_source(
        key=DelphiValidationMixture.SEPLF_T9_SEP_KL0P3,
        display_name="separate-heads Table-9 KL=0.3",
        method="seplf_t9_sep_kl0p3",
        target_metric=TABLE9_TARGET_METRIC,
        wandb_series_tag="delphi-sep-lf-kl-sweep-table9",
        expected_max_simulated_epoch=4.328800,
    ),
    DelphiValidationMixture.SEPLF_T9_SEP_KL0P4: _sep_lf_kl_sweep_source(
        key=DelphiValidationMixture.SEPLF_T9_SEP_KL0P4,
        display_name="separate-heads Table-9 KL=0.4",
        method="seplf_t9_sep_kl0p4",
        target_metric=TABLE9_TARGET_METRIC,
        wandb_series_tag="delphi-sep-lf-kl-sweep-table9",
        expected_max_simulated_epoch=3.853026,
    ),
    DelphiValidationMixture.SEPLF_T9_LF2P_KL0P2: _sep_lf_kl_sweep_source(
        key=DelphiValidationMixture.SEPLF_T9_LF2P_KL0P2,
        display_name="LEARN-FORGET Table-9 KL=0.2 (confirming ~eff-exp)",
        method="seplf_t9_lf2p_kl0p2",
        target_metric=TABLE9_TARGET_METRIC,
        wandb_series_tag="delphi-sep-lf-kl-sweep-table9",
        expected_max_simulated_epoch=4.648602,
    ),
    DelphiValidationMixture.BESTPHASE_UNCH_2P_KL5: _best_phase_model_source(
        key=DelphiValidationMixture.BESTPHASE_UNCH_2P_KL5,
        display_name="Best phase model Uncheatable two-phase KL=5",
        target_metric="eval/uncheatable_eval/bpb",
        expected_max_simulated_epoch=1.164605,
    ),
    DelphiValidationMixture.BESTPHASE_UNCH_TIED_KL5: _best_phase_model_source(
        key=DelphiValidationMixture.BESTPHASE_UNCH_TIED_KL5,
        display_name="Best phase model Uncheatable aggregate-matched tied KL=5",
        target_metric="eval/uncheatable_eval/bpb",
        expected_max_simulated_epoch=1.164605,
    ),
    DelphiValidationMixture.BESTPHASE_UNCH_1P_KL5: _best_phase_model_source(
        key=DelphiValidationMixture.BESTPHASE_UNCH_1P_KL5,
        display_name="Best phase model Uncheatable optimized one-phase KL=5",
        target_metric="eval/uncheatable_eval/bpb",
        expected_max_simulated_epoch=1.143068,
    ),
    DelphiValidationMixture.BESTPHASE_UNCH_2P_KL10: _best_phase_model_source(
        key=DelphiValidationMixture.BESTPHASE_UNCH_2P_KL10,
        display_name="Best phase model Uncheatable two-phase KL=10",
        target_metric="eval/uncheatable_eval/bpb",
        expected_max_simulated_epoch=1.026664,
    ),
    DelphiValidationMixture.BESTPHASE_UNCH_TIED_KL10: _best_phase_model_source(
        key=DelphiValidationMixture.BESTPHASE_UNCH_TIED_KL10,
        display_name="Best phase model Uncheatable aggregate-matched tied KL=10",
        target_metric="eval/uncheatable_eval/bpb",
        expected_max_simulated_epoch=1.026664,
    ),
    DelphiValidationMixture.BESTPHASE_UNCH_1P_KL10: _best_phase_model_source(
        key=DelphiValidationMixture.BESTPHASE_UNCH_1P_KL10,
        display_name="Best phase model Uncheatable optimized one-phase KL=10",
        target_metric="eval/uncheatable_eval/bpb",
        expected_max_simulated_epoch=1.020232,
    ),
    DelphiValidationMixture.BESTPHASE_T9_2P_KL5: _best_phase_model_source(
        key=DelphiValidationMixture.BESTPHASE_T9_2P_KL5,
        display_name="Best phase model Table-9 two-phase KL=5",
        target_metric=TABLE9_TARGET_METRIC,
        expected_max_simulated_epoch=1.454163,
    ),
    DelphiValidationMixture.BESTPHASE_T9_TIED_KL5: _best_phase_model_source(
        key=DelphiValidationMixture.BESTPHASE_T9_TIED_KL5,
        display_name="Best phase model Table-9 aggregate-matched tied KL=5",
        target_metric=TABLE9_TARGET_METRIC,
        expected_max_simulated_epoch=1.454163,
    ),
    DelphiValidationMixture.BESTPHASE_T9_1P_KL5: _best_phase_model_source(
        key=DelphiValidationMixture.BESTPHASE_T9_1P_KL5,
        display_name="Best phase model Table-9 optimized one-phase KL=5",
        target_metric=TABLE9_TARGET_METRIC,
        expected_max_simulated_epoch=1.355453,
    ),
    DelphiValidationMixture.BESTPHASE_T9_2P_KL10: _best_phase_model_source(
        key=DelphiValidationMixture.BESTPHASE_T9_2P_KL10,
        display_name="Best phase model Table-9 two-phase KL=10",
        target_metric=TABLE9_TARGET_METRIC,
        expected_max_simulated_epoch=1.144575,
    ),
    DelphiValidationMixture.BESTPHASE_T9_TIED_KL10: _best_phase_model_source(
        key=DelphiValidationMixture.BESTPHASE_T9_TIED_KL10,
        display_name="Best phase model Table-9 aggregate-matched tied KL=10",
        target_metric=TABLE9_TARGET_METRIC,
        expected_max_simulated_epoch=1.144575,
    ),
    DelphiValidationMixture.BESTPHASE_T9_1P_KL10: _best_phase_model_source(
        key=DelphiValidationMixture.BESTPHASE_T9_1P_KL10,
        display_name="Best phase model Table-9 optimized one-phase KL=10",
        target_metric=TABLE9_TARGET_METRIC,
        expected_max_simulated_epoch=1.116847,
    ),
    DelphiValidationMixture.SEPFRONT_UNCH_2P_S0: _sep_frontier_tied_source(
        key=DelphiValidationMixture.SEPFRONT_UNCH_2P_S0,
        objective="uncheatable",
        policy="2p",
        repeat=0,
        data_seed=680000,
        expected_max_simulated_epoch=6.070029,
    ),
    DelphiValidationMixture.SEPFRONT_UNCH_TIED_S0: _sep_frontier_tied_source(
        key=DelphiValidationMixture.SEPFRONT_UNCH_TIED_S0,
        objective="uncheatable",
        policy="tied",
        repeat=0,
        data_seed=680000,
        expected_max_simulated_epoch=6.070029,
    ),
    DelphiValidationMixture.SEPFRONT_UNCH_2P_S1: _sep_frontier_tied_source(
        key=DelphiValidationMixture.SEPFRONT_UNCH_2P_S1,
        objective="uncheatable",
        policy="2p",
        repeat=1,
        data_seed=680001,
        expected_max_simulated_epoch=6.070029,
    ),
    DelphiValidationMixture.SEPFRONT_UNCH_TIED_S1: _sep_frontier_tied_source(
        key=DelphiValidationMixture.SEPFRONT_UNCH_TIED_S1,
        objective="uncheatable",
        policy="tied",
        repeat=1,
        data_seed=680001,
        expected_max_simulated_epoch=6.070029,
    ),
    DelphiValidationMixture.SEPFRONT_UNCH_2P_S2: _sep_frontier_tied_source(
        key=DelphiValidationMixture.SEPFRONT_UNCH_2P_S2,
        objective="uncheatable",
        policy="2p",
        repeat=2,
        data_seed=680002,
        expected_max_simulated_epoch=6.070029,
    ),
    DelphiValidationMixture.SEPFRONT_UNCH_TIED_S2: _sep_frontier_tied_source(
        key=DelphiValidationMixture.SEPFRONT_UNCH_TIED_S2,
        objective="uncheatable",
        policy="tied",
        repeat=2,
        data_seed=680002,
        expected_max_simulated_epoch=6.070029,
    ),
    DelphiValidationMixture.SEPFRONT_T9_2P_S0: _sep_frontier_tied_source(
        key=DelphiValidationMixture.SEPFRONT_T9_2P_S0,
        objective="table9",
        policy="2p",
        repeat=0,
        data_seed=680100,
        expected_max_simulated_epoch=6.448497,
    ),
    DelphiValidationMixture.SEPFRONT_T9_TIED_S0: _sep_frontier_tied_source(
        key=DelphiValidationMixture.SEPFRONT_T9_TIED_S0,
        objective="table9",
        policy="tied",
        repeat=0,
        data_seed=680100,
        expected_max_simulated_epoch=6.448497,
    ),
    DelphiValidationMixture.SEPFRONT_T9_2P_S1: _sep_frontier_tied_source(
        key=DelphiValidationMixture.SEPFRONT_T9_2P_S1,
        objective="table9",
        policy="2p",
        repeat=1,
        data_seed=680101,
        expected_max_simulated_epoch=6.448497,
    ),
    DelphiValidationMixture.SEPFRONT_T9_TIED_S1: _sep_frontier_tied_source(
        key=DelphiValidationMixture.SEPFRONT_T9_TIED_S1,
        objective="table9",
        policy="tied",
        repeat=1,
        data_seed=680101,
        expected_max_simulated_epoch=6.448497,
    ),
    DelphiValidationMixture.SEPFRONT_T9_2P_S2: _sep_frontier_tied_source(
        key=DelphiValidationMixture.SEPFRONT_T9_2P_S2,
        objective="table9",
        policy="2p",
        repeat=2,
        data_seed=680102,
        expected_max_simulated_epoch=6.448497,
    ),
    DelphiValidationMixture.SEPFRONT_T9_TIED_S2: _sep_frontier_tied_source(
        key=DelphiValidationMixture.SEPFRONT_T9_TIED_S2,
        objective="table9",
        policy="tied",
        repeat=2,
        data_seed=680102,
        expected_max_simulated_epoch=6.448497,
    ),
    DelphiValidationMixture.CENTREC_SEP_UNCH_OKL1: _centered_recency_reorder_source(
        key=DelphiValidationMixture.CENTREC_SEP_UNCH_OKL1,
        objective="uncheatable",
        order_kl=1.0,
        data_seed=680000,
        expected_max_simulated_epoch=6.070029,
    ),
    DelphiValidationMixture.CENTREC_SEP_UNCH_OKL3: _centered_recency_reorder_source(
        key=DelphiValidationMixture.CENTREC_SEP_UNCH_OKL3,
        objective="uncheatable",
        order_kl=3.0,
        data_seed=680000,
        expected_max_simulated_epoch=6.070029,
    ),
    DelphiValidationMixture.CENTREC_SEP_T9_OKL1: _centered_recency_reorder_source(
        key=DelphiValidationMixture.CENTREC_SEP_T9_OKL1,
        objective="table9",
        order_kl=1.0,
        data_seed=680100,
        expected_max_simulated_epoch=6.448497,
    ),
    DelphiValidationMixture.CENTREC_SEP_T9_OKL3: _centered_recency_reorder_source(
        key=DelphiValidationMixture.CENTREC_SEP_T9_OKL3,
        objective="table9",
        order_kl=3.0,
        data_seed=680100,
        expected_max_simulated_epoch=6.448497,
    ),
    DelphiValidationMixture.GENPOW_SEP_UNCH_OKL0P3: _generalized_power_reorder_source(
        key=DelphiValidationMixture.GENPOW_SEP_UNCH_OKL0P3,
        objective="uncheatable",
        order_kl=0.3,
        data_seed=680000,
        expected_max_simulated_epoch=6.070029,
    ),
    DelphiValidationMixture.GENPOW_SEP_UNCH_OKL1: _generalized_power_reorder_source(
        key=DelphiValidationMixture.GENPOW_SEP_UNCH_OKL1,
        objective="uncheatable",
        order_kl=1.0,
        data_seed=680000,
        expected_max_simulated_epoch=6.070029,
    ),
    DelphiValidationMixture.GENPOW_SEP_T9_OKL0P3: _generalized_power_reorder_source(
        key=DelphiValidationMixture.GENPOW_SEP_T9_OKL0P3,
        objective="table9",
        order_kl=0.3,
        data_seed=680100,
        expected_max_simulated_epoch=6.448497,
    ),
    DelphiValidationMixture.GENPOW_SEP_T9_OKL1: _generalized_power_reorder_source(
        key=DelphiValidationMixture.GENPOW_SEP_T9_OKL1,
        objective="table9",
        order_kl=1.0,
        data_seed=680100,
        expected_max_simulated_epoch=6.448497,
    ),
}

_SYMMETRIC_FRONTIER_SPECS = {
    DelphiValidationMixture.SYMSEP_UNCH_1P_KL0P05: ("separate-heads", "uncheatable", "1p", 0.05, 9.380395),
    DelphiValidationMixture.SYMSEP_UNCH_1P_KL0P1: ("separate-heads", "uncheatable", "1p", 0.1, 7.220985),
    DelphiValidationMixture.SYMSEP_UNCH_1P_KL0P2: ("separate-heads", "uncheatable", "1p", 0.2, 5.302521),
    DelphiValidationMixture.SYMSEP_UNCH_2P_KL0P05: ("separate-heads", "uncheatable", "2p", 0.05, 5.666745),
    DelphiValidationMixture.SYMSEP_UNCH_2P_KL0P1: ("separate-heads", "uncheatable", "2p", 0.1, 5.086449),
    DelphiValidationMixture.SYMSEP_UNCH_2P_KL0P2: ("separate-heads", "uncheatable", "2p", 0.2, 4.331176),
    DelphiValidationMixture.SYMSEP_T9_1P_KL0P05: ("separate-heads", "table9", "1p", 0.05, 17.540714),
    DelphiValidationMixture.SYMSEP_T9_1P_KL0P1: ("separate-heads", "table9", "1p", 0.1, 13.352402),
    DelphiValidationMixture.SYMSEP_T9_1P_KL0P2: ("separate-heads", "table9", "1p", 0.2, 6.648886),
    DelphiValidationMixture.SYMSEP_T9_2P_KL0P05: ("separate-heads", "table9", "2p", 0.05, 9.311626),
    DelphiValidationMixture.SYMSEP_T9_2P_KL0P1: ("separate-heads", "table9", "2p", 0.1, 6.435724),
    DelphiValidationMixture.SYMSEP_T9_2P_KL0P2: ("separate-heads", "table9", "2p", 0.2, 4.308822),
    DelphiValidationMixture.GEOMFRONT_UNCH_1P_KL0P2: ("eff-exp+geometry", "uncheatable", "1p", 0.2, 7.315369),
    DelphiValidationMixture.GEOMFRONT_UNCH_1P_KL0P3: ("eff-exp+geometry", "uncheatable", "1p", 0.3, 5.380262),
    DelphiValidationMixture.GEOMFRONT_UNCH_1P_KL0P5: ("eff-exp+geometry", "uncheatable", "1p", 0.5, 3.688639),
    DelphiValidationMixture.GEOMFRONT_UNCH_2P_KL0P2: ("eff-exp+geometry", "uncheatable", "2p", 0.2, 4.296505),
    DelphiValidationMixture.GEOMFRONT_UNCH_2P_KL0P3: ("eff-exp+geometry", "uncheatable", "2p", 0.3, 3.876880),
    DelphiValidationMixture.GEOMFRONT_UNCH_2P_KL0P5: ("eff-exp+geometry", "uncheatable", "2p", 0.5, 3.183509),
    DelphiValidationMixture.GEOMFRONT_UNCH_TIED_KL0P2: ("eff-exp+geometry", "uncheatable", "tied", 0.2, 4.296505),
    DelphiValidationMixture.GEOMFRONT_UNCH_TIED_KL0P3: ("eff-exp+geometry", "uncheatable", "tied", 0.3, 3.876880),
    DelphiValidationMixture.GEOMFRONT_UNCH_TIED_KL0P5: ("eff-exp+geometry", "uncheatable", "tied", 0.5, 3.183509),
    DelphiValidationMixture.GEOMFRONT_T9_1P_KL0P15: ("eff-exp+geometry", "table9", "1p", 0.15, 13.000752),
    DelphiValidationMixture.GEOMFRONT_T9_1P_KL0P2: ("eff-exp+geometry", "table9", "1p", 0.2, 11.703045),
    DelphiValidationMixture.GEOMFRONT_T9_1P_KL0P3: ("eff-exp+geometry", "table9", "1p", 0.3, 9.634832),
    DelphiValidationMixture.GEOMFRONT_T9_2P_KL0P15: ("eff-exp+geometry", "table9", "2p", 0.15, 5.746636),
    DelphiValidationMixture.GEOMFRONT_T9_2P_KL0P2: ("eff-exp+geometry", "table9", "2p", 0.2, 5.465891),
    DelphiValidationMixture.GEOMFRONT_T9_2P_KL0P3: ("eff-exp+geometry", "table9", "2p", 0.3, 4.928776),
    DelphiValidationMixture.GEOMFRONT_T9_TIED_KL0P15: ("eff-exp+geometry", "table9", "tied", 0.15, 5.746636),
    DelphiValidationMixture.GEOMFRONT_T9_TIED_KL0P2: ("eff-exp+geometry", "table9", "tied", 0.2, 5.465891),
    DelphiValidationMixture.GEOMFRONT_T9_TIED_KL0P3: ("eff-exp+geometry", "table9", "tied", 0.3, 4.928776),
}
MIXTURE_SOURCES.update(
    {
        key: _symmetric_frontier_source(
            key=key,
            family=family,
            objective=objective,
            policy=policy,
            kl_reg=kl_reg,
            expected_max_simulated_epoch=expected_max_simulated_epoch,
        )
        for key, (family, objective, policy, kl_reg, expected_max_simulated_epoch) in _SYMMETRIC_FRONTIER_SPECS.items()
    }
)

_ORIGINAL_STYLE_MATCHED_SEPHEADS_SPECS = {
    DelphiValidationMixture.ORIGSTYLE_SEP_UNCH_1P_KL0P05: ("uncheatable", "1p", 0.05, 12.918367),
    DelphiValidationMixture.ORIGSTYLE_SEP_UNCH_1P_KL0P075: ("uncheatable", "1p", 0.075, 11.434867),
    DelphiValidationMixture.ORIGSTYLE_SEP_UNCH_1P_KL0P1: ("uncheatable", "1p", 0.1, 10.369754),
    DelphiValidationMixture.ORIGSTYLE_SEP_UNCH_1P_KL0P15: ("uncheatable", "1p", 0.15, 8.902462),
    DelphiValidationMixture.ORIGSTYLE_SEP_UNCH_1P_KL0P2: ("uncheatable", "1p", 0.2, 7.915496),
    DelphiValidationMixture.ORIGSTYLE_SEP_UNCH_1P_KL0P3: ("uncheatable", "1p", 0.3, 6.627064),
    DelphiValidationMixture.ORIGSTYLE_SEP_UNCH_2P_KL0P05: ("uncheatable", "2p", 0.05, 7.742066),
    DelphiValidationMixture.ORIGSTYLE_SEP_UNCH_2P_KL0P075: ("uncheatable", "2p", 0.075, 6.722458),
    DelphiValidationMixture.ORIGSTYLE_SEP_UNCH_2P_KL0P1: ("uncheatable", "2p", 0.1, 6.043568),
    DelphiValidationMixture.ORIGSTYLE_SEP_UNCH_2P_KL0P15: ("uncheatable", "2p", 0.15, 5.163211),
    DelphiValidationMixture.ORIGSTYLE_SEP_UNCH_2P_KL0P2: ("uncheatable", "2p", 0.2, 4.584567),
    DelphiValidationMixture.ORIGSTYLE_SEP_UNCH_2P_KL0P3: ("uncheatable", "2p", 0.3, 3.844410),
    DelphiValidationMixture.ORIGSTYLE_SEP_T9_1P_KL0P05: ("table9", "1p", 0.05, 13.324778),
    DelphiValidationMixture.ORIGSTYLE_SEP_T9_1P_KL0P075: ("table9", "1p", 0.075, 11.894401),
    DelphiValidationMixture.ORIGSTYLE_SEP_T9_1P_KL0P1: ("table9", "1p", 0.1, 10.834981),
    DelphiValidationMixture.ORIGSTYLE_SEP_T9_1P_KL0P15: ("table9", "1p", 0.15, 9.368723),
    DelphiValidationMixture.ORIGSTYLE_SEP_T9_1P_KL0P2: ("table9", "1p", 0.2, 8.362538),
    DelphiValidationMixture.ORIGSTYLE_SEP_T9_1P_KL0P3: ("table9", "1p", 0.3, 7.042844),
    DelphiValidationMixture.ORIGSTYLE_SEP_T9_2P_KL0P05: ("table9", "2p", 0.05, 7.986666),
    DelphiValidationMixture.ORIGSTYLE_SEP_T9_2P_KL0P075: ("table9", "2p", 0.075, 7.087201),
    DelphiValidationMixture.ORIGSTYLE_SEP_T9_2P_KL0P1: ("table9", "2p", 0.1, 6.459543),
    DelphiValidationMixture.ORIGSTYLE_SEP_T9_2P_KL0P15: ("table9", "2p", 0.15, 5.637771),
    DelphiValidationMixture.ORIGSTYLE_SEP_T9_2P_KL0P2: ("table9", "2p", 0.2, 5.088830),
    DelphiValidationMixture.ORIGSTYLE_SEP_T9_2P_KL0P3: ("table9", "2p", 0.3, 4.352144),
}
MIXTURE_SOURCES.update(
    {
        key: _original_style_matched_sepheads_source(
            key=key,
            objective=objective,
            policy=policy,
            kl_reg=kl_reg,
            expected_max_simulated_epoch=expected_max_simulated_epoch,
        )
        for key, (
            objective,
            policy,
            kl_reg,
            expected_max_simulated_epoch,
        ) in _ORIGINAL_STYLE_MATCHED_SEPHEADS_SPECS.items()
    }
)


_EMBEDDED_MIXTURE_WEIGHT_CSVS: dict[DelphiValidationMixture, str] = {
    DelphiValidationMixture.OLMIX_D001_KL005_CAP4: (
        """domain,phase_0_weight,phase_1_weight,simulated_epochs
dolma3_arxiv,0.007877294593986481,0.03635606452360494,3.0403477019912675
dolma3_cc/art_and_design_high,0.019545498593021696,8.909429720890187e-05,0.8672650871365877
dolma3_cc/art_and_design_low,0.010559552793511558,1.4178515221360545e-05,1.1534396594560405
dolma3_cc/crime_and_law_high,0.0350226967748449,2.3302335359916474e-05,1.1534070479003278
dolma3_cc/crime_and_law_low,0.004444682047110862,0.00018463120527987726,0.3291871115125159
dolma3_cc/education_and_jobs_high,0.02769255808963606,5.397760991762897e-05,0.55797264751664
dolma3_cc/education_and_jobs_low,0.013209173864450583,7.917510118851106e-08,0.543094141834381
dolma3_cc/electronics_and_hardware_high,0.005247668930145996,4.396399977411061e-07,0.20030218024588944
dolma3_cc/electronics_and_hardware_low,0.010727686200166602,6.59459897955764e-06,0.8712681324978486
dolma3_cc/entertainment_high,0.07726800801307862,0.00028258350863736084,0.9242637707306016
dolma3_cc/entertainment_low,0.026996886657516674,2.367135322374188e-06,0.886595644868703
dolma3_cc/finance_and_business_high,0.07760905940328383,0.00018272102282068198,0.7223722035529303
dolma3_cc/finance_and_business_low,0.04349776302900368,2.6405194886169533e-06,0.8492025306231142
dolma3_cc/food_and_dining_high,0.03299884658156384,1.0566800465957603e-05,0.977481643559478
dolma3_cc/food_and_dining_low,0.007385770274324416,5.605031620699921e-07,0.4444490351412025
dolma3_cc/games_high,0.05614035870985765,2.7289288307112236e-05,1.257456981766967
dolma3_cc/games_low,0.009016072631240187,4.82044607840731e-07,0.5385800080589155
dolma3_cc/health_high,0.0481271848004615,1.3943237532411648e-05,0.5592788323776305
dolma3_cc/health_low,0.024205053021422954,7.316178873678304e-07,0.6599353173096475
dolma3_cc/history_and_geography_high,0.03309734432407341,0.00010039062286777329,1.2404876770179627
dolma3_cc/history_and_geography_low,0.006717721213536731,1.3899621783751634e-06,0.9142665882938764
dolma3_cc/industrial_high,0.014182816573017766,8.130294143831049e-06,0.8788443496954568
dolma3_cc/industrial_low,0.005080310894108908,2.0871428104730416e-06,0.5945394055875848
dolma3_cc/literature_high,0.015992248004829252,0.003890390398153202,0.33229225875028834
dolma3_cc/literature_low,0.01573398897331479,1.2379914229727515e-05,1.1560714601742572
dolma3_cc/science_math_and_technology_high,0.04563389575405995,0.00016872404620079144,0.9509964956876682
dolma3_cc/science_math_and_technology_low,0.036095959019950066,0.005142212704291994,1.704982311651281
dolma3_finemath_3plus,0.007230096389844306,1.4656763005626623e-05,1.076526296404441
dolma3_stack_edu,3.3034799816058055e-06,0.4239145709133125,4.000000010484327
dolma3_wikipedia,8.902715629914813e-05,1.1324309380465921e-09,0.12277867212100839
dolmino_common_crawl_hq,0.13784731244802997,0.00018506099690930963,0.529575772386458
dolmino_olmocr_pdfs_hq,0.04567876841609875,0.020512910853750344,1.2486055698761394
dolmino_stack_edu_fim,4.6230379254813127e-07,0.42333152777432126,4.000000013287558
dolmino_stem_heavy_crawl,0.00028873226847738267,1.4690930209578784e-09,0.2802260657207796
dolmino_synth_code,8.063119726239299e-05,0.0593146659964898,4.000000079432466
dolmino_synth_instruction,0.001855576000032917,2.0187990837630566e-05,0.5236260931662129
dolmino_synth_math,0.005530836836822305,0.0001429933298053315,1.2908608741494498
dolmino_synth_qa,0.08486734804427445,0.025985264500008917,0.8770001537214391
dolmino_synth_thinking,0.006421805693565501,2.0561625634680298e-07,0.8144671989820418
"""
    ),
    DelphiValidationMixture.DSP_EFFECTIVE_EXPOSURE_KL01: (
        """domain,phase_0_weight,phase_1_weight,simulated_epochs
dolma3_arxiv,0.0051155766373394505,0.01854371428944535,1.7474610088202231
dolma3_cc/art_and_design_high,0.013744238117693347,0.00025625142231538924,0.6119990177926778
dolma3_cc/art_and_design_low,0.00454817730617484,4.991236437238256e-05,0.49800173168972456
dolma3_cc/crime_and_law_high,0.022760137836410187,0.006323880119517936,0.8014955663615003
dolma3_cc/crime_and_law_low,0.01087821519164199,0.0067580402079638275,0.9212381894847661
dolma3_cc/education_and_jobs_high,0.03276850691384163,0.0016118800445175207,0.6680409201971444
dolma3_cc/education_and_jobs_low,0.014117712875959748,0.00019995379739571283,0.5825030499301324
dolma3_cc/electronics_and_hardware_high,0.020189632879773614,0.008137137830140672,0.8482634714354707
dolma3_cc/electronics_and_hardware_low,0.009987453748804097,0.0077078511292916685,0.9675018615867099
dolma3_cc/entertainment_high,0.06410116857390159,0.0235099969500146,0.8363055746207865
dolma3_cc/entertainment_low,0.021140174496557335,0.0024478805880340723,0.7143391835440304
dolma3_cc/finance_and_business_high,0.07497932883154632,0.008372862660610355,0.7169564156161192
dolma3_cc/finance_and_business_low,0.02164244042255889,3.7932770469705914e-05,0.42270195082035433
dolma3_cc/food_and_dining_high,0.02412643618074609,0.0036172256110543442,0.7413935756838379
dolma3_cc/food_and_dining_low,0.007588355223559377,4.149816510193049e-05,0.4572554944921937
dolma3_cc/games_high,0.028771343508736377,0.0013595628829366452,0.6519672770193224
dolma3_cc/games_low,0.011070316183926452,0.0007680319492347343,0.6727520747516524
dolma3_cc/health_high,0.04909793035645304,0.00043765841949835045,0.5717897981821503
dolma3_cc/health_low,0.026868032358364803,0.005635900866222897,0.7709487328962129
dolma3_cc/history_and_geography_high,0.021402040121816582,0.014236627225983303,0.9348361818275015
dolma3_cc/history_and_geography_low,0.005774037395029583,0.0027737987589755405,0.8801647634592228
dolma3_cc/industrial_high,0.013168842771057962,0.010993801738491061,0.9861806108566931
dolma3_cc/industrial_low,0.005126705545786675,0.00014245129173535816,0.6040745460066631
dolma3_cc/literature_high,0.04415072820495569,0.07522076646892732,1.2331243848179203
dolma3_cc/literature_low,0.01092155538625683,0.007224637671886842,0.9349983696728944
dolma3_cc/science_math_and_technology_high,0.041088297898680966,0.061000887642721775,1.1729933633720497
dolma3_cc/science_math_and_technology_low,0.01892633585969825,0.0317663475868531,1.2254542515636415
dolma3_finemath_3plus,0.005577482100055393,0.005452500762526859,1.0328993029550533
dolma3_stack_edu,0.025719911147963983,0.18843849413284333,2.748752498993329
dolma3_wikipedia,0.0006723503175560467,0.006323235595895515,3.107355423516416
dolmino_common_crawl_hq,0.1819513856350587,0.020889615818311297,0.7188348133954006
dolmino_olmocr_pdfs_hq,0.03947889697723864,0.2883745619268828,2.741947021652079
dolmino_stack_edu_fim,0.024221203355460918,0.08989333414011078,1.7648327142137648
dolmino_stem_heavy_crawl,0.00085811346240548,0.0017320372360831956,1.253084029308579
dolmino_synth_code,0.0037594709533817546,0.03947676543992924,3.656419735265803
dolmino_synth_instruction,0.003214634853798436,0.009694668245809166,1.5867599226477924
dolmino_synth_math,0.0039365087266317825,0.012379813853897277,1.630557900985739
dolmino_synth_qa,0.08130647543191534,0.03775423887508535,0.8710619314606202
dolmino_synth_thinking,0.005249846211261922,0.0004142435189125362,0.6789585025208178
"""
    ),
    DelphiValidationMixture.DSP_CANONICAL_KL01: (
        """domain,phase_0_weight,phase_1_weight,simulated_epochs
dolma3_arxiv,0.01201784024613365,0.014864854312305095,2.8195283679009804
dolma3_cc/art_and_design_high,0.03805705444901422,0.0004260914111440201,1.6914515311742888
dolma3_cc/art_and_design_low,0.002244637421547694,0.00012816217473236017,0.24860233243317767
dolma3_cc/crime_and_law_high,0.03387406944216076,0.00024217597476987662,1.117387193895774
dolma3_cc/crime_and_law_low,0.014848829196421486,0.015138242910614694,1.3658629596724516
dolma3_cc/education_and_jobs_high,0.00797480523038342,0.0010880972934410068,0.1660830362257602
dolma3_cc/education_and_jobs_low,0.00435527299378773,0.0001277902057498704,0.1803799564679302
dolma3_cc/electronics_and_hardware_high,0.007653558286399227,0.00013397519207277253,0.2934066657675398
dolma3_cc/electronics_and_hardware_low,0.021099445985763323,0.000321684805312015,1.7198963019410052
dolma3_cc/entertainment_high,0.05441391150355928,0.0012331296643950775,0.6539775228906396
dolma3_cc/entertainment_low,0.028235377326223145,0.0001572954003765224,0.9285395662912843
dolma3_cc/finance_and_business_high,0.06507237295082377,0.0023311505719335194,0.6107478209448866
dolma3_cc/finance_and_business_low,0.027005563667412944,0.00022172524661847055,0.5283011141224426
dolma3_cc/food_and_dining_high,0.02846915714900511,0.00044558263747524867,0.8465365408527974
dolma3_cc/food_and_dining_low,0.003925019383151708,0.00010882830328640503,0.2378262314586005
dolma3_cc/games_high,0.04285754910208404,0.0001838793224200761,0.9608555306382474
dolma3_cc/games_low,0.011952685064681202,0.002547706698356032,0.7520372606166003
dolma3_cc/health_high,0.05568377545211494,0.0008189167394229707,0.6494249346757164
dolma3_cc/health_low,0.03701380145143279,0.00010684299804476066,1.0098782861476365
dolma3_cc/history_and_geography_high,0.033185886301789154,0.0009565367474517036,1.2518197192569516
dolma3_cc/history_and_geography_low,0.002553570463915716,0.00010025378173587318,0.3509280614126178
dolma3_cc/industrial_high,0.02059080037354194,0.00022238609936002228,1.2791795716855532
dolma3_cc/industrial_low,0.0027558896007895753,0.000991184471151667,0.35147969520642125
dolma3_cc/literature_high,0.03232828779164631,0.005719783930591238,0.6612261622582951
dolma3_cc/literature_low,0.017667692868805245,0.0007281763332820511,1.3112703424013588
dolma3_cc/science_math_and_technology_high,0.019585117269499967,0.021150540461245017,0.5178619100200778
dolma3_cc/science_math_and_technology_low,0.004388158624537967,0.05409370184853902,0.8169528798431147
dolma3_finemath_3plus,0.007301562970828354,0.00023281146319178815,1.095278364584023
dolma3_stack_edu,4.009233804672937e-05,0.35681141891346563,3.3682320618991963
dolma3_wikipedia,0.0014565784189044444,7.391355717284336e-06,2.0113308943421986
dolmino_common_crawl_hq,0.20819611827251772,0.0015308309645392247,0.8010401552721311
dolmino_olmocr_pdfs_hq,0.055952616135912556,0.17414192199152193,2.444966265602824
dolmino_stack_edu_fim,6.353430208795893e-05,0.15954036014498924,1.5098691156414754
dolmino_stem_heavy_crawl,0.0029367994259667215,1.7487925431158873e-05,2.8545193373872895
dolmino_synth_code,3.743494675221391e-05,0.08376343255426603,5.6282445837793835
dolmino_synth_instruction,0.001167903871822112,0.025952096786692632,2.154570651299081
dolmino_synth_math,0.004192670318781768,0.0016058663867584472,1.0653554044950464
dolmino_synth_qa,0.07558406213063826,0.07173514165857404,0.8976781679963518
dolmino_synth_thinking,0.013260497271114668,7.254431902505372e-05,1.68409404743456
"""
    ),
}


@dataclass(frozen=True)
class MixtureDiagnostics:
    """Static diagnostics for one selected mixture."""

    mixture: str
    source_csv: str
    phase0_sum: float
    phase1_sum: float
    aggregate_sum: float
    max_simulated_epoch: float
    q95_simulated_epoch: float
    max_weight: float
    min_weight: float
    mean_phase_tv_to_proportional: float


@dataclass(frozen=True)
class DelphiOptimizedRunSpec:
    """One Delphi scaling validation training run."""

    run_order: int
    run_id: int
    run_name: str
    mixture: str
    mixture_display_name: str
    source_csv: str
    github_issue: int
    target_metric: str
    method: str
    target_flops: float
    tpu_type: str
    tpu_region: str
    tpu_zone: str
    batch_size: int
    train_tokens: int
    train_steps: int
    realized_train_tokens: int
    expected_checkpoint_step: int
    model_hidden_dim: int
    model_layers: int
    non_embedding_params: int
    total_trainable_params: int
    tensor_parallel_size: int
    data_seed: int
    trainer_seed: int
    phase_boundary: float
    phase_0_fraction: float
    phase_1_fraction: float
    simulated_epoch_target_budget: int
    available_top_level_tokens: int
    max_simulated_epoch: float
    q95_simulated_epoch: float
    mean_phase_tv_to_proportional: float
    phase_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class DelphiOptimizedTrainingConfig:
    """Config resolved by the executor before one Delphi optimized-mixture train."""

    analysis_output_path: str
    target_flops: float
    tpu_type: str
    tpu_region: str
    tpu_zone: str
    batch_size: int
    mixture: DelphiValidationMixture
    label: str
    output_path: str
    run_id: int
    run_name: str
    data_seed: int
    train_tokens_override: int | None
    trainer_seed: int = 0
    validation_configs: dict[str, DatasetComponent] | None = None


@dataclass(frozen=True)
class SaveDelphiOptimizedManifestConfig:
    """Config for persisting a resolved launcher manifest."""

    output_path: str
    analysis_output_path: str
    mixtures: tuple[DelphiValidationMixture, ...]
    target_budgets_json: str
    tpu_region: str
    tpu_zone: str
    run_order_offset: int = 0


@dataclass(frozen=True)
class LaunchArtifacts:
    """Resolved optimized-mixture launcher graph."""

    manifest_step: ExecutorStep
    training_steps: list[ExecutorStep]
    eval_steps: list[ExecutorStep]

    @property
    def steps(self) -> list[ExecutorStep]:
        return [self.manifest_step, *self.training_steps, *self.eval_steps]


def _proportional_weights() -> dict[str, float]:
    total_tokens = float(TOP_LEVEL_TOTAL_AVAILABLE_TOKENS)
    return {domain_name: TOP_LEVEL_DOMAIN_TOKEN_COUNTS[domain_name] / total_tokens for domain_name in DOMAIN_NAMES}


def _phase_lengths() -> dict[str, float]:
    if set(PHASE_FRACTIONS) != set(PHASE_NAMES):
        raise ValueError(f"Phase names mismatch: {PHASE_FRACTIONS.keys()} vs {PHASE_NAMES}")
    if abs(sum(PHASE_FRACTIONS.values()) - 1.0) > 1e-12:
        raise ValueError(f"Phase fractions sum to {sum(PHASE_FRACTIONS.values())}, expected 1")
    if abs(PHASE_FRACTIONS["phase_0"] - 0.8) > 1e-12 or abs(PHASE_FRACTIONS["phase_1"] - 0.2) > 1e-12:
        raise ValueError(f"This launcher expects historical 80/20 phase fractions, got {PHASE_FRACTIONS}")
    return dict(PHASE_FRACTIONS)


def _q95(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute q95 of empty list")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = 0.95 * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _parse_weight_rows(csv_text: str, *, source_label: str) -> list[dict[str, str]]:
    required_columns = {"domain", "phase_0_weight", "phase_1_weight"}
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    missing = required_columns.difference(reader.fieldnames or [])
    if missing:
        raise ValueError(f"{source_label} missing required columns: {sorted(missing)}")
    rows = list(reader)
    if not rows:
        raise ValueError(f"{source_label} has no rows")
    return rows


def _validate_embedded_matches_local(
    embedded_rows: list[dict[str, str]],
    local_rows: list[dict[str, str]],
    *,
    source: MixtureSource,
) -> None:
    embedded_by_domain = {row["domain"]: row for row in embedded_rows}
    local_by_domain = {row["domain"]: row for row in local_rows}
    if set(embedded_by_domain) != set(local_by_domain):
        missing = sorted(set(embedded_by_domain).difference(local_by_domain))
        extra = sorted(set(local_by_domain).difference(embedded_by_domain))
        raise ValueError(f"{source.key.value} embedded/local domain mismatch: missing={missing}, extra={extra}")
    for domain, embedded in embedded_by_domain.items():
        local = local_by_domain[domain]
        for column in ["phase_0_weight", "phase_1_weight", "simulated_epochs"]:
            if column not in embedded or column not in local:
                continue
            embedded_value = float(embedded[column])
            local_value = float(local[column])
            if abs(embedded_value - local_value) > max(1e-12, 1e-12 * abs(local_value)):
                raise ValueError(
                    f"{source.key.value}/{domain}/{column} embedded={embedded_value} "
                    f"does not match local CSV {local_value}"
                )


def _read_phase_weight_rows(source: MixtureSource) -> list[dict[str, str]]:
    embedded_csv = _EMBEDDED_MIXTURE_WEIGHT_CSVS.get(source.key)
    embedded_rows = (
        _parse_weight_rows(
            embedded_csv,
            source_label=f"embedded:{source.key.value}",
        )
        if embedded_csv is not None
        else None
    )
    if source.source_csv.startswith("gs://"):
        with fsspec.open(source.source_csv, "r") as handle:
            remote_rows = _parse_weight_rows(handle.read(), source_label=source.source_csv)
        if embedded_rows is not None:
            _validate_embedded_matches_local(embedded_rows, remote_rows, source=source)
        return remote_rows
    path = Path(source.source_csv)
    if path.exists():
        local_rows = _parse_weight_rows(path.read_text(), source_label=str(path))
        if embedded_rows is not None:
            _validate_embedded_matches_local(embedded_rows, local_rows, source=source)
        return local_rows
    if embedded_rows is None:
        raise FileNotFoundError(f"{source.key.value} mixture CSV is required but absent at {path}")
    logger.info("Using embedded weights for %s; local provenance CSV is absent at %s", source.key.value, path)
    return embedded_rows


def _read_phase_weights(source: MixtureSource) -> tuple[dict[str, dict[str, float]], MixtureDiagnostics]:
    rows = _read_phase_weight_rows(source)

    domains = [row["domain"] for row in rows]
    if sorted(domains) != sorted(DOMAIN_NAMES):
        missing = sorted(set(DOMAIN_NAMES).difference(domains))
        extra = sorted(set(domains).difference(DOMAIN_NAMES))
        raise ValueError(f"{source.key.value} domain mismatch: missing={missing}, extra={extra}")
    if len(domains) != len(set(domains)):
        raise ValueError(f"{source.key.value} has duplicate domains")

    phase_weights = {"phase_0": {}, "phase_1": {}}
    phase_lengths = _phase_lengths()
    aggregate_weights: dict[str, float] = {}
    simulated_epochs: list[float] = []
    for row in rows:
        domain = row["domain"]
        phase0 = float(row["phase_0_weight"])
        phase1 = float(row["phase_1_weight"])
        if phase0 < 0 or phase1 < 0:
            raise ValueError(f"{source.key.value}/{domain} has negative phase weights")
        phase_weights["phase_0"][domain] = phase0
        phase_weights["phase_1"][domain] = phase1
        aggregate = phase_lengths["phase_0"] * phase0 + phase_lengths["phase_1"] * phase1
        aggregate_weights[domain] = aggregate
        simulated_epoch = SIMULATED_EPOCH_TARGET_BUDGET * aggregate / TOP_LEVEL_DOMAIN_TOKEN_COUNTS[domain]
        simulated_epochs.append(simulated_epoch)
        if row.get("simulated_epochs"):
            recorded = float(row["simulated_epochs"])
            if abs(recorded - simulated_epoch) > max(1e-6, 1e-6 * abs(simulated_epoch)):
                raise ValueError(
                    f"{source.key.value}/{domain} recorded simulated_epochs={recorded} but recomputed {simulated_epoch}"
                )

    phase0_sum = sum(phase_weights["phase_0"].values())
    phase1_sum = sum(phase_weights["phase_1"].values())
    aggregate_sum = sum(aggregate_weights.values())
    for phase_name, total in [("phase_0", phase0_sum), ("phase_1", phase1_sum)]:
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"{source.key.value}/{phase_name} sums to {total}, expected 1")
    if abs(aggregate_sum - 1.0) > 1e-9:
        raise ValueError(f"{source.key.value} aggregate weights sum to {aggregate_sum}, expected 1")
    max_simulated_epoch = max(simulated_epochs)
    if source.expected_max_simulated_epoch is not None:
        if max_simulated_epoch > source.expected_max_simulated_epoch + 2e-6:
            raise ValueError(
                f"{source.key.value} max simulated epoch {max_simulated_epoch} exceeds expected "
                f"{source.expected_max_simulated_epoch}"
            )

    proportional = _proportional_weights()
    phase_tv = []
    for phase_name in PHASE_NAMES:
        tv = 0.5 * sum(abs(phase_weights[phase_name][domain] - proportional[domain]) for domain in DOMAIN_NAMES)
        phase_tv.append(tv)
    diagnostics = MixtureDiagnostics(
        mixture=source.key.value,
        source_csv=source.source_csv,
        phase0_sum=phase0_sum,
        phase1_sum=phase1_sum,
        aggregate_sum=aggregate_sum,
        max_simulated_epoch=max_simulated_epoch,
        q95_simulated_epoch=_q95(simulated_epochs),
        max_weight=max(max(weights.values()) for weights in phase_weights.values()),
        min_weight=min(min(weights.values()) for weights in phase_weights.values()),
        mean_phase_tv_to_proportional=sum(phase_tv) / len(phase_tv),
    )
    return phase_weights, diagnostics


def _weights_for_mixture(mixture: DelphiValidationMixture) -> tuple[dict[str, dict[str, float]], MixtureDiagnostics]:
    return _read_phase_weights(MIXTURE_SOURCES[mixture])


def _validate_runtime_phase_weights(phase_weights: dict[str, dict[str, float]], *, run_name: str) -> None:
    if set(phase_weights) != set(PHASE_NAMES):
        raise ValueError(f"{run_name} phase names mismatch: {sorted(phase_weights)}")
    for phase_name, weights in phase_weights.items():
        if set(weights) != set(DOMAIN_NAMES):
            raise ValueError(f"{run_name}/{phase_name} domain names mismatch")
        total = sum(float(weight) for weight in weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"{run_name}/{phase_name} weights sum to {total}, expected 1.0")
        negative = {domain: weight for domain, weight in weights.items() if weight < 0}
        if negative:
            raise ValueError(f"{run_name}/{phase_name} has negative weights: {negative}")


def _build_mixture_data(
    mixture: DelphiValidationMixture,
    train_tokens: int,
    model_config,
    batch_size: int,
    train_steps: int,
):
    phase_weights, _ = _weights_for_mixture(mixture)
    _validate_runtime_phase_weights(phase_weights, run_name=mixture.value)
    experiment = MixtureExperiment(
        name=EXPERIMENT_NAME,
        domains=build_top_level_domains(runtime_cache_region=DEFAULT_RUNTIME_CACHE_REGION),
        phase_schedule=PHASE_SCHEDULE,
        model_config=model_config,
        batch_size=batch_size,
        seq_len=SEQ_LEN_DELPHI,
        num_train_steps=train_steps,
        target_budget=None,
        resources=ResourceConfig.with_tpu("v5p-8", regions=[DEFAULT_TPU_REGION], zone=DEFAULT_TPU_ZONE),
        eval_harness_tasks=(),
        optimizer_config=None,
        eval_datasets_cache_path=None,
        hierarchical_runtime_domains=True,
    )
    data = experiment.create_mixture_config(WeightConfig(run_id=0, phase_weights=phase_weights))
    if train_tokens > SIMULATED_EPOCH_TARGET_BUDGET:
        raise ValueError(
            f"Delphi train_tokens={train_tokens} exceeds simulated-epoch target budget "
            f"{SIMULATED_EPOCH_TARGET_BUDGET}; simulated epoching would be ill-defined."
        )
    return (
        replace(
            data,
            target_budget=SIMULATED_EPOCH_TARGET_BUDGET,
            experiment_budget=train_tokens,
            simulated_epoch_subset_seed=None,
        ),
        phase_weights,
    )


def run_delphi_optimized_training(config: DelphiOptimizedTrainingConfig) -> None:
    """Run one Delphi optimized-mixture training job."""
    scaling_fits = _read_scaling_fits(config.analysis_output_path)
    candidate = _candidate_for_budget(
        scaling_fits=scaling_fits,
        target_flops=config.target_flops,
        batch_size=config.batch_size,
    )
    if config.train_tokens_override is not None:
        if config.train_tokens_override <= 0:
            raise ValueError(f"train_tokens_override must be positive, got {config.train_tokens_override}")
        train_steps = round(config.train_tokens_override / (config.batch_size * SEQ_LEN_DELPHI))
        realized_override_tokens = train_steps * config.batch_size * SEQ_LEN_DELPHI
        candidate = replace(
            candidate,
            tokens=float(realized_override_tokens),
            train_steps=train_steps,
            optimizer_config=completed_adamh_heuristic.build_optimizer_config(
                config.batch_size,
                realized_override_tokens,
            ),
        )
    params = candidate.model_config.total_trainable_params(completed_adamh_heuristic.vocab_size)
    realized_train_tokens = candidate.train_steps * config.batch_size * SEQ_LEN_DELPHI
    tp = _tensor_parallel_size(candidate.model_config.hidden_dim, config.tpu_type)

    source = MIXTURE_SOURCES[config.mixture]
    logger.info(
        "Delphi optimized %s/%s: hidden_dim=%d layers=%d params=%.2e tokens=%.2e "
        "realized_tokens=%d batch_size=%d steps=%d tpu=%s tp=%d phase_boundary=%.3f",
        config.mixture.value,
        _slug(config.target_flops),
        candidate.model_config.hidden_dim,
        candidate.model_config.num_layers,
        params,
        candidate.tokens,
        realized_train_tokens,
        config.batch_size,
        candidate.train_steps,
        config.tpu_type,
        tp,
        PHASE_BOUNDARIES[0],
    )

    data, phase_weights = _build_mixture_data(
        config.mixture,
        realized_train_tokens,
        candidate.model_config,
        config.batch_size,
        candidate.train_steps,
    )
    _validate_runtime_phase_weights(phase_weights, run_name=config.run_name)
    data = _add_validation_components(data, config.validation_configs)

    inner_config = train_lm.TrainLmConfig(
        data=data,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                entity="marin-community",
                project="marin",
                tags=[
                    f"issue-{source.github_issue}",
                    source.wandb_series_tag,
                    "delphi-optimized-mixtures",
                    "completed-adamh",
                    config.mixture.value,
                    source.method,
                    f"FLOPs={config.target_flops:.1e}",
                    f"D={realized_train_tokens:.1e}",
                    f"D/N={realized_train_tokens / params:.3f}",
                    f"label={config.label}",
                    f"N={params:.1e}",
                    f"data_seed={config.data_seed}",
                    f"trainer_seed={config.trainer_seed}",
                ],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=candidate.batch_size,
            per_device_parallelism=-1,
            num_train_steps=candidate.train_steps,
            steps_per_eval=1000,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[dict(every=5000)],
            ),
            mesh=MeshConfig(
                axes={"data": -1, "replica": 1, "model": tp},
                compute_mapping={
                    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                },
            ),
            seed=config.trainer_seed,
            allow_nondivisible_batch_size=True,
        ),
        train_seq_len=SEQ_LEN_DELPHI,
        model=candidate.model_config,
        optimizer=candidate.optimizer_config,
        data_seed=config.data_seed,
    )

    resources = ResourceConfig.with_tpu(config.tpu_type, regions=[config.tpu_region], zone=config.tpu_zone)
    pod_config = TrainLmOnPodConfig(
        train_config=inner_config,
        resources=resources,
        output_path=config.output_path,
        env_vars={
            "MARIN_PREFIX": marin_prefix_for_region(config.tpu_region),
            SKIP_EVAL_HARNESS_ENV_VAR: "1",
        },
    )
    run_levanter_train_lm(pod_config)


def _predict_run_spec(
    *,
    scaling_fits,
    mixture: DelphiValidationMixture,
    target_flops: float,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    batch_size: int,
    run_order: int,
) -> DelphiOptimizedRunSpec:
    source = MIXTURE_SOURCES[mixture]
    candidate = _candidate_for_budget(
        scaling_fits=scaling_fits,
        target_flops=target_flops,
        batch_size=batch_size,
    )
    train_tokens = round(candidate.tokens)
    realized_train_tokens = candidate.train_steps * batch_size * SEQ_LEN_DELPHI
    phase_weights, diagnostics = _weights_for_mixture(mixture)
    run_name = f"{mixture.value}_{_slug(target_flops)}"
    _validate_runtime_phase_weights(phase_weights, run_name=run_name)
    non_embedding_params = int(candidate.model_config.total_trainable_params(0))
    total_params = int(candidate.model_config.total_trainable_params(completed_adamh_heuristic.vocab_size))
    return DelphiOptimizedRunSpec(
        run_order=run_order,
        run_id=RUN_ID_BASE + run_order,
        run_name=run_name,
        mixture=mixture.value,
        mixture_display_name=source.display_name,
        source_csv=source.source_csv,
        github_issue=source.github_issue,
        target_metric=source.target_metric,
        method=source.method,
        target_flops=target_flops,
        tpu_type=tpu_type,
        tpu_region=tpu_region,
        tpu_zone=tpu_zone,
        batch_size=batch_size,
        train_tokens=train_tokens,
        train_steps=candidate.train_steps,
        realized_train_tokens=realized_train_tokens,
        expected_checkpoint_step=candidate.train_steps - 1,
        model_hidden_dim=int(candidate.model_config.hidden_dim),
        model_layers=int(candidate.model_config.num_layers),
        non_embedding_params=non_embedding_params,
        total_trainable_params=total_params,
        tensor_parallel_size=_tensor_parallel_size(candidate.model_config.hidden_dim, tpu_type),
        data_seed=(source.data_seed_override if source.data_seed_override is not None else RUN_ID_BASE + run_order),
        trainer_seed=0,
        phase_boundary=PHASE_BOUNDARIES[0],
        phase_0_fraction=PHASE_FRACTIONS["phase_0"],
        phase_1_fraction=PHASE_FRACTIONS["phase_1"],
        simulated_epoch_target_budget=SIMULATED_EPOCH_TARGET_BUDGET,
        available_top_level_tokens=TOP_LEVEL_TOTAL_AVAILABLE_TOKENS,
        max_simulated_epoch=diagnostics.max_simulated_epoch,
        q95_simulated_epoch=diagnostics.q95_simulated_epoch,
        mean_phase_tv_to_proportional=diagnostics.mean_phase_tv_to_proportional,
        phase_weights=phase_weights,
    )


def save_delphi_optimized_manifest(config: SaveDelphiOptimizedManifestConfig) -> None:
    """Persist run specs as JSON and CSV artifacts."""
    target_budgets = {
        float(item["target_flops"]): (str(item["tpu_type"]), int(item["batch_size"]))
        for item in json.loads(config.target_budgets_json)
    }
    scaling_fits = _read_scaling_fits(config.analysis_output_path)
    run_specs: list[DelphiOptimizedRunSpec] = []
    diagnostics: list[MixtureDiagnostics] = []
    for mixture in config.mixtures:
        _, diag = _weights_for_mixture(mixture)
        diagnostics.append(diag)
    for target_flops, (tpu_type, batch_size) in target_budgets.items():
        for mixture in config.mixtures:
            run_specs.append(
                _predict_run_spec(
                    scaling_fits=scaling_fits,
                    mixture=mixture,
                    target_flops=target_flops,
                    tpu_type=tpu_type,
                    tpu_region=config.tpu_region,
                    tpu_zone=config.tpu_zone,
                    batch_size=batch_size,
                    run_order=config.run_order_offset + len(run_specs),
                )
            )

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fs.open(os.path.join(config.output_path, "run_specs.json"), "w") as handle:
        json.dump([asdict(run_spec) for run_spec in run_specs], handle, indent=2, sort_keys=True)
    with fs.open(os.path.join(config.output_path, "selected_mixtures.json"), "w") as handle:
        json.dump([asdict(item) for item in diagnostics], handle, indent=2, sort_keys=True)
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=[
            "run_order",
            "run_id",
            "run_name",
            "mixture",
            "mixture_display_name",
            "github_issue",
            "target_metric",
            "method",
            "target_flops",
            "tpu_type",
            "tpu_region",
            "tpu_zone",
            "batch_size",
            "train_tokens",
            "train_steps",
            "realized_train_tokens",
            "expected_checkpoint_step",
            "model_hidden_dim",
            "model_layers",
            "non_embedding_params",
            "total_trainable_params",
            "tensor_parallel_size",
            "data_seed",
            "trainer_seed",
            "phase_boundary",
            "phase_0_fraction",
            "phase_1_fraction",
            "simulated_epoch_target_budget",
            "available_top_level_tokens",
            "max_simulated_epoch",
            "q95_simulated_epoch",
            "mean_phase_tv_to_proportional",
            "source_csv",
        ],
    )
    writer.writeheader()
    for run_spec in run_specs:
        row = asdict(run_spec)
        row.pop("phase_weights")
        writer.writerow(row)
    with fs.open(os.path.join(config.output_path, "training_manifest.csv"), "w") as handle:
        handle.write(csv_buffer.getvalue())
    summary: dict[str, Any] = {
        "n_runs": len(run_specs),
        "mixtures": sorted({run_spec.mixture for run_spec in run_specs}),
        "target_flops": sorted({run_spec.target_flops for run_spec in run_specs}),
        "source_experiment": EXPERIMENT_NAME,
        "analysis_output_path": config.analysis_output_path,
        "phase_boundary": PHASE_BOUNDARIES[0],
        "phase_fractions": dict(PHASE_FRACTIONS),
        "run_order_offset": config.run_order_offset,
        "simulated_epoch_target_budget": SIMULATED_EPOCH_TARGET_BUDGET,
        "available_top_level_tokens": TOP_LEVEL_TOTAL_AVAILABLE_TOKENS,
        "diagnostics": [asdict(item) for item in diagnostics],
    }
    with fs.open(os.path.join(config.output_path, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def _selected_target_budgets(values: tuple[str, ...]) -> dict[float, tuple[str, int]]:
    if not values:
        return dict(TARGET_BUDGETS)
    selected: dict[float, tuple[str, int]] = {}
    unknown: list[str] = []
    for value in values:
        target = float(value)
        if target not in TARGET_BUDGETS:
            unknown.append(value)
            continue
        selected[target] = TARGET_BUDGETS[target]
    if unknown:
        allowed = ", ".join(f"{budget:.0e}" for budget in TARGET_BUDGETS)
        raise ValueError(f"Unknown target budget(s): {unknown}. Allowed: {allowed}")
    return selected


def _parse_mixtures(values: tuple[str, ...]) -> tuple[DelphiValidationMixture, ...]:
    if not values:
        return tuple(DelphiValidationMixture)
    return tuple(DelphiValidationMixture(value) for value in values)


def build_launch_artifacts(
    *,
    analysis_output_path: str,
    validation_configs: dict[str, DatasetComponent],
    mixtures: tuple[DelphiValidationMixture, ...],
    target_budgets: dict[float, tuple[str, int]],
    tpu_region: str,
    tpu_zone: str,
    run_order_offset: int = 0,
    include_table9_native_eval: bool = False,
) -> LaunchArtifacts:
    """Build the executor graph for selected mixtures and FLOP budgets."""
    training_steps: list[ExecutorStep] = []
    eval_steps: list[ExecutorStep] = []
    scaling_fits = _read_scaling_fits(analysis_output_path)
    for target_flops, (tpu_type, batch_size) in target_budgets.items():
        for mixture in mixtures:
            run_order = run_order_offset + len(training_steps)
            run_name = f"{mixture.value}_{_slug(target_flops)}"
            candidate = _candidate_for_budget(
                scaling_fits=scaling_fits,
                target_flops=target_flops,
                batch_size=batch_size,
            )
            source = MIXTURE_SOURCES[mixture]
            data_seed = source.data_seed_override if source.data_seed_override is not None else RUN_ID_BASE + run_order
            training_step = ExecutorStep(
                name=f"{EXPERIMENT_NAME}/{run_name}",
                fn=run_delphi_optimized_training,
                resources=ResourceConfig.with_tpu(tpu_type, regions=[tpu_region], zone=tpu_zone),
                config=DelphiOptimizedTrainingConfig(
                    analysis_output_path=analysis_output_path,
                    target_flops=target_flops,
                    tpu_type=tpu_type,
                    tpu_region=tpu_region,
                    tpu_zone=tpu_zone,
                    batch_size=batch_size,
                    mixture=mixture,
                    label=LABEL,
                    output_path=this_output_path(),
                    run_id=RUN_ID_BASE + run_order,
                    run_name=run_name,
                    data_seed=data_seed,
                    train_tokens_override=None,
                    trainer_seed=0,
                    validation_configs=validation_configs,
                ),
            )
            training_steps.append(training_step)
            if include_table9_native_eval:
                eval_steps.append(
                    olmo_base_eval_step(
                        name=f"t9_{run_name}",
                        checkpoint=training_step / f"hf/step-{candidate.train_steps - 1}",
                        request_set_dir=TABLE9_REQUEST_SET_DIR,
                        resource_config=TABLE9_EVAL_RESOURCES,
                        wandb_group="olmo_base_eval_table9_scaling_validation",
                        provenance={
                            "evaluator": "marin-native-table9-bpb",
                            "panel": "delphi_optimized_mixtures",
                            "scale": _slug(target_flops),
                            "source_run_name": run_name,
                            "mixture": mixture.value,
                            "method": source.method,
                        },
                    )
                )

    manifest_step = ExecutorStep(
        name=f"{EXPERIMENT_NAME}/manifest",
        fn=save_delphi_optimized_manifest,
        config=SaveDelphiOptimizedManifestConfig(
            output_path=this_output_path(),
            analysis_output_path=analysis_output_path,
            mixtures=mixtures,
            target_budgets_json=json.dumps(
                [
                    {"target_flops": target_flops, "tpu_type": tpu_type, "batch_size": batch_size}
                    for target_flops, (tpu_type, batch_size) in target_budgets.items()
                ],
                sort_keys=True,
            ),
            tpu_region=tpu_region,
            tpu_zone=tpu_zone,
            run_order_offset=run_order_offset,
        ),
    )
    return LaunchArtifacts(manifest_step=manifest_step, training_steps=training_steps, eval_steps=eval_steps)


def _write_local_static_manifest(
    *,
    mixtures: tuple[DelphiValidationMixture, ...],
    target_budgets: dict[float, tuple[str, int]],
    tpu_region: str,
    tpu_zone: str,
    run_order_offset: int = 0,
) -> None:
    """Write a local dependency-light manifest validating static mixture semantics."""
    LOCAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    diagnostics: list[MixtureDiagnostics] = []
    rows: list[dict[str, Any]] = []
    for mixture in mixtures:
        _, diag = _weights_for_mixture(mixture)
        diagnostics.append(diag)
    run_order = run_order_offset
    for target_flops, (tpu_type, batch_size) in target_budgets.items():
        for mixture in mixtures:
            source = MIXTURE_SOURCES[mixture]
            diag = next(item for item in diagnostics if item.mixture == mixture.value)
            rows.append(
                {
                    "run_order": run_order,
                    "run_id": RUN_ID_BASE + run_order,
                    "data_seed": (
                        source.data_seed_override if source.data_seed_override is not None else RUN_ID_BASE + run_order
                    ),
                    "trainer_seed": 0,
                    "run_name": f"{mixture.value}_{_slug(target_flops)}",
                    "mixture": mixture.value,
                    "mixture_display_name": source.display_name,
                    "github_issue": source.github_issue,
                    "target_metric": source.target_metric,
                    "method": source.method,
                    "target_flops": target_flops,
                    "tpu_type": tpu_type,
                    "tpu_region": tpu_region,
                    "tpu_zone": tpu_zone,
                    "batch_size": batch_size,
                    "phase_boundary": PHASE_BOUNDARIES[0],
                    "phase_0_fraction": PHASE_FRACTIONS["phase_0"],
                    "phase_1_fraction": PHASE_FRACTIONS["phase_1"],
                    "simulated_epoch_target_budget": SIMULATED_EPOCH_TARGET_BUDGET,
                    "max_simulated_epoch": diag.max_simulated_epoch,
                    "q95_simulated_epoch": diag.q95_simulated_epoch,
                    "mean_phase_tv_to_proportional": diag.mean_phase_tv_to_proportional,
                    "source_csv": source.source_csv,
                }
            )
            run_order += 1

    with (LOCAL_ARTIFACT_DIR / "selected_mixtures.json").open("w") as handle:
        json.dump([asdict(item) for item in diagnostics], handle, indent=2, sort_keys=True)
    with (LOCAL_ARTIFACT_DIR / "run_specs_static.json").open("w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
    with (LOCAL_ARTIFACT_DIR / "training_manifest_static.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "n_runs": len(rows),
        "mixtures": [mixture.value for mixture in mixtures],
        "target_flops": sorted(target_budgets),
        "phase_boundary": PHASE_BOUNDARIES[0],
        "phase_fractions": dict(PHASE_FRACTIONS),
        "run_order_offset": run_order_offset,
        "simulated_epoch_target_budget": SIMULATED_EPOCH_TARGET_BUDGET,
        "available_top_level_tokens": TOP_LEVEL_TOTAL_AVAILABLE_TOKENS,
        "diagnostics": [asdict(item) for item in diagnostics],
        "note": "Static local manifest; remote executor manifest adds model/token predictions from scaling fits.",
    }
    with (LOCAL_ARTIFACT_DIR / "summary_static.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixtures", nargs="*", default=[])
    parser.add_argument("--target-budgets", nargs="*", default=[])
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument(
        "--run-order-offset",
        type=int,
        default=0,
        help=(
            "Offset added to generated run_order/run_id and default data_seed. Explicit per-mixture "
            "seed overrides are unchanged. Keep 0 for normal launches; use only for scoped retries "
            "that need to preserve the original manifest row seed."
        ),
    )
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help=(
            "Do not include the manifest step in the executor graph. Use only for scoped retries after "
            "the full local dry-run manifest has been captured; this avoids overwriting shared manifest "
            "provenance with a one-row retry manifest."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-table9-training-without-native-eval-plan",
        action="store_true",
        help=(
            "Acknowledge that this launcher trains Table-9-targeted checkpoints but does not itself "
            "schedule Marin-native OLMoBaseEval Table-9. Use only when a separate native Table-9 eval "
            "job is already tracked for the generated checkpoints."
        ),
    )
    parser.add_argument(
        "--include-table9-native-eval",
        action="store_true",
        help=(
            "Append Marin-native OLMoBaseEval Table-9 executor steps for Table-9-targeted checkpoints. "
            "This is the preferred path for new Table-9 scaling-validation runs."
        ),
    )
    parser.add_argument("--analysis-output-path", default=DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--local-artifact-dir", default=str(LOCAL_ARTIFACT_DIR))
    return parser.parse_known_args()


def _has_table9_target(mixtures: tuple[DelphiValidationMixture, ...]) -> bool:
    return any(MIXTURE_SOURCES[mixture].target_metric == TABLE9_TARGET_METRIC for mixture in mixtures)


def main() -> None:
    global EXPERIMENT_NAME, LOCAL_ARTIFACT_DIR
    logging.basicConfig(level=logging.INFO)
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]
    EXPERIMENT_NAME = args.experiment_name
    LOCAL_ARTIFACT_DIR = Path(args.local_artifact_dir)

    if args.tpu_region != DEFAULT_TPU_REGION or args.tpu_zone != DEFAULT_TPU_ZONE:
        raise ValueError(f"This launcher is pinned to {DEFAULT_TPU_REGION}/{DEFAULT_TPU_ZONE}")
    expected_prefix = marin_prefix_for_region(args.tpu_region)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != expected_prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match required east5 prefix {expected_prefix!r}")
    os.environ["MARIN_PREFIX"] = expected_prefix

    mixtures = _parse_mixtures(tuple(args.mixtures))
    target_budgets = _selected_target_budgets(tuple(args.target_budgets))
    if (
        _has_table9_target(mixtures)
        and not args.dry_run
        and not args.include_table9_native_eval
        and not args.allow_table9_training_without_native_eval_plan
    ):
        raise ValueError(
            "This training launcher does not schedule Marin-native OLMoBaseEval Table-9. "
            "For Table-9-targeted mixtures, pass --include-table9-native-eval or submit/track a "
            "separate native Table-9 eval job and rerun with --allow-table9-training-without-native-eval-plan."
        )
    if not args.analysis_output_path:
        raise ValueError("--analysis-output-path must be set; do not rerun isoflop analysis in this parent")
    if args.run_order_offset < 0:
        raise ValueError("--run-order-offset must be nonnegative")

    if args.dry_run:
        _write_local_static_manifest(
            mixtures=mixtures,
            target_budgets=target_budgets,
            tpu_region=args.tpu_region,
            tpu_zone=args.tpu_zone,
            run_order_offset=args.run_order_offset,
        )
        logger.info("Wrote static dry-run specs under %s", LOCAL_ARTIFACT_DIR)
        return

    validation_steps = default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }

    artifacts = build_launch_artifacts(
        analysis_output_path=args.analysis_output_path,
        validation_configs=validation_configs,
        mixtures=mixtures,
        target_budgets=target_budgets,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
        run_order_offset=args.run_order_offset,
        include_table9_native_eval=args.include_table9_native_eval,
    )
    if os.getenv("CI") is not None:
        logger.info(
            "Built Delphi optimized-mixture graph with %d training steps and %d eval steps; skipping executor launch.",
            len(artifacts.training_steps),
            len(artifacts.eval_steps),
        )
        return

    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=[*artifacts.training_steps, *artifacts.eval_steps] if args.skip_manifest else artifacts.steps,
        description=f"{EXPERIMENT_NAME}: Delphi optimized-mixture scaling validation",
    )


if __name__ == "__main__":
    main()
