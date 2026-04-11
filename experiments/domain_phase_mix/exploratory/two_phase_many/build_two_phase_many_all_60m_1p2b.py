# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "numpy", "pandas", "wandb"]
# ///
"""Build a broader 60M/1.2B two-phase-many export including validated optima."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib import import_module
import json
from pathlib import Path
import re
from collections.abc import Callable

import fsspec
import numpy as np
import pandas as pd
import wandb

import experiments.domain_phase_mix.two_phase_many_ccglobalpremium_baselines as ccglobal
import experiments.domain_phase_mix.two_phase_many_ccpairtotal_baseline as ccpair
import experiments.domain_phase_mix.two_phase_many_dsre_baselines as dsre
import experiments.domain_phase_mix.two_phase_many_dsre_predicted_baselines as dsre_predicted
import experiments.domain_phase_mix.two_phase_many_dsre_predicted_topic_collapsed as dsre_topic
import experiments.domain_phase_mix.two_phase_many_genericfamily_lbfgsb_baseline as gf_lbfgsb
import experiments.domain_phase_mix.two_phase_many_genericfamily_no_groups_baseline as gf_no_groups
import experiments.domain_phase_mix.two_phase_many_genericfamily_no_penalty_baseline as gf_no_penalty
import experiments.domain_phase_mix.two_phase_many_genericfamily_no_quality_splits_no_groups_baseline as gf_no_qs
import experiments.domain_phase_mix.two_phase_many_genericfamily_no_retention_baseline as gf_no_retention
import experiments.domain_phase_mix.two_phase_many_genericfamily_observed_only_trustblend_baseline as gf_trustblend
import experiments.domain_phase_mix.two_phase_many_genericfamily_penalty_raw_optima_baselines as gf_penalty_raw
import experiments.domain_phase_mix.two_phase_many_genericfamily_powell_baseline as gf_powell
import experiments.domain_phase_mix.two_phase_many_genericfamily_recovered_hull_baseline as gf_recovered
import experiments.domain_phase_mix.two_phase_many_genericfamily_recovered_hull_subset_optima as gf_recovered_subset
import experiments.domain_phase_mix.two_phase_many_genericfamily_retuned_subset_optima as gf_retuned_subset
import experiments.domain_phase_mix.two_phase_many_genericfamily_subset_optima as gf_subset
import experiments.domain_phase_mix.two_phase_many_genericfamily_top8actual_hull_subset_optima as gf_top8_subset
import experiments.domain_phase_mix.two_phase_many_genericfamily_tuned_baseline as gf_tuned
import experiments.domain_phase_mix.two_phase_many_observed_runs as observed_runs
import experiments.domain_phase_mix.two_phase_many_olmix_loglinear_sl_verb as olmix_slverb
import experiments.domain_phase_mix.two_phase_many_olmix_loglinear_sl_verb_seedpanel as olmix_slverb_seedpanel
import experiments.domain_phase_mix.two_phase_many_olmix_loglinear_uncheatable as olmix_uncheatable
import experiments.domain_phase_mix.two_phase_many_phasecomp_sparse_pls_baseline as phasecomp
import experiments.domain_phase_mix.two_phase_many_power_ridge_single as power_ridge
import experiments.domain_phase_mix.two_phase_many_surrogate_baselines as surrogate
import experiments.domain_phase_mix.two_phase_many_thresholdtotal_overfit as thresholdtotal
from experiments.domain_phase_mix import (
    two_phase_many_genericfamily_family_curvature_observed_only_trustblend_baselines as gf_family_curvature,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    create_two_phase_dolma3_dolmino_top_level_experiment,
)

gf_trustblend_subset = import_module(
    "experiments.domain_phase_mix.two_phase_many_genericfamily_observed_only_trustblend_subset_optima"
)
gf_power_trustblend = import_module(
    "experiments.domain_phase_mix.two_phase_many_genericfamily_power_observed_only_trustblend_baseline"
)

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_CSV = SCRIPT_DIR / "two_phase_many.csv"
OUTPUT_CSV = SCRIPT_DIR / "two_phase_many_all_60m_1p2b.csv"
OUTPUT_JSON = SCRIPT_DIR / "two_phase_many_all_60m_1p2b_summary.json"

CHECKPOINT_ROOT = "marin-us-east5/checkpoints"
SWARM_RUN_PATTERN = re.compile(r"run_\d{5}$")
WANDB_RUN_URL_PREFIX = "https://wandb.ai/marin-community/marin/runs/"
METRIC_PREFIXES = ("eval/", "lm_eval/")


@dataclass(frozen=True)
class ValidationRunSpec:
    """One extra 60M/1.2B validated run to append to the base swarm export."""

    collection: str
    variant: str
    source_experiment: str
    run_id: int
    run_name: str
    phase_weights_fn: Callable[[], dict[str, dict[str, float]]]
    subset_size: int | None = None


def _weight_config_phase_weights(factory: Callable[[], object]) -> Callable[[], dict[str, dict[str, float]]]:
    return lambda: factory().phase_weights


def _subset_weight_config_phase_weights(
    factory: Callable[[int], object],
    subset_size: int,
) -> Callable[[], dict[str, dict[str, float]]]:
    return lambda subset_size=subset_size: factory(subset_size).phase_weights


@cache
def _top_level_natural_proportions_and_phase_fractions() -> tuple[np.ndarray, np.ndarray]:
    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(name="two_phase_many_all_60m_1p2b")
    natural_proportions = np.asarray([float(domain.total_weight) for domain in experiment.domains], dtype=float)
    natural_proportions = natural_proportions / natural_proportions.sum()
    phase_fractions = np.asarray(
        [phase.end_fraction - phase.start_fraction for phase in experiment.phase_schedule.phases],
        dtype=float,
    )
    return natural_proportions, phase_fractions


@cache
def _olmix_uncheatable_phase_weights() -> dict[str, dict[str, float]]:
    natural_proportions, phase_fractions = _top_level_natural_proportions_and_phase_fractions()
    fit = olmix_uncheatable.load_fit_from_local_results(
        natural_proportions=natural_proportions,
        phase_fractions=phase_fractions,
    )
    return fit.phase_weights


@cache
def _olmix_slverb_phase_weights() -> dict[str, dict[str, float]]:
    natural_proportions, phase_fractions = _top_level_natural_proportions_and_phase_fractions()
    fit = olmix_slverb.load_fit_from_results(
        natural_proportions=natural_proportions,
        phase_fractions=phase_fractions,
    )
    return fit.phase_weights


@cache
def _olmix_slverb_seedpanel_phase_weights() -> dict[str, dict[str, float]]:
    natural_proportions, phase_fractions = _top_level_natural_proportions_and_phase_fractions()
    fit = olmix_slverb_seedpanel.load_fit_from_seedpanel_results(
        natural_proportions=natural_proportions,
        phase_fractions=phase_fractions,
    )
    return fit.phase_weights


def _extra_validation_specs() -> tuple[ValidationRunSpec, ...]:
    specs: list[ValidationRunSpec] = [
        ValidationRunSpec(
            collection="validated_baseline",
            variant="clr_ridge_balanced",
            source_experiment=surrogate.SURROGATE_BASELINES_SOURCE_EXPERIMENT,
            run_id=surrogate.CLR_RIDGE_RUN_ID,
            run_name=surrogate.CLR_RIDGE_RUN_NAME,
            phase_weights_fn=_weight_config_phase_weights(surrogate.create_clr_ridge_weight_config),
        ),
        ValidationRunSpec(
            collection="validated_baseline",
            variant="dsre_ceq_st_lite",
            source_experiment=surrogate.SURROGATE_BASELINES_SOURCE_EXPERIMENT,
            run_id=surrogate.DSRE_CEQ_ST_LITE_RUN_ID,
            run_name=surrogate.DSRE_CEQ_ST_LITE_RUN_NAME,
            phase_weights_fn=_weight_config_phase_weights(surrogate.create_dsre_ceq_st_lite_weight_config),
        ),
        ValidationRunSpec(
            collection="validated_baseline",
            variant="dsre_ensemble",
            source_experiment=dsre.DSRE_BASELINES_SOURCE_EXPERIMENT,
            run_id=dsre.DSRE_ENSEMBLE_RUN_ID,
            run_name=dsre.DSRE_ENSEMBLE_RUN_NAME,
            phase_weights_fn=_weight_config_phase_weights(dsre.create_dsre_ensemble_weight_config),
        ),
        ValidationRunSpec(
            collection="validated_baseline",
            variant="dsre_observed_consensus",
            source_experiment=dsre.DSRE_BASELINES_SOURCE_EXPERIMENT,
            run_id=dsre.DSRE_OBSERVED_CONSENSUS_RUN_ID,
            run_name=dsre.DSRE_OBSERVED_CONSENSUS_RUN_NAME,
            phase_weights_fn=_weight_config_phase_weights(dsre.create_dsre_observed_consensus_weight_config),
        ),
        ValidationRunSpec(
            collection="validated_baseline",
            variant="power_ridge_single",
            source_experiment=power_ridge.POWER_RIDGE_SINGLE_SOURCE_EXPERIMENT,
            run_id=power_ridge.POWER_RIDGE_SINGLE_RUN_ID,
            run_name=power_ridge.POWER_RIDGE_SINGLE_RUN_NAME,
            phase_weights_fn=_weight_config_phase_weights(power_ridge.create_power_ridge_single_weight_config),
        ),
        ValidationRunSpec(
            collection="validated_baseline",
            variant="dsre_ceq_predicted",
            source_experiment=dsre_predicted.DSRE_PREDICTED_BASELINES_SOURCE_EXPERIMENT,
            run_id=dsre_predicted.DSRE_CEQ_PREDICTED_RUN_ID,
            run_name=dsre_predicted.DSRE_CEQ_PREDICTED_RUN_NAME,
            phase_weights_fn=_weight_config_phase_weights(dsre_predicted.create_dsre_ceq_predicted_weight_config),
        ),
        ValidationRunSpec(
            collection="validated_baseline",
            variant="dsre_ceq_predicted_quality_collapsed",
            source_experiment=dsre_predicted.DSRE_PREDICTED_BASELINES_SOURCE_EXPERIMENT,
            run_id=dsre_predicted.DSRE_CEQ_PREDICTED_QUALITY_COLLAPSED_RUN_ID,
            run_name=dsre_predicted.DSRE_CEQ_PREDICTED_QUALITY_COLLAPSED_RUN_NAME,
            phase_weights_fn=_weight_config_phase_weights(
                dsre_predicted.create_dsre_ceq_predicted_quality_collapsed_weight_config
            ),
        ),
        ValidationRunSpec(
            collection="validated_baseline",
            variant="dsre_ceq_predicted_topic_collapsed",
            source_experiment=dsre_topic.DSRE_CEQ_PREDICTED_TOPIC_COLLAPSED_SOURCE_EXPERIMENT,
            run_id=dsre_topic.DSRE_CEQ_PREDICTED_TOPIC_COLLAPSED_RUN_ID,
            run_name=dsre_topic.DSRE_CEQ_PREDICTED_TOPIC_COLLAPSED_RUN_NAME,
            phase_weights_fn=_weight_config_phase_weights(
                dsre_topic.create_dsre_ceq_predicted_topic_collapsed_weight_config
            ),
        ),
        ValidationRunSpec(
            collection="validated_baseline",
            variant="olmix_loglinear_uncheatable",
            source_experiment=olmix_uncheatable.SOURCE_EXPERIMENT,
            run_id=olmix_uncheatable.RUN_ID,
            run_name=olmix_uncheatable.RUN_NAME,
            phase_weights_fn=_olmix_uncheatable_phase_weights,
        ),
        ValidationRunSpec(
            collection="validated_baseline",
            variant="olmix_loglinear_sl_verb",
            source_experiment=olmix_slverb.SOURCE_EXPERIMENT,
            run_id=olmix_slverb.RUN_ID,
            run_name=olmix_slverb.RUN_NAME,
            phase_weights_fn=_olmix_slverb_phase_weights,
        ),
        ValidationRunSpec(
            collection="validated_baseline",
            variant="olmix_loglinear_sl_verb_seedpanel",
            source_experiment=olmix_slverb_seedpanel.SOURCE_EXPERIMENT,
            run_id=olmix_slverb_seedpanel.RUN_ID,
            run_name=olmix_slverb_seedpanel.RUN_NAME,
            phase_weights_fn=_olmix_slverb_seedpanel_phase_weights,
        ),
        ValidationRunSpec(
            collection="validated_baseline",
            variant="thresholdtotal_overfit",
            source_experiment=thresholdtotal.THRESHOLDTOTAL_OVERFIT_SOURCE_EXPERIMENT,
            run_id=thresholdtotal.THRESHOLDTOTAL_OVERFIT_RUN_ID,
            run_name=thresholdtotal.THRESHOLDTOTAL_OVERFIT_RUN_NAME,
            phase_weights_fn=_weight_config_phase_weights(thresholdtotal.create_thresholdtotal_overfit_weight_config),
        ),
        ValidationRunSpec(
            collection="validated_baseline",
            variant="ccglobalpremium_threshold",
            source_experiment=ccglobal.CCGLOBALPREMIUM_SOURCE_EXPERIMENT,
            run_id=ccglobal.CCGLOBALPREMIUM_THRESHOLD_RUN_ID,
            run_name=ccglobal.CCGLOBALPREMIUM_THRESHOLD_RUN_NAME,
            phase_weights_fn=_weight_config_phase_weights(ccglobal.create_ccglobalpremium_threshold_weight_config),
        ),
        ValidationRunSpec(
            collection="validated_baseline",
            variant="ccglobalpremium_retainedtotal",
            source_experiment=ccglobal.CCGLOBALPREMIUM_SOURCE_EXPERIMENT,
            run_id=ccglobal.CCGLOBALPREMIUM_RETAINEDTOTAL_RUN_ID,
            run_name=ccglobal.CCGLOBALPREMIUM_RETAINEDTOTAL_RUN_NAME,
            phase_weights_fn=_weight_config_phase_weights(ccglobal.create_ccglobalpremium_retainedtotal_weight_config),
        ),
        ValidationRunSpec(
            collection="validated_baseline",
            variant="ccpairtotal_retainedtotal",
            source_experiment=ccpair.CCPAIRTOTAL_RETAINEDTOTAL_SOURCE_EXPERIMENT,
            run_id=ccpair.CCPAIRTOTAL_RETAINEDTOTAL_RUN_ID,
            run_name=ccpair.CCPAIRTOTAL_RETAINEDTOTAL_RUN_NAME,
            phase_weights_fn=_weight_config_phase_weights(ccpair.create_ccpairtotal_retainedtotal_weight_config),
        ),
        ValidationRunSpec(
            collection="validated_baseline",
            variant="genericfamily_tuned",
            source_experiment=gf_tuned.GENERICFAMILY_TUNED_SOURCE_EXPERIMENT,
            run_id=gf_tuned.GENERICFAMILY_TUNED_RUN_ID,
            run_name=gf_tuned.GENERICFAMILY_TUNED_RUN_NAME,
            phase_weights_fn=_weight_config_phase_weights(gf_tuned.create_genericfamily_tuned_weight_config),
        ),
        ValidationRunSpec(
            collection="validated_baseline",
            variant="genericfamily_no_retention",
            source_experiment=gf_no_retention.GENERICFAMILY_NO_RETENTION_SOURCE_EXPERIMENT,
            run_id=gf_no_retention.GENERICFAMILY_NO_RETENTION_RUN_ID,
            run_name=gf_no_retention.GENERICFAMILY_NO_RETENTION_RUN_NAME,
            phase_weights_fn=_weight_config_phase_weights(
                gf_no_retention.create_genericfamily_no_retention_weight_config
            ),
        ),
        ValidationRunSpec(
            collection="validated_baseline",
            variant="genericfamily_no_quality_splits_no_groups",
            source_experiment=gf_no_qs.GENERICFAMILY_NO_QUALITY_SPLITS_NO_GROUPS_SOURCE_EXPERIMENT,
            run_id=gf_no_qs.GENERICFAMILY_NO_QUALITY_SPLITS_NO_GROUPS_RUN_ID,
            run_name=gf_no_qs.GENERICFAMILY_NO_QUALITY_SPLITS_NO_GROUPS_RUN_NAME,
            phase_weights_fn=_weight_config_phase_weights(
                gf_no_qs.create_genericfamily_no_quality_splits_no_groups_weight_config
            ),
        ),
        ValidationRunSpec(
            collection="validated_baseline",
            variant="genericfamily_no_groups",
            source_experiment=gf_no_groups.GENERICFAMILY_NO_GROUPS_SOURCE_EXPERIMENT,
            run_id=gf_no_groups.GENERICFAMILY_NO_GROUPS_RUN_ID,
            run_name=gf_no_groups.GENERICFAMILY_NO_GROUPS_RUN_NAME,
            phase_weights_fn=_weight_config_phase_weights(gf_no_groups.create_genericfamily_no_groups_weight_config),
        ),
        ValidationRunSpec(
            collection="validated_baseline",
            variant="genericfamily_no_penalty",
            source_experiment=gf_no_penalty.GENERICFAMILY_NO_PENALTY_SOURCE_EXPERIMENT,
            run_id=gf_no_penalty.GENERICFAMILY_NO_PENALTY_RUN_ID,
            run_name=gf_no_penalty.GENERICFAMILY_NO_PENALTY_RUN_NAME,
            phase_weights_fn=_weight_config_phase_weights(gf_no_penalty.create_genericfamily_no_penalty_weight_config),
        ),
        ValidationRunSpec(
            collection="validated_baseline",
            variant="phasecomp_sparse_pls",
            source_experiment=phasecomp.PHASECOMP_SPARSE_PLS_SOURCE_EXPERIMENT,
            run_id=phasecomp.PHASECOMP_SPARSE_PLS_RUN_ID,
            run_name=phasecomp.PHASECOMP_SPARSE_PLS_RUN_NAME,
            phase_weights_fn=_weight_config_phase_weights(phasecomp.create_phasecomp_sparse_pls_weight_config),
        ),
        ValidationRunSpec(
            collection="validated_baseline",
            variant="genericfamily_lbfgsb",
            source_experiment=gf_lbfgsb.GENERICFAMILY_LBFGSB_SOURCE_EXPERIMENT,
            run_id=gf_lbfgsb.GENERICFAMILY_LBFGSB_RUN_ID,
            run_name=gf_lbfgsb.GENERICFAMILY_LBFGSB_RUN_NAME,
            phase_weights_fn=_weight_config_phase_weights(gf_lbfgsb.create_genericfamily_lbfgsb_weight_config),
        ),
        ValidationRunSpec(
            collection="validated_baseline",
            variant="genericfamily_powell",
            source_experiment=gf_powell.GENERICFAMILY_POWELL_SOURCE_EXPERIMENT,
            run_id=gf_powell.GENERICFAMILY_POWELL_RUN_ID,
            run_name=gf_powell.GENERICFAMILY_POWELL_RUN_NAME,
            phase_weights_fn=_weight_config_phase_weights(gf_powell.create_genericfamily_powell_weight_config),
        ),
        ValidationRunSpec(
            collection="validated_baseline",
            variant="genericfamily_recovered_hull",
            source_experiment=gf_recovered.GENERICFAMILY_RECOVERED_HULL_SOURCE_EXPERIMENT,
            run_id=gf_recovered.GENERICFAMILY_RECOVERED_HULL_RUN_ID,
            run_name=gf_recovered.GENERICFAMILY_RECOVERED_HULL_RUN_NAME,
            phase_weights_fn=_weight_config_phase_weights(
                gf_recovered.create_genericfamily_recovered_hull_weight_config
            ),
        ),
        ValidationRunSpec(
            collection="validated_baseline",
            variant="genericfamily_observed_only_trustblend",
            source_experiment=gf_trustblend.GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SOURCE_EXPERIMENT,
            run_id=gf_trustblend.GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_RUN_ID,
            run_name=gf_trustblend.GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_RUN_NAME,
            phase_weights_fn=_weight_config_phase_weights(
                gf_trustblend.create_genericfamily_observed_only_trustblend_weight_config
            ),
        ),
        ValidationRunSpec(
            collection="validated_baseline",
            variant="genericfamily_power_observed_only_trustblend",
            source_experiment=gf_power_trustblend.GENERICFAMILY_POWER_OBSERVED_ONLY_TRUSTBLEND_SOURCE_EXPERIMENT,
            run_id=gf_power_trustblend.GENERICFAMILY_POWER_OBSERVED_ONLY_TRUSTBLEND_RUN_ID,
            run_name=gf_power_trustblend.GENERICFAMILY_POWER_OBSERVED_ONLY_TRUSTBLEND_RUN_NAME,
            phase_weights_fn=_weight_config_phase_weights(
                gf_power_trustblend.create_genericfamily_power_observed_only_trustblend_weight_config
            ),
        ),
        ValidationRunSpec(
            collection="validated_baseline",
            variant="genericfamily_power_family_observed_only_trustblend",
            source_experiment=gf_family_curvature.GENERICFAMILY_FAMILY_CURVATURE_OBSERVED_ONLY_TRUSTBLEND_SOURCE_EXPERIMENT,
            run_id=gf_family_curvature.GENERICFAMILY_FAMILY_CURVATURE_VARIANT_SPECS[0].run_id,
            run_name=gf_family_curvature.GENERICFAMILY_FAMILY_CURVATURE_VARIANT_SPECS[0].run_name,
            phase_weights_fn=_weight_config_phase_weights(
                lambda: gf_family_curvature.create_genericfamily_family_curvature_observed_only_trustblend_weight_config(
                    "power_family"
                )
            ),
        ),
        ValidationRunSpec(
            collection="validated_baseline",
            variant="genericfamily_boxcox_family_observed_only_trustblend",
            source_experiment=gf_family_curvature.GENERICFAMILY_FAMILY_CURVATURE_OBSERVED_ONLY_TRUSTBLEND_SOURCE_EXPERIMENT,
            run_id=gf_family_curvature.GENERICFAMILY_FAMILY_CURVATURE_VARIANT_SPECS[1].run_id,
            run_name=gf_family_curvature.GENERICFAMILY_FAMILY_CURVATURE_VARIANT_SPECS[1].run_name,
            phase_weights_fn=_weight_config_phase_weights(
                lambda: gf_family_curvature.create_genericfamily_family_curvature_observed_only_trustblend_weight_config(
                    "boxcox_family"
                )
            ),
        ),
        ValidationRunSpec(
            collection="validated_baseline",
            variant="genericfamily_power_boxcox_family_observed_only_trustblend",
            source_experiment=gf_family_curvature.GENERICFAMILY_FAMILY_CURVATURE_OBSERVED_ONLY_TRUSTBLEND_SOURCE_EXPERIMENT,
            run_id=gf_family_curvature.GENERICFAMILY_FAMILY_CURVATURE_VARIANT_SPECS[2].run_id,
            run_name=gf_family_curvature.GENERICFAMILY_FAMILY_CURVATURE_VARIANT_SPECS[2].run_name,
            phase_weights_fn=_weight_config_phase_weights(
                lambda: gf_family_curvature.create_genericfamily_family_curvature_observed_only_trustblend_weight_config(
                    "power_boxcox_family"
                )
            ),
        ),
    ]

    for raw_variant in gf_penalty_raw.GENERICFAMILY_PENALTY_RAW_OPTIMUM_VARIANT_SPECS:
        specs.append(
            ValidationRunSpec(
                collection="validated_baseline",
                variant=f"genericfamily_{raw_variant.variant_name}_raw_optimum",
                source_experiment=gf_penalty_raw.GENERICFAMILY_PENALTY_RAW_OPTIMA_SOURCE_EXPERIMENT,
                run_id=raw_variant.run_id,
                run_name=raw_variant.run_name,
                phase_weights_fn=_weight_config_phase_weights(
                    lambda variant_name=raw_variant.variant_name: (
                        gf_penalty_raw.create_genericfamily_penalty_raw_optimum_weight_config(variant_name)
                    )
                ),
            )
        )

    for subset_size in gf_retuned_subset.GENERICFAMILY_RETUNED_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES:
        specs.append(
            ValidationRunSpec(
                collection="subset_validation",
                variant="genericfamily_subset_optimum",
                source_experiment=gf_subset.GENERICFAMILY_SUBSET_OPTIMA_SOURCE_EXPERIMENT,
                run_id=gf_subset.create_genericfamily_subset_optimum_weight_config(subset_size).run_id,
                run_name=gf_subset.genericfamily_subset_optimum_run_name(subset_size),
                subset_size=subset_size,
                phase_weights_fn=_subset_weight_config_phase_weights(
                    gf_subset.create_genericfamily_subset_optimum_weight_config,
                    subset_size,
                ),
            )
        )
        specs.append(
            ValidationRunSpec(
                collection="subset_validation",
                variant="genericfamily_retuned_subset_optimum",
                source_experiment=gf_retuned_subset.GENERICFAMILY_RETUNED_SUBSET_OPTIMA_SOURCE_EXPERIMENT,
                run_id=gf_retuned_subset.genericfamily_retuned_subset_optimum_run_id(subset_size),
                run_name=gf_retuned_subset.genericfamily_retuned_subset_optimum_run_name(subset_size),
                subset_size=subset_size,
                phase_weights_fn=_subset_weight_config_phase_weights(
                    gf_retuned_subset.create_genericfamily_retuned_subset_optimum_weight_config,
                    subset_size,
                ),
            )
        )
        specs.append(
            ValidationRunSpec(
                collection="subset_validation",
                variant="genericfamily_recovered_hull_subset_optimum",
                source_experiment=gf_recovered_subset.GENERICFAMILY_RECOVERED_HULL_SUBSET_OPTIMA_SOURCE_EXPERIMENT,
                run_id=gf_recovered_subset.genericfamily_recovered_hull_subset_optimum_run_id(subset_size),
                run_name=gf_recovered_subset.genericfamily_recovered_hull_subset_optimum_run_name(subset_size),
                subset_size=subset_size,
                phase_weights_fn=_subset_weight_config_phase_weights(
                    gf_recovered_subset.create_genericfamily_recovered_hull_subset_optimum_weight_config,
                    subset_size,
                ),
            )
        )
        specs.append(
            ValidationRunSpec(
                collection="subset_validation",
                variant="genericfamily_top8actual_hull_subset_optimum",
                source_experiment=gf_top8_subset.GENERICFAMILY_TOP8ACTUAL_HULL_SUBSET_OPTIMA_SOURCE_EXPERIMENT,
                run_id=gf_top8_subset.genericfamily_top8actual_hull_subset_optimum_run_id(subset_size),
                run_name=gf_top8_subset.genericfamily_top8actual_hull_subset_optimum_run_name(subset_size),
                subset_size=subset_size,
                phase_weights_fn=_subset_weight_config_phase_weights(
                    gf_top8_subset.create_genericfamily_top8actual_hull_subset_optimum_weight_config,
                    subset_size,
                ),
            )
        )

    for (
        subset_size
    ) in gf_trustblend_subset.GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES:
        specs.append(
            ValidationRunSpec(
                collection="subset_validation",
                variant="genericfamily_observed_only_trustblend_subset_optimum",
                source_experiment=gf_trustblend_subset.GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_SOURCE_EXPERIMENT,
                run_id=gf_trustblend_subset.genericfamily_observed_only_trustblend_subset_optimum_run_id(subset_size),
                run_name=gf_trustblend_subset.genericfamily_observed_only_trustblend_subset_optimum_run_name(
                    subset_size
                ),
                subset_size=subset_size,
                phase_weights_fn=_subset_weight_config_phase_weights(
                    gf_trustblend_subset.create_genericfamily_observed_only_trustblend_subset_optimum_weight_config,
                    subset_size,
                ),
            )
        )

    return tuple(specs)


def _base_collection(row: pd.Series) -> str:
    run_name = str(row["run_name"])
    source_experiment = str(row["source_experiment"])
    if run_name in observed_runs.CORE_BASELINE_RUN_NAMES:
        return "core_baseline"
    if source_experiment == observed_runs.ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT and SWARM_RUN_PATTERN.fullmatch(run_name):
        return "original_swarm"
    return "historical_baseline"


def _load_last_json_record(fs: fsspec.AbstractFileSystem, path: str) -> dict[str, float]:
    last_line: str | None = None
    with fs.open(path, "r") as f:
        for line in f:
            text = line.strip()
            if text:
                last_line = text
    if last_line is None:
        raise ValueError(f"No non-empty JSONL records found in {path}")
    payload = json.loads(last_line)
    return {str(key): value for key, value in payload.items()}


def _wandb_url(wandb_run_id: object) -> str | None:
    if wandb_run_id is None or pd.isna(wandb_run_id):
        return None
    run_id = str(wandb_run_id).strip()
    return None if not run_id else f"{WANDB_RUN_URL_PREFIX}{run_id}"


@cache
def _wandb_api() -> wandb.Api:
    return wandb.Api(timeout=60)


def _load_wandb_metric_summary(wandb_run_id: str) -> dict[str, float]:
    run = _wandb_api().run(f"marin-community/marin/{wandb_run_id}")
    metrics: dict[str, float] = {}
    for key, value in run.summary.items():
        if not any(key.startswith(prefix) for prefix in METRIC_PREFIXES):
            continue
        if not isinstance(value, int | float):
            continue
        metrics[str(key)] = float(value)
    return metrics


def _resolve_eval_metrics_path(fs: fsspec.AbstractFileSystem, spec: ValidationRunSpec) -> str | None:
    pattern = f"{CHECKPOINT_ROOT}/{spec.source_experiment}/{spec.run_name}-*/checkpoints/eval_metrics.jsonl"
    matches = sorted(fs.glob(pattern))
    return None if not matches else matches[-1]


def _validation_row(
    fs: fsspec.AbstractFileSystem,
    spec: ValidationRunSpec,
    *,
    broad_run_id: int,
) -> dict[str, object] | None:
    metrics_path = _resolve_eval_metrics_path(fs, spec)
    if metrics_path is None:
        return None

    checkpoint_root = "/".join(metrics_path.split("/")[:-2])
    wandb_run_id = checkpoint_root.split("/")[-1]
    metrics = _load_last_json_record(fs, metrics_path)
    metrics.update(_load_wandb_metric_summary(wandb_run_id))
    phase_weights = spec.phase_weights_fn()

    row: dict[str, object] = {
        "broad_run_id": broad_run_id,
        "wandb_run_id": wandb_run_id,
        "wandb_url": _wandb_url(wandb_run_id),
        "source_experiment": spec.source_experiment,
        "run_id": spec.run_id,
        "run_name": spec.run_name,
        "status": "completed",
        "broad_collection": spec.collection,
        "broad_variant": spec.variant,
        "validated_subset_size": spec.subset_size,
        "checkpoint_eval_metrics_path": f"gs://{metrics_path}",
    }
    for phase_name, domain_weights in phase_weights.items():
        for domain_name, weight in domain_weights.items():
            row[f"{phase_name}_{domain_name}"] = float(weight)
    row.update(metrics)
    return row


def main() -> None:
    base = pd.read_csv(BASE_CSV)
    base = base.copy()
    base["broad_collection"] = base.apply(_base_collection, axis=1)
    base["broad_variant"] = pd.NA
    base["validated_subset_size"] = pd.NA
    base["checkpoint_eval_metrics_path"] = pd.NA
    base["wandb_url"] = base["wandb_run_id"].map(_wandb_url)

    fs = fsspec.filesystem("gs")
    existing = {(str(row.source_experiment), str(row.run_name)) for row in base.itertuples(index=False)}

    extra_rows: list[dict[str, object]] = []
    missing_specs: list[dict[str, object]] = []
    broad_run_id = len(base)
    for spec in _extra_validation_specs():
        if (spec.source_experiment, spec.run_name) in existing:
            continue
        row = _validation_row(fs, spec, broad_run_id=broad_run_id)
        if row is None:
            missing_specs.append(
                {
                    "collection": spec.collection,
                    "variant": spec.variant,
                    "source_experiment": spec.source_experiment,
                    "run_name": spec.run_name,
                    "subset_size": spec.subset_size,
                }
            )
            continue
        extra_rows.append(row)
        broad_run_id += 1

    if extra_rows:
        extra_frame = pd.DataFrame(extra_rows).dropna(axis=1, how="all")
        broad = pd.concat([base, extra_frame], ignore_index=True, sort=False)
    else:
        broad = base.copy()
    if "broad_run_id" not in broad.columns:
        broad["broad_run_id"] = np.arange(len(broad), dtype=int)
    broad["broad_run_id"] = broad["broad_run_id"].fillna(pd.Series(np.arange(len(broad), dtype=int))).astype(int)
    broad = broad.sort_values(["broad_run_id", "run_name"], kind="stable").reset_index(drop=True)
    broad.to_csv(OUTPUT_CSV, index=False)

    summary = {
        "base_rows": len(base),
        "extra_completed_rows": len(extra_rows),
        "total_rows": len(broad),
        "counts_by_collection": {str(k): int(v) for k, v in broad["broad_collection"].value_counts().items()},
        "added_run_names": [str(row["run_name"]) for row in extra_rows],
        "missing_specs": missing_specs,
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
