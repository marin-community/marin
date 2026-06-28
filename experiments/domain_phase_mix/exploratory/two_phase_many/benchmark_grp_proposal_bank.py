# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas"]
# ///
"""Benchmark proposal-bank deployment regularization for observed-only GRP."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.exploratory.two_phase_many.dataset_metadata import (
    load_two_phase_many_candidate_summary_spec,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GenericFamilyRetainedTotalSurrogate,
    load_generic_family_packet,
)
from experiments.domain_phase_mix.static_batch_selection import (
    average_phase_tv_distance,
    prospective_generic_selection,
    replay_proposals_to_observed,
    retrospective_generic_selection,
    sobol_weight_configs,
    weight_configs_to_tensor,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    create_two_phase_dolma3_dolmino_top_level_experiment,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_observed_only_trustblend_subset_optima import (
    CSV_PATH,
    GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES,
    OBJECTIVE_METRIC,
    _phase_weights_from_array,
    _subset_packet,
    deploy_genericfamily_trustblend_topkactual,
)

plt.rcParams["text.usetex"] = False

SCRIPT_DIR = Path(__file__).resolve().parent
DETAIL_CSV = SCRIPT_DIR / "two_phase_many_grp_proposal_bank_curve_points.csv"
SUMMARY_JSON = SCRIPT_DIR / "two_phase_many_grp_proposal_bank_summary.json"
PLOT_PATH = SCRIPT_DIR / "two_phase_many_grp_proposal_bank_variants.png"
TRUSTBLEND_CONVERGENCE_CSV = SCRIPT_DIR / "two_phase_many_grp_observed_only_trustblend_top8actual_cap_curve_points.csv"


@dataclass(frozen=True)
class ProposalBankVariant:
    """One proposal-bank deployment variant."""

    name: str
    label: str
    kind: str
    pool_size: int | None = None
    thin_size: int | None = None
    exclusion: str = "subset"
    locality_source: str | None = None
    locality_topk: int | None = None
    tv_cap: float | None = None


VARIANTS: tuple[ProposalBankVariant, ...] = (
    ProposalBankVariant(
        name="trustblend_top8actual_cap",
        label="Trustblend",
        kind="trustblend",
    ),
    ProposalBankVariant(
        name="sobol_argmin_pool512_subset",
        label="Sobol argmin (512)",
        kind="sobol_argmin",
        pool_size=512,
        exclusion="subset",
    ),
    ProposalBankVariant(
        name="sobol_fbl_top8_pool512_subset",
        label="Sobol -> FBL-8 (512)",
        kind="fbl_argmin",
        pool_size=512,
        thin_size=8,
        exclusion="subset",
    ),
    ProposalBankVariant(
        name="sobol_fbl_top16_pool2048_subset",
        label="Sobol -> FBL-16 (2048)",
        kind="fbl_argmin",
        pool_size=2048,
        thin_size=16,
        exclusion="subset",
    ),
    ProposalBankVariant(
        name="sobol_fbl_top16_pool2048_all_observed",
        label="Sobol -> FBL-16 (all-observed exclusion)",
        kind="fbl_argmin",
        pool_size=2048,
        thin_size=16,
        exclusion="all_observed",
    ),
    ProposalBankVariant(
        name="sobol_local_trustblend_cap015_pool2048_subset",
        label="Local Sobol (trustblend, TV<=0.15)",
        kind="local_sobol_argmin",
        pool_size=2048,
        exclusion="subset",
        locality_source="trustblend",
        tv_cap=0.15,
    ),
    ProposalBankVariant(
        name="sobol_local_top4actual_cap015_pool2048_subset",
        label="Local Sobol (top4 actual, TV<=0.15)",
        kind="local_sobol_argmin",
        pool_size=2048,
        exclusion="subset",
        locality_source="top_actual",
        locality_topk=4,
        tv_cap=0.15,
    ),
    ProposalBankVariant(
        name="sobol_local_top4plustrustblend_cap015_pool2048_subset",
        label="Local Sobol (top4 + trustblend, TV<=0.15)",
        kind="local_sobol_argmin",
        pool_size=2048,
        exclusion="subset",
        locality_source="top_actual_plus_trustblend",
        locality_topk=4,
        tv_cap=0.15,
    ),
)

LOCAL_PROPOSAL_MIN_KEEP = 64

TUNING_PARAM_KEYS = ("alpha", "eta", "lam", "tau", "reg", "beta")
TUNING_METRIC_KEYS = ("tuning_objective", "tuning_cv_rmse", "tuning_cv_r2", "tuning_cv_regret_at_1")
OBSERVED_ONLY_TUNING_KEYS = (*TUNING_PARAM_KEYS, "tuning_cv_foldmean_regret_at_1", "tuning_lower_tail_optimism")


def _weight_config_from_weights(
    *,
    run_id: int,
    domain_names: list[str],
    weights: np.ndarray,
) -> WeightConfig:
    return WeightConfig(
        run_id=run_id,
        phase_weights=_phase_weights_from_array(domain_names, np.asarray(weights, dtype=float)),
    )


@cache
def _cached_trustblend_tuning_frame() -> pd.DataFrame:
    frame = pd.read_csv(TRUSTBLEND_CONVERGENCE_CSV)
    subset_frame = frame[
        (frame["variant"] == "trustblend_top8actual_cap") & (frame["tuning_procedure"] == "observed_only")
    ].copy()
    if subset_frame.empty:
        raise ValueError(f"No observed-only trustblend tuning rows found in {TRUSTBLEND_CONVERGENCE_CSV}")
    return subset_frame.set_index("subset_size")


def _observed_only_tuning_for_subset(subset_size: int) -> dict[str, float]:
    tuning_row = _cached_trustblend_tuning_frame().loc[subset_size]
    metrics = {
        key: float(tuning_row[key])
        for key in (*TUNING_PARAM_KEYS, "tuning_objective", "tuning_cv_rmse", "tuning_cv_r2", "tuning_cv_regret_at_1")
    }
    metrics["tuning_cv_foldmean_regret_at_1"] = float(tuning_row["tuning_cv_foldmean_regret_at_1"])
    metrics["tuning_lower_tail_optimism"] = float(tuning_row["tuning_lower_tail_optimism"])
    metrics["objective"] = float(tuning_row["tuning_objective"])
    metrics["cv_rmse"] = float(tuning_row["tuning_cv_rmse"])
    metrics["cv_r2"] = float(tuning_row["tuning_cv_r2"])
    metrics["cv_regret_at_1"] = float(tuning_row["tuning_cv_regret_at_1"])
    metrics["cv_foldmean_regret_at_1"] = float(tuning_row["tuning_cv_foldmean_regret_at_1"])
    metrics["lower_tail_optimism"] = float(tuning_row["tuning_lower_tail_optimism"])
    return metrics


@cache
def _sampling_experiment():
    return create_two_phase_dolma3_dolmino_top_level_experiment(
        name="two_phase_many_grp_proposal_bank_benchmark",
        eval_harness_tasks=(),
    )


def _proposal_configs_for_variant(
    variant: ProposalBankVariant,
    *,
    train_weights: np.ndarray,
    train_spec,
    packet,
    max_proposals: int,
    trustblend_weights: np.ndarray | None = None,
    actual_order: np.ndarray | None = None,
) -> tuple[list[WeightConfig], dict[str, float]]:
    experiment = _sampling_experiment()
    if variant.exclusion == "subset":
        existing_configs = [
            _weight_config_from_weights(run_id=int(idx), domain_names=packet.base.domain_names, weights=weights)
            for idx, weights in zip(range(len(train_weights)), train_weights, strict=True)
        ]
    elif variant.exclusion == "all_observed":
        existing_configs = [
            _weight_config_from_weights(run_id=int(idx), domain_names=packet.base.domain_names, weights=weights)
            for idx, weights in enumerate(packet.base.w)
        ]
    else:
        raise ValueError(f"Unsupported exclusion mode: {variant.exclusion}")

    if variant.kind == "trustblend":
        return [], {}
    if variant.kind == "sobol_argmin":
        if variant.pool_size is None:
            raise ValueError("sobol_argmin variant requires pool_size")
        proposals = sobol_weight_configs(
            experiment,
            n=variant.pool_size,
            seed=0,
            existing_configs=existing_configs,
            min_accepted=variant.pool_size,
        )
        diagnostics = {
            "candidate_pool_requested": float(variant.pool_size),
            "candidate_pool_size": float(len(proposals)),
        }
        return proposals, diagnostics
    if variant.kind == "local_sobol_argmin":
        if variant.pool_size is None or variant.tv_cap is None or variant.locality_source is None:
            raise ValueError("local_sobol_argmin variant requires pool_size, tv_cap, and locality_source")
        max_keep = min(max_proposals, variant.pool_size)
        proposals = sobol_weight_configs(
            experiment,
            n=variant.pool_size,
            seed=0,
            existing_configs=existing_configs,
            min_accepted=variant.pool_size,
        )
        proposal_weights = weight_configs_to_tensor(
            proposals,
            phase_names=train_spec.phase_names,
            domain_names=packet.base.domain_names,
        )
        if variant.locality_source == "trustblend":
            if trustblend_weights is None:
                raise ValueError("trustblend_weights required for trustblend-local proposals")
            anchor_weights = trustblend_weights[None, :, :]
        elif variant.locality_source == "top_actual":
            if actual_order is None or variant.locality_topk is None:
                raise ValueError("actual_order and locality_topk required for top_actual-local proposals")
            anchor_weights = train_weights[actual_order[: min(int(variant.locality_topk), len(actual_order))]]
        elif variant.locality_source == "top_actual_plus_trustblend":
            if trustblend_weights is None or actual_order is None or variant.locality_topk is None:
                raise ValueError(
                    "trustblend_weights, actual_order, and locality_topk required for mixed-local proposals"
                )
            anchor_weights = np.concatenate(
                [
                    train_weights[actual_order[: min(int(variant.locality_topk), len(actual_order))]],
                    trustblend_weights[None, :, :],
                ],
                axis=0,
            )
        else:
            raise ValueError(f"Unsupported locality_source: {variant.locality_source}")

        anchor_distances = np.column_stack(
            [
                average_phase_tv_distance(proposal_weights, anchor_weights[idx : idx + 1])
                for idx in range(len(anchor_weights))
            ]
        )
        min_anchor_tv = anchor_distances.min(axis=1)
        kept_indices = np.flatnonzero(min_anchor_tv <= float(variant.tv_cap))
        cap_relaxed = False
        min_keep = min(LOCAL_PROPOSAL_MIN_KEEP, max_keep, len(proposals))
        if len(kept_indices) < min_keep:
            kept_indices = np.argsort(min_anchor_tv)[:min_keep]
            cap_relaxed = True
        elif len(kept_indices) > max_keep:
            kept_indices = kept_indices[np.argsort(min_anchor_tv[kept_indices])[:max_keep]]
        filtered_proposals = [proposals[int(idx)] for idx in kept_indices.tolist()]
        diagnostics = {
            "candidate_pool_requested": float(variant.pool_size),
            "candidate_pool_size": float(len(proposals)),
            "local_anchor_count": float(len(anchor_weights)),
            "local_tv_cap": float(variant.tv_cap),
            "local_kept_count": float(len(filtered_proposals)),
            "local_cap_relaxed": float(cap_relaxed),
            "local_mean_min_anchor_tv": float(min_anchor_tv[kept_indices].mean()),
            "local_max_min_anchor_tv": float(min_anchor_tv[kept_indices].max()),
        }
        return filtered_proposals, diagnostics
    if variant.kind == "fbl_argmin":
        if variant.pool_size is None or variant.thin_size is None:
            raise ValueError("fbl_argmin variant requires pool_size and thin_size")
        thin_size = min(int(variant.thin_size), max_proposals)
        proposals, selection = prospective_generic_selection(
            train_spec,
            experiment,
            method="feature_bayes_linear",
            n_select=thin_size,
            seed=0,
            pool_size=variant.pool_size,
            existing_configs=existing_configs,
        )
        diagnostics = dict(selection.diagnostics)
        diagnostics["candidate_pool_requested"] = float(variant.pool_size)
        diagnostics["candidate_pool_size"] = float(len(proposals))
        diagnostics["thinned_candidate_count"] = float(len(proposals))
        return proposals, diagnostics
    raise ValueError(f"Unsupported variant kind: {variant.kind}")


def _replay_single_deployment(
    deployment_weights: np.ndarray,
    *,
    holdout_weights: np.ndarray,
    holdout_indices: np.ndarray,
    packet,
) -> dict[str, object]:
    replay = replay_proposals_to_observed(deployment_weights[None, :, :], holdout_weights)
    holdout_local_idx = int(replay.selected_indices[0])
    holdout_global_idx = int(holdout_indices[holdout_local_idx])
    distance = float(
        average_phase_tv_distance(
            holdout_weights[holdout_local_idx : holdout_local_idx + 1],
            deployment_weights[None, :, :],
        )[0]
    )
    return {
        "chosen_holdout_local_idx": holdout_local_idx,
        "chosen_holdout_global_idx": holdout_global_idx,
        "chosen_holdout_run_name": str(packet.base.frame.iloc[holdout_global_idx][packet.base.name_col]),
        "chosen_holdout_value": float(packet.base.y[holdout_global_idx]),
        "chosen_holdout_distance": distance,
        "proposal_count": 1,
        "bank_oracle_best_holdout_value": float(packet.base.y[holdout_global_idx]),
        "bank_oracle_gap": 0.0,
        "mean_replay_distance": float(replay.mean_distance),
        "max_replay_distance": float(replay.max_distance),
    }


def _replay_proposal_bank(
    proposals: list[WeightConfig],
    proposal_predictions: np.ndarray,
    *,
    phase_names: list[str],
    domain_names: list[str],
    holdout_weights: np.ndarray,
    holdout_indices: np.ndarray,
    packet,
) -> dict[str, object]:
    proposal_weights = weight_configs_to_tensor(
        proposals,
        phase_names=phase_names,
        domain_names=domain_names,
    )
    replay = replay_proposals_to_observed(proposal_weights, holdout_weights)
    replay_local = np.asarray(replay.selected_indices, dtype=int)
    replay_global = holdout_indices[replay_local]
    replay_values = packet.base.y[replay_global]
    chosen_proposal_idx = int(np.argmin(proposal_predictions))
    chosen_local_idx = int(replay_local[chosen_proposal_idx])
    chosen_global_idx = int(replay_global[chosen_proposal_idx])
    chosen_distance = float(
        average_phase_tv_distance(
            holdout_weights[chosen_local_idx : chosen_local_idx + 1],
            proposal_weights[chosen_proposal_idx : chosen_proposal_idx + 1],
        )[0]
    )
    chosen_rank = int(np.argsort(np.argsort(replay_values))[chosen_proposal_idx]) + 1
    oracle_best_value = float(np.min(replay_values))
    return {
        "chosen_holdout_local_idx": chosen_local_idx,
        "chosen_holdout_global_idx": chosen_global_idx,
        "chosen_holdout_run_name": str(packet.base.frame.iloc[chosen_global_idx][packet.base.name_col]),
        "chosen_holdout_value": float(packet.base.y[chosen_global_idx]),
        "chosen_holdout_distance": chosen_distance,
        "proposal_count": len(proposals),
        "proposal_predicted_value": float(proposal_predictions[chosen_proposal_idx]),
        "chosen_replay_rank_within_bank": chosen_rank,
        "bank_oracle_best_holdout_value": oracle_best_value,
        "bank_oracle_gap": float(packet.base.y[chosen_global_idx] - oracle_best_value),
        "mean_replay_distance": float(replay.mean_distance),
        "max_replay_distance": float(replay.max_distance),
    }


def main() -> None:
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    _, spec, _ = load_two_phase_many_candidate_summary_spec(
        CSV_PATH,
        objective_metric=OBJECTIVE_METRIC,
        name="two_phase_many_grp_proposal_bank_benchmark",
    )
    best_global_value = float(np.min(packet.base.y))
    best_global_idx = int(np.argmin(packet.base.y))

    rows: list[dict[str, object]] = []

    for subset_size in GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES:
        selection = retrospective_generic_selection(spec, method="feature_bayes_linear", k=subset_size, seed=0)
        subset_indices = np.asarray(selection.selected_indices, dtype=int)
        holdout_indices = np.asarray(sorted(set(range(len(packet.base.y))) - set(subset_indices.tolist())), dtype=int)
        if len(holdout_indices) == 0:
            raise ValueError(f"Subset size {subset_size} leaves no holdout rows")

        train_packet = _subset_packet(packet, subset_indices)
        train_spec = spec.subset(subset_indices)
        actual_order = np.argsort(train_packet.base.y)
        tuning_metrics = _observed_only_tuning_for_subset(subset_size)
        tuned_params = {key: tuning_metrics[key] for key in TUNING_PARAM_KEYS}
        model = GenericFamilyRetainedTotalSurrogate(train_packet, params=tuned_params).fit(
            train_packet.base.w,
            train_packet.base.y,
        )
        trustblend = deploy_genericfamily_trustblend_topkactual(train_packet, model, tuning_metrics)
        trustblend_weights = np.asarray(trustblend["weights"], dtype=float)
        best_holdout_local_idx = int(np.argmin(packet.base.y[holdout_indices]))
        best_holdout_global_idx = int(holdout_indices[best_holdout_local_idx])
        best_holdout_value = float(packet.base.y[best_holdout_global_idx])
        subset_best_idx = int(subset_indices[np.argmin(packet.base.y[subset_indices])])

        for variant in VARIANTS:
            if variant.kind == "trustblend":
                replay_summary = _replay_single_deployment(
                    trustblend_weights,
                    holdout_weights=packet.base.w[holdout_indices],
                    holdout_indices=holdout_indices,
                    packet=packet,
                )
                predicted_value = float(trustblend["predicted_optimum_value"])
                diagnostics: dict[str, float] = {
                    "deployment_delta": float(trustblend["delta"]),
                    "deployment_gain_budget": float(trustblend["gain_budget"]),
                    "deployment_raw_predicted_optimum_value": float(trustblend["raw_predicted_optimum_value"]),
                    "deployment_hull_predicted_optimum_value": float(trustblend["hull_predicted_optimum_value"]),
                }
                chosen_weights = trustblend_weights
            else:
                proposals, diagnostics = _proposal_configs_for_variant(
                    variant,
                    train_weights=train_packet.base.w,
                    train_spec=train_spec,
                    packet=packet,
                    max_proposals=len(holdout_indices),
                    trustblend_weights=trustblend_weights,
                    actual_order=actual_order,
                )
                proposal_weights = weight_configs_to_tensor(
                    proposals,
                    phase_names=spec.phase_names,
                    domain_names=packet.base.domain_names,
                )
                proposal_predictions = np.asarray(model.predict(proposal_weights), dtype=float)
                if variant.kind == "sobol_argmin":
                    chosen_idx = int(np.argmin(proposal_predictions))
                    replay_summary = _replay_single_deployment(
                        proposal_weights[chosen_idx],
                        holdout_weights=packet.base.w[holdout_indices],
                        holdout_indices=holdout_indices,
                        packet=packet,
                    )
                    replay_summary["proposal_count"] = len(proposals)
                else:
                    replay_summary = _replay_proposal_bank(
                        proposals,
                        proposal_predictions,
                        phase_names=spec.phase_names,
                        domain_names=packet.base.domain_names,
                        holdout_weights=packet.base.w[holdout_indices],
                        holdout_indices=holdout_indices,
                        packet=packet,
                    )
                predicted_value = float(np.min(proposal_predictions))
                chosen_weights = proposal_weights[int(np.argmin(proposal_predictions))]

            rows.append(
                {
                    "subset_size": subset_size,
                    "variant": variant.name,
                    "variant_label": variant.label,
                    "variant_kind": variant.kind,
                    "variant_exclusion": variant.exclusion,
                    "predicted_value": predicted_value,
                    "chosen_holdout_run_name": replay_summary["chosen_holdout_run_name"],
                    "chosen_holdout_value": replay_summary["chosen_holdout_value"],
                    "chosen_holdout_regret_at_1": float(replay_summary["chosen_holdout_value"]) - best_holdout_value,
                    "chosen_global_regret_at_1": float(replay_summary["chosen_holdout_value"]) - best_global_value,
                    "chosen_holdout_distance": replay_summary["chosen_holdout_distance"],
                    "chosen_nearest_full_observed_tv": float(
                        np.min(average_phase_tv_distance(packet.base.w, chosen_weights[None, :, :]))
                    ),
                    "proposal_count": replay_summary["proposal_count"],
                    "bank_oracle_best_holdout_value": replay_summary["bank_oracle_best_holdout_value"],
                    "bank_oracle_gap": replay_summary["bank_oracle_gap"],
                    "mean_replay_distance": replay_summary["mean_replay_distance"],
                    "max_replay_distance": replay_summary["max_replay_distance"],
                    "best_holdout_run_name": str(packet.base.frame.iloc[best_holdout_global_idx][packet.base.name_col]),
                    "best_holdout_value": best_holdout_value,
                    "subset_best_observed_run_name": str(packet.base.frame.iloc[subset_best_idx][packet.base.name_col]),
                    "subset_best_observed_value": float(packet.base.y[subset_best_idx]),
                    "best_global_run_name": str(packet.base.frame.iloc[best_global_idx][packet.base.name_col]),
                    "best_global_value": best_global_value,
                    "tuning_cv_rmse": float(tuning_metrics["cv_rmse"]),
                    "tuning_cv_foldmean_regret_at_1": float(tuning_metrics["cv_foldmean_regret_at_1"]),
                    "tuning_lower_tail_optimism": float(tuning_metrics["lower_tail_optimism"]),
                    **diagnostics,
                    **{
                        key: replay_summary[key]
                        for key in replay_summary
                        if key not in {"chosen_holdout_run_name", "chosen_holdout_value"}
                    },
                }
            )

    frame = pd.DataFrame(rows).sort_values(["variant", "subset_size"]).reset_index(drop=True)
    frame.to_csv(DETAIL_CSV, index=False)

    summary: dict[str, dict[str, float | int]] = {}
    for variant in VARIANTS:
        variant_rows = frame[frame["variant"] == variant.name]
        later_rows = variant_rows[variant_rows["subset_size"] >= 80]
        summary[variant.name] = {
            "mean_predicted_value_after80": float(later_rows["predicted_value"].mean()),
            "mean_holdout_value_after80": float(later_rows["chosen_holdout_value"].mean()),
            "mean_holdout_regret_after80": float(later_rows["chosen_holdout_regret_at_1"].mean()),
            "mean_global_regret_after80": float(later_rows["chosen_global_regret_at_1"].mean()),
            "num_zero_holdout_regret_after80": int((later_rows["chosen_holdout_regret_at_1"] == 0.0).sum()),
            "mean_nearest_full_observed_tv_after80": float(later_rows["chosen_nearest_full_observed_tv"].mean()),
            "mean_bank_oracle_gap_after80": float(later_rows["bank_oracle_gap"].mean()),
        }
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "objective_metric": OBJECTIVE_METRIC,
                "detail_csv": str(DETAIL_CSV),
                "variants": summary,
            },
            indent=2,
        )
    )

    cmap = plt.colormaps["RdYlGn_r"]
    color_positions = np.linspace(0.15, 0.85, num=len(VARIANTS))
    color_map = {variant.name: cmap(position) for variant, position in zip(VARIANTS, color_positions, strict=True)}
    label_map = {variant.name: variant.label for variant in VARIANTS}

    fig, (ax_bpb, ax_regret, ax_dist) = plt.subplots(
        3,
        1,
        figsize=(10.5, 9.0),
        dpi=180,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.3, 1.0, 1.0], "hspace": 0.08},
    )

    holdout_reference = frame.drop_duplicates("subset_size").sort_values("subset_size")
    ax_bpb.plot(
        holdout_reference["subset_size"],
        holdout_reference["best_holdout_value"],
        color="#4C78A8",
        linewidth=2.0,
        linestyle=":",
        marker="P",
        label="Best holdout observed BPB",
    )

    for variant in VARIANTS:
        variant_rows = frame[frame["variant"] == variant.name].sort_values("subset_size")
        color = color_map[variant.name]
        ax_bpb.plot(
            variant_rows["subset_size"],
            variant_rows["chosen_holdout_value"],
            color=color,
            marker="o",
            linewidth=2.0,
            label=label_map[variant.name],
        )
        ax_regret.plot(
            variant_rows["subset_size"],
            variant_rows["chosen_holdout_regret_at_1"],
            color=color,
            marker="s",
            linewidth=2.0,
        )
        ax_dist.plot(
            variant_rows["subset_size"],
            variant_rows["chosen_nearest_full_observed_tv"],
            color=color,
            marker="D",
            linewidth=2.0,
        )

    ax_regret.axhline(0.0, color="0.55", linewidth=1.0, linestyle=":")

    ax_bpb.set_title("Two-phase many-domain: GRP proposal-bank regularization benchmark")
    ax_bpb.set_ylabel("Replayed holdout BPB")
    ax_regret.set_ylabel("Holdout Regret@1")
    ax_dist.set_ylabel("Nearest full-cloud TV")
    ax_dist.set_xlabel("Observed runs used for fitting")
    ax_dist.set_xticks(sorted(frame["subset_size"].unique().tolist()))

    for axis in (ax_bpb, ax_regret, ax_dist):
        axis.grid(True, alpha=0.25)

    ax_bpb.legend(loc="best", fontsize=8)
    fig.savefig(PLOT_PATH, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
