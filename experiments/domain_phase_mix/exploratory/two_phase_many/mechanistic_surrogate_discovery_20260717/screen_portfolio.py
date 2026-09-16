# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Run the predeclared cheap-falsification screen for mechanistic families."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717.freeze_baseline_gate import (  # noqa: E402
    assert_sealed_absent,
    metrics,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717.mechanistic_models import (  # noqa: E402
    ModelConfig,
    Panel,
    build_design,
    candidate_configs,
    fit_nonnegative_ridge,
    record,
    round2_candidate_configs,
    round3_dynamics_candidate_configs,
    round4_foundation_candidate_configs,
    round5_prior_candidate_configs,
    round8_bounded_coverage_candidate_configs,
    round9_ces_candidate_configs,
    round10_replay_hazard_candidate_configs,
    round16_plasticity_candidate_configs,
    round17_gradient_noise_candidate_configs,
    round18_parallel_reliability_candidate_configs,
    round19_posterior_precision_candidate_configs,
    round20_capacity_gated_candidate_configs,
    round21_finite_subset_candidate_configs,
    round23_power_law_memory_candidate_configs,
    round24_riccati_uncertainty_candidate_configs,
    round25_two_pool_consolidation_candidate_configs,
    round26_concentration_displacement_candidate_configs,
    round27_diversity_gated_candidate_configs,
    round28_learned_state_competition_candidate_configs,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
DASHBOARD = RESEARCH_DIR / "mixture_fit_debugger/src/generated/dashboard_data.json"
DEFAULT_OUTPUT_ROOT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717"
L2_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0)
PANEL_IDS = (
    "300m_uncheatable",
    "300m_table9",
    "delphi_3e18_uncheatable",
    "delphi_3e18_table9",
    "production_uncheatable",
    "starcoder_cosine_starcoder_bpb",
    "starcoder_wsd80_starcoder_bpb",
)


def split_panel_id(panel_id: str) -> tuple[str, str]:
    for swarm in ("starcoder_cosine", "starcoder_wsd80", "delphi_3e18", "production", "300m"):
        prefix = f"{swarm}_"
        if panel_id.startswith(prefix):
            return swarm, panel_id.removeprefix(prefix)
    raise ValueError(f"Unknown panel ID {panel_id!r}")


def load_raw_dataset(swarm: str, target: str) -> Any:
    if swarm == "300m":
        return observatory.pooled.load_300m_dataset(target)
    if swarm == "delphi_3e18":
        return observatory.load_delphi_3e18_fit_dataset(target)
    if swarm == "production":
        if target != "uncheatable":
            raise ValueError(target)
        return observatory.pooled.load_production_dataset()
    cosine = observatory.load_cosine_starcoder()
    if swarm == "starcoder_cosine":
        return cosine
    if swarm == "starcoder_wsd80":
        return observatory.load_wsd80_starcoder(cosine)
    raise ValueError(swarm)


def hierarchical_partition(
    domains: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[np.ndarray, ...], tuple[str, ...], tuple[np.ndarray, ...], np.ndarray]:
    if len(domains) == 39 and "dolma3_arxiv" in domains:
        tech = {
            "dolma3_arxiv",
            "dolma3_finemath_3plus",
            "dolma3_stack_edu",
            "dolmino_stack_edu_fim",
            "dolmino_synth_code",
            "dolmino_synth_math",
        }
        reasoning = {"dolmino_synth_instruction", "dolmino_synth_thinking"}
        groups = {
            "broad_text": [index for index, domain in enumerate(domains) if domain not in tech | reasoning],
            "tech_code": [index for index, domain in enumerate(domains) if domain in tech],
            "reasoning": [index for index, domain in enumerate(domains) if domain in reasoning],
        }
    elif all(re.fullmatch(r"c\d+q\d+|tail", domain) for domain in domains):
        groups: dict[str, list[int]] = {}
        for index, domain in enumerate(domains):
            match = re.fullmatch(r"(c\d+)q\d+", domain)
            family = match.group(1) if match is not None else domain
            groups.setdefault(family, []).append(index)
    else:
        groups = {domain: [index] for index, domain in enumerate(domains)}
    family_names = tuple(groups)
    family_members = tuple(np.asarray(groups[name], dtype=int) for name in family_names)
    if sorted(np.concatenate(family_members).tolist()) != list(range(len(domains))):
        raise ValueError("Family partition does not cover domains exactly once")

    if len(domains) == 39 and "dolma3_arxiv" in domains:
        group_map: dict[str, list[int]] = {}
        for index, domain in enumerate(domains):
            match = re.fullmatch(r"dolma3_cc/(.+)_(high|low)", domain)
            group = f"cc:{match.group(1)}" if match is not None else domain
            group_map.setdefault(group, []).append(index)
    else:
        group_map = {domain: [index] for index, domain in enumerate(domains)}
    group_names = tuple(group_map)
    group_members = tuple(np.asarray(group_map[name], dtype=int) for name in group_names)
    domain_to_family = np.empty(len(domains), dtype=int)
    for family_index, family in enumerate(family_members):
        domain_to_family[family] = family_index
    group_family_indices = np.asarray([domain_to_family[members[0]] for members in group_members], dtype=int)
    if any(
        np.any(domain_to_family[members] != group_family_indices[index]) for index, members in enumerate(group_members)
    ):
        raise ValueError("A structural group crosses semantic families")
    return family_names, family_members, group_names, group_members, group_family_indices


def dashboard_fit_rows(bundle: dict[str, Any], swarm: str) -> list[dict[str, Any]]:
    return [row for row in bundle["swarms"][swarm]["rows"] if row["split"] == "fit"]


def load_panel(bundle: dict[str, Any], panel_id: str) -> tuple[Panel, Any]:
    swarm, target = split_panel_id(panel_id)
    dataset = load_raw_dataset(swarm, target)
    rows = dashboard_fit_rows(bundle, swarm)
    dashboard_weights = np.asarray([[row["phase0"], row["phase1"]] for row in rows], dtype=float)
    dashboard_observed = np.asarray([row["observed"][target] for row in rows], dtype=float)
    if dashboard_weights.shape != dataset.weights.shape or not np.allclose(
        dashboard_weights, dataset.weights, atol=2e-7
    ):
        raise ValueError(f"{panel_id}: raw and frozen-dashboard weights disagree")
    target_disagreement = np.flatnonzero(~np.isclose(dashboard_observed, dataset.y, atol=2e-7))
    if target_disagreement.size:
        expected_repeat_mean_override = (
            swarm == "300m" and target_disagreement.tolist() == [0] and rows[0]["name"] == "baseline_proportional"
        )
        if not expected_repeat_mean_override:
            raise ValueError(f"{panel_id}: raw and frozen-dashboard targets disagree")
    domains = tuple(dataset.domain_names)
    family_names, family_members, group_names, group_members, group_family_indices = hierarchical_partition(domains)
    proportional = np.asarray(
        [domain["proportionalWeight"] for domain in bundle["swarms"][swarm]["domains"]],
        dtype=float,
    )
    if len(proportional) != len(domains) or not np.isclose(proportional.sum(), 1.0):
        raise ValueError(f"{panel_id}: invalid proportional reference")
    phase_fractions = np.asarray(bundle["swarms"][swarm]["dataset"]["phaseFractions"], dtype=float)
    panel = Panel(
        name=panel_id,
        target=target,
        weights=np.asarray(dataset.weights, dtype=float),
        # The canonical 300M panel replaces the original proportional seed with
        # the repeat-panel mean. All other raw targets are required to agree.
        observed=dashboard_observed,
        phase_epoch_factors=np.stack([dataset.c0, dataset.c1]),
        phase_fractions=phase_fractions,
        domains=domains,
        proportional=proportional,
        family_names=family_names,
        family_members=family_members,
        group_names=group_names,
        group_members=group_members,
        group_family_indices=group_family_indices,
    )
    return panel, dataset


def oof_prediction(panel: Panel, dataset: Any, config: ModelConfig, l2: float, seeds: Iterable[int]) -> np.ndarray:
    seed_predictions: list[np.ndarray] = []
    design = build_design(panel, panel.weights, config)
    for seed in seeds:
        prediction = np.full(panel.n, np.nan, dtype=float)
        for train, test in observatory.folds(dataset, seed):
            model = fit_nonnegative_ridge(design, panel.observed, train, config, l2)
            prediction[test] = model.predict_design(type(design)(values=design.values[test], names=design.names))
        if not np.isfinite(prediction).all():
            raise ValueError(f"Incomplete OOF prediction for {panel.name}/{config.key}/seed={seed}")
        seed_predictions.append(prediction)
    return np.mean(seed_predictions, axis=0)


def region_oof_prediction(panel: Panel, config: ModelConfig, l2: float) -> np.ndarray:
    if panel.m != 2:
        raise ValueError("Leave-region-out is defined only for two-domain surfaces")
    coordinates = panel.weights[:, :, 1]
    labels = KMeans(n_clusters=5, random_state=0, n_init=20).fit_predict(coordinates)
    prediction = np.full(panel.n, np.nan, dtype=float)
    design = build_design(panel, panel.weights, config)
    for region in sorted(set(labels)):
        test = np.flatnonzero(labels == region)
        train = np.flatnonzero(labels != region)
        model = fit_nonnegative_ridge(design, panel.observed, train, config, l2)
        prediction[test] = model.predict_design(type(design)(values=design.values[test], names=design.names))
    if not np.isfinite(prediction).all():
        raise ValueError("Incomplete leave-region-out prediction")
    return prediction


def heldout_data(
    bundle: dict[str, Any],
    swarm: str,
    target: str,
    policy: str = "two_phase",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]] | None:
    selected = [
        row
        for row in bundle["swarms"][swarm]["rows"]
        if row["split"] == "heldout" and not row["isSharedAlias"] and row["policyFamily"] == policy
    ]
    if not selected:
        return None
    weights = np.asarray([[row["phase0"], row["phase1"]] for row in selected], dtype=float)
    observed = np.asarray([row["observed"][target] for row in selected], dtype=float)
    return weights, observed, selected


def screen_one_panel(
    panel_id: str,
    output_dir: Path,
    bundle: dict[str, Any],
    configs: tuple[ModelConfig, ...],
) -> None:
    swarm, target = split_panel_id(panel_id)
    panel, dataset = load_panel(bundle, panel_id)
    screen_rows: list[dict[str, Any]] = []
    best_by_family: dict[str, tuple[float, float, ModelConfig, float]] = {}
    print(f"{panel_id}: screening {len(configs)} shapes x {len(L2_GRID)} ridge values", flush=True)
    for config_index, config in enumerate(configs, start=1):
        for l2 in L2_GRID:
            prediction = oof_prediction(panel, dataset, config, l2, seeds=(0,))
            summary, _bins = metrics(panel.observed, prediction)
            row = {
                "panel": panel_id,
                "family": config.family,
                "config": config.key,
                "parameters": json.dumps(dict(config.parameters), sort_keys=True),
                "l2": l2,
                **summary,
            }
            screen_rows.append(row)
            candidate = (float(summary["rmse"]), -float(summary["spearman"]), config, l2)
            current = best_by_family.get(config.family)
            if current is None or candidate[:2] < current[:2]:
                best_by_family[config.family] = candidate
        print(f"{panel_id}: [{config_index}/{len(configs)}] {config.key}", flush=True)

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    heldout = heldout_data(bundle, swarm, target)
    for family, (_rmse, _negative_spearman, config, l2) in sorted(best_by_family.items()):
        oof = oof_prediction(panel, dataset, config, l2, seeds=(0, 1, 2))
        oof_summary, _bins = metrics(panel.observed, oof)
        metric_rows.append(
            {
                "panel": panel_id,
                "family": family,
                "config": config.key,
                "l2": l2,
                "split": "fit_oof",
                **oof_summary,
            }
        )
        prediction_rows.extend(
            {
                "panel": panel_id,
                "family": family,
                "config": config.key,
                "split": "fit_oof",
                "row_id": rows_id,
                "observed": observed,
                "predicted": predicted,
                "policy": "two_phase",
            }
            for rows_id, observed, predicted in zip(
                [row["name"] for row in dashboard_fit_rows(bundle, swarm)],
                panel.observed,
                oof,
                strict=True,
            )
        )
        design = build_design(panel, panel.weights, config)
        model = fit_nonnegative_ridge(design, panel.observed, np.arange(panel.n), config, l2)
        model_record = record(model)
        parameter_rows.extend(
            {
                "panel": panel_id,
                "family": family,
                **model_record,
                "feature": name,
                "coefficient": coefficient,
            }
            for name, coefficient in zip(model.feature_names, model.coefficients, strict=True)
        )
        if heldout is not None:
            heldout_weights, heldout_observed, heldout_rows = heldout
            heldout_design = build_design(panel, heldout_weights, config)
            heldout_prediction = model.predict_design(heldout_design)
            heldout_summary, _bins = metrics(heldout_observed, heldout_prediction)
            metric_rows.append(
                {
                    "panel": panel_id,
                    "family": family,
                    "config": config.key,
                    "l2": l2,
                    "split": "heldout_policy_matched",
                    **heldout_summary,
                }
            )
            prediction_rows.extend(
                {
                    "panel": panel_id,
                    "family": family,
                    "config": config.key,
                    "split": "heldout_policy_matched",
                    "row_id": row["name"],
                    "observed": observed,
                    "predicted": predicted,
                    "policy": row["policyFamily"],
                    "support_distance": row["diagnostics"]["supportDistance"],
                    "max_epoch": row["diagnostics"]["maxEpoch"],
                    "phase_tv": row["diagnostics"]["phaseTv"],
                    "aggregate_tv_to_proportional": row["diagnostics"]["aggregateTvToProportional"],
                }
                for row, observed, predicted in zip(heldout_rows, heldout_observed, heldout_prediction, strict=True)
            )
        if panel.m == 2:
            region_prediction = region_oof_prediction(panel, config, l2)
            region_summary, _bins = metrics(panel.observed, region_prediction)
            metric_rows.append(
                {
                    "panel": panel_id,
                    "family": family,
                    "config": config.key,
                    "l2": l2,
                    "split": "leave_region_out",
                    **region_summary,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(screen_rows).to_csv(output_dir / "hyperparameter_screen.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output_dir / "selected_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "selected_predictions.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(output_dir / "selected_parameters.csv", index=False)
    selection = {
        family: {"config": config.key, "parameters": dict(config.parameters), "l2": l2}
        for family, (_rmse, _negative_spearman, config, l2) in sorted(best_by_family.items())
    }
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    print(f"{panel_id}: wrote {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", choices=PANEL_IDS, required=True)
    parser.add_argument(
        "--portfolio",
        choices=(
            "initial",
            "round2",
            "round3_dynamics",
            "round4_foundation",
            "round5_prior",
            "round8_bounded_coverage",
            "round9_ces",
            "round10_replay_hazard",
            "round16_plasticity",
            "round17_gradient_noise",
            "round18_parallel_reliability",
            "round19_posterior_precision",
            "round20_capacity_gated",
            "round21_finite_subset",
            "round23_power_law_memory",
            "round24_riccati_uncertainty",
            "round25_two_pool_consolidation",
            "round26_concentration_displacement",
            "round27_diversity_gated",
            "round28_learned_state_competition",
        ),
        default="initial",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    assert_sealed_absent(DASHBOARD)
    bundle = json.loads(DASHBOARD.read_text())
    if args.portfolio == "initial":
        configs = candidate_configs()
    elif args.portfolio == "round2":
        configs = round2_candidate_configs()
    elif args.portfolio == "round3_dynamics":
        configs = round3_dynamics_candidate_configs()
    elif args.portfolio == "round4_foundation":
        configs = round4_foundation_candidate_configs()
    elif args.portfolio == "round5_prior":
        configs = round5_prior_candidate_configs()
    elif args.portfolio == "round8_bounded_coverage":
        configs = round8_bounded_coverage_candidate_configs()
    elif args.portfolio == "round9_ces":
        configs = round9_ces_candidate_configs()
    elif args.portfolio == "round10_replay_hazard":
        configs = round10_replay_hazard_candidate_configs()
    elif args.portfolio == "round16_plasticity":
        configs = round16_plasticity_candidate_configs()
    elif args.portfolio == "round17_gradient_noise":
        configs = round17_gradient_noise_candidate_configs()
    elif args.portfolio == "round18_parallel_reliability":
        configs = round18_parallel_reliability_candidate_configs()
    elif args.portfolio == "round19_posterior_precision":
        configs = round19_posterior_precision_candidate_configs()
    elif args.portfolio == "round20_capacity_gated":
        configs = round20_capacity_gated_candidate_configs()
    elif args.portfolio == "round21_finite_subset":
        configs = round21_finite_subset_candidate_configs()
    elif args.portfolio == "round23_power_law_memory":
        configs = round23_power_law_memory_candidate_configs()
    elif args.portfolio == "round24_riccati_uncertainty":
        configs = round24_riccati_uncertainty_candidate_configs()
    elif args.portfolio == "round25_two_pool_consolidation":
        configs = round25_two_pool_consolidation_candidate_configs()
    elif args.portfolio == "round26_concentration_displacement":
        configs = round26_concentration_displacement_candidate_configs()
    elif args.portfolio == "round27_diversity_gated":
        configs = round27_diversity_gated_candidate_configs()
    else:
        configs = round28_learned_state_competition_candidate_configs()
    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / f"{args.portfolio}_screen" / args.panel
    screen_one_panel(args.panel, output_dir, bundle, configs)


if __name__ == "__main__":
    main()
