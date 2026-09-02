# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy>=1.7",
#   "fsspec>=2025.7",
#   "gcsfs>=2025.7",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scikit-learn>=1.6",
#   "scipy>=1.15",
#   "tabulate>=0.9",
# ]
# ///
"""Falsify the exposure-gated competence cascade on both StarCoder surfaces."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_potential_round2 as starcoder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    exposure_gated_cascade_models as cascade,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    paired_dynamics_models as paired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_paired_dynamics_round1 as paired_screen,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round10_exposure_gated_cascade_starcoder"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
SEED = 20260719
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
RATE_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
OFFSET_GRID = (0.03, 0.1, 0.3, 1.0)
L2_GRID = (0.1, 1.0, 10.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configs() -> list[cascade.CascadeConfig]:
    return [
        cascade.CascadeConfig(feature_rate, conversion_rate, offset, l2, replay)
        for feature_rate in RATE_GRID
        for conversion_rate in RATE_GRID
        for offset in OFFSET_GRID
        for l2 in L2_GRID
        for replay in (False, True)
    ]


def geometry(panel: paired.PairedPanel) -> cascade.CascadeGeometry:
    return cascade.CascadeGeometry(
        domain_names=panel.domain_names,
        family_names=panel.family_names,
        family_members=panel.family_members,
        proportional_weights=panel.proportional_weights,
        phase0_epoch_coefficients=panel.c0,
        phase1_epoch_coefficients=panel.c1,
    )


def prediction_from_design(
    design: np.ndarray,
    names: tuple[str, ...],
    signs: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    l2: float,
) -> np.ndarray:
    head = paired.fit_linear_head(design[train], target[train], names, signs, l2)
    return head.predict(design[test])


def surface_oof(
    panel: paired.PairedPanel,
    config: cascade.CascadeConfig,
    folds: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray:
    folds = starcoder.surface_folds(panel) if folds is None else folds
    design, names, signs = cascade.cascade_design(geometry(panel), panel.weights, config)
    prediction = np.full(panel.n, np.nan, dtype=float)
    for train, test in folds:
        prediction[test] = prediction_from_design(
            design,
            names,
            signs,
            panel.two_phase_target,
            train,
            test,
            config.l2,
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete cascade OOF prediction for {panel.name}")
    return prediction


def select_config(panel: paired.PairedPanel) -> tuple[cascade.CascadeConfig, np.ndarray, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    for config in configs():
        prediction = surface_oof(panel, config)
        predictions[config.key] = prediction
        rows.append(
            {
                "surface": panel.name,
                "config": config.key,
                **asdict(config),
                **paired_screen.scalar_metrics(panel.two_phase_target, prediction),
            }
        )
    table = pd.DataFrame(rows).sort_values(["rmse", "regret_at_1", "worst_optimism"])
    best = table.iloc[0]
    selected = cascade.CascadeConfig(
        feature_rate=float(best["feature_rate"]),
        conversion_rate=float(best["conversion_rate"]),
        response_offset=float(best["response_offset"]),
        l2=float(best["l2"]),
        include_replay_harm=bool(best["include_replay_harm"]),
    )
    return selected, predictions[selected.key], table


def nested_folds(
    panel: paired.PairedPanel, indices: np.ndarray, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    tied = indices[panel.paired_mask[indices]]
    untied = indices[~panel.paired_mask[indices]]
    tied_splits = min(n_splits, len(tied))
    untied_splits = min(n_splits, len(untied))
    if tied_splits < 2 or untied_splits < 2:
        splitter = KFold(min(n_splits, len(indices)), shuffle=True, random_state=seed)
        return [(indices[train], indices[test]) for train, test in splitter.split(indices)]
    tied_folds = list(KFold(tied_splits, shuffle=True, random_state=seed).split(tied))
    untied_folds = list(KFold(untied_splits, shuffle=True, random_state=seed + 1).split(untied))
    count = min(len(tied_folds), len(untied_folds))
    result = []
    for fold in range(count):
        tied_train, tied_test = tied_folds[fold]
        untied_train, untied_test = untied_folds[fold]
        result.append(
            (
                np.sort(np.concatenate([tied[tied_train], untied[untied_train]])),
                np.sort(np.concatenate([tied[tied_test], untied[untied_test]])),
            )
        )
    return result


def nested_surface_selection(
    panel: paired.PairedPanel,
) -> tuple[np.ndarray, pd.DataFrame]:
    all_configs = configs()
    geom = geometry(panel)
    design_cache = {config.key: cascade.cascade_design(geom, panel.weights, config) for config in all_configs}
    prediction = np.full(panel.n, np.nan, dtype=float)
    selections = []
    for outer_fold, (outer_train, outer_test) in enumerate(starcoder.surface_folds(panel)):
        inner_folds = nested_folds(panel, outer_train, 4, SEED + 100 * outer_fold)
        scored = []
        for config in all_configs:
            design, names, signs = design_cache[config.key]
            inner_prediction = np.full(panel.n, np.nan, dtype=float)
            for inner_train, inner_test in inner_folds:
                inner_prediction[inner_test] = prediction_from_design(
                    design,
                    names,
                    signs,
                    panel.two_phase_target,
                    inner_train,
                    inner_test,
                    config.l2,
                )
            inner_test = np.concatenate([test for _train, test in inner_folds])
            scored.append(
                (
                    float(np.sqrt(np.mean((inner_prediction[inner_test] - panel.two_phase_target[inner_test]) ** 2))),
                    config,
                )
            )
        _score, selected = min(scored, key=lambda item: item[0])
        design, names, signs = design_cache[selected.key]
        prediction[outer_test] = prediction_from_design(
            design,
            names,
            signs,
            panel.two_phase_target,
            outer_train,
            outer_test,
            selected.l2,
        )
        selections.append(
            {
                "surface": panel.name,
                "outer_fold": outer_fold,
                "selected_config": selected.key,
                "inner_rmse": _score,
                **asdict(selected),
            }
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete nested prediction for {panel.name}")
    return prediction, pd.DataFrame(selections)


def tied_independent_oof(
    panel: paired.PairedPanel,
) -> tuple[cascade.CascadeConfig, np.ndarray, pd.DataFrame]:
    tied = np.flatnonzero(panel.paired_mask)
    n_splits = min(5, len(tied))
    folds = [
        (tied[train], tied[test]) for train, test in KFold(n_splits, shuffle=True, random_state=SEED + 77).split(tied)
    ]
    geom = geometry(panel)
    rows = []
    predictions: dict[str, np.ndarray] = {}
    for config in configs():
        design, names, signs = cascade.cascade_design(geom, panel.weights, config)
        prediction = np.full(panel.n, np.nan, dtype=float)
        for train, test in folds:
            prediction[test] = prediction_from_design(
                design,
                names,
                signs,
                panel.two_phase_target,
                train,
                test,
                config.l2,
            )
        predictions[config.key] = prediction
        rows.append(
            {
                "surface": panel.name,
                "config": config.key,
                **asdict(config),
                **paired_screen.scalar_metrics(panel.two_phase_target[tied], prediction[tied]),
            }
        )
    table = pd.DataFrame(rows).sort_values(["rmse", "regret_at_1", "worst_optimism"])
    best = table.iloc[0]
    selected = cascade.CascadeConfig(
        feature_rate=float(best["feature_rate"]),
        conversion_rate=float(best["conversion_rate"]),
        response_offset=float(best["response_offset"]),
        l2=float(best["l2"]),
        include_replay_harm=bool(best["include_replay_harm"]),
    )
    return selected, predictions[selected.key], table


def leave_region_out(panel: paired.PairedPanel, config: cascade.CascadeConfig) -> list[dict[str, Any]]:
    rare_index = 1
    contrast = panel.weights[:, 1, rare_index] - panel.weights[:, 0, rare_index]
    regions = {
        "late_rare_enriched": contrast > 0.1,
        "early_rare_enriched": contrast < -0.1,
        "near_phase_tied": np.abs(contrast) <= 0.1,
    }
    geom = geometry(panel)
    design, names, signs = cascade.cascade_design(geom, panel.weights, config)
    rows = []
    for region, test_mask in regions.items():
        train = np.flatnonzero(~test_mask)
        test = np.flatnonzero(test_mask)
        if len(train) <= design.shape[1] or len(test) == 0:
            continue
        prediction = prediction_from_design(
            design,
            names,
            signs,
            panel.two_phase_target,
            train,
            test,
            config.l2,
        )
        rows.append(
            {
                "surface": panel.name,
                "region": region,
                "n_train": len(train),
                "n_test": len(test),
                **paired_screen.scalar_metrics(panel.two_phase_target[test], prediction),
            }
        )
    return rows


def algebraic_audit() -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    feature = rng.uniform(size=(64, 7))
    competence = rng.uniform(size=(64, 7))
    first = rng.uniform(0.0, 5.0, size=(64, 7))
    second = rng.uniform(0.0, 5.0, size=(64, 7))
    errors = []
    zero_errors = []
    for feature_rate in RATE_GRID:
        for conversion_rate in RATE_GRID:
            split_f, split_c = cascade.exposure_update(feature, competence, first, feature_rate, conversion_rate)
            split_f, split_c = cascade.exposure_update(split_f, split_c, second, feature_rate, conversion_rate)
            whole_f, whole_c = cascade.exposure_update(
                feature,
                competence,
                first + second,
                feature_rate,
                conversion_rate,
            )
            errors.append(float(max(np.max(np.abs(split_f - whole_f)), np.max(np.abs(split_c - whole_c)))))
            zero_f, zero_c = cascade.exposure_update(
                feature, competence, np.zeros_like(first), feature_rate, conversion_rate
            )
            zero_errors.append(float(max(np.max(np.abs(zero_f - feature)), np.max(np.abs(zero_c - competence)))))
    return {
        "maximum_semigroup_error": max(errors),
        "maximum_zero_exposure_error": max(zero_errors),
        "minimum_state": 0.0,
        "maximum_state": 1.0,
    }


def optimum_and_surface(
    panel: paired.PairedPanel,
    config: cascade.CascadeConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    geom = geometry(panel)
    model = cascade.fit_cascade(geom, panel.weights, panel.two_phase_target, np.arange(panel.n), config)
    grid = np.linspace(0.0, 1.0, 201)
    rare0, rare1 = np.meshgrid(grid, grid, indexing="ij")
    weights = np.stack(
        [
            np.column_stack([1.0 - rare0.ravel(), rare0.ravel()]),
            np.column_stack([1.0 - rare1.ravel(), rare1.ravel()]),
        ],
        axis=1,
    )
    prediction = model.predict(weights)
    best = int(np.argmin(prediction))
    observed_best = int(np.argmin(panel.two_phase_target))
    return (
        {
            "surface": panel.name,
            "phase0_rare": float(rare0.ravel()[best]),
            "phase1_rare": float(rare1.ravel()[best]),
            "predicted_bpb": float(prediction[best]),
            "observed_best_phase0_rare": float(panel.weights[observed_best, 0, 1]),
            "observed_best_phase1_rare": float(panel.weights[observed_best, 1, 1]),
            "observed_best_bpb": float(panel.two_phase_target[observed_best]),
            "distance_to_observed_best": float(
                np.hypot(
                    rare0.ravel()[best] - panel.weights[observed_best, 0, 1],
                    rare1.ravel()[best] - panel.weights[observed_best, 1, 1],
                )
            ),
        },
        pd.DataFrame({"phase0_rare": rare0.ravel(), "phase1_rare": rare1.ravel(), "predicted_bpb": prediction}),
    )


def render_surface(panel: paired.PairedPanel, surface: pd.DataFrame, optimum: dict[str, Any], output: Path) -> None:
    grid_size = round(np.sqrt(len(surface)))
    axis = surface["phase0_rare"].to_numpy().reshape(grid_size, grid_size)[:, 0]
    z = surface["predicted_bpb"].to_numpy().reshape(grid_size, grid_size)
    figure = go.Figure(
        [
            go.Surface(x=axis, y=axis, z=z.T, colorscale="RdYlGn_r", opacity=0.72, name="Predicted"),
            go.Scatter3d(
                x=panel.weights[:, 0, 1],
                y=panel.weights[:, 1, 1],
                z=panel.two_phase_target,
                mode="markers",
                marker={"size": 4, "color": panel.two_phase_target, "colorscale": "RdYlGn_r"},
                name="Observed",
            ),
            go.Scatter3d(
                x=[optimum["phase0_rare"]],
                y=[optimum["phase1_rare"]],
                z=[optimum["predicted_bpb"]],
                mode="markers",
                marker={"size": 9, "symbol": "diamond", "color": "#111827"},
                name="Predicted optimum",
            ),
        ]
    )
    figure.update_layout(
        title=f"{panel.name}: exposure-gated competence cascade",
        template="plotly_white",
        scene={
            "xaxis_title": "Phase 0 StarCoder weight",
            "yaxis_title": "Phase 1 StarCoder weight",
            "zaxis_title": "BPB",
        },
        height=850,
    )
    figure.write_html(output, include_plotlyjs="cdn", config=PLOT_CONFIG)


def update_status(status: str, evidence: str, output_dir: Path) -> None:
    registry = pd.read_csv(REGISTRY)
    mask = registry["id"].eq("EGCC")
    registry.loc[mask, "status"] = status
    registry.loc[mask, "status_evidence"] = evidence
    registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    ledger_row = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "round_id": "round_10_starcoder_gate",
        "candidate_id": "EGCC",
        "candidate_family": "Exposure-gated competence cascade",
        "hyperparameters": "Frozen preregistered grid; nested StarCoder selection",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "See round_10_preregistration",
        "novelty_class": "Boundary-free triangular feature-to-competence acquisition cascade",
        "evaluation_status": status,
        "evidence_path": str(output_dir.relative_to(OUTPUT_ROOT)),
        "notes": evidence,
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    key = tuple(ledger_row[column] for column in identity)
    if key not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([ledger_row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cosine = observatory.load_cosine_starcoder()
    panels = [
        starcoder.panel_from_dataset(cosine),
        starcoder.panel_from_dataset(starcoder.load_refined_wsd80(cosine)),
    ]
    algebra = algebraic_audit()
    metric_rows = []
    selection_tables = []
    nested_selection_tables = []
    tied_tables = []
    restriction_rows = []
    region_rows = []
    optimum_rows = []
    for panel in panels:
        selected, oof_prediction, grid = select_config(panel)
        nested_prediction, nested_selections = nested_surface_selection(panel)
        tied_config, tied_prediction, tied_grid = tied_independent_oof(panel)
        tied = panel.paired_mask
        selection_tables.append(grid)
        nested_selection_tables.append(nested_selections)
        tied_tables.append(tied_grid)
        design, _, _signs = cascade.cascade_design(geometry(panel), panel.weights, selected)
        metric_rows.append(
            {
                "surface": panel.name,
                "selected_config": selected.key,
                "response_feature_count": design.shape[1],
                "total_parameter_count": design.shape[1] + 4,
                "feature_rate_boundary": selected.feature_rate in (min(RATE_GRID), max(RATE_GRID)),
                "conversion_rate_boundary": selected.conversion_rate in (min(RATE_GRID), max(RATE_GRID)),
                **{
                    f"selection_{key}": value
                    for key, value in paired_screen.scalar_metrics(panel.two_phase_target, oof_prediction).items()
                },
                **{
                    f"nested_{key}": value
                    for key, value in paired_screen.scalar_metrics(panel.two_phase_target, nested_prediction).items()
                },
            }
        )
        restriction_rows.append(
            {
                "surface": panel.name,
                "n_tied": int(tied.sum()),
                "two_phase_selected_config": selected.key,
                "independent_tied_selected_config": tied_config.key,
                **{
                    f"algebraic_two_phase_oof_{key}": value
                    for key, value in paired_screen.scalar_metrics(
                        panel.two_phase_target[tied], oof_prediction[tied]
                    ).items()
                },
                **{
                    f"independent_tied_oof_{key}": value
                    for key, value in paired_screen.scalar_metrics(
                        panel.two_phase_target[tied], tied_prediction[tied]
                    ).items()
                },
            }
        )
        region_rows.extend(leave_region_out(panel, selected))
        optimum, surface = optimum_and_surface(panel, selected)
        optimum_rows.append(optimum)
        surface.to_csv(args.output_dir / f"{panel.name}__surface.csv", index=False)
        render_surface(panel, surface, optimum, args.output_dir / f"{panel.name}__surface.html")

    metrics = pd.DataFrame(metric_rows)
    nested_selections = pd.concat(nested_selection_tables, ignore_index=True)
    restrictions = pd.DataFrame(restriction_rows)
    regions = pd.DataFrame(region_rows)
    optima = pd.DataFrame(optimum_rows)
    pd.concat(selection_tables, ignore_index=True).to_csv(args.output_dir / "hyperparameter_grid.csv", index=False)
    pd.concat(tied_tables, ignore_index=True).to_csv(
        args.output_dir / "independent_tied_hyperparameter_grid.csv", index=False
    )
    metrics.to_csv(args.output_dir / "surface_oof_metrics.csv", index=False)
    nested_selections.to_csv(args.output_dir / "nested_fold_selections.csv", index=False)
    restrictions.to_csv(args.output_dir / "single_phase_restriction.csv", index=False)
    regions.to_csv(args.output_dir / "leave_region_out.csv", index=False)
    optima.to_csv(args.output_dir / "predicted_optima.csv", index=False)
    (args.output_dir / "algebraic_audit.json").write_text(json.dumps(algebra, indent=2) + "\n")

    prior = pd.read_csv(OUTPUT_ROOT / "round1_starcoder_shape_refined107/surface_oof_metrics.csv")
    strongest_prior = prior.groupby("surface", as_index=False)["rmse"].min().rename(columns={"rmse": "prior_best_rmse"})
    comparison = metrics[["surface", "nested_rmse"]].merge(strongest_prior, on="surface")
    comparison["relative_rmse"] = comparison["nested_rmse"] / comparison["prior_best_rmse"] - 1.0
    comparison.to_csv(args.output_dir / "prior_comparison.csv", index=False)

    semigroup_ok = algebra["maximum_semigroup_error"] < 1e-10 and algebra["maximum_zero_exposure_error"] < 1e-12
    no_rate_boundaries = not metrics[["feature_rate_boundary", "conversion_rate_boundary"]].to_numpy().any()
    shape_ok = bool((comparison["relative_rmse"] <= 0.05).all())
    optimum_ok = bool((optima["distance_to_observed_best"] <= 0.15).all())
    passed = semigroup_ok and no_rate_boundaries and shape_ok and optimum_ok
    status = "promoted_to_multi_swarm" if passed else "blocked_before_multi_swarm"
    evidence = (
        f"semigroup_ok={semigroup_ok}; no_rate_boundaries={no_rate_boundaries}; "
        f"within_5pct_prior_shape={shape_ok}; optimum_distance_ok={optimum_ok}."
    )
    update_status(status, evidence, args.output_dir)

    report = [
        "# Round 10: exposure-gated competence cascade",
        "",
        "## Frozen candidate",
        "",
        r"The latent state follows $df_i/dE_i=k_f(1-f_i)$ and $dc_i/dE_i=k_c f_i(1-c_i)$, where $E_i$ is realized bucket exposure in epochs. The response is a nonnegative log competence debt relative to proportional, optionally plus literal family replay harm.",
        "",
        "## Algebraic audit",
        "",
        f"- Maximum subdivision-semigroup error: `{algebra['maximum_semigroup_error']:.3e}`.",
        f"- Maximum zero-exposure error: `{algebra['maximum_zero_exposure_error']:.3e}`.",
        "- The tied restriction is exactly invariant to the artificial phase boundary.",
        "",
        "## Nested StarCoder results",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Existing-shape comparison",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Leave-region-out",
        "",
        regions.to_markdown(index=False),
        "",
        "## Single-phase restriction",
        "",
        restrictions.to_markdown(index=False),
        "",
        "## Raw optima",
        "",
        optima.to_markdown(index=False),
        "",
        "## Gate decision",
        "",
        f"Status: **{status}**. {evidence}",
        "",
        "No adversarial candidate predictions were evaluated in this round.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(metrics.to_string(index=False))
    print(comparison.to_string(index=False))
    print(optima.to_string(index=False))
    print(status, evidence)


if __name__ == "__main__":
    main()
