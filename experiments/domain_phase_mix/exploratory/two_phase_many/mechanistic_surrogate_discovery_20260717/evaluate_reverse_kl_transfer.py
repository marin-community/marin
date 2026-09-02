# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
"""Evaluate the promoted reverse-KL support mechanism across core panels."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_deficit_output_link_20260716 as output_link,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_deficit_response_20260716 as deficit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_production_grp_quality_variants as family_grp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (  # noqa: E402
    freeze_baseline_gate as gate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (  # noqa: E402
    screen_nested_support_invariants as nested,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (  # noqa: E402
    screen_portfolio as portfolio,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717/reverse_kl_transfer"
PANELS = (
    "300m_uncheatable",
    "300m_table9",
    "production_uncheatable",
    "starcoder_cosine_starcoder_bpb",
    "starcoder_wsd80_starcoder_bpb",
)
NUM_SHAPES = 8
TOP_SHAPE_FLOOR_PAIRS = 2
FINAL_SEEDS = (0, 1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panels", default=",".join(PANELS))
    return parser.parse_args()


def family_dataset(panel: Any, raw: Any) -> family_grp.Dataset:
    return family_grp.Dataset(
        frame=raw.frame.copy(),
        target=np.asarray(panel.observed, dtype=float),
        weights=np.asarray(panel.weights, dtype=float),
        c0=np.asarray(panel.phase_epoch_factors[0], dtype=float),
        c1=np.asarray(panel.phase_epoch_factors[1], dtype=float),
        domains=panel.domains,
        family_names=panel.family_names,
        family_members=panel.family_members,
        quality=np.full(panel.m, -1, dtype=int),
    )


def splits(raw: Any, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(np.asarray(train), np.asarray(test)) for train, test in observatory.folds(raw, seed)]


def select_base_config(dataset: family_grp.Dataset, raw: Any) -> tuple[deficit.Config, list[dict[str, Any]]]:
    shapes = family_grp.shape_candidates(family_grp.Variant.BUCKET_RESOLVED, NUM_SHAPES)
    local_splits = splits(raw, 7152)
    configs, first_stage = deficit.deficit_configs(
        nested.DEFICIT_VARIANT,
        shapes,
        TOP_SHAPE_FLOOR_PAIRS,
        dataset,
        local_splits,
    )
    config, _prediction, second_stage = deficit.score_configs(dataset, configs, local_splits)
    return config, [{"stage": "shape_floor", **row} for row in first_stage] + [
        {"stage": "full", **row} for row in second_stage
    ]


def select_link_config(
    dataset: family_grp.Dataset,
    raw: Any,
    deficit_config: deficit.Config,
) -> tuple[output_link.LinkConfig, list[dict[str, Any]]]:
    local_splits = splits(raw, 7152)
    rows: list[dict[str, Any]] = []
    best: tuple[float, float, output_link.LinkConfig] | None = None
    baseline = nested.SupportConfig(nested.Mechanism.BASELINE)
    for config in output_link.candidate_configs():
        prediction = nested.oof_prediction(dataset, deficit_config, config, baseline, local_splits)
        summary, _bins = gate.metrics(dataset.target, prediction)
        rows.append(
            {
                "link": config.link.value,
                "floor_fraction": config.floor_fraction,
                "l2": config.l2,
                **summary,
            }
        )
        candidate = (float(summary["rmse"]), -float(summary["spearman"]), config)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("No output link selected")
    return best[2], rows


def select_support_config(
    dataset: family_grp.Dataset,
    raw: Any,
    deficit_config: deficit.Config,
    link_config: output_link.LinkConfig,
) -> tuple[nested.SupportConfig, list[dict[str, Any]]]:
    local_splits = splits(raw, 7152)
    configs = [nested.SupportConfig(nested.Mechanism.BASELINE)]
    configs.extend(nested.SupportConfig(nested.Mechanism.BUCKET_REVERSE_KL, floor) for floor in nested.SUPPORT_FLOORS)
    rows: list[dict[str, Any]] = []
    best: tuple[float, float, nested.SupportConfig] | None = None
    for config in configs:
        prediction = nested.oof_prediction(dataset, deficit_config, link_config, config, local_splits)
        summary, _bins = gate.metrics(dataset.target, prediction)
        rows.append({"support_config": config.key, **summary})
        candidate = (float(summary["rmse"]), -float(summary["spearman"]), config)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("No support configuration selected")
    return best[2], rows


def repeated_oof(
    dataset: family_grp.Dataset,
    raw: Any,
    deficit_config: deficit.Config,
    link_config: output_link.LinkConfig,
    support_config: nested.SupportConfig,
) -> np.ndarray:
    predictions = [
        nested.oof_prediction(dataset, deficit_config, link_config, support_config, splits(raw, seed))
        for seed in FINAL_SEEDS
    ]
    return np.mean(predictions, axis=0)


def leave_region_out(
    panel: Any,
    dataset: family_grp.Dataset,
    deficit_config: deficit.Config,
    link_config: output_link.LinkConfig,
    support_config: nested.SupportConfig,
) -> np.ndarray:
    if panel.m != 2:
        raise ValueError("Leave-region-out requires a two-domain panel")
    coordinates = panel.weights[:, :, 1]
    labels = KMeans(n_clusters=5, random_state=0, n_init=20).fit_predict(coordinates)
    prediction = np.full(panel.n, np.nan, dtype=float)
    for region in sorted(set(labels)):
        test = np.flatnonzero(labels == region)
        train = np.flatnonzero(labels != region)
        model = nested.fit_model(dataset, deficit_config, link_config, support_config, train)
        prediction[test] = model.predict(dataset.weights[test])
    if not np.isfinite(prediction).all():
        raise RuntimeError("Incomplete leave-region-out prediction")
    return prediction


def benchmark_panel(
    bundle: dict[str, Any],
    panel_id: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    panel, raw = portfolio.load_panel(bundle, panel_id)
    dataset = family_dataset(panel, raw)
    print(f"{panel_id}: selecting base response", flush=True)
    deficit_config, base_screen = select_base_config(dataset, raw)
    print(f"{panel_id}: selecting output link", flush=True)
    link_config, link_screen = select_link_config(dataset, raw, deficit_config)
    print(f"{panel_id}: selecting reverse-KL floor", flush=True)
    support_config, support_screen = select_support_config(dataset, raw, deficit_config, link_config)

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    configs = (
        nested.SupportConfig(nested.Mechanism.BASELINE),
        support_config,
    )
    for support in configs:
        prediction = repeated_oof(dataset, raw, deficit_config, link_config, support)
        summary, _bins = gate.metrics(dataset.target, prediction)
        model = nested.fit_model(dataset, deficit_config, link_config, support, np.arange(dataset.n))
        metric_rows.append(
            {
                "panel": panel_id,
                "mechanism": support.mechanism.value,
                "support_config": support.key,
                "split": "fit_oof",
                **nested.model_record(model),
                **summary,
            }
        )
        prediction_rows.extend(
            {
                "panel": panel_id,
                "mechanism": support.mechanism.value,
                "support_config": support.key,
                "split": "fit_oof",
                "row_id": str(dataset.frame.iloc[index].get("run_name", index)),
                "observed": observed,
                "predicted": predicted,
            }
            for index, (observed, predicted) in enumerate(zip(dataset.target, prediction, strict=True))
        )
        if panel.m == 2:
            region_prediction = leave_region_out(panel, dataset, deficit_config, link_config, support)
            region_summary, _bins = gate.metrics(dataset.target, region_prediction)
            metric_rows.append(
                {
                    "panel": panel_id,
                    "mechanism": support.mechanism.value,
                    "support_config": support.key,
                    "split": "leave_region_out",
                    **nested.model_record(model),
                    **region_summary,
                }
            )
            prediction_rows.extend(
                {
                    "panel": panel_id,
                    "mechanism": support.mechanism.value,
                    "support_config": support.key,
                    "split": "leave_region_out",
                    "row_id": str(dataset.frame.iloc[index].get("run_name", index)),
                    "observed": observed,
                    "predicted": predicted,
                }
                for index, (observed, predicted) in enumerate(zip(dataset.target, region_prediction, strict=True))
            )
        for name, coefficient in zip(model.names, model.coefficients, strict=True):
            parameter_rows.append(
                {
                    "panel": panel_id,
                    "mechanism": support.mechanism.value,
                    "support_config": support.key,
                    "name": name,
                    "coefficient": coefficient,
                }
            )

    screen_rows = (
        [{"panel": panel_id, "screen": "base", **row} for row in base_screen]
        + [{"panel": panel_id, "screen": "link", **row} for row in link_screen]
        + [{"panel": panel_id, "screen": "support", **row} for row in support_screen]
    )
    selection = {
        "panel": panel_id,
        "deficit_config": {
            "variant": deficit_config.variant.value,
            "shape": asdict(deficit_config.base.shape),
            "l2": deficit_config.base.l2,
            "residual_shrink": deficit_config.base.residual_shrink,
            "deficit_floor": deficit_config.deficit_floor,
            "surplus_credit": deficit_config.surplus_credit,
        },
        "link_config": {
            "link": link_config.link.value,
            "floor_fraction": link_config.floor_fraction,
            "l2": link_config.l2,
        },
        "support_config": {
            "mechanism": support_config.mechanism.value,
            "floor": support_config.floor,
        },
    }
    return metric_rows, screen_rows, prediction_rows, parameter_rows, selection


def main() -> None:
    args = parse_args()
    gate.assert_sealed_absent(portfolio.DASHBOARD)
    bundle = json.loads(portfolio.DASHBOARD.read_text())
    metric_rows: list[dict[str, Any]] = []
    screen_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for panel_id in args.panels.split(","):
        metrics, screen, predictions, parameters, selection = benchmark_panel(bundle, panel_id)
        metric_rows.extend(metrics)
        screen_rows.extend(screen)
        prediction_rows.extend(predictions)
        parameter_rows.extend(parameters)
        selections.append(selection)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame(screen_rows).to_csv(args.output_dir / "hyperparameter_screen.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(args.output_dir / "predictions.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(args.output_dir / "parameters.csv", index=False)
    (args.output_dir / "selections.json").write_text(json.dumps(selections, indent=2) + "\n")


if __name__ == "__main__":
    main()
