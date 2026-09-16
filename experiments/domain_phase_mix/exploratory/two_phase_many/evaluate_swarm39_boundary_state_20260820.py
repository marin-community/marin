# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Evaluate exposed phase-boundary state arms without adversarial-target fitting.

This is the leakage-corrected deployment audit for ATOM-023. The model is fit
only on non-adversarial Delphi archive rows. The completed adversarial stress
panel is used once, after the protocol is frozen, for prediction-only scoring.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for entry in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import fit_swarm39_split_damage_20260817 as split_damage  # noqa: E402
import fit_swarm39_state_geometry_20260820 as state_geometry  # noqa: E402
import fit_swarm39_trajectory_20260817 as trajectory  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import swarm39_harness_20260725 as swarm39  # noqa: E402
from scipy.optimize import differential_evolution  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    general_mixture_surrogate_20260809 as gen,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_swarm39_phase0_20260817 as phase0,
)

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "swarm39_state_geometry_20260820"
PROTOCOL_PATH = OUTPUT_DIR / "boundary_state_protocol.json"
ARMS = ("endpoint", "endpoint+y0", "trajectory")
VARIANTS = ("split", "blended")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=swarm39.UNCHEATABLE, choices=(swarm39.UNCHEATABLE, swarm39.TABLE9))
    parser.add_argument("--variant", default="split", choices=VARIANTS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def protocol_hash() -> str:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    assert protocol["status"] == "frozen_before_adversarial_prediction"
    assert tuple(protocol["targets"]) == (swarm39.UNCHEATABLE, swarm39.TABLE9)
    assert set(protocol["variants"]) == set(VARIANTS)
    assert tuple(protocol["arms"]) == ARMS
    return hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()


def vector(payload: str, buckets: tuple[str, ...]) -> np.ndarray:
    mapping = json.loads(payload)
    return np.asarray([float(mapping.get(bucket, 0.0)) for bucket in buckets], dtype=float)


def load_delphi(target: str) -> tuple[gen.Panel, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    fit_panel, _ = swarm39.load_scale("delphi_3e18")
    rows, readout_path = phase0.sources("delphi_3e18")
    readouts = pd.read_csv(readout_path)
    merged = rows.merge(readouts, on="heldout_id", how="inner")
    merged = state_geometry.recover_adversarial_provenance(merged)
    keep = merged[target].notna() & merged["phase0_uncheatable_bpb"].notna()
    merged = merged.loc[keep].reset_index(drop=True)

    phase_0 = np.stack([vector(value, fit_panel.buckets) for value in merged["phase_0_weights_json"]])
    phase_1 = np.stack([vector(value, fit_panel.buckets) for value in merged["phase_1_weights_json"]])
    endpoint = merged[target].to_numpy(float)
    readout = merged["phase0_uncheatable_bpb"].to_numpy(float)
    aggregate = fit_panel.alpha * phase_0 + (1.0 - fit_panel.alpha) * phase_1
    panel = gen.Panel(np.stack([phase_0, phase_1], axis=1), fit_panel.c0, fit_panel.c1, fit_panel.family_index)
    return panel, endpoint, readout, aggregate, merged


def subset(panel: gen.Panel, rows: np.ndarray) -> gen.Panel:
    return gen.Panel(panel.weights[rows], panel.epochs_early, panel.epochs_late, panel.family_index)


def aggregate_folds(aggregate: np.ndarray, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    cells: dict[tuple[float, ...], list[int]] = collections.defaultdict(list)
    for index, row in enumerate(np.round(aggregate, 6)):
        cells[tuple(row)].append(index)
    blocks = [np.asarray(rows, dtype=int) for rows in cells.values()]
    order = np.random.default_rng(seed).permutation(len(blocks))
    result = []
    for fold in range(folds):
        test = np.concatenate([blocks[index] for index in order[fold::folds]])
        train = np.setdiff1d(np.arange(len(aggregate)), test)
        result.append((train, test))
    return result


def arm_data(endpoint: np.ndarray, readout: np.ndarray, arm: str) -> tuple[np.ndarray, np.ndarray | None]:
    if arm == "endpoint":
        return endpoint, None
    if arm == "endpoint+y0":
        return endpoint, readout
    if arm == "trajectory":
        return endpoint - readout, readout
    raise ValueError(arm)


def unpack(panel: gen.Panel, variant: str, vector_value: np.ndarray):
    if variant == "split":
        shape, ridge = gen.unpack(vector_value[:-1], panel.n_families)
        return shape, ridge, 10.0 ** vector_value[-1]
    shape, ridge = gen.unpack(vector_value, panel.n_families)
    return shape, ridge, 0.0


def design(
    panel: gen.Panel,
    rows: np.ndarray,
    shape,
    readout_feature: np.ndarray | None,
    variant: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    return trajectory.columns(
        subset(panel, rows),
        shape,
        None if readout_feature is None else readout_feature[rows],
        variant,
    )


def select_parameters(
    panel: gen.Panel,
    endpoint: np.ndarray,
    readout: np.ndarray,
    arm: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
    variant: str,
    seed: int,
) -> np.ndarray:
    target, readout_feature = arm_data(endpoint, readout, arm)

    def objective(vector_value: np.ndarray) -> float:
        shape, ridge, departure_weight = unpack(panel, variant, vector_value)
        total = 0.0
        for train, test in folds:
            free, constrained, pooled = design(panel, train, shape, readout_feature, variant)
            offsets, amplitudes = gen.fit_head(
                free,
                constrained,
                target[train],
                ridge,
                pooled,
                split_damage.departure_pairs(panel, variant),
                departure_weight,
            )
            free_test, constrained_test, _ = design(panel, test, shape, readout_feature, variant)
            residual = free_test @ offsets + constrained_test @ amplitudes - target[test]
            if not np.isfinite(residual).all():
                return 1e6
            total += float(residual @ residual)
        return total

    bounds = list(gen.bounds(panel.n_families))
    if variant == "split":
        bounds.append(split_damage.DEPARTURE_BOUND)
    return differential_evolution(
        objective,
        bounds,
        rng=np.random.default_rng(20260817 + seed),
        popsize=8,
        maxiter=12,
        tol=1e-12,
        polish=True,
        init="sobol",
    ).x


def cross_fitted_prediction(
    panel: gen.Panel,
    endpoint: np.ndarray,
    readout: np.ndarray,
    arm: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
    variant: str,
    vector_value: np.ndarray,
) -> np.ndarray:
    target, readout_feature = arm_data(endpoint, readout, arm)
    shape, ridge, departure_weight = unpack(panel, variant, vector_value)
    predicted = np.empty(len(endpoint))
    for train, test in folds:
        free, constrained, pooled = design(panel, train, shape, readout_feature, variant)
        offsets, amplitudes = gen.fit_head(
            free,
            constrained,
            target[train],
            ridge,
            pooled,
            split_damage.departure_pairs(panel, variant),
            departure_weight,
        )
        free_test, constrained_test, _ = design(panel, test, shape, readout_feature, variant)
        predicted[test] = free_test @ offsets + constrained_test @ amplitudes
    if arm == "trajectory":
        predicted += readout
    return predicted


def fit_full_predict(
    train_panel: gen.Panel,
    train_endpoint: np.ndarray,
    train_readout: np.ndarray,
    test_panel: gen.Panel,
    test_readout: np.ndarray,
    arm: str,
    variant: str,
    vector_value: np.ndarray,
) -> np.ndarray:
    target, readout_feature = arm_data(train_endpoint, train_readout, arm)
    shape, ridge, departure_weight = unpack(train_panel, variant, vector_value)
    train_rows = np.arange(len(train_endpoint))
    free, constrained, pooled = design(train_panel, train_rows, shape, readout_feature, variant)
    offsets, amplitudes = gen.fit_head(
        free,
        constrained,
        target,
        ridge,
        pooled,
        split_damage.departure_pairs(train_panel, variant),
        departure_weight,
    )
    test_rows = np.arange(len(test_readout))
    test_feature = None if arm == "endpoint" else test_readout
    free_test, constrained_test, _ = design(test_panel, test_rows, shape, test_feature, variant)
    predicted = free_test @ offsets + constrained_test @ amplitudes
    if arm == "trajectory":
        predicted += test_readout
    return predicted


def matched_cell_metrics(
    aggregate: np.ndarray,
    weights: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float | int]:
    cells: dict[tuple[float, ...], list[int]] = collections.defaultdict(list)
    for index, row in enumerate(np.round(aggregate, 6)):
        cells[tuple(row)].append(index)
    phase_tv = 0.5 * np.abs(weights[:, 1] - weights[:, 0]).sum(axis=1)
    rhos = []
    regrets = []
    for rows_list in cells.values():
        rows = np.asarray(rows_list, dtype=int)
        tied = rows[phase_tv[rows] <= 1e-9]
        alternatives = rows[phase_tv[rows] > 1e-9]
        if len(tied) == 0 or len(alternatives) < 4:
            continue
        observed_delta = observed[alternatives] - observed[tied].mean()
        predicted_delta = predicted[alternatives] - predicted[tied].mean()
        rho = float(spearmanr(predicted_delta, observed_delta).statistic)
        if np.isfinite(rho):
            rhos.append(rho)
        selected = alternatives[int(np.argmin(predicted_delta))]
        regrets.append(float(observed[selected] - observed[alternatives].min()))
    return {
        "matched_cells": len(regrets),
        "median_within_cell_spearman": float(np.median(rhos)) if rhos else float("nan"),
        "positive_rho_cells": int(np.sum(np.asarray(rhos) > 0.0)),
        "mean_decision_regret": float(np.mean(regrets)) if regrets else float("nan"),
    }


def main() -> None:
    args = parse_args()
    frozen_hash = protocol_hash()
    panel, endpoint, readout, aggregate, metadata = load_delphi(args.target)
    development_mask = metadata["training_series"].astype(str).to_numpy() != state_geometry.ADVERSARIAL_SERIES
    adversarial_mask = ~development_mask

    development_panel = subset(panel, np.flatnonzero(development_mask))
    adversarial_panel = subset(panel, np.flatnonzero(adversarial_mask))
    development_endpoint = endpoint[development_mask]
    development_readout = readout[development_mask]
    development_aggregate = aggregate[development_mask]
    folds = aggregate_folds(development_aggregate, args.folds, args.seed)

    development_rows = []
    adversarial_predictions: dict[str, np.ndarray] = {}
    parameter_rows = []
    for arm_index, arm in enumerate(ARMS):
        vector_value = select_parameters(
            development_panel,
            development_endpoint,
            development_readout,
            arm,
            folds,
            args.variant,
            args.seed + arm_index,
        )
        name = f"{args.variant}_{arm.replace('+', '_plus_')}"
        development_prediction = cross_fitted_prediction(
            development_panel,
            development_endpoint,
            development_readout,
            arm,
            folds,
            args.variant,
            vector_value,
        )
        row = {
            "model": name,
            **state_geometry.scalar_metrics(development_endpoint, development_prediction),
            **matched_cell_metrics(
                development_aggregate,
                development_panel.weights,
                development_endpoint,
                development_prediction,
            ),
        }
        development_rows.append(row)
        adversarial_predictions[name] = fit_full_predict(
            development_panel,
            development_endpoint,
            development_readout,
            adversarial_panel,
            readout[adversarial_mask],
            arm,
            args.variant,
            vector_value,
        )
        parameter_rows.append(
            {
                "model": name,
                "protocol_sha256": frozen_hash,
                "parameter_vector_json": json.dumps(vector_value.tolist()),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.target.replace('_bpb', '')}_{args.variant}"
    pd.DataFrame(development_rows).to_csv(args.output_dir / f"boundary_state_development_{stem}.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(args.output_dir / f"boundary_state_parameters_{stem}.csv", index=False)

    adversarial_metadata = metadata.loc[adversarial_mask].reset_index(drop=True)
    adversarial_endpoint = endpoint[adversarial_mask]
    grouped = state_geometry.grouped_metrics(
        adversarial_metadata,
        args.target,
        adversarial_endpoint,
        adversarial_predictions,
    )
    grouped.to_csv(args.output_dir / f"boundary_state_adversarial_metrics_{stem}.csv", index=False)
    prediction_frame = adversarial_metadata[
        [
            "heldout_id",
            "proposal_target",
            "policy_class",
            "selection_stratum",
            "proposal_series",
            "phase0_uncheatable_bpb",
        ]
    ].copy()
    prediction_frame["observed"] = adversarial_endpoint
    for name, values in adversarial_predictions.items():
        prediction_frame[name] = values
    prediction_frame.to_csv(args.output_dir / f"boundary_state_adversarial_predictions_{stem}.csv", index=False)

    print(
        json.dumps(
            {
                "target": args.target,
                "variant": args.variant,
                "development_rows": int(development_mask.sum()),
                "adversarial_rows": int(adversarial_mask.sum()),
                "protocol_sha256": frozen_hash,
                "development": development_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
