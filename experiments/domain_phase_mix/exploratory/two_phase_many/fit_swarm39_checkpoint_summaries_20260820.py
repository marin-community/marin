# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""Screen pre-boundary checkpoint summaries as state for the split surrogate.

The frozen split prediction is the baseline. Every candidate fits only its
residual on non-adversarial archive rows. The exposed adversarial panel is
scored once after model, feature, fold, and regularization choices are frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for entry in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import fit_swarm39_state_geometry_20260820 as state_geometry  # noqa: E402
import materialize_swarm39_checkpoint_summaries_20260820 as materialize  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import swarm39_harness_20260725 as swarm39  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "swarm39_checkpoint_summaries_20260820"
PROTOCOL_PATH = OUTPUT_DIR / "checkpoint_summary_protocol.json"
SUMMARY_PATH = OUTPUT_DIR / "checkpoint_summaries.csv"
BASELINE_DIR = SCRIPT_DIR / "reference_outputs" / "swarm39_state_geometry_20260820"

MODEL_ORDER = (
    "split",
    "split_action",
    "split_state",
    "split_state_action_additive",
    "split_state_action_interaction",
    "split_shuffled_state_interaction",
)
ARMS = MODEL_ORDER[1:]
ALPHAS = np.asarray([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
STATE_COMPONENTS = 4
ACTION_COMPONENTS = 4


@dataclass(frozen=True)
class Projection:
    state_scaler: StandardScaler
    state_pca: PCA
    action_scaler: StandardScaler
    action_pca: PCA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=swarm39.TABLE9, choices=(swarm39.UNCHEATABLE, swarm39.TABLE9))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def validate_protocol() -> tuple[dict[str, object], str]:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    assert protocol["status"] == "frozen_before_adversarial_prediction"
    assert tuple(protocol["models"]) == MODEL_ORDER
    assert tuple(protocol["state_features"]) == materialize.FEATURE_COLUMNS
    assert np.array_equal(np.asarray(protocol["selection"]["ridge_alpha_grid"], float), ALPHAS)
    return protocol, hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()


def prefix_groups(phase0: np.ndarray) -> np.ndarray:
    return np.asarray(
        [hashlib.sha256(np.round(row, 8).tobytes()).hexdigest() for row in phase0],
        dtype=object,
    )


def grouped_folds(groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    assert len(unique) >= n_splits
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(unique)), -counts))
    fold_load = np.zeros(n_splits, dtype=int)
    assignment = np.empty(len(unique), dtype=int)
    for group_index in order:
        fold = int(np.argmin(fold_load))
        assignment[group_index] = fold
        fold_load[fold] += counts[group_index]
    result = []
    for fold in range(n_splits):
        test = np.flatnonzero(assignment[inverse] == fold)
        train = np.flatnonzero(assignment[inverse] != fold)
        assert len(train) and len(test) and not set(groups[train]) & set(groups[test])
        result.append((train, test))
    return result


def action_features(panel: swarm39.Panel) -> tuple[np.ndarray, tuple[str, ...]]:
    phase1_family = panel.family_pool(panel.phase1)
    contrast_family = panel.family_pool(panel.phase1 - panel.phase0)
    late_epochs = panel.c1 * panel.phase1
    late_family = panel.family_pool(late_epochs)
    values = np.column_stack(
        [
            phase1_family,
            contrast_family,
            late_family,
            panel.phase_tv,
            late_epochs.max(axis=1),
        ]
    )
    names = (
        *(f"phase1_share::{name}" for name in panel.family_names),
        *(f"phase_contrast::{name}" for name in panel.family_names),
        *(f"late_epochs::{name}" for name in panel.family_names),
        "phase_tv",
        "max_late_epochs",
    )
    assert values.shape[1] == len(names)
    return values, names


def fit_projection(state: np.ndarray, action: np.ndarray) -> Projection:
    state_scaler = StandardScaler().fit(state)
    action_scaler = StandardScaler().fit(action)
    state_pca = PCA(n_components=STATE_COMPONENTS, svd_solver="full").fit(state_scaler.transform(state))
    action_pca = PCA(n_components=ACTION_COMPONENTS, svd_solver="full").fit(action_scaler.transform(action))
    return Projection(state_scaler, state_pca, action_scaler, action_pca)


def project(projection: Projection, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = projection.state_pca.transform(projection.state_scaler.transform(state))
    a = projection.action_pca.transform(projection.action_scaler.transform(action))
    return z, a


def design(z: np.ndarray, a: np.ndarray, arm: str) -> np.ndarray:
    if arm == "split_action":
        return a
    if arm == "split_state":
        return z
    if arm == "split_state_action_additive":
        return np.column_stack([z, a])
    interaction = np.einsum("ni,nj->nij", z, a).reshape(len(z), -1)
    if arm in ("split_state_action_interaction", "split_shuffled_state_interaction"):
        return np.column_stack([z, a, interaction])
    raise ValueError(arm)


def permute_rows(values: np.ndarray, seed: int) -> np.ndarray:
    return values[np.random.default_rng(seed).permutation(len(values))]


def fit_predict(
    train: np.ndarray,
    test: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    response: np.ndarray,
    arm: str,
    alpha: float,
    seed: int,
) -> np.ndarray:
    train_state, test_state = state[train], state[test]
    if arm == "split_shuffled_state_interaction":
        train_state = permute_rows(train_state, seed)
        test_state = permute_rows(test_state, seed + 1)
    projection = fit_projection(train_state, action[train])
    train_z, train_a = project(projection, train_state, action[train])
    test_z, test_a = project(projection, test_state, action[test])
    model = Ridge(alpha=alpha, fit_intercept=True).fit(design(train_z, train_a, arm), response[train])
    return model.predict(design(test_z, test_a, arm))


def choose_alpha(
    rows: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    response: np.ndarray,
    groups: np.ndarray,
    arm: str,
    seed: int,
) -> float:
    folds = grouped_folds(groups[rows], 4, seed)
    losses = np.zeros(len(ALPHAS))
    for fold_index, (local_train, local_test) in enumerate(folds):
        train, test = rows[local_train], rows[local_test]
        for alpha_index, alpha in enumerate(ALPHAS):
            prediction = fit_predict(
                train,
                test,
                state,
                action,
                response,
                arm,
                float(alpha),
                seed + 1000 * fold_index,
            )
            losses[alpha_index] += float(np.square(prediction - response[test]).sum())
    return float(ALPHAS[int(np.argmin(losses))])


def nested_oof(
    state: np.ndarray,
    action: np.ndarray,
    response: np.ndarray,
    groups: np.ndarray,
    arm: str,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    prediction = np.full(len(response), np.nan)
    selected = []
    for fold_index, (train, test) in enumerate(grouped_folds(groups, 5, seed)):
        alpha = choose_alpha(
            train,
            state,
            action,
            response,
            groups,
            arm,
            seed + 100 * fold_index,
        )
        selected.append(alpha)
        prediction[test] = fit_predict(
            train,
            test,
            state,
            action,
            response,
            arm,
            alpha,
            seed + 10000 * fold_index,
        )
    assert np.isfinite(prediction).all()
    return prediction, selected


def load_rows(target: str):
    panel_full, heldout = swarm39.load_scale("delphi_3e18")
    del panel_full
    metadata = pd.read_csv(swarm39.DELPHI_HELDOUTS)
    metadata = metadata[metadata["fit_panel_overlap"] == "coordinate_disjoint"].reset_index(drop=True)
    metadata = state_geometry.recover_adversarial_provenance(metadata)
    assert metadata["heldout_id"].astype(str).tolist() == heldout.row_id.tolist()

    summaries = pd.read_csv(SUMMARY_PATH)
    summaries = summaries[summaries["error"].isna()].drop_duplicates("heldout_id", keep="last")
    summaries = summaries[
        (summaries["summary_rows"] == materialize.WINDOW_ROWS) & (summaries["summary_fraction"] >= 0.95)
    ]
    metadata["panel_index"] = np.arange(len(metadata))
    merged = metadata.merge(summaries, on="heldout_id", how="inner", validate="one_to_one")
    merged = merged.sort_values("panel_index").reset_index(drop=True)
    rows = merged["panel_index"].to_numpy(int)
    panel = heldout.subset(rows)

    suffix = target.replace("_bpb", "")
    baseline = pd.read_csv(BASELINE_DIR / f"predictions_{suffix}.csv")
    baseline = baseline.set_index("heldout_id").loc[merged["heldout_id"]]
    observed = panel.targets[target]
    assert np.allclose(observed, baseline["observed"].to_numpy(float), atol=1e-12, rtol=0.0)
    split = baseline["predicted::split"].to_numpy(float)
    state = merged[list(materialize.FEATURE_COLUMNS)].to_numpy(float)
    assert np.isfinite(state).all()
    action, action_names = action_features(panel)
    return panel, merged, observed, split, state, action, action_names


def main() -> None:
    args = parse_args()
    protocol, protocol_sha256 = validate_protocol()
    panel, metadata, observed, split, state, action, action_names = load_rows(args.target)
    adversarial = metadata["training_series"].astype(str).to_numpy() == state_geometry.ADVERSARIAL_SERIES
    development = ~adversarial
    development_rows = np.flatnonzero(development)
    adversarial_rows = np.flatnonzero(adversarial)
    assert len(development_rows) and len(adversarial_rows)

    response = observed - split
    groups = prefix_groups(panel.phase0)
    development_predictions: dict[str, np.ndarray] = {"split": split[development]}
    adversarial_predictions: dict[str, np.ndarray] = {"split": split[adversarial]}
    parameter_rows = []

    for arm_index, arm in enumerate(ARMS):
        oof_residual, outer_alphas = nested_oof(
            state[development],
            action[development],
            response[development],
            groups[development],
            arm,
            args.seed + 10000 * arm_index,
        )
        development_predictions[arm] = split[development] + oof_residual
        alpha = choose_alpha(
            development_rows,
            state,
            action,
            response,
            groups,
            arm,
            args.seed + 50000 + 10000 * arm_index,
        )
        residual_prediction = fit_predict(
            development_rows,
            adversarial_rows,
            state,
            action,
            response,
            arm,
            alpha,
            args.seed + 90000 + 10000 * arm_index,
        )
        adversarial_predictions[arm] = split[adversarial] + residual_prediction
        parameter_rows.append(
            {
                "target": args.target,
                "model": arm,
                "full_development_alpha": alpha,
                "outer_alphas_json": json.dumps(outer_alphas),
            }
        )

    development_metrics = pd.DataFrame(
        [
            {
                "target": args.target,
                "model": model,
                **state_geometry.scalar_metrics(observed[development], prediction),
            }
            for model, prediction in development_predictions.items()
        ]
    )
    adversarial_metadata = metadata.loc[adversarial].reset_index(drop=True)
    adversarial_metrics = state_geometry.grouped_metrics(
        adversarial_metadata,
        args.target,
        observed[adversarial],
        adversarial_predictions,
    )
    development_cells = state_geometry.matched_cell_metrics(
        panel.subset(development_rows),
        observed[development],
        development_predictions,
    )
    adversarial_cells = state_geometry.matched_cell_metrics(
        panel.subset(adversarial_rows),
        observed[adversarial],
        adversarial_predictions,
    )

    combined_predictions = np.full((len(panel), len(MODEL_ORDER)), np.nan)
    for column, model in enumerate(MODEL_ORDER):
        combined_predictions[development, column] = development_predictions[model]
        combined_predictions[adversarial, column] = adversarial_predictions[model]
    prediction_frame = metadata[
        [
            "heldout_id",
            "training_series",
            "policy_class",
            "proposal_target",
            "selection_stratum",
            "proposal_series",
            "summary_step",
            "summary_fraction",
        ]
    ].copy()
    prediction_frame["target"] = args.target
    prediction_frame["data_use"] = np.where(adversarial, "adversarial_prediction_only", "development_oof")
    prediction_frame["observed"] = observed
    for column, model in enumerate(MODEL_ORDER):
        prediction_frame[f"predicted::{model}"] = combined_predictions[:, column]

    suffix = args.target.replace("_bpb", "")
    prediction_frame.to_csv(OUTPUT_DIR / f"predictions_{suffix}.csv", index=False)
    development_metrics.to_csv(OUTPUT_DIR / f"development_metrics_{suffix}.csv", index=False)
    adversarial_metrics.to_csv(OUTPUT_DIR / f"adversarial_metrics_{suffix}.csv", index=False)
    development_cells.to_csv(OUTPUT_DIR / f"development_matched_cells_{suffix}.csv", index=False)
    adversarial_cells.to_csv(OUTPUT_DIR / f"adversarial_matched_cells_{suffix}.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(OUTPUT_DIR / f"parameters_{suffix}.csv", index=False)

    summary = {
        "target": args.target,
        "protocol_sha256": protocol_sha256,
        "protocol_created_at": protocol["created_at"],
        "rows_with_complete_state": len(panel),
        "development_rows": len(development_rows),
        "adversarial_rows": len(adversarial_rows),
        "unique_development_prefixes": len(np.unique(groups[development])),
        "unique_adversarial_prefixes": len(np.unique(groups[adversarial])),
        "state_feature_count": state.shape[1],
        "action_feature_count": action.shape[1],
        "action_features": action_names,
    }
    (OUTPUT_DIR / f"summary_{suffix}.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"Checkpoint-summary audit: {args.target}")
    print(json.dumps(summary, indent=2))
    print("\nDevelopment prefix-grouped OOF")
    print(
        development_metrics[
            ["model", "rmse", "r2", "spearman", "observed_on_predicted_slope", "regret_at_1", "regret_at_20"]
        ].to_string(index=False)
    )
    target_name = "table9" if args.target == swarm39.TABLE9 else "uncheatable"
    matched = adversarial_metrics[
        (adversarial_metrics["stratum"] == "adversarial_proposal_target")
        & (adversarial_metrics["stratum_value"].astype(str).str.lower() == target_name)
    ]
    print("\nAdversarial target-matched prediction-only")
    print(
        matched[
            [
                "model",
                "n",
                "rmse",
                "spearman",
                "observed_on_predicted_slope",
                "optimism_gt_0p05",
                "regret_at_1",
                "regret_at_20",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
