# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy"]
# ///
"""Test static content geometry as a correction to the 39-bucket split surrogate.

This round is deliberately narrower than a boundary-state critic. Exact Table-9
boundary measurements are not yet available for all 280 canonical fitting rows,
whereas the frozen content-Hellinger basis is available for every fit and heldout
policy. The candidate therefore asks whether action geometry supplies residual
structure that the mechanistic split model misses.

The candidate family is frozen in ``state_geometry_protocol.json`` before any
heldout predictions are scored. Weight-Hellinger is the mandatory geometry
control for content-Hellinger. No heldout target is used for fitting, selection,
calibration, or feature design.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for entry in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import benchmark_hellinger_krr_delphi_3e18_20260727 as hkrr  # noqa: E402
import fit_swarm39_split_damage_20260817 as split_damage  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import swarm39_harness_20260725 as swarm39  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "swarm39_state_geometry_20260820"
PROTOCOL_PATH = OUTPUT_DIR / "state_geometry_protocol.json"
RESIDUAL_TARGET = "cross_fitted_split_residual"
MODEL_ORDER = (
    "split",
    "weight_krr",
    "content_krr",
    "split_weight_blend",
    "split_content_blend",
    "split_weight_residual",
    "split_content_residual",
)
ADVERSARIAL_SERIES = "delphi_3e18_adversarial_stress_panel_20260716"


@dataclass(frozen=True)
class GeometryFit:
    """One Hellinger model and its heldout diagnostics."""

    model: hkrr.KernelFit
    heldout_prediction: np.ndarray
    support_distance: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=swarm39.TABLE9, choices=(swarm39.UNCHEATABLE, swarm39.TABLE9))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outer-folds", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def protocol_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_protocol(path: Path, target: str, outer_folds: int) -> dict[str, object]:
    protocol = json.loads(path.read_text())
    assert protocol["status"] == "frozen_before_heldout_evaluation"
    assert target in protocol["targets"]
    assert outer_folds == protocol["outer_folds"]
    assert tuple(protocol["model_order"]) == MODEL_ORDER
    return protocol


def fit_split(panel: swarm39.Panel, response: np.ndarray, seed: int):
    return split_damage.fit_variant(split_damage.bridge(panel), response, "split", seed)


def predict_split(panel: swarm39.Panel, fitted) -> np.ndarray:
    return split_damage.predict(split_damage.bridge(panel), fitted, "split")


def cross_fitted_split(
    panel: swarm39.Panel,
    response: np.ndarray,
    seed: int,
    outer_folds: int,
) -> np.ndarray:
    """Nested spatial OOF predictions; each outer test block is unseen by shape selection."""
    prediction = np.full(len(panel), np.nan)
    folds = swarm39.mixture_blocked_splits(panel, n_splits=outer_folds, seed=seed)
    for fold_index, (train, test) in enumerate(folds):
        fitted = fit_split(panel.subset(train), response[train], seed + 100 + fold_index)
        prediction[test] = predict_split(panel.subset(test), fitted)
    assert np.isfinite(prediction).all()
    return prediction


def fit_geometry(
    fit_panel: swarm39.Panel,
    heldout_panel: swarm39.Panel,
    basis: np.ndarray,
    kernel_space: str,
    target: str,
    seed: int,
) -> GeometryFit:
    fitted = hkrr.fit_kernel_model(fit_panel, basis, kernel_space, target, seed)
    prediction, distance, _ = hkrr.predict_weights(fitted, heldout_panel.phase0, heldout_panel.phase1)
    return GeometryFit(fitted, prediction, distance)


def fit_residual_geometry(
    fit_panel: swarm39.Panel,
    heldout_panel: swarm39.Panel,
    basis: np.ndarray,
    kernel_space: str,
    residual: np.ndarray,
    seed: int,
) -> GeometryFit:
    residual_panel = replace(fit_panel, targets={**fit_panel.targets, RESIDUAL_TARGET: residual})
    return fit_geometry(residual_panel, heldout_panel, basis, kernel_space, RESIDUAL_TARGET, seed)


def convex_blend_weight(observed: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    """Least-squares weight in ``(1-weight)*left + weight*right``, clipped to a convex blend."""
    direction = right - left
    denominator = float(direction @ direction)
    if denominator <= 1e-20:
        return 0.0
    return float(np.clip(direction @ (observed - left) / denominator, 0.0, 1.0))


def scalar_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    residual = observed - predicted
    spread = float(np.sum((observed - observed.mean()) ** 2))
    slope = float(np.polyfit(predicted, observed, 1)[0]) if np.std(predicted) > 1e-12 else float("nan")
    result: dict[str, float | int] = {
        "n": len(observed),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias_observed_minus_predicted": float(np.mean(residual)),
        "r2": 1.0 - float(np.sum(residual**2)) / spread if spread > 0.0 else float("nan"),
        "spearman": float(spearmanr(predicted, observed).statistic),
        "observed_on_predicted_slope": slope,
        "optimism_gt_0p05": int(np.sum(residual > 0.05)),
    }
    order = np.argsort(predicted)
    best = float(np.min(observed))
    for k in (1, 3, 5, 20):
        selected = float(np.min(observed[order[: min(k, len(order))]]))
        result[f"best_observed_in_top_{k}"] = selected
        result[f"regret_at_{k}"] = selected - best
    return result


def tag_fields(raw_tags: object) -> dict[str, str]:
    if not isinstance(raw_tags, str) or not raw_tags:
        return {}
    tags = json.loads(raw_tags)
    assert isinstance(tags, list)
    return {key: value for tag in tags if isinstance(tag, str) and "=" in tag for key, value in (tag.split("=", 1),)}


def recover_adversarial_provenance(metadata: pd.DataFrame) -> pd.DataFrame:
    """Recover fields omitted by the registry join from the frozen run tags."""
    enriched = metadata.copy()
    parsed = [tag_fields(value) for value in enriched["tags_json"]]
    tag_frame = pd.DataFrame(parsed, index=enriched.index)
    for destination, source in (
        ("proposal_target", "target"),
        ("candidate_kind", "policy"),
        ("selection_stratum", "selection"),
        ("proposal_source_run", "source_run"),
    ):
        recovered = tag_frame[source] if source in tag_frame else pd.Series(index=enriched.index, dtype=object)
        if destination in enriched:
            enriched[destination] = enriched[destination].combine_first(recovered)
        else:
            enriched[destination] = recovered
    enriched["proposal_series"] = enriched["proposal_source_run"].map(
        lambda value: value.rsplit("_", 1)[0] if isinstance(value, str) and value.rsplit("_", 1)[-1].isdigit() else value
    )
    return enriched


def grouped_metrics(
    metadata: pd.DataFrame,
    target: str,
    observed: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    matched = "table9" if target == swarm39.TABLE9 else "uncheatable"
    adversarial = metadata["training_series"].to_numpy() == ADVERSARIAL_SERIES
    proposal_target = metadata["proposal_target"].fillna("").astype(str).str.lower().to_numpy()
    masks: list[tuple[str, str, np.ndarray]] = [
        ("pooled", "all", np.ones(len(metadata), dtype=bool)),
        ("data_use", "non_adversarial", ~adversarial),
        ("data_use", "adversarial_all", adversarial),
        ("data_use", "adversarial_target_matched", adversarial & (proposal_target == matched)),
        ("data_use", "adversarial_cross_target", adversarial & (proposal_target != matched)),
    ]
    masks.extend(
        ("policy_class", str(value), metadata["policy_class"].astype(str).to_numpy() == str(value))
        for value in sorted(metadata["policy_class"].dropna().unique())
    )
    masks.extend(
        ("training_series", str(value), metadata["training_series"].astype(str).to_numpy() == str(value))
        for value in sorted(metadata["training_series"].dropna().unique())
    )
    for column in ("proposal_target", "policy_class", "selection_stratum", "proposal_series"):
        masks.extend(
            (
                f"adversarial_{column}",
                str(value),
                adversarial & (metadata[column].astype(str).to_numpy() == str(value)),
            )
            for value in sorted(metadata.loc[adversarial, column].dropna().unique())
        )
    combinations = metadata.loc[adversarial, ["proposal_target", "policy_class", "selection_stratum"]].drop_duplicates()
    for proposal, policy, selection in combinations.itertuples(index=False, name=None):
        value = f"{proposal}|{policy}|{selection}"
        masks.append(
            (
                "adversarial_target_policy_selection",
                value,
                adversarial
                & (metadata["proposal_target"].astype(str).to_numpy() == str(proposal))
                & (metadata["policy_class"].astype(str).to_numpy() == str(policy))
                & (metadata["selection_stratum"].astype(str).to_numpy() == str(selection)),
            )
        )
    rows = []
    for stratum, value, mask in masks:
        if int(mask.sum()) < 2:
            continue
        for model, predicted in predictions.items():
            rows.append(
                {
                    "target": target,
                    "model": model,
                    "stratum": stratum,
                    "stratum_value": value,
                    **scalar_metrics(observed[mask], predicted[mask]),
                }
            )
    return pd.DataFrame(rows)


def matched_cell_metrics(
    panel: swarm39.Panel,
    observed: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    untied = panel.phase_tv > 1e-9
    cells = split_damage.matched_cells(panel)
    rows = []
    for model, predicted in predictions.items():
        rhos, agreements, regrets, tied_regrets = [], [], [], []
        for cell in cells:
            base, alternatives = cell[~untied[cell]], cell[untied[cell]]
            observed_delta = observed[alternatives] - observed[base].mean()
            predicted_delta = predicted[alternatives] - predicted[base].mean()
            if len(alternatives) >= 4:
                rho = float(spearmanr(predicted_delta, observed_delta).statistic)
                if np.isfinite(rho):
                    rhos.append(rho)
            agreements.append(float(np.mean(np.sign(predicted_delta) == np.sign(observed_delta))))
            best = min(float(observed[alternatives].min()), float(observed[base].mean()))
            choice = alternatives[int(np.argmin(predicted_delta))] if predicted_delta.min() < 0.0 else None
            chosen = float(observed[choice]) if choice is not None else float(observed[base].mean())
            regrets.append(chosen - best)
            tied_regrets.append(float(observed[base].mean()) - best)
        rows.append(
            {
                "model": model,
                "matched_cell_count": len(cells),
                "median_cell_spearman": float(np.median(rhos)),
                "positive_spearman_cells": int(np.sum(np.asarray(rhos) > 0.0)),
                "spearman_cells": len(rhos),
                "mean_sign_agreement": float(np.mean(agreements)),
                "mean_decision_regret": float(np.mean(regrets)),
                "mean_stay_tied_regret": float(np.mean(tied_regrets)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = validate_protocol(PROTOCOL_PATH, args.target, args.outer_folds)

    fit_panel, heldout_panel = swarm39.load_scale("delphi_3e18")
    fit_observed = fit_panel.targets[args.target]
    heldout_observed = heldout_panel.targets[args.target]
    assert np.isfinite(fit_observed).all() and np.isfinite(heldout_observed).all()

    basis, basis_provenance = hkrr.load_embedding_basis(
        fit_panel.buckets,
        hkrr.DEFAULT_HISTOGRAM_DIR,
        hkrr.DEFAULT_LOOKUP,
    )
    split_oof = cross_fitted_split(fit_panel, fit_observed, args.seed, args.outer_folds)
    split_fit = fit_split(fit_panel, fit_observed, args.seed)
    split_heldout = predict_split(heldout_panel, split_fit)

    geometry: dict[str, GeometryFit] = {}
    residual_geometry: dict[str, GeometryFit] = {}
    blend_weights: dict[str, float] = {}
    predictions: dict[str, np.ndarray] = {"split": split_heldout}
    fit_oof: dict[str, np.ndarray] = {"split": split_oof}

    for kernel_space in ("weight", "content"):
        geometry[kernel_space] = fit_geometry(
            fit_panel,
            heldout_panel,
            basis,
            kernel_space,
            args.target,
            args.seed,
        )
        fit_oof[f"{kernel_space}_krr"] = geometry[kernel_space].model.oof_predictions
        predictions[f"{kernel_space}_krr"] = geometry[kernel_space].heldout_prediction

        blend = convex_blend_weight(fit_observed, split_oof, geometry[kernel_space].model.oof_predictions)
        blend_weights[kernel_space] = blend
        predictions[f"split_{kernel_space}_blend"] = (1.0 - blend) * split_heldout + blend * geometry[
            kernel_space
        ].heldout_prediction
        fit_oof[f"split_{kernel_space}_blend"] = (1.0 - blend) * split_oof + blend * geometry[
            kernel_space
        ].model.oof_predictions

        residual_geometry[kernel_space] = fit_residual_geometry(
            fit_panel,
            heldout_panel,
            basis,
            kernel_space,
            fit_observed - split_oof,
            args.seed,
        )
        predictions[f"split_{kernel_space}_residual"] = (
            split_heldout + residual_geometry[kernel_space].heldout_prediction
        )
        fit_oof[f"split_{kernel_space}_residual"] = split_oof + residual_geometry[kernel_space].model.oof_predictions

    predictions = {model: predictions[model] for model in MODEL_ORDER}
    fit_oof = {model: fit_oof[model] for model in MODEL_ORDER}
    metadata = pd.read_csv(swarm39.DELPHI_HELDOUTS)
    metadata = metadata[metadata["fit_panel_overlap"] == "coordinate_disjoint"].reset_index(drop=True)
    metadata = recover_adversarial_provenance(metadata)
    assert metadata["heldout_id"].astype(str).tolist() == heldout_panel.row_id.tolist()

    prediction_frame = metadata[
        [
            "heldout_id",
            "training_series",
            "policy_class",
            "proposal_target",
            "candidate_kind",
            "selection_stratum",
            "proposal_source_run",
            "proposal_series",
        ]
    ].copy()
    prediction_frame["target"] = args.target
    prediction_frame["observed"] = heldout_observed
    for model in MODEL_ORDER:
        prediction_frame[f"predicted::{model}"] = predictions[model]
    for kernel_space in ("weight", "content"):
        prediction_frame[f"support_distance::{kernel_space}"] = geometry[kernel_space].support_distance

    metrics = grouped_metrics(metadata, args.target, heldout_observed, predictions)
    cells = matched_cell_metrics(heldout_panel, heldout_observed, predictions)
    fit_metrics = pd.DataFrame(
        [
            {"target": args.target, "model": model, **scalar_metrics(fit_observed, predicted)}
            for model, predicted in fit_oof.items()
        ]
    )
    fit_metrics["evaluation"] = "canonical_280_cross_fitted"

    suffix = args.target.replace("_bpb", "")
    prediction_frame.to_csv(args.output_dir / f"predictions_{suffix}.csv", index=False)
    metrics.to_csv(args.output_dir / f"heldout_metrics_{suffix}.csv", index=False)
    cells.to_csv(args.output_dir / f"matched_cell_metrics_{suffix}.csv", index=False)
    fit_metrics.to_csv(args.output_dir / f"fit_oof_metrics_{suffix}.csv", index=False)

    summary = {
        "target": args.target,
        "seed": args.seed,
        "fit_rows": len(fit_panel),
        "heldout_rows": len(heldout_panel),
        "outer_folds": args.outer_folds,
        "protocol_sha256": protocol_hash(PROTOCOL_PATH),
        "protocol_created_at": protocol["created_at"],
        "basis": basis_provenance,
        "blend_weights": blend_weights,
        "kernel_parameters": {
            space: {
                "gamma": geometry[space].model.gamma,
                "ridge_alpha": geometry[space].model.ridge_alpha,
                "residual_gamma": residual_geometry[space].model.gamma,
                "residual_ridge_alpha": residual_geometry[space].model.ridge_alpha,
            }
            for space in ("weight", "content")
        },
    }
    (args.output_dir / f"summary_{suffix}.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    pooled = metrics[(metrics["stratum"] == "pooled") & (metrics["stratum_value"] == "all")]
    print(f"Frozen state/action-geometry round: {args.target}")
    print(f"protocol {summary['protocol_sha256']}; blend weights {blend_weights}")
    print(pooled[["model", "rmse", "r2", "spearman", "regret_at_1", "regret_at_20"]].to_string(index=False))
    print("\nAggregate-matched decisions")
    print(cells.to_string(index=False))


if __name__ == "__main__":
    main()
