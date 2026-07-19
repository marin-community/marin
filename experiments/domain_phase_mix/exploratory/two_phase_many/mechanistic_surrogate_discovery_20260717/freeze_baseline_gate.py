# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scipy",
# ]
# ///
"""Freeze the pre-search surrogate baseline and acceptance gate.

This script intentionally reads only the pre-existing Observatory snapshot and
pre-search deficit-model artifacts. It rejects any input containing the sealed
adversarial-panel identifiers. The emitted manifest is content-addressed so a
later search cannot silently move the acceptance gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
DEFAULT_DASHBOARD = RESEARCH_DIR / "mixture_fit_debugger/src/generated/dashboard_data.json"
DEFAULT_OUTPUT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717/frozen_gate"

SEALED_TOKENS = (
    "adversarial_stress",
    "dm-delphi-adversarial-stress-3e18-20260716",
    "stress_000_adv3e18c",
)
BASELINE_MODELS = (
    "effective_exposure",
    "effective_exposure_geometry",
    "separate_heads",
    "grp",
    "compact_retained_state",
    "bucket_family_grp",
    "hierarchical_phase_bucket_replay",
    "bucket_family_power_separate_heads",
)
CORE_OOF_PANELS = (
    ("300m", "uncheatable", "two_phase"),
    ("300m", "table9", "two_phase"),
    ("production", "uncheatable", "two_phase"),
    ("delphi_3e18", "uncheatable", "two_phase"),
    ("delphi_3e18", "table9", "two_phase"),
    ("starcoder_cosine", "starcoder_bpb", "two_phase"),
    ("starcoder_wsd80", "starcoder_bpb", "two_phase"),
)
LOWER_TAIL_FRACTION = 0.15
LOWER_TAIL_MIN_COUNT = 5
OPTIMISM_THRESHOLD = 0.05
CALIBRATION_BINS = 5


@dataclass(frozen=True)
class Gate:
    """Immutable quantitative promotion criteria."""

    core_oof_rmse_relative_tolerance: float = 0.05
    policy_matched_regret_at_1_absolute_tolerance: float = 0.002
    optimism_threshold_bpb: float = OPTIMISM_THRESHOLD
    optimism_count_rule: str = "candidate <= strongest baseline on both Delphi 3e18 targets"
    calibration_rule: str = "absolute distance of observed-on-predicted slope from 1 must not increase"
    material_improvement_rule: str = (
        "At least one: policy-matched heldout RMSE improves >=10%; optimism count drops >=2; "
        "worst optimism drops >=0.02 BPB; or |1-slope| drops >=20%."
    )
    mechanism_rule: str = "Every retained mechanism must survive a nested ablation."
    stability_rule: str = (
        "Parameter signs must agree in >=80% of grouped folds and transformed magnitudes must have "
        "bootstrap median absolute deviation <=50% of the absolute median unless the coefficient is zero."
    )
    raw_optimum_rule: str = (
        "Raw optimum must be finite, bootstrap-stable, and avoid a single-bucket corner or unsupported "
        "epoch explosion without relying on deployment regularization."
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_sealed_absent(path: Path) -> None:
    raw = path.read_bytes()
    for token in SEALED_TOKENS:
        if token.encode() in raw:
            raise ValueError(f"Sealed confirmatory token {token!r} found in {path}")


def finite_pairs(observed: Iterable[object], predicted: Iterable[object]) -> tuple[np.ndarray, np.ndarray]:
    y = pd.to_numeric(pd.Series(list(observed)), errors="coerce").to_numpy(dtype=float)
    prediction = pd.to_numeric(pd.Series(list(predicted)), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(prediction)
    return y[valid], prediction[valid]


def regret_at_k(observed: np.ndarray, predicted: np.ndarray, k: int) -> float:
    selected = np.argsort(predicted)[: min(k, len(predicted))]
    return float(np.min(observed[selected]) - np.min(observed))


def calibration_bins(observed: np.ndarray, predicted: np.ndarray) -> list[dict[str, float | int]]:
    order = np.argsort(predicted)
    output: list[dict[str, float | int]] = []
    for bin_index, indices in enumerate(np.array_split(order, min(CALIBRATION_BINS, len(order)))):
        if not len(indices):
            continue
        output.append(
            {
                "bin": bin_index,
                "n": len(indices),
                "mean_predicted": float(np.mean(predicted[indices])),
                "mean_observed": float(np.mean(observed[indices])),
                "mean_residual_predicted_minus_observed": float(np.mean(predicted[indices] - observed[indices])),
            }
        )
    return output


def metrics(
    observed: Iterable[object], predicted: Iterable[object]
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    y, prediction = finite_pairs(observed, predicted)
    if len(y) < 3:
        raise ValueError(f"Need at least three finite observations, got {len(y)}")
    residual = prediction - y
    optimism = y - prediction
    slope, intercept = np.polyfit(prediction, y, 1)
    tail_count = min(len(y), max(LOWER_TAIL_MIN_COUNT, math.ceil(LOWER_TAIL_FRACTION * len(y))))
    tail = np.argsort(prediction)[:tail_count]
    selected = int(np.argmin(prediction))
    summary: dict[str, float | int] = {
        "n": len(y),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "spearman": float(spearmanr(y, prediction).statistic),
        "bias_predicted_minus_observed": float(np.mean(residual)),
        "calibration_slope_observed_on_predicted": float(slope),
        "calibration_intercept_observed_on_predicted": float(intercept),
        "regret_at_1": regret_at_k(y, prediction, 1),
        "regret_at_3": regret_at_k(y, prediction, 3),
        "regret_at_5": regret_at_k(y, prediction, 5),
        "lower_tail_optimism": float(np.mean(np.maximum(optimism[tail], 0.0))),
        "low_tail_rmse": float(np.sqrt(np.mean(residual[tail] ** 2))),
        "optimism_gt_0p05_count": int(np.sum(optimism > OPTIMISM_THRESHOLD)),
        "worst_optimism": float(np.max(optimism)),
        "selected_optimism": float(optimism[selected]),
        "selected_observed": float(y[selected]),
        "selected_predicted": float(prediction[selected]),
    }
    return summary, calibration_bins(y, prediction)


def row_mask(rows: list[Mapping[str, Any]], split: str, policy: str | None) -> np.ndarray:
    mask = np.asarray([row["split"] == split for row in rows], dtype=bool)
    if split == "heldout":
        mask &= np.asarray([not bool(row["isSharedAlias"]) for row in rows], dtype=bool)
    if policy is not None:
        mask &= np.asarray([row["policyFamily"] == policy for row in rows], dtype=bool)
    return mask


def dashboard_metric_rows(bundle: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    for swarm_id, swarm in bundle["swarms"].items():
        rows = swarm["rows"]
        for target_id, policies in swarm["predictions"].items():
            for policy, models in policies.items():
                for model_id in BASELINE_MODELS:
                    if model_id not in models:
                        continue
                    prediction = np.asarray(models[model_id]["prediction"], dtype=float)
                    observed = np.asarray([row["observed"].get(target_id) for row in rows], dtype=float)
                    split_specs = [("fit_oof", "fit", None)]
                    if any(row["split"] == "heldout" and not row["isSharedAlias"] for row in rows):
                        split_specs.extend(
                            [
                                ("heldout_all", "heldout", None),
                                ("heldout_policy_matched", "heldout", policy),
                            ]
                        )
                    for label, split, policy_filter in split_specs:
                        mask = row_mask(rows, split, policy_filter)
                        if int(mask.sum()) < 3:
                            continue
                        summary, bins = metrics(observed[mask], prediction[mask])
                        parameter_count = bundle["swarms"][swarm_id]["fits"][target_id][policy][model_id][
                            "parameterCount"
                        ]
                        metric_rows.append(
                            {
                                "source": "dashboard",
                                "swarm": swarm_id,
                                "target": target_id,
                                "policy": policy,
                                "model": model_id,
                                "split": label,
                                "parameter_count": parameter_count,
                                **summary,
                            }
                        )
                        bin_rows.extend(
                            {
                                "source": "dashboard",
                                "swarm": swarm_id,
                                "target": target_id,
                                "policy": policy,
                                "model": model_id,
                                "split": label,
                                **row,
                            }
                            for row in bins
                        )
    return metric_rows, bin_rows


def external_metric_rows(research_dir: Path, bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read pre-search inverse-deficit baselines not yet wired into Observatory."""

    selections = (
        (
            "model_improvement_round2_conditioned_replay_link_20260716/metrics.csv",
            "inverse_power_deficit_conditioned_replay",
            "log_reducible_bpb",
            "inverse_deficit_log_link",
        ),
        (
            "deficit_output_link_asymmetric_20260716/metrics.csv",
            "inverse_power_deficit_early_family_asymmetric_surplus",
            None,
            "early_family_asymmetric",
        ),
    )
    output: list[dict[str, Any]] = []
    delphi_rows = {
        row["name"]: row
        for row in bundle["swarms"]["delphi_3e18"]["rows"]
        if row["split"] == "heldout" and not row["isSharedAlias"]
    }
    for relative_path, variant, link, model_name in selections:
        path = research_dir / "reference_outputs" / relative_path
        assert_sealed_absent(path)
        prediction_path = path.with_name("predictions.csv")
        assert_sealed_absent(prediction_path)
        predictions = pd.read_csv(prediction_path)
        selected_predictions = predictions.loc[predictions["deficit_variant"].eq(variant)]
        if link is not None:
            selected_predictions = selected_predictions.loc[selected_predictions["link"].eq(link)]
        else:
            selected_predictions = selected_predictions.loc[
                (
                    selected_predictions["dataset"].str.contains("uncheatable")
                    & selected_predictions["link"].eq("identity_raw_bpb")
                )
                | (
                    selected_predictions["dataset"].str.contains("table9")
                    & selected_predictions["link"].eq("log_reducible_bpb")
                )
            ]
        for dataset, local in selected_predictions.groupby("dataset", sort=True):
            target = "table9" if "table9" in dataset else "uncheatable"
            for split, split_frame in local.groupby("split", sort=True):
                split_specs = [("fit_oof", split_frame)] if split == "fit_oof" else [("heldout_all", split_frame)]
                if split == "heldout":
                    policy_matched = split_frame.loc[
                        split_frame["row_id"].map(lambda row_id: delphi_rows[str(row_id)]["policyFamily"]) == "two_phase"
                    ]
                    split_specs.append(("heldout_policy_matched", policy_matched))
                for split_name, metric_frame in split_specs:
                    summary, _bins = metrics(metric_frame["observed"], metric_frame["predicted"])
                    output.append(
                        {
                            "source": str(prediction_path.relative_to(research_dir)),
                            "swarm": "delphi_3e18",
                            "target": target,
                            "policy": "two_phase",
                            "model": model_name,
                            "split": split_name,
                            "parameter_count": "see source fit artifact",
                            **summary,
                        }
                    )
    return output


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, divider]
    for record in frame[columns].to_dict(orient="records"):
        values = []
        for column in columns:
            value = record[column]
            if isinstance(value, float):
                values.append(f"{value:.5f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard", type=Path, default=DEFAULT_DASHBOARD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    assert_sealed_absent(args.dashboard)
    bundle = json.loads(args.dashboard.read_text())
    metric_rows, bin_rows = dashboard_metric_rows(bundle)
    metric_rows.extend(external_metric_rows(RESEARCH_DIR, bundle))
    metrics_frame = pd.DataFrame(metric_rows)
    bins_frame = pd.DataFrame(bin_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(args.output_dir / "baseline_metrics.csv", index=False)
    bins_frame.to_csv(args.output_dir / "calibration_bins.csv", index=False)

    gate = Gate()
    gate_record = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "dashboard": str(args.dashboard.relative_to(RESEARCH_DIR)),
        "dashboard_sha256": sha256(args.dashboard),
        "sealed_tokens_checked_absent": list(SEALED_TOKENS),
        "baseline_models": list(BASELINE_MODELS),
        "core_oof_panels": [list(panel) for panel in CORE_OOF_PANELS],
        "acceptance_gate": asdict(gate),
    }
    gate_path = args.output_dir / "acceptance_gate.json"
    gate_path.write_text(json.dumps(gate_record, indent=2, sort_keys=True) + "\n")
    manifest = {
        "acceptance_gate_sha256": sha256(gate_path),
        "baseline_metrics_sha256": sha256(args.output_dir / "baseline_metrics.csv"),
        "calibration_bins_sha256": sha256(args.output_dir / "calibration_bins.csv"),
    }
    manifest_path = args.output_dir / "frozen_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    focus = metrics_frame.loc[
        metrics_frame["swarm"].eq("delphi_3e18") & metrics_frame["split"].isin(["fit_oof", "heldout_policy_matched"])
    ].copy()
    focus = focus.sort_values(["target", "split", "rmse", "model"])
    report = [
        "# Frozen surrogate baseline and acceptance gate",
        "",
        f"Frozen manifest digest: `{sha256(manifest_path)}`.",
        "",
        "The adversarial stress panel is sealed and absent from every input. The 12 exact-coordinate aliases are excluded from heldout metrics.",
        "",
        "## Delphi 3e18 baseline",
        "",
        markdown_table(
            focus,
            [
                "target",
                "split",
                "model",
                "n",
                "rmse",
                "spearman",
                "regret_at_1",
                "calibration_slope_observed_on_predicted",
                "optimism_gt_0p05_count",
                "worst_optimism",
            ],
        ),
        "",
        "## Immutable promotion gate",
        "",
        *[f"- `{key}`: {value}" for key, value in asdict(gate).items()],
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps({**manifest, "manifest_sha256": sha256(manifest_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
