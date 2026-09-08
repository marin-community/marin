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
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
#   "wandb",
# ]
# ///
"""Evaluate the frozen intervention-identified signed dose potential.

This evaluator is deliberately staged. It first selects the aggregate shape on
the full 60M conditional-dose panel, excluding x32. It will not materialize or
read the Delphi full-panel outcomes until ``selected_60m.json`` has been
written. The 300M endpoint panel is evaluated only after the same structure is
frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pandas as pd
import wandb
from scipy import stats
from scipy.optimize import lsq_linear, minimize

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_bucket_epoch_dose_pilot_20260730 as pilot,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_compact_tied_backbone_20260730 as compact_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_physical_hpr_tied_spine_20260731 as physical_spine,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_intervention_identified_signed_dose_potential_20260731 as model,
)

DEFAULT_OUTPUT_DIR = model.DEFAULT_OUTPUT_DIR
PANEL_DIR = model.SCRIPT_DIR / "reference_outputs" / "bucket_epoch_dose_response_20260729" / "full"
PILOT_RESULTS_PATH = pilot.DEFAULT_OUTPUT_DIR / "observations.csv"
PILOT_PANEL_DIR = pilot.PANEL_DIR
PARENT_PROTOCOL_PATH = DEFAULT_OUTPUT_DIR / "protocol.json"
EVALUATION_PROTOCOL_VERSION = "signed-dose-potential-evaluation-v2"

TRAIN_PROJECT = "marin-community/marin"
EVAL_PROJECT = "marin-community/marin-eval"
UNCHEATABLE_METRIC = "eval/uncheatable_eval/bpb"
TABLE9_METRIC = "olmo_base_easy/table9_macro_bpb"
EXPECTED_ROWS = 277
TARGET_COLUMNS = {
    "uncheatable": "uncheatable_bpb",
    "table9": "table9_macro_bpb",
}

SCALE_CONFIGS = {
    "60m": {
        "train_filter": {"tags": "pinlin_calvin_xu/data_mixture/be60_20260729"},
        "eval_groups": ("olmo_base_eval_table9_bucket_epoch_dose_60m_full_20260729",),
    },
    "delphi_3e18": {
        "train_filter": {"tags": "bucket-epoch-dose-response"},
        "eval_groups": (
            "olmo_base_eval_table9_bucket_epoch_dose_delphi_full_20260729",
            "olmo_base_eval_table9_bucket_epoch_dose_delphi_3e18_full_20260729",
        ),
    },
}

INNER_BOOTSTRAP_DRAWS = 2_000
INNER_BOOTSTRAP_SEED = 7_317_304
CURVATURE_ACTIVE_TOLERANCE = 1e-8
SHAPE_FAMILYWISE_ALPHA = 0.05
AFFINE_SCALE_ACTIVE_TOLERANCE = 1e-10
OPTIMIZER_STARTS = 8
OPTIMIZER_SEED = 7_317_305


@dataclass(frozen=True, order=True)
class FitConfig:
    """One frozen nonlinear shape and estimator setting."""

    generator_order: float
    curvature_mode: str
    ridge: float


@dataclass(frozen=True)
class FittedPotential:
    """One target-specific fitted convex aggregate response."""

    config: FitConfig
    coefficients: np.ndarray
    feature_scale: np.ndarray
    geometry: model.Geometry
    effective_df: float

    def predict(self, weights: np.ndarray) -> np.ndarray:
        design = config_design(weights, self.geometry, self.config)
        return design.matrix @ self.coefficients

    def bucket_utility(self) -> np.ndarray:
        design = config_design(self.geometry.proportional[None, :], self.geometry, self.config)
        gauge = self.coefficients[design.utility_slice]
        return model.recover_bucket_utility(gauge, self.geometry)

    def curvature(self) -> np.ndarray:
        design = config_design(self.geometry.proportional[None, :], self.geometry, self.config)
        return self.coefficients[design.curvature_slice]


@dataclass(frozen=True)
class SelectedConfig:
    """Nested structural-selection result and its evidence."""

    config: FitConfig
    entropy_global_rmse: float
    entropy_family_rmse: float
    family_minus_global_ci_low: float
    family_minus_global_ci_high: float
    selected_entropy_rmse: float
    best_nonentropy_rmse: float
    nonentropy_minus_entropy_ci_low: float
    nonentropy_minus_entropy_ci_high: float
    shape_comparison_confidence: float
    retained_family_extension: bool
    retained_nonentropy_extension: bool
    nonentropy_comparisons: tuple[dict[str, float | bool], ...]


@dataclass(frozen=True)
class TrainingOutcome:
    """One selected training run and its persisted final evaluation."""

    run: Any
    value: float
    source: str
    source_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "prepare",
            "materialize-60m",
            "evaluate-60m",
            "materialize-delphi",
            "evaluate-delphi",
            "evaluate-300m",
        ),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--panel-dir", type=Path, default=PANEL_DIR)
    parser.add_argument("--wandb-timeout", type=int, default=180)
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    return model.json_ready(value)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return model.canonical_hash(payload)


def evaluation_protocol_payload(output_dir: Path, panel_dir: Path) -> dict[str, Any]:
    parent_path = output_dir / "protocol.json"
    if not parent_path.exists():
        raise FileNotFoundError(f"Freeze the parent protocol first: {parent_path}")
    parent = json.loads(parent_path.read_text())
    if parent["protocol_sha256"] != model.protocol_payload()["protocol_sha256"]:
        raise ValueError("Parent source differs from the frozen aggregate protocol")

    manifest_hashes = {}
    for scale in SCALE_CONFIGS:
        path = panel_dir / scale / "run_manifest.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        manifest_hashes[str(path.relative_to(REPO_ROOT))] = sha256(path)

    evidence_paths = (
        PILOT_RESULTS_PATH,
        PILOT_PANEL_DIR / "delphi_3e18" / "phase_weights.csv",
        physical_spine.benchmark.PACKET,
        physical_spine.benchmark.ONE_PHASE_SOURCE,
    )
    evidence_hashes = {}
    for path in evidence_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        evidence_hashes[str(path.relative_to(REPO_ROOT))] = sha256(path)

    payload: dict[str, Any] = {
        "candidate_id": model.CANDIDATE_ID,
        "version": EVALUATION_PROTOCOL_VERSION,
        "parent_protocol_sha256": parent["protocol_sha256"],
        "staged_disclosure": {
            "stage_1": "select structure on 60M full multipliers through x16",
            "stage_1_holdout": "x32 cannot select structure or ridge",
            "stage_2": "Delphi full outcomes remain unread until selected_60m.json exists",
            "stage_3": "300M endpoint evaluation uses the stage-1 structure without shape-grid extension",
        },
        "materialization": {
            "training_identity": "exact run-name tag, with scale-specific display-name fallback",
            "training_metric": (
                "use a finite W&B summary when available; otherwise recover the exact expected final step "
                "from the run's persisted checkpoints/eval_metrics.jsonl"
            ),
            "retry_consistency": "all successful values for one coordinate must agree within 1e-10 BPB",
            "table9_metric": "finite native Table-9 W&B summary keyed by manifest run_name provenance",
        },
        "estimator": {
            "head": "bounded least squares with free intercept and utility, nonnegative curvature",
            "ridge": "utility-gauge coefficients only after train-fold RMS scaling",
            "curvature": "unpenalized; every selected curvature must exceed the active tolerance",
            "active_tolerance": CURVATURE_ACTIVE_TOLERANCE,
        },
        "nested_selection": {
            "outer_folds": model.OUTER_FOLDS,
            "outer_seed": model.OUTER_SEED,
            "outer_design": "bucket-stratified deterministic rows; every bucket contributes across folds",
            "inner_design": "leave one multiplier out across buckets; proportional anchor remains train-only",
            "linear_ablation": "same signed utility head with all curvature columns removed",
            "family_extension": "retain only if bucket-cluster bootstrap RMSE-difference upper bound is below zero",
            "shape_extension": (
                "retain q!=0 only when its bucket-cluster bootstrap upper bound relative to q=0 "
                "is below zero after Bonferroni correction over the three non-entropy orders"
            ),
            "bootstrap_draws": INNER_BOOTSTRAP_DRAWS,
            "bootstrap_seed": INNER_BOOTSTRAP_SEED,
            "shape_familywise_alpha": SHAPE_FAMILYWISE_ALPHA,
            "tie_break": "lower RMSE, q=0, global curvature, larger ridge",
        },
        "cross_scale": {
            "form_transfer": "freeze q, curvature mode, and ridge at 60M; refit only coefficients in fixed outer folds",
            "strict_transfer": (
                "fit the 60M source potential once; estimate a positive affine BPB map from the eight "
                "already-exposed unique Delphi pilot coordinates; score only nonoverlapping full-panel coordinates"
            ),
            "strict_metrics": "absolute BPB metrics plus intervention-effect Spearman and sign accuracy",
            "x32": "report after selection; never use for gates or affine calibration",
        },
        "300m": {
            "dataset": "282 physically tied policies from benchmark_physical_hpr_tied_spine_20260731.tied_dataset",
            "folds": "five correspondence-group folds using the frozen outer seed",
            "strict_transfer": "fixed 60M source potential plus train-fold positive affine BPB map",
            "form_refit": "freeze q, curvature mode, and ridge; refit only coefficients in each train fold",
            "linear_ablation": "frozen 60M-selected linear ridge with coefficients refit in each train fold",
            "raw_optimum": "convex simplex optimization before deployment regularization",
            "raw_optimism": "best-observed tied BPB minus predicted-optimum BPB may not exceed two candidate OOF RMSEs",
            "bootstrap": {
                "replicates": model.OPTIMUM_BOOTSTRAP_REPLICATES,
                "seed": model.OPTIMUM_BOOTSTRAP_SEED,
                "unit": "phase_correspondence_key",
            },
        },
        "targets": model.TARGETS,
        "selection_multipliers": model.SELECTION_MULTIPLIERS,
        "extrapolation_multiplier": model.EXTRAPOLATION_MULTIPLIER,
        "gates": model.GATES,
        "manifest_hashes": manifest_hashes,
        "evidence_hashes": evidence_hashes,
        "source_hashes": {
            str(Path(__file__).relative_to(REPO_ROOT)): sha256(Path(__file__)),
            str(Path(model.__file__).relative_to(REPO_ROOT)): sha256(Path(model.__file__)),
            str(Path(pilot.__file__).relative_to(REPO_ROOT)): sha256(Path(pilot.__file__)),
            str(Path(compact_audit.__file__).relative_to(REPO_ROOT)): sha256(Path(compact_audit.__file__)),
            str(Path(physical_spine.__file__).relative_to(REPO_ROOT)): sha256(Path(physical_spine.__file__)),
        },
    }
    payload["evaluation_protocol_sha256"] = canonical_hash(payload)
    return payload


def write_if_absent_or_equal(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"Frozen artifact differs from current source: {path}")
        return
    path.write_text(content)


def write_frame_if_absent_or_equal(path: Path, frame: pd.DataFrame) -> None:
    write_if_absent_or_equal(path, frame.to_csv(index=False))


def prepare(output_dir: Path, panel_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json_ready(evaluation_protocol_payload(output_dir, panel_dir))
    write_if_absent_or_equal(
        output_dir / "evaluation_protocol.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def verify_evaluation_protocol(output_dir: Path, panel_dir: Path) -> dict[str, Any]:
    path = output_dir / "evaluation_protocol.json"
    if not path.exists():
        raise FileNotFoundError(f"Freeze the evaluation protocol first: {path}")
    frozen = json.loads(path.read_text())
    current = json_ready(evaluation_protocol_payload(output_dir, panel_dir))
    if frozen != current:
        raise ValueError("Evaluator source, parent protocol, or design manifests differ from the frozen protocol")
    return frozen


def finite_summary(run: Any, key: str) -> float | None:
    try:
        value = float(run.summary.get(key, np.nan))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def run_name_matches(scale: str, run: Any, run_name: str) -> bool:
    if run_name in set(run.tags or ()):
        return True
    if scale == "60m":
        return str(run.name).endswith(f"/{run_name}")
    return str(run.name).startswith(f"{run_name}-")


def persisted_training_metric(run: Any, key: str, expected_step: int) -> tuple[float, str] | None:
    trainer = dict(run.config.get("trainer") or {})
    checkpointer = dict(trainer.get("checkpointer") or {})
    checkpoint_path = str(checkpointer.get("base_path") or "")
    if not checkpoint_path:
        return None

    metrics_path = f"{checkpoint_path.rstrip('/')}/eval_metrics.jsonl"
    filesystem, path = fsspec.core.url_to_fs(metrics_path)
    if not filesystem.exists(path):
        return None

    values = []
    with filesystem.open(path, "rt") as handle:
        for line in handle:
            payload = json.loads(line)
            if int(payload.get("step", -1)) != expected_step:
                continue
            try:
                value = float(payload.get(key, np.nan))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
    if not values:
        return None
    if max(values) - min(values) > 1e-10:
        raise ValueError(f"{run.name}: persisted final-step outcomes disagree: {values}")
    return values[-1], metrics_path


def training_metric(run: Any, key: str, expected_step: int) -> tuple[float, str, str] | None:
    summary_value = finite_summary(run, key)
    if summary_value is not None:
        return summary_value, "wandb_summary", ""
    persisted = persisted_training_metric(run, key, expected_step)
    if persisted is None:
        return None
    value, path = persisted
    return value, "gcs_checkpoint_eval_metrics", path


def selected_training_outcome(
    scale: str,
    runs: list[Any],
    run_name: str,
    expected_step: int,
) -> TrainingOutcome:
    candidates = []
    for run in runs:
        if not run_name_matches(scale, run, run_name):
            continue
        metric = training_metric(run, UNCHEATABLE_METRIC, expected_step)
        if metric is not None:
            candidates.append((run, *metric))
    if not candidates:
        raise ValueError(f"{scale}/{run_name}: no finite training outcome")
    values = [value for _, value, _, _ in candidates]
    if max(values) - min(values) > 1e-10:
        raise ValueError(f"{scale}/{run_name}: successful training retries disagree: {values}")
    run, value, source, path = max(candidates, key=lambda item: str(item[0].created_at))
    return TrainingOutcome(run=run, value=value, source=source, source_path=path)


def selected_eval_runs(runs: list[Any]) -> dict[str, Any]:
    candidates: dict[str, list[Any]] = {}
    for run in runs:
        provenance = dict(run.config.get("provenance") or {})
        run_name = str(provenance.get("run_name") or "")
        if not run_name or finite_summary(run, TABLE9_METRIC) is None:
            continue
        candidates.setdefault(run_name, []).append(run)

    selected = {}
    for run_name, attempts in candidates.items():
        values = [float(finite_summary(run, TABLE9_METRIC)) for run in attempts]
        if max(values) - min(values) > 1e-10:
            raise ValueError(f"{run_name}: successful Table-9 retries disagree: {values}")
        selected[run_name] = max(attempts, key=lambda run: str(run.created_at))
    return selected


def materialize_scale(scale: str, output_dir: Path, panel_dir: Path, timeout: int) -> None:
    protocol = verify_evaluation_protocol(output_dir, panel_dir)
    if scale == model.CROSS_SCALE_VALIDATION:
        require_60m_gate(output_dir, protocol)

    config = SCALE_CONFIGS[scale]
    manifest = pd.read_csv(panel_dir / scale / "run_manifest.csv")
    if len(manifest) != EXPECTED_ROWS or manifest["run_name"].duplicated().any():
        raise ValueError(f"{scale}: expected {EXPECTED_ROWS} unique manifest rows")

    api = wandb.Api(timeout=timeout)
    training_runs = list(api.runs(TRAIN_PROJECT, filters=config["train_filter"], per_page=500))
    eval_runs: list[Any] = []
    for group in config["eval_groups"]:
        eval_runs.extend(api.runs(EVAL_PROJECT, filters={"group": group}, per_page=500))
    evaluations = selected_eval_runs(eval_runs)

    rows = []
    missing = []
    for _, spec in manifest.iterrows():
        run_name = str(spec["run_name"])
        try:
            training = selected_training_outcome(
                scale,
                training_runs,
                run_name,
                int(spec["expected_checkpoint_step"]),
            )
            evaluation = evaluations[run_name]
        except (KeyError, ValueError) as error:
            missing.append({"run_name": run_name, "reason": str(error)})
            continue
        rows.append(
            {
                **spec.to_dict(),
                "uncheatable_bpb": training.value,
                "table9_macro_bpb": finite_summary(evaluation, TABLE9_METRIC),
                "training_metric_source": training.source,
                "training_metric_source_path": training.source_path,
                "training_wandb_id": training.run.id,
                "training_wandb_url": training.run.url,
                "table9_wandb_id": evaluation.id,
                "table9_wandb_url": evaluation.url,
            }
        )

    if missing:
        (output_dir / f"materialization_missing_{scale}.json").write_text(
            json.dumps(missing, indent=2, sort_keys=True) + "\n"
        )
        raise ValueError(f"{scale}: {len(missing)} of {EXPECTED_ROWS} outcomes are incomplete")
    observations = pd.DataFrame(rows)
    if len(observations) != EXPECTED_ROWS or observations[["uncheatable_bpb", "table9_macro_bpb"]].isna().any().any():
        raise ValueError(f"{scale}: materialized outcomes are incomplete")

    path = output_dir / f"observations_{scale}.csv"
    write_frame_if_absent_or_equal(path, observations)
    append_data_use(
        output_dir,
        stage=f"materialize_{scale}",
        outcomes_inspected=f"complete {scale} full outcomes",
        purpose="materialize source selection panel" if scale == "60m" else "materialize frozen cross-scale validation",
        protocol_sha256=protocol["evaluation_protocol_sha256"],
    )
    print(json.dumps({"scale": scale, "rows": len(observations), "path": str(path)}, indent=2))


def append_data_use(
    output_dir: Path,
    *,
    stage: str,
    outcomes_inspected: str,
    purpose: str,
    protocol_sha256: str,
) -> None:
    path = output_dir / "data_use_ledger.csv"
    rows = list(csv.DictReader(path.open())) if path.exists() else []
    if any(row["stage"] == stage for row in rows):
        return
    with path.open("a", newline="") as handle:
        fieldnames = ("timestamp", "candidate_id", "stage", "outcomes_inspected", "purpose", "protocol_sha256")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not rows:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": pd.Timestamp.now(tz="America/Los_Angeles").isoformat(),
                "candidate_id": model.CANDIDATE_ID,
                "stage": stage,
                "outcomes_inspected": outcomes_inspected,
                "purpose": purpose,
                "protocol_sha256": protocol_sha256,
            }
        )


def panel_rows(observations: pd.DataFrame) -> tuple[pd.DataFrame, model.Geometry, np.ndarray]:
    points, geometry, weights = model.panel_design()
    point_frame = pd.DataFrame(
        {
            "point_id": [point.point_id for point in points],
            "point_kind_model": [point.point_kind for point in points],
            "focal_index_model": [point.focal_index for point in points],
            "focal_domain_model": [point.focal_domain for point in points],
            "epoch_multiplier_model": [point.epoch_multiplier for point in points],
            "model_row": np.arange(len(points)),
        }
    )
    frame = observations.merge(point_frame, on="point_id", how="left", validate="one_to_one")
    if frame["model_row"].isna().any():
        raise ValueError("Materialized outcomes contain an unknown point ID")
    frame = frame.sort_values("model_row").reset_index(drop=True)
    ordered_weights = weights[frame["model_row"].to_numpy(int)]
    if not np.allclose(ordered_weights.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("Joined model policies are not normalized")
    return frame, geometry, ordered_weights


def config_design(weights: np.ndarray, geometry: model.Geometry, config: FitConfig) -> model.FeatureDesign:
    if config.curvature_mode == "linear":
        base = model.feature_design(weights, geometry, generator_order=0.0, curvature_mode="global")
        matrix = base.matrix[:, : base.curvature_slice.start]
        return model.FeatureDesign(
            matrix=matrix,
            utility_slice=base.utility_slice,
            curvature_slice=slice(matrix.shape[1], matrix.shape[1]),
            parameter_names=base.parameter_names[: matrix.shape[1]],
        )
    return model.feature_design(
        weights,
        geometry,
        generator_order=config.generator_order,
        curvature_mode=config.curvature_mode,
    )


def fit_potential(
    weights: np.ndarray,
    target: np.ndarray,
    geometry: model.Geometry,
    config: FitConfig,
) -> FittedPotential:
    design = config_design(weights, geometry, config)
    matrix = design.matrix
    scale = np.ones(matrix.shape[1])
    scale[1:] = np.maximum(np.sqrt(np.mean(matrix[:, 1:] ** 2, axis=0)), 1e-10)
    scaled = matrix / scale[None, :]

    penalty_indices = np.arange(design.utility_slice.start, design.utility_slice.stop)
    if config.ridge > 0.0:
        penalty = np.zeros((len(penalty_indices), matrix.shape[1]))
        penalty[np.arange(len(penalty_indices)), penalty_indices] = math.sqrt(config.ridge)
        augmented_matrix = np.vstack([scaled, penalty])
        augmented_target = np.concatenate([target, np.zeros(len(penalty_indices))])
    else:
        augmented_matrix = scaled
        augmented_target = target

    lower = np.full(matrix.shape[1], -np.inf)
    upper = np.full(matrix.shape[1], np.inf)
    lower[design.curvature_slice] = 0.0
    result = lsq_linear(
        augmented_matrix,
        augmented_target,
        bounds=(lower, upper),
        method="trf",
        max_iter=5_000,
        tol=1e-12,
        lsmr_tol=1e-12,
    )
    if not np.isfinite(result.x).all():
        raise RuntimeError(f"Non-finite fit for {config}")
    coefficients = result.x / scale

    active = np.ones(matrix.shape[1], dtype=bool)
    if design.curvature_slice.stop > design.curvature_slice.start:
        active[design.curvature_slice] = coefficients[design.curvature_slice] > CURVATURE_ACTIVE_TOLERANCE
    active_matrix = scaled[:, active]
    penalty_diagonal = np.zeros(int(active.sum()))
    original_active_indices = np.flatnonzero(active)
    penalty_diagonal[np.isin(original_active_indices, penalty_indices)] = config.ridge
    gram = active_matrix.T @ active_matrix + np.diag(penalty_diagonal)
    effective_df = float(np.trace(active_matrix @ np.linalg.pinv(gram) @ active_matrix.T))
    return FittedPotential(
        config=config,
        coefficients=coefficients,
        feature_scale=scale,
        geometry=geometry,
        effective_df=effective_df,
    )


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(predicted) - np.asarray(observed)) ** 2)))


def metric_summary(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = np.asarray(predicted) - np.asarray(observed)
    slope, intercept = np.polyfit(predicted, observed, deg=1)
    return {
        "rmse": rmse(observed, predicted),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "spearman": float(stats.spearmanr(observed, predicted).statistic),
        "observed_on_predicted_slope": float(slope),
        "observed_on_predicted_intercept": float(intercept),
    }


def selection_mask(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["point_kind_model"].eq("focal_bucket_dose").to_numpy()
        & frame["epoch_multiplier_model"].isin(model.SELECTION_MULTIPLIERS).to_numpy()
    )


def anchor_indices(frame: pd.DataFrame) -> np.ndarray:
    indices = np.flatnonzero(frame["point_kind_model"].eq("proportional_anchor").to_numpy())
    if len(indices) != 1:
        raise ValueError(f"Expected one proportional anchor, found {len(indices)}")
    return indices


def inner_multiplier_predictions(
    weights: np.ndarray,
    target: np.ndarray,
    frame: pd.DataFrame,
    geometry: model.Geometry,
    config: FitConfig,
    available: np.ndarray,
) -> np.ndarray:
    predictions = np.full(len(frame), np.nan)
    anchors = anchor_indices(frame)
    multipliers = frame["epoch_multiplier_model"].to_numpy(float)
    available_multipliers = sorted(set(multipliers[available]))
    for multiplier in available_multipliers:
        test = available[np.isclose(multipliers[available], multiplier)]
        train = available[~np.isclose(multipliers[available], multiplier)]
        train = np.unique(np.concatenate([train, anchors]))
        fitted = fit_potential(weights[train], target[train], geometry, config)
        predictions[test] = fitted.predict(weights[test])
    if not np.isfinite(predictions[available]).all():
        raise ValueError(f"Incomplete inner predictions for {config}")
    return predictions


def best_ridge(
    weights: np.ndarray,
    target: np.ndarray,
    frame: pd.DataFrame,
    geometry: model.Geometry,
    available: np.ndarray,
    *,
    generator_order: float,
    curvature_mode: str,
) -> tuple[FitConfig, np.ndarray, float]:
    rows = []
    for ridge in model.RIDGE_GRID:
        config = FitConfig(generator_order, curvature_mode, ridge)
        predictions = inner_multiplier_predictions(weights, target, frame, geometry, config, available)
        rows.append((rmse(target[available], predictions[available]), -ridge, config, predictions))
    score, _negative_ridge, config, predictions = min(rows, key=lambda item: (item[0], item[1]))
    return config, predictions, score


def cluster_bootstrap_rmse_difference(
    observed: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    clusters: np.ndarray,
    *,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"Bootstrap alpha must lie in (0, 1), found {alpha}")
    unique = np.unique(clusters)
    generator = np.random.default_rng(seed)
    draws = np.empty(INNER_BOOTSTRAP_DRAWS)
    for draw in range(INNER_BOOTSTRAP_DRAWS):
        sampled = generator.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(clusters == cluster) for cluster in sampled])
        draws[draw] = rmse(observed[indices], candidate[indices]) - rmse(observed[indices], reference[indices])
    point = rmse(observed, candidate) - rmse(observed, reference)
    low, high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return point, float(low), float(high)


def select_config(
    weights: np.ndarray,
    target: np.ndarray,
    frame: pd.DataFrame,
    geometry: model.Geometry,
    available: np.ndarray,
    *,
    seed: int,
) -> SelectedConfig:
    global_config, global_prediction, global_rmse = best_ridge(
        weights,
        target,
        frame,
        geometry,
        available,
        generator_order=0.0,
        curvature_mode="global",
    )
    family_config, family_prediction, family_rmse = best_ridge(
        weights,
        target,
        frame,
        geometry,
        available,
        generator_order=0.0,
        curvature_mode="family",
    )
    clusters = frame.iloc[available]["focal_domain_model"].astype(str).to_numpy()
    _difference, family_low, family_high = cluster_bootstrap_rmse_difference(
        target[available],
        family_prediction[available],
        global_prediction[available],
        clusters,
        seed=seed,
    )
    retain_family = family_high < model.GATES["family_curvature_retain_only_if_bootstrap_difference_high_max"]
    entropy_config = family_config if retain_family else global_config
    entropy_prediction = family_prediction if retain_family else global_prediction
    entropy_rmse = family_rmse if retain_family else global_rmse

    nonentropy_rows = []
    for order in model.GENERATOR_ORDERS:
        if order == 0.0:
            continue
        config, prediction, score = best_ridge(
            weights,
            target,
            frame,
            geometry,
            available,
            generator_order=order,
            curvature_mode=entropy_config.curvature_mode,
        )
        nonentropy_rows.append((score, abs(order), order, config, prediction))
    best_nonentropy_rmse, _absolute_order, _order, nonentropy_config, _nonentropy_prediction = min(
        nonentropy_rows,
        key=lambda item: (item[0], item[1], item[2]),
    )
    comparison_alpha = SHAPE_FAMILYWISE_ALPHA / len(nonentropy_rows)
    comparison_rows = []
    for comparison_index, (score, _absolute_order, order, _config, prediction) in enumerate(nonentropy_rows):
        difference, low, high = cluster_bootstrap_rmse_difference(
            target[available],
            prediction[available],
            entropy_prediction[available],
            clusters,
            seed=seed + 1 + comparison_index,
            alpha=comparison_alpha,
        )
        comparison_rows.append(
            {
                "generator_order": order,
                "rmse": score,
                "rmse_difference": difference,
                "ci_low": low,
                "ci_high": high,
                "retained": high < model.GATES["nonentropy_shape_retain_only_if_bootstrap_difference_high_max"],
            }
        )
    selected_comparison = next(
        row for row in comparison_rows if row["generator_order"] == nonentropy_config.generator_order
    )
    shape_low = float(selected_comparison["ci_low"])
    shape_high = float(selected_comparison["ci_high"])
    retain_nonentropy = bool(selected_comparison["retained"])
    selected = nonentropy_config if retain_nonentropy else entropy_config
    return SelectedConfig(
        config=selected,
        entropy_global_rmse=global_rmse,
        entropy_family_rmse=family_rmse,
        family_minus_global_ci_low=family_low,
        family_minus_global_ci_high=family_high,
        selected_entropy_rmse=entropy_rmse,
        best_nonentropy_rmse=best_nonentropy_rmse,
        nonentropy_minus_entropy_ci_low=shape_low,
        nonentropy_minus_entropy_ci_high=shape_high,
        shape_comparison_confidence=1.0 - comparison_alpha,
        retained_family_extension=retain_family,
        retained_nonentropy_extension=retain_nonentropy,
        nonentropy_comparisons=tuple(comparison_rows),
    )


def outer_fold_ids(frame: pd.DataFrame, available: np.ndarray) -> np.ndarray:
    multipliers = list(model.SELECTION_MULTIPLIERS)
    fold_ids = np.full(len(frame), -1, dtype=int)
    for index in available:
        focal = int(frame.iloc[index]["focal_index_model"])
        multiplier = float(frame.iloc[index]["epoch_multiplier_model"])
        multiplier_index = multipliers.index(multiplier)
        fold_ids[index] = (focal + 2 * multiplier_index + model.OUTER_SEED) % model.OUTER_FOLDS
    counts = np.bincount(fold_ids[available], minlength=model.OUTER_FOLDS)
    if np.any(counts < 40):
        raise ValueError(f"Unbalanced outer folds: {counts.tolist()}")
    return fold_ids


def nested_oof(
    weights: np.ndarray,
    target: np.ndarray,
    frame: pd.DataFrame,
    geometry: model.Geometry,
) -> tuple[np.ndarray, np.ndarray, list[SelectedConfig], list[FitConfig]]:
    available = np.flatnonzero(selection_mask(frame))
    folds = outer_fold_ids(frame, available)
    anchors = anchor_indices(frame)
    candidate_prediction = np.full(len(frame), np.nan)
    linear_prediction = np.full(len(frame), np.nan)
    selections = []
    linear_configs = []

    for fold in range(model.OUTER_FOLDS):
        test = available[folds[available] == fold]
        train = available[folds[available] != fold]
        selection = select_config(
            weights,
            target,
            frame,
            geometry,
            train,
            seed=INNER_BOOTSTRAP_SEED + 100 * fold,
        )
        selections.append(selection)
        fit_rows = np.unique(np.concatenate([train, anchors]))
        fitted = fit_potential(weights[fit_rows], target[fit_rows], geometry, selection.config)
        candidate_prediction[test] = fitted.predict(weights[test])

        linear_config, _linear_inner, _score = best_ridge(
            weights,
            target,
            frame,
            geometry,
            train,
            generator_order=0.0,
            curvature_mode="linear",
        )
        linear_configs.append(linear_config)
        linear_fit = fit_potential(weights[fit_rows], target[fit_rows], geometry, linear_config)
        linear_prediction[test] = linear_fit.predict(weights[test])

    if not np.isfinite(candidate_prediction[available]).all() or not np.isfinite(linear_prediction[available]).all():
        raise ValueError("Incomplete nested OOF predictions")
    return candidate_prediction, linear_prediction, selections, linear_configs


def config_mode_fraction(config: FitConfig, selections: list[SelectedConfig]) -> float:
    return float(np.mean([selection.config.generator_order == config.generator_order for selection in selections]))


def evaluate_60m(output_dir: Path, panel_dir: Path) -> None:
    protocol = verify_evaluation_protocol(output_dir, panel_dir)
    observation_path = output_dir / "observations_60m.csv"
    if not observation_path.exists():
        raise FileNotFoundError("Materialize 60M outcomes before evaluation")
    observations = pd.read_csv(observation_path)
    frame, geometry, weights = panel_rows(observations)
    available = np.flatnonzero(selection_mask(frame))
    anchors = anchor_indices(frame)
    fit_rows = np.unique(np.concatenate([available, anchors]))
    x32 = np.flatnonzero(
        frame["point_kind_model"].eq("focal_bucket_dose").to_numpy()
        & np.isclose(frame["epoch_multiplier_model"].to_numpy(float), model.EXTRAPOLATION_MULTIPLIER)
    )

    summaries = []
    selected_payload: dict[str, Any] = {
        "candidate_id": model.CANDIDATE_ID,
        "evaluation_protocol_sha256": protocol["evaluation_protocol_sha256"],
        "observation_sha256": sha256(observation_path),
        "source_scale": "60m",
        "targets": {},
    }
    oof_frame = frame[["point_id", "focal_domain_model", "epoch_multiplier_model"]].copy()
    parameter_rows = []
    all_pass = True

    for target_name, target_column in (("uncheatable", "uncheatable_bpb"), ("table9", "table9_macro_bpb")):
        target = frame[target_column].to_numpy(float)
        candidate_oof, linear_oof, selections, linear_configs = nested_oof(weights, target, frame, geometry)
        full_selection = select_config(
            weights,
            target,
            frame,
            geometry,
            available,
            seed=INNER_BOOTSTRAP_SEED + 10_000 + len(summaries),
        )
        fitted = fit_potential(weights[fit_rows], target[fit_rows], geometry, full_selection.config)
        linear_config, _linear_inner, _linear_score = best_ridge(
            weights,
            target,
            frame,
            geometry,
            available,
            generator_order=0.0,
            curvature_mode="linear",
        )
        candidate_metrics = metric_summary(target[available], candidate_oof[available])
        linear_metrics = metric_summary(target[available], linear_oof[available])
        relative_rmse = candidate_metrics["rmse"] / linear_metrics["rmse"]
        x32_metrics = metric_summary(target[x32], fitted.predict(weights[x32])) if len(x32) >= 3 else {}
        curvature = fitted.curvature()
        all_curvature_active = bool(len(curvature) > 0 and np.all(curvature > CURVATURE_ACTIVE_TOLERANCE))
        order_mode_fraction = config_mode_fraction(full_selection.config, selections)
        passed = bool(
            relative_rmse <= model.GATES["dose_linear_ablation_relative_rmse_max"]
            and all_curvature_active
            and order_mode_fraction >= model.GATES["generator_order_fold_mode_fraction_min"]
        )
        all_pass = all_pass and passed

        oof_frame[f"observed_{target_name}"] = target
        oof_frame[f"candidate_oof_{target_name}"] = candidate_oof
        oof_frame[f"linear_oof_{target_name}"] = linear_oof
        utility = fitted.bucket_utility()
        for index, domain in enumerate(geometry.domains):
            parameter_rows.append(
                {
                    "target": target_name,
                    "parameter_type": "bucket_utility",
                    "name": domain,
                    "value": utility[index],
                    "units": "BPB",
                }
            )
        curvature_names = ("global",) if full_selection.config.curvature_mode == "global" else geometry.family_names
        for name, value in zip(curvature_names, curvature, strict=True):
            parameter_rows.append(
                {
                    "target": target_name,
                    "parameter_type": "curvature",
                    "name": name,
                    "value": value,
                    "units": "BPB",
                }
            )

        target_payload = {
            "selected_config": asdict(full_selection.config),
            "selected_linear_config": asdict(linear_config),
            "selection_evidence": asdict(full_selection),
            "outer_selected_configs": [asdict(selection.config) for selection in selections],
            "outer_linear_configs": [asdict(config) for config in linear_configs],
            "candidate_metrics": candidate_metrics,
            "linear_metrics": linear_metrics,
            "relative_rmse": relative_rmse,
            "x32_extrapolation_metrics": x32_metrics,
            "curvature": curvature,
            "all_curvature_active": all_curvature_active,
            "generator_order_fold_mode_fraction": order_mode_fraction,
            "effective_df": fitted.effective_df,
            "passed_60m_gate": passed,
        }
        selected_payload["targets"][target_name] = json_ready(target_payload)
        summaries.append(
            {
                "target": target_name,
                "candidate_oof_rmse": candidate_metrics["rmse"],
                "linear_oof_rmse": linear_metrics["rmse"],
                "relative_rmse": relative_rmse,
                "generator_order": full_selection.config.generator_order,
                "curvature_mode": full_selection.config.curvature_mode,
                "ridge": full_selection.config.ridge,
                "all_curvature_active": all_curvature_active,
                "order_mode_fraction": order_mode_fraction,
                "passed": passed,
            }
        )

    selected_payload["passed_both_targets"] = all_pass
    write_frame_if_absent_or_equal(output_dir / "dose_oof_60m.csv", oof_frame)
    write_frame_if_absent_or_equal(output_dir / "dose_parameters_60m.csv", pd.DataFrame(parameter_rows))
    write_frame_if_absent_or_equal(output_dir / "dose_summary_60m.csv", pd.DataFrame(summaries))
    selected_path = output_dir / "selected_60m.json"
    write_if_absent_or_equal(
        selected_path,
        json.dumps(json_ready(selected_payload), indent=2, sort_keys=True) + "\n",
    )
    append_data_use(
        output_dir,
        stage="evaluate_60m",
        outcomes_inspected="complete 60M full outcomes including x32 after structure selection",
        purpose="freeze source-scale aggregate structure and apply 60M mechanism gate",
        protocol_sha256=protocol["evaluation_protocol_sha256"],
    )
    print(json.dumps(json_ready(selected_payload), indent=2, sort_keys=True))


def fit_config(payload: dict[str, Any]) -> FitConfig:
    return FitConfig(
        generator_order=float(payload["generator_order"]),
        curvature_mode=str(payload["curvature_mode"]),
        ridge=float(payload["ridge"]),
    )


def selected_60m_payload(output_dir: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    path = output_dir / "selected_60m.json"
    if not path.exists():
        raise RuntimeError("60M structure is not frozen")
    payload = json.loads(path.read_text())
    if payload.get("evaluation_protocol_sha256") != protocol["evaluation_protocol_sha256"]:
        raise ValueError("60M selection was produced under a different evaluation protocol")
    return payload


def require_60m_gate(output_dir: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    payload = selected_60m_payload(output_dir, protocol)
    if payload.get("passed_both_targets") is not True:
        raise RuntimeError("The frozen 60M nonlinear dose gate failed; downstream outcomes remain sealed")
    return payload


def fixed_panel_oof(
    weights: np.ndarray,
    target: np.ndarray,
    frame: pd.DataFrame,
    geometry: model.Geometry,
    config: FitConfig,
) -> np.ndarray:
    available = np.flatnonzero(selection_mask(frame))
    folds = outer_fold_ids(frame, available)
    anchors = anchor_indices(frame)
    prediction = np.full(len(frame), np.nan)
    for fold in range(model.OUTER_FOLDS):
        test = available[folds[available] == fold]
        train = available[folds[available] != fold]
        fit_rows = np.unique(np.concatenate([train, anchors]))
        fitted = fit_potential(weights[fit_rows], target[fit_rows], geometry, config)
        prediction[test] = fitted.predict(weights[test])
    if not np.isfinite(prediction[available]).all():
        raise ValueError(f"Incomplete fixed-form OOF prediction for {config}")
    return prediction


def positive_affine_fit(predictor: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    x = np.asarray(predictor, dtype=float)
    y = np.asarray(target, dtype=float)
    matrix = np.column_stack([np.ones(len(x)), x])
    result = lsq_linear(
        matrix,
        y,
        bounds=(np.asarray([-np.inf, 0.0]), np.asarray([np.inf, np.inf])),
        method="trf",
        max_iter=2_000,
        tol=1e-13,
        lsmr_tol=1e-13,
    )
    if not np.isfinite(result.x).all():
        raise RuntimeError("Positive affine fit returned non-finite coefficients")
    return float(result.x[0]), float(result.x[1])


def affine_predict(predictor: np.ndarray, intercept: float, scale: float) -> np.ndarray:
    return intercept + scale * np.asarray(predictor, dtype=float)


def tied_policy_matrix(
    phase_weights_path: Path,
    run_names: list[str],
    geometry: model.Geometry,
) -> np.ndarray:
    phase_weights = pd.read_csv(phase_weights_path)
    phases = sorted(phase_weights["phase"].astype(str).unique())
    if phases != ["phase_0", "phase_1"]:
        raise ValueError(f"Unexpected phase names in {phase_weights_path}: {phases}")
    matrices = []
    for phase in phases:
        subset = phase_weights.loc[phase_weights["phase"].eq(phase)]
        pivot = subset.pivot(index="run_name", columns="domain", values="weight")
        missing = sorted(set(run_names) - set(pivot.index))
        if missing:
            raise ValueError(f"Missing policy rows in {phase_weights_path}: {missing[:5]}")
        matrices.append(pivot.reindex(index=run_names, columns=geometry.domains).to_numpy(float))
    if not np.allclose(matrices[0], matrices[1], atol=1e-12, rtol=0.0):
        raise ValueError(f"Pilot policy file contains asymmetric rows: {phase_weights_path}")
    return matrices[0]


def policy_key(weights: np.ndarray) -> str:
    return hashlib.sha256(np.round(np.asarray(weights, dtype=float), 12).tobytes()).hexdigest()


def policy_keys(weights: np.ndarray) -> np.ndarray:
    return np.asarray([policy_key(row) for row in np.asarray(weights)], dtype=object)


def source_fit(
    output_dir: Path,
    selected: dict[str, Any],
    target_name: str,
) -> tuple[FittedPotential, FittedPotential]:
    observations = pd.read_csv(output_dir / "observations_60m.csv")
    frame, geometry, weights = panel_rows(observations)
    available = np.flatnonzero(selection_mask(frame))
    fit_rows = np.unique(np.concatenate([available, anchor_indices(frame)]))
    target = frame[TARGET_COLUMNS[target_name]].to_numpy(float)
    target_payload = selected["targets"][target_name]
    candidate = fit_potential(
        weights[fit_rows],
        target[fit_rows],
        geometry,
        fit_config(target_payload["selected_config"]),
    )
    linear = fit_potential(
        weights[fit_rows],
        target[fit_rows],
        geometry,
        fit_config(target_payload["selected_linear_config"]),
    )
    return candidate, linear


def pilot_affine_calibration(
    target_name: str,
    geometry: model.Geometry,
    source: FittedPotential,
) -> tuple[float, float, set[str], int]:
    if not PILOT_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing exposed pilot outcomes: {PILOT_RESULTS_PATH}")
    observations = pd.read_csv(PILOT_RESULTS_PATH)
    observations = observations.loc[observations["scale"].eq("delphi_3e18")].reset_index(drop=True)
    weights = tied_policy_matrix(
        PILOT_PANEL_DIR / "delphi_3e18" / "phase_weights.csv",
        observations["run_name"].astype(str).tolist(),
        geometry,
    )
    observations["policy_key"] = policy_keys(weights)
    observations["source_prediction"] = source.predict(weights)
    collapsed = observations.groupby("policy_key", sort=True).agg(
        source_prediction=("source_prediction", "mean"),
        target=(TARGET_COLUMNS[target_name], "mean"),
    )
    if len(collapsed) != 8:
        raise ValueError(f"Expected eight unique exposed pilot coordinates, found {len(collapsed)}")
    intercept, scale = positive_affine_fit(
        collapsed["source_prediction"].to_numpy(float),
        collapsed["target"].to_numpy(float),
    )
    return intercept, scale, set(collapsed.index.astype(str)), len(observations)


def effect_summary(
    observed: np.ndarray,
    predicted: np.ndarray,
    score_indices: np.ndarray,
    anchor_index: int,
) -> dict[str, float | int]:
    observed_effect = observed[score_indices] - observed[anchor_index]
    predicted_effect = predicted[score_indices] - predicted[anchor_index]
    nonzero = np.abs(observed_effect) > 1e-12
    if not np.any(nonzero):
        raise ValueError("No nonzero intervention effects to score")
    return {
        "n": int(nonzero.sum()),
        "effect_rmse": rmse(observed_effect[nonzero], predicted_effect[nonzero]),
        "effect_spearman": float(stats.spearmanr(observed_effect[nonzero], predicted_effect[nonzero]).statistic),
        "sign_accuracy": float(np.mean(np.sign(observed_effect[nonzero]) == np.sign(predicted_effect[nonzero]))),
    }


def evaluate_delphi(output_dir: Path, panel_dir: Path) -> None:
    protocol = verify_evaluation_protocol(output_dir, panel_dir)
    selected = require_60m_gate(output_dir, protocol)
    observation_path = output_dir / "observations_delphi_3e18.csv"
    if not observation_path.exists():
        raise FileNotFoundError("Materialize Delphi outcomes before evaluation")
    observations = pd.read_csv(observation_path)
    frame, geometry, weights = panel_rows(observations)
    available = np.flatnonzero(selection_mask(frame))
    x32 = np.flatnonzero(
        frame["point_kind_model"].eq("focal_bucket_dose").to_numpy()
        & np.isclose(frame["epoch_multiplier_model"].to_numpy(float), model.EXTRAPOLATION_MULTIPLIER)
    )
    anchor = int(anchor_indices(frame)[0])
    full_keys = policy_keys(weights)
    prediction_frame = frame[["point_id", "focal_domain_model", "epoch_multiplier_model"]].copy()
    summary_rows = []
    calibration_rows = []
    payload: dict[str, Any] = {
        "candidate_id": model.CANDIDATE_ID,
        "evaluation_protocol_sha256": protocol["evaluation_protocol_sha256"],
        "observation_sha256": sha256(observation_path),
        "scale": "delphi_3e18",
        "targets": {},
    }

    for target_name, target_column in TARGET_COLUMNS.items():
        target = frame[target_column].to_numpy(float)
        target_selection = selected["targets"][target_name]
        config = fit_config(target_selection["selected_config"])
        linear_config = fit_config(target_selection["selected_linear_config"])
        candidate_oof = fixed_panel_oof(weights, target, frame, geometry, config)
        linear_oof = fixed_panel_oof(weights, target, frame, geometry, linear_config)
        candidate_metrics = metric_summary(target[available], candidate_oof[available])
        linear_metrics = metric_summary(target[available], linear_oof[available])
        form_relative_rmse = candidate_metrics["rmse"] / linear_metrics["rmse"]

        source, _source_linear = source_fit(output_dir, selected, target_name)
        pilot_intercept, pilot_scale, pilot_keys, pilot_rows = pilot_affine_calibration(
            target_name,
            geometry,
            source,
        )
        strict_prediction = affine_predict(source.predict(weights), pilot_intercept, pilot_scale)
        strict_score = available[np.fromiter((full_keys[index] not in pilot_keys for index in available), dtype=bool)]
        if len(strict_score) < 200:
            raise ValueError(f"Unexpectedly small nonoverlap Delphi score set: {len(strict_score)}")
        strict_metrics = metric_summary(target[strict_score], strict_prediction[strict_score])
        strict_effect = effect_summary(target, strict_prediction, strict_score, anchor)
        strict_x32 = metric_summary(target[x32], strict_prediction[x32])

        fit_rows = np.unique(np.concatenate([available, np.asarray([anchor])]))
        delphi_fit = fit_potential(weights[fit_rows], target[fit_rows], geometry, config)
        curvature = delphi_fit.curvature()
        all_curvature_active = bool(len(curvature) > 0 and np.all(curvature > CURVATURE_ACTIVE_TOLERANCE))
        passed = bool(
            form_relative_rmse <= model.GATES["cross_scale_form_relative_rmse_max"]
            and strict_effect["effect_spearman"] >= model.GATES["cross_scale_strict_transfer_spearman_min"]
            and strict_effect["sign_accuracy"] >= model.GATES["cross_scale_strict_transfer_sign_accuracy_min"]
            and all_curvature_active
        )

        prediction_frame[f"observed_{target_name}"] = target
        prediction_frame[f"fixed_form_oof_{target_name}"] = candidate_oof
        prediction_frame[f"linear_oof_{target_name}"] = linear_oof
        prediction_frame[f"strict_transfer_{target_name}"] = strict_prediction
        prediction_frame[f"strict_transfer_scored_{target_name}"] = False
        prediction_frame.loc[strict_score, f"strict_transfer_scored_{target_name}"] = True
        calibration_rows.append(
            {
                "target": target_name,
                "pilot_rows": pilot_rows,
                "pilot_unique_coordinates": len(pilot_keys),
                "intercept": pilot_intercept,
                "positive_scale": pilot_scale,
            }
        )
        target_payload = {
            "selected_config": asdict(config),
            "selected_linear_config": asdict(linear_config),
            "fixed_form_metrics": candidate_metrics,
            "linear_metrics": linear_metrics,
            "fixed_form_relative_rmse": form_relative_rmse,
            "strict_transfer_metrics": strict_metrics,
            "strict_transfer_effect_metrics": strict_effect,
            "strict_transfer_x32_metrics": strict_x32,
            "pilot_affine_intercept": pilot_intercept,
            "pilot_affine_positive_scale": pilot_scale,
            "strict_transfer_scored_rows": len(strict_score),
            "curvature": curvature,
            "all_curvature_active": all_curvature_active,
            "passed_delphi_gate": passed,
        }
        payload["targets"][target_name] = json_ready(target_payload)
        summary_rows.append(
            {
                "target": target_name,
                "fixed_form_oof_rmse": candidate_metrics["rmse"],
                "linear_oof_rmse": linear_metrics["rmse"],
                "fixed_form_relative_rmse": form_relative_rmse,
                "strict_transfer_rmse": strict_metrics["rmse"],
                "strict_effect_spearman": strict_effect["effect_spearman"],
                "strict_sign_accuracy": strict_effect["sign_accuracy"],
                "pilot_affine_positive_scale": pilot_scale,
                "all_curvature_active": all_curvature_active,
                "passed": passed,
            }
        )

    payload["passed_both_targets"] = all(row["passed"] for row in summary_rows)
    write_frame_if_absent_or_equal(output_dir / "dose_predictions_delphi_3e18.csv", prediction_frame)
    write_frame_if_absent_or_equal(
        output_dir / "dose_calibration_delphi_3e18.csv",
        pd.DataFrame(calibration_rows),
    )
    write_frame_if_absent_or_equal(output_dir / "dose_summary_delphi_3e18.csv", pd.DataFrame(summary_rows))
    write_if_absent_or_equal(
        output_dir / "evaluation_delphi_3e18.json",
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n",
    )
    append_data_use(
        output_dir,
        stage="evaluate_delphi_3e18",
        outcomes_inspected="complete Delphi full outcomes after frozen 60M gate",
        purpose="cross-scale fixed-form and pilot-calibrated strict-transfer falsification",
        protocol_sha256=protocol["evaluation_protocol_sha256"],
    )
    print(json.dumps(json_ready(payload), indent=2, sort_keys=True))


def correspondence_folds(frame: pd.DataFrame) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    indices = np.arange(len(frame))
    groups = frame["phase_correspondence_key"].astype(str).to_numpy()
    return compact_audit.group_folds(indices, groups, model.OUTER_FOLDS, model.OUTER_SEED)


def fixed_300m_oof(
    weights: np.ndarray,
    target: np.ndarray,
    geometry: model.Geometry,
    config: FitConfig,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> np.ndarray:
    prediction = np.full(len(target), np.nan)
    for train, test in folds:
        fitted = fit_potential(weights[train], target[train], geometry, config)
        prediction[test] = fitted.predict(weights[test])
    if not np.isfinite(prediction).all():
        raise ValueError(f"Incomplete 300M OOF prediction for {config}")
    return prediction


def strict_affine_oof(
    source_prediction: np.ndarray,
    target: np.ndarray,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> tuple[np.ndarray, list[dict[str, float]]]:
    prediction = np.full(len(target), np.nan)
    rows = []
    for fold, (train, test) in enumerate(folds):
        intercept, scale = positive_affine_fit(source_prediction[train], target[train])
        prediction[test] = affine_predict(source_prediction[test], intercept, scale)
        rows.append({"fold": fold, "intercept": intercept, "positive_scale": scale})
    if not np.isfinite(prediction).all():
        raise ValueError("Incomplete strict-transfer 300M OOF prediction")
    return prediction, rows


def regret_at_k(observed: np.ndarray, predicted: np.ndarray, k: int) -> float:
    selected = np.argsort(predicted)[:k]
    return float(np.min(observed[selected]) - np.min(observed))


def generator_derivative(relative_dose: np.ndarray, order: float) -> np.ndarray:
    values = np.maximum(np.asarray(relative_dose, dtype=float), 1e-12)
    if order == 0.0:
        return np.log(values)
    return (np.power(values, order) - 1.0) / order


def potential_gradient(weights: np.ndarray, fitted: FittedPotential) -> np.ndarray:
    utility = fitted.bucket_utility()
    curvature = fitted.curvature()
    if len(curvature) == 1:
        bucket_curvature = np.full(len(weights), curvature[0])
    else:
        bucket_curvature = curvature[fitted.geometry.family_index]
    relative_dose = np.asarray(weights, dtype=float) / fitted.geometry.proportional
    return utility + bucket_curvature * generator_derivative(relative_dose, fitted.config.generator_order)


def optimization_starts(
    geometry: model.Geometry,
    observed_best: np.ndarray,
    seed: int,
    count: int,
) -> tuple[np.ndarray, ...]:
    starts = [geometry.proportional.copy(), observed_best.copy(), np.full(len(observed_best), 1.0 / len(observed_best))]
    generator = np.random.default_rng(seed)
    concentrations = (0.25, 1.0, 4.0)
    while len(starts) < count:
        concentration = concentrations[(len(starts) - 3) % len(concentrations)]
        starts.append(generator.dirichlet(np.full(len(observed_best), concentration)))
    return tuple(starts[:count])


def optimize_potential(
    fitted: FittedPotential,
    observed_best: np.ndarray,
    *,
    seed: int,
    starts: int = OPTIMIZER_STARTS,
) -> tuple[np.ndarray, float, int, float]:
    bucket_count = len(observed_best)

    def objective(weights: np.ndarray) -> float:
        return float(fitted.predict(np.asarray(weights)[None, :])[0])

    candidates = []
    successes = 0
    for start in optimization_starts(fitted.geometry, observed_best, seed, starts):
        result = minimize(
            objective,
            start,
            jac=lambda weights: potential_gradient(weights, fitted),
            method="SLSQP",
            bounds=[(1e-12, 1.0)] * bucket_count,
            constraints={
                "type": "eq",
                "fun": lambda weights: float(np.sum(weights) - 1.0),
                "jac": lambda _: np.ones(bucket_count),
            },
            options={"maxiter": 5_000, "ftol": 1e-13, "disp": False},
        )
        if result.success:
            successes += 1
        weights = np.maximum(np.asarray(result.x, dtype=float), 0.0)
        weights /= weights.sum()
        value = objective(weights)
        if np.isfinite(value):
            candidates.append((value, weights))
    if not candidates:
        raise RuntimeError("No finite raw optimum for signed dose potential")
    value, optimum = min(candidates, key=lambda item: item[0])
    objective_spread = float(max(item[0] for item in candidates) - min(item[0] for item in candidates))
    return optimum, value, successes, objective_spread


def nearest_support_tv(optimum: np.ndarray, training_weights: np.ndarray) -> float:
    return float(np.min(0.5 * np.abs(training_weights - optimum[None, :]).sum(axis=1)))


def bootstrap_300m_optima(
    weights: np.ndarray,
    target: np.ndarray,
    frame: pd.DataFrame,
    geometry: model.Geometry,
    config: FitConfig,
    full_optimum: np.ndarray,
    observed_best: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = frame["phase_correspondence_key"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    generator = np.random.default_rng(model.OPTIMUM_BOOTSTRAP_SEED)
    optimum_rows = []
    parameter_rows = []
    for replicate in range(model.OPTIMUM_BOOTSTRAP_REPLICATES):
        sampled_groups = generator.choice(unique_groups, size=len(unique_groups), replace=True)
        sample = np.concatenate([np.flatnonzero(groups == group) for group in sampled_groups])
        fitted = fit_potential(weights[sample], target[sample], geometry, config)
        optimum, value, successful, spread = optimize_potential(
            fitted,
            observed_best,
            seed=OPTIMIZER_SEED + 10_000 + replicate,
            starts=3,
        )
        optimum_rows.append(
            {
                "replicate": replicate,
                "predicted_optimum_bpb": value,
                "tv_to_full_optimum": float(0.5 * np.abs(optimum - full_optimum).sum()),
                "maximum_weight": float(np.max(optimum)),
                "successful_optimizer_starts": successful,
                "optimizer_objective_spread": spread,
                **{f"weight::{domain}": value for domain, value in zip(geometry.domains, optimum, strict=True)},
            }
        )
        for domain, value in zip(geometry.domains, fitted.bucket_utility(), strict=True):
            parameter_rows.append(
                {"replicate": replicate, "parameter_type": "bucket_utility", "name": domain, "value": value}
            )
        curvature_names = ("global",) if len(fitted.curvature()) == 1 else geometry.family_names
        for name, value in zip(curvature_names, fitted.curvature(), strict=True):
            parameter_rows.append({"replicate": replicate, "parameter_type": "curvature", "name": name, "value": value})
    return pd.DataFrame(optimum_rows), pd.DataFrame(parameter_rows)


def evaluate_300m(output_dir: Path, panel_dir: Path) -> None:
    protocol = verify_evaluation_protocol(output_dir, panel_dir)
    selected = require_60m_gate(output_dir, protocol)
    geometry = model.geometry()
    summary_rows = []
    prediction_frames = []
    calibration_rows = []
    optimum_rows = []
    bootstrap_optima = []
    bootstrap_parameters = []
    full_parameters = []
    payload: dict[str, Any] = {
        "candidate_id": model.CANDIDATE_ID,
        "evaluation_protocol_sha256": protocol["evaluation_protocol_sha256"],
        "source_data_sha256": {
            str(physical_spine.benchmark.PACKET.relative_to(REPO_ROOT)): sha256(physical_spine.benchmark.PACKET),
            str(physical_spine.benchmark.ONE_PHASE_SOURCE.relative_to(REPO_ROOT)): sha256(
                physical_spine.benchmark.ONE_PHASE_SOURCE
            ),
        },
        "scale": "300m",
        "targets": {},
    }

    for target_index, target_name in enumerate(model.TARGETS):
        dataset = physical_spine.tied_dataset(target_name)
        domain_lookup = {domain: index for index, domain in enumerate(dataset.domain_names)}
        if set(domain_lookup) != set(geometry.domains):
            raise ValueError("300M tied data and conditional-dose geometry use different domains")
        order = np.asarray([domain_lookup[domain] for domain in geometry.domains], dtype=int)
        weights = dataset.weights[:, 0, :][:, order]
        target = dataset.y
        frame = dataset.frame.reset_index(drop=True)
        folds = correspondence_folds(frame)
        target_selection = selected["targets"][target_name]
        config = fit_config(target_selection["selected_config"])
        linear_config = fit_config(target_selection["selected_linear_config"])

        candidate_oof = fixed_300m_oof(weights, target, geometry, config, folds)
        linear_oof = fixed_300m_oof(weights, target, geometry, linear_config, folds)
        source, _source_linear = source_fit(output_dir, selected, target_name)
        source_prediction = source.predict(weights)
        strict_oof, strict_calibration = strict_affine_oof(source_prediction, target, folds)

        candidate_metrics = metric_summary(target, candidate_oof)
        linear_metrics = metric_summary(target, linear_oof)
        strict_metrics = metric_summary(target, strict_oof)
        candidate_metrics.update({f"regret_at_{k}": regret_at_k(target, candidate_oof, k) for k in (1, 3, 5)})
        linear_metrics.update({f"regret_at_{k}": regret_at_k(target, linear_oof, k) for k in (1, 3, 5)})
        strict_metrics.update({f"regret_at_{k}": regret_at_k(target, strict_oof, k) for k in (1, 3, 5)})

        fitted = fit_potential(weights, target, geometry, config)
        observed_best = weights[int(np.argmin(target))]
        optimum, optimum_bpb, successful, objective_spread = optimize_potential(
            fitted,
            observed_best,
            seed=OPTIMIZER_SEED + target_index,
        )
        support_tv = nearest_support_tv(optimum, weights)
        observed_best_tv = float(0.5 * np.abs(optimum - observed_best).sum())
        raw_optimism = float(np.min(target) - optimum_bpb)
        raw_optimism_rmse_multiple = max(raw_optimism, 0.0) / candidate_metrics["rmse"]
        exposure_per_weight = (dataset.c0 + dataset.c1)[order]
        maximum_epochs = float(np.max(optimum * exposure_per_weight))
        optimum_bootstrap, parameter_bootstrap = bootstrap_300m_optima(
            weights,
            target,
            frame,
            geometry,
            config,
            optimum,
            observed_best,
        )
        optimum_bootstrap.insert(0, "target", target_name)
        parameter_bootstrap.insert(0, "target", target_name)
        bootstrap_optima.append(optimum_bootstrap)
        bootstrap_parameters.append(parameter_bootstrap)
        median_bootstrap_tv = float(optimum_bootstrap["tv_to_full_optimum"].median())
        curvature = fitted.curvature()
        full_utility = fitted.bucket_utility()
        utility_sign_agreement = []
        for domain, value in zip(geometry.domains, full_utility, strict=True):
            draws = parameter_bootstrap.loc[
                parameter_bootstrap["parameter_type"].eq("bucket_utility") & parameter_bootstrap["name"].eq(domain),
                "value",
            ].to_numpy(float)
            utility_sign_agreement.append(float(np.mean(np.sign(draws) == np.sign(value))))
        curvature_draws = parameter_bootstrap.loc[
            parameter_bootstrap["parameter_type"].eq("curvature"),
            "value",
        ].to_numpy(float)
        parameter_stability = {
            "median_bucket_utility_sign_agreement": float(np.median(utility_sign_agreement)),
            "minimum_bucket_utility_sign_agreement": float(np.min(utility_sign_agreement)),
            "curvature_positive_fraction": float(np.mean(curvature_draws > CURVATURE_ACTIVE_TOLERANCE)),
        }
        all_curvature_active = bool(len(curvature) > 0 and np.all(curvature > CURVATURE_ACTIVE_TOLERANCE))
        reference = model.FROZEN_300M_REFERENCES[target_name]
        relative_to_reference = candidate_metrics["rmse"] / reference
        passed = bool(
            relative_to_reference <= model.GATES["300m_tied_oof_relative_to_reference_max"]
            and support_tv <= model.GATES["300m_raw_support_tv_max"]
            and float(np.max(optimum)) <= model.GATES["300m_raw_max_bucket_weight_max"]
            and raw_optimism_rmse_multiple <= model.GATES["300m_raw_optimism_oof_rmse_multiple_max"]
            and median_bootstrap_tv <= model.GATES["300m_optimum_bootstrap_median_tv_max"]
            and all_curvature_active
        )

        for fold_row in strict_calibration:
            calibration_rows.append({"target": target_name, **fold_row})
        for domain, value in zip(geometry.domains, fitted.bucket_utility(), strict=True):
            full_parameters.append(
                {"target": target_name, "parameter_type": "bucket_utility", "name": domain, "value": value}
            )
        curvature_names = ("global",) if len(curvature) == 1 else geometry.family_names
        for name, value in zip(curvature_names, curvature, strict=True):
            full_parameters.append({"target": target_name, "parameter_type": "curvature", "name": name, "value": value})
        optimum_rows.append(
            {
                "target": target_name,
                "predicted_optimum_bpb": optimum_bpb,
                "observed_best_tied_bpb": float(np.min(target)),
                "raw_optimism": raw_optimism,
                "raw_optimism_oof_rmse_multiple": raw_optimism_rmse_multiple,
                "support_tv": support_tv,
                "tv_to_observed_best_tied": observed_best_tv,
                "maximum_weight": float(np.max(optimum)),
                "maximum_epochs": maximum_epochs,
                "median_bootstrap_tv": median_bootstrap_tv,
                "p90_bootstrap_tv": float(optimum_bootstrap["tv_to_full_optimum"].quantile(0.90)),
                "successful_optimizer_starts": successful,
                "optimizer_objective_spread": objective_spread,
                **{f"weight::{domain}": value for domain, value in zip(geometry.domains, optimum, strict=True)},
            }
        )
        prediction_frame = frame[["run_name", "phase_correspondence_key", "policy_family", "source_panel"]].copy()
        prediction_frame.insert(0, "target", target_name)
        prediction_frame["observed"] = target
        prediction_frame["candidate_oof"] = candidate_oof
        prediction_frame["linear_oof"] = linear_oof
        prediction_frame["strict_source_oof"] = strict_oof
        prediction_frames.append(prediction_frame)

        target_payload = {
            "selected_config": asdict(config),
            "selected_linear_config": asdict(linear_config),
            "candidate_metrics": candidate_metrics,
            "linear_metrics": linear_metrics,
            "strict_source_metrics": strict_metrics,
            "frozen_reference_rmse": reference,
            "candidate_relative_rmse_to_reference": relative_to_reference,
            "raw_optimum": {
                "predicted_bpb": optimum_bpb,
                "observed_best_tied_bpb": float(np.min(target)),
                "raw_optimism": raw_optimism,
                "raw_optimism_oof_rmse_multiple": raw_optimism_rmse_multiple,
                "weights": optimum,
                "support_tv": support_tv,
                "tv_to_observed_best_tied": observed_best_tv,
                "maximum_weight": float(np.max(optimum)),
                "maximum_epochs": maximum_epochs,
                "median_bootstrap_tv": median_bootstrap_tv,
                "p90_bootstrap_tv": float(optimum_bootstrap["tv_to_full_optimum"].quantile(0.90)),
                "successful_optimizer_starts": successful,
                "optimizer_objective_spread": objective_spread,
            },
            "curvature": curvature,
            "bootstrap_parameter_stability": parameter_stability,
            "all_curvature_active": all_curvature_active,
            "passed_300m_gate": passed,
        }
        payload["targets"][target_name] = json_ready(target_payload)
        summary_rows.append(
            {
                "target": target_name,
                "candidate_oof_rmse": candidate_metrics["rmse"],
                "linear_oof_rmse": linear_metrics["rmse"],
                "strict_source_oof_rmse": strict_metrics["rmse"],
                "frozen_reference_rmse": reference,
                "candidate_relative_rmse_to_reference": relative_to_reference,
                "candidate_regret_at_1": candidate_metrics["regret_at_1"],
                "raw_optimism": raw_optimism,
                "raw_optimism_oof_rmse_multiple": raw_optimism_rmse_multiple,
                "support_tv": support_tv,
                "tv_to_observed_best_tied": observed_best_tv,
                "maximum_weight": float(np.max(optimum)),
                "maximum_epochs": maximum_epochs,
                "median_bootstrap_tv": median_bootstrap_tv,
                "median_bucket_utility_sign_agreement": parameter_stability["median_bucket_utility_sign_agreement"],
                "curvature_positive_fraction": parameter_stability["curvature_positive_fraction"],
                "all_curvature_active": all_curvature_active,
                "passed": passed,
            }
        )

    payload["passed_both_targets"] = all(row["passed"] for row in summary_rows)
    write_frame_if_absent_or_equal(
        output_dir / "dose_predictions_300m.csv",
        pd.concat(prediction_frames, ignore_index=True),
    )
    write_frame_if_absent_or_equal(
        output_dir / "dose_strict_calibration_300m.csv",
        pd.DataFrame(calibration_rows),
    )
    write_frame_if_absent_or_equal(output_dir / "dose_summary_300m.csv", pd.DataFrame(summary_rows))
    write_frame_if_absent_or_equal(output_dir / "dose_raw_optima_300m.csv", pd.DataFrame(optimum_rows))
    write_frame_if_absent_or_equal(
        output_dir / "dose_optimum_bootstrap_300m.csv",
        pd.concat(bootstrap_optima, ignore_index=True),
    )
    parameter_bootstrap_frame = pd.concat(bootstrap_parameters, ignore_index=True)
    parameter_bootstrap_summary = (
        parameter_bootstrap_frame.groupby(["target", "parameter_type", "name"], sort=True)["value"]
        .agg(
            median="median",
            mean="mean",
            sd="std",
            q025=lambda values: values.quantile(0.025),
            q975=lambda values: values.quantile(0.975),
            positive_fraction=lambda values: float(np.mean(values > CURVATURE_ACTIVE_TOLERANCE)),
        )
        .reset_index()
    )
    write_frame_if_absent_or_equal(
        output_dir / "dose_parameter_bootstrap_300m.csv",
        parameter_bootstrap_frame,
    )
    write_frame_if_absent_or_equal(
        output_dir / "dose_parameter_bootstrap_summary_300m.csv",
        parameter_bootstrap_summary,
    )
    write_frame_if_absent_or_equal(output_dir / "dose_parameters_300m.csv", pd.DataFrame(full_parameters))
    write_if_absent_or_equal(
        output_dir / "evaluation_300m.json",
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n",
    )
    append_data_use(
        output_dir,
        stage="evaluate_300m",
        outcomes_inspected="known-development 300M physically tied outcomes after frozen 60M gate",
        purpose="strict source transfer, fixed-form refit, and raw-optimum falsification",
        protocol_sha256=protocol["evaluation_protocol_sha256"],
    )
    print(json.dumps(json_ready(payload), indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        prepare(args.output_dir, args.panel_dir)
    elif args.mode == "materialize-60m":
        materialize_scale("60m", args.output_dir, args.panel_dir, args.wandb_timeout)
    elif args.mode == "evaluate-60m":
        evaluate_60m(args.output_dir, args.panel_dir)
    elif args.mode == "materialize-delphi":
        materialize_scale("delphi_3e18", args.output_dir, args.panel_dir, args.wandb_timeout)
    elif args.mode == "evaluate-delphi":
        evaluate_delphi(args.output_dir, args.panel_dir)
    else:
        evaluate_300m(args.output_dir, args.panel_dir)


if __name__ == "__main__":
    main()
