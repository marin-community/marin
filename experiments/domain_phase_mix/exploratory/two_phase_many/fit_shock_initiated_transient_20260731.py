# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
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
"""Test a bounded shock-initiated transient on post-switch target dynamics.

The initial state is SUR-070's blocked-OOF policy prediction of the observed
gradient shock. This script fits only a nonnegative response amplitude and one
dimensionless exponential decay rate on smooth-target residuals at steps
19,000--21,000. Step 22,000 and the final endpoint are temporal holdouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import diagnose_switch_gradient_shock_20260731 as switch_shock
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "shock_initiated_transient_20260731"
POLICY_SHOCK_DIR = SCRIPT_DIR / "reference_outputs" / "policy_predictable_switch_shock_20260731"
POLICY_SHOCK_PROTOCOL = POLICY_SHOCK_DIR / "protocol.json"
POLICY_SHOCK_PREDICTIONS = POLICY_SHOCK_DIR / "oof_predictions.csv"

CANDIDATE_ID = "WSD80-SUR-071"
FIT_STEPS = (19_000, 20_000, 21_000)
HOLDOUT_STEPS = (22_000, switch_shock.FINAL_STEP)
MAX_DECAY_RATE = 50.0
BOOTSTRAP_SAMPLES = 5_000
BOOTSTRAP_SEED = 20_260_733


@dataclass(frozen=True)
class DecayFit:
    """Shock response in BPB per log-gradient unit and phase-1 decay rate."""

    amplitude: float
    decay_rate: float
    objective: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preregister", "evaluate"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def protocol_payload() -> dict[str, object]:
    return {
        "candidate_id": CANDIDATE_ID,
        "title": "Shock-initiated phase-1 transient",
        "scope": "development transition diagnostic; not a full endpoint surrogate",
        "nearest_prior_routes": [
            "WSD80-SUR-060",
            "WSD80-SUR-061",
            "WSD80-SUR-068",
            "WSD80-SUR-069",
            "WSD80-SUR-070",
            "prior_AB",
        ],
        "material_novelty": (
            "The initial state is a policy prediction of independently logged optimizer telemetry, not an "
            "endpoint-selected phase coordinate. Its feature equation and OOF predictions are frozen by SUR-070."
        ),
        "state": {
            "initial": "q_p = SUR-070 OOF cross_phase prediction of asymmetric-minus-tied log-gradient shock",
            "transition": "h_p(s) = q_p * exp(-lambda*s)",
            "response": "r_p(s) = a*h_p(s)",
            "progress": (
                f"s=(step-{switch_shock.PHASE_BOUNDARY_STEP})/"
                f"({switch_shock.FINAL_STEP}-{switch_shock.PHASE_BOUNDARY_STEP})"
            ),
            "units": {
                "q_p": "dimensionless log-gradient jump",
                "s": "dimensionless phase-1 progress",
                "lambda": "inverse normalized phase-1 duration",
                "a": "BPB per unit log-gradient jump",
            },
            "constraints": "a>=0, 0<=lambda<=50; no intercept; tied state is exactly zero",
        },
        "fit": {
            "steps": FIT_STEPS,
            "outcome": "common residual across seven smooth Uncheatable components left by frozen SUR-068",
            "outer_folds": "reuse SUR-070 mixture-blocked outer-fold labels",
            "estimation": "profile nonnegative amplitude and bounded scalar decay by training-pair squared error",
            "parameter_count": 2,
        },
        "temporal_holdout": {
            "steps": HOLDOUT_STEPS,
            "prediction": "single full fit on steps 19000--21000, applied without refitting",
        },
        "baselines": {
            "zero": "SUR-068 unchanged",
            "static_shock": "same q_p and fitted nonnegative amplitude with lambda=0",
        },
        "data_use": {
            "exposed_before_freeze": [
                "SUR-068 residual summaries at all fit and holdout steps",
                "SUR-069 observed-shock transfer at all fit and holdout steps",
                "SUR-070 policy-predicted-shock transfer correlations at all fit and holdout steps",
            ],
            "interpretation": (
                "This is development evidence. The functional form is frozen after observing rank decay, so "
                "neither step 22000 nor final can be treated as confirmatory."
            ),
        },
        "uncertainty": {
            "paired_bootstrap_samples": BOOTSTRAP_SAMPLES,
            "paired_bootstrap_seed": BOOTSTRAP_SEED,
            "outer_fold_wins_required": 4,
        },
        "gates": {
            "fit_oof_zero_improvement_min": 0.10,
            "fit_dynamic_vs_static_bootstrap_upper_max": 0.0,
            "fit_dynamic_vs_static_fold_wins_min": 4,
            "full_decay_rate_min": 0.05,
            "full_decay_rate_max": 45.0,
            "fold_decay_positive_min": 4,
            "fold_amplitude_positive_min": 5,
            "step22000_zero_regression_max": 0.02,
            "step22000_must_beat_static": True,
            "final_zero_regression_max": 0.02,
            "final_must_beat_static": True,
        },
        "forbidden_repairs": [
            "persistent offset selected on step 22000 or final",
            "second decay timescale",
            "component-specific rates or amplitudes",
            "endpoint calibration or intercept",
            "changing the SUR-070 shock map",
        ],
        "decision_boundary": (
            "A pass identifies only a transient state and its decay. Because the state may vanish before final, "
            "it cannot by itself be promoted as the endpoint temporal correction."
        ),
    }


def wrapped_protocol() -> dict[str, object]:
    payload = protocol_payload()
    digest = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    return {"protocol_sha256": digest, "protocol": payload}


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def freeze_protocol(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = wrapped_protocol()
    path = output_dir / "protocol.json"
    if path.exists():
        observed = json.loads(path.read_text())
        if canonical_json(observed) != canonical_json(expected):
            raise RuntimeError(f"Existing protocol differs from current code: {path}")
    else:
        write_json(path, expected)
    print(expected["protocol_sha256"])


def require_frozen_protocol(output_dir: Path) -> dict[str, object]:
    path = output_dir / "protocol.json"
    if not path.exists():
        raise RuntimeError("Run --mode preregister before evaluation")
    observed = json.loads(path.read_text())
    expected = wrapped_protocol()
    if canonical_json(observed) != canonical_json(expected):
        raise RuntimeError("Frozen protocol does not match the evaluation code")
    source_protocol = json.loads(POLICY_SHOCK_PROTOCOL.read_text())
    expected_source = "b1d76f86d7ebe0dabfc6e5ae7f7b2c76049884873b05d1291d32db65128a2c3e"
    if source_protocol["protocol_sha256"] != expected_source:
        raise RuntimeError("SUR-070 source protocol changed")
    return expected


def progress(step: np.ndarray | float) -> np.ndarray:
    return (np.asarray(step, dtype=float) - switch_shock.PHASE_BOUNDARY_STEP) / (
        switch_shock.FINAL_STEP - switch_shock.PHASE_BOUNDARY_STEP
    )


def amplitude_for_rate(q: np.ndarray, s: np.ndarray, target: np.ndarray, decay_rate: float) -> float:
    design = q * np.exp(-decay_rate * s)
    denominator = float(np.dot(design, design))
    if denominator <= 1e-15:
        return 0.0
    return max(float(np.dot(design, target) / denominator), 0.0)


def fit_decay(frame: pd.DataFrame, dynamic: bool) -> DecayFit:
    q = frame["predicted_gradient_shock"].to_numpy(float)
    s = progress(frame["global_step"].to_numpy(float))
    target = frame["common_residual"].to_numpy(float)
    if not dynamic:
        amplitude = amplitude_for_rate(q, s, target, 0.0)
        prediction = amplitude * q
        return DecayFit(amplitude, 0.0, float(np.mean(np.square(target - prediction))))

    def objective(decay_rate: float) -> float:
        amplitude = amplitude_for_rate(q, s, target, decay_rate)
        prediction = amplitude * q * np.exp(-decay_rate * s)
        return float(np.mean(np.square(target - prediction)))

    result = minimize_scalar(objective, bounds=(0.0, MAX_DECAY_RATE), method="bounded")
    candidates = [
        (float(result.fun), float(result.x)),
        (objective(0.0), 0.0),
        (objective(MAX_DECAY_RATE), MAX_DECAY_RATE),
    ]
    value, decay_rate = min(candidates)
    amplitude = amplitude_for_rate(q, s, target, decay_rate)
    return DecayFit(amplitude, decay_rate, value)


def predict(frame: pd.DataFrame, fit: DecayFit) -> np.ndarray:
    return (
        fit.amplitude
        * frame["predicted_gradient_shock"].to_numpy(float)
        * np.exp(-fit.decay_rate * progress(frame["global_step"].to_numpy(float)))
    )


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(predicted) - np.asarray(observed)))))


def calibration_slope(observed: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.dot(predicted, predicted))
    if denominator <= 1e-15:
        return float("nan")
    return float(np.dot(predicted, observed) / denominator)


def paired_bootstrap_difference(
    frame: pd.DataFrame,
    candidate_column: str,
    baseline_column: str,
    seed: int,
) -> tuple[float, float, float]:
    squared_errors = (
        frame.assign(
            candidate_squared_error=np.square(frame[candidate_column] - frame["common_residual"]),
            baseline_squared_error=np.square(frame[baseline_column] - frame["common_residual"]),
        )
        .groupby("pair_id", sort=True, as_index=False)
        .agg(
            candidate_squared_error=("candidate_squared_error", "sum"),
            baseline_squared_error=("baseline_squared_error", "sum"),
            rows=("pair_id", "size"),
        )
    )
    generator = np.random.default_rng(seed)
    pair_count = len(squared_errors)
    samples = generator.integers(0, pair_count, size=(BOOTSTRAP_SAMPLES, pair_count))
    row_counts = squared_errors["rows"].to_numpy(float)[samples].sum(axis=1)
    candidate_rmse = np.sqrt(squared_errors["candidate_squared_error"].to_numpy(float)[samples].sum(axis=1) / row_counts)
    baseline_rmse = np.sqrt(squared_errors["baseline_squared_error"].to_numpy(float)[samples].sum(axis=1) / row_counts)
    differences = candidate_rmse - baseline_rmse
    return (
        float(np.mean(differences)),
        float(np.quantile(differences, 0.025)),
        float(np.quantile(differences, 0.975)),
    )


def load_frame() -> pd.DataFrame:
    shock_predictions = pd.read_csv(POLICY_SHOCK_PREDICTIONS)
    shock_predictions = shock_predictions.loc[shock_predictions["target"].eq("gradient_log_jump")].copy()
    shock_predictions = shock_predictions[["pair_id", "outer_fold", "predicted__cross_phase"]].rename(
        columns={"predicted__cross_phase": "predicted_gradient_shock"}
    )
    residuals = switch_shock.relaxation_residuals()
    common = residuals.groupby(["pair_id", "global_step"], as_index=False).agg(
        common_residual=("residual", "mean"),
        component_residual_sd=("residual", "std"),
    )
    frame = common.merge(shock_predictions, on="pair_id", how="inner", validate="many_to_one")
    frame = frame.loc[frame["global_step"].isin((*FIT_STEPS, *HOLDOUT_STEPS))].copy()
    if frame["outer_fold"].nunique() != 5:
        raise RuntimeError("SUR-070 outer-fold labels are incomplete")
    return frame


def out_of_fold_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_rows = frame.loc[frame["global_step"].isin(FIT_STEPS)].copy()
    fit_rows["predicted_dynamic"] = np.nan
    fit_rows["predicted_static"] = np.nan
    parameters = []
    for fold in sorted(fit_rows["outer_fold"].unique()):
        train = fit_rows.loc[fit_rows["outer_fold"].ne(fold)]
        test = fit_rows["outer_fold"].eq(fold)
        dynamic_fit = fit_decay(train, dynamic=True)
        static_fit = fit_decay(train, dynamic=False)
        fit_rows.loc[test, "predicted_dynamic"] = predict(fit_rows.loc[test], dynamic_fit)
        fit_rows.loc[test, "predicted_static"] = predict(fit_rows.loc[test], static_fit)
        parameters.extend(
            [
                {
                    "outer_fold": int(fold),
                    "model": "dynamic",
                    "amplitude": dynamic_fit.amplitude,
                    "decay_rate": dynamic_fit.decay_rate,
                    "training_objective": dynamic_fit.objective,
                },
                {
                    "outer_fold": int(fold),
                    "model": "static",
                    "amplitude": static_fit.amplitude,
                    "decay_rate": static_fit.decay_rate,
                    "training_objective": static_fit.objective,
                },
            ]
        )
    if fit_rows[["predicted_dynamic", "predicted_static"]].isna().any().any():
        raise RuntimeError("Incomplete outer-fold transition predictions")
    fit_rows["predicted_zero"] = 0.0
    return fit_rows, pd.DataFrame(parameters)


def full_fit_holdout(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    training = frame.loc[frame["global_step"].isin(FIT_STEPS)]
    holdout = frame.loc[frame["global_step"].isin(HOLDOUT_STEPS)].copy()
    dynamic_fit = fit_decay(training, dynamic=True)
    static_fit = fit_decay(training, dynamic=False)
    holdout["predicted_dynamic"] = predict(holdout, dynamic_fit)
    holdout["predicted_static"] = predict(holdout, static_fit)
    holdout["predicted_zero"] = 0.0
    parameters = pd.DataFrame(
        [
            {
                "outer_fold": "full",
                "model": "dynamic",
                "amplitude": dynamic_fit.amplitude,
                "decay_rate": dynamic_fit.decay_rate,
                "training_objective": dynamic_fit.objective,
            },
            {
                "outer_fold": "full",
                "model": "static",
                "amplitude": static_fit.amplitude,
                "decay_rate": static_fit.decay_rate,
                "training_objective": static_fit.objective,
            },
        ]
    )
    return holdout, parameters


def metrics_for_block(frame: pd.DataFrame, scope: str, step: int | str) -> list[dict[str, object]]:
    observed = frame["common_residual"].to_numpy(float)
    zero_rmse = rmse(observed, np.zeros_like(observed))
    rows = []
    for model in ("dynamic", "static", "zero"):
        predicted = frame[f"predicted_{model}"].to_numpy(float)
        rho, p_value = spearmanr(predicted, observed)
        candidate_rmse = rmse(observed, predicted)
        rows.append(
            {
                "scope": scope,
                "global_step": step,
                "model": model,
                "rows": len(frame),
                "rmse": candidate_rmse,
                "zero_improvement": 1.0 - candidate_rmse / zero_rmse,
                "spearman": float(rho),
                "spearman_p": float(p_value),
                "calibration_slope": calibration_slope(observed, predicted),
                "bias": float(np.mean(predicted - observed)),
                "amplitude_ratio": float(np.std(predicted) / np.std(observed)),
            }
        )
    return rows


def build_metrics(fit_rows: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    rows = metrics_for_block(fit_rows, "fit_oof", "pooled")
    for step, block in fit_rows.groupby("global_step", sort=True):
        rows.extend(metrics_for_block(block, "fit_oof", int(step)))
    for step, block in holdout.groupby("global_step", sort=True):
        rows.extend(metrics_for_block(block, "temporal_holdout", int(step)))
    return pd.DataFrame(rows)


def incremental_metrics(fit_rows: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, label, block in (
        ("fit_oof", "pooled", fit_rows),
        *(("temporal_holdout", int(step), local) for step, local in holdout.groupby("global_step", sort=True)),
    ):
        for baseline in ("predicted_static", "predicted_zero"):
            mean, low, high = paired_bootstrap_difference(
                block.reset_index(drop=True),
                "predicted_dynamic",
                baseline,
                BOOTSTRAP_SEED + len(rows),
            )
            rows.append(
                {
                    "scope": scope,
                    "global_step": label,
                    "comparison": f"dynamic_minus_{baseline.removeprefix('predicted_')}",
                    "rmse_difference_mean": mean,
                    "rmse_difference_low": low,
                    "rmse_difference_high": high,
                }
            )
    return pd.DataFrame(rows)


def decide(
    metrics: pd.DataFrame,
    incremental: pd.DataFrame,
    parameters: pd.DataFrame,
    fit_rows: pd.DataFrame,
) -> dict[str, object]:
    indexed = metrics.set_index(["scope", "global_step", "model"])
    increments = incremental.set_index(["scope", "global_step", "comparison"])
    dynamic_parameters = parameters.loc[parameters["model"].eq("dynamic")]
    fold_parameters = dynamic_parameters.loc[dynamic_parameters["outer_fold"].ne("full")]
    full = dynamic_parameters.loc[dynamic_parameters["outer_fold"].eq("full")].iloc[0]
    fold_rmses = []
    for fold, block in fit_rows.groupby("outer_fold", sort=True):
        observed = block["common_residual"].to_numpy(float)
        fold_rmses.append(
            {
                "fold": fold,
                "dynamic": rmse(observed, block["predicted_dynamic"].to_numpy(float)),
                "static": rmse(observed, block["predicted_static"].to_numpy(float)),
            }
        )
    fold_frame = pd.DataFrame(fold_rmses)
    fold_wins = int((fold_frame["dynamic"] < fold_frame["static"]).sum())
    step22000_dynamic = float(indexed.loc[("temporal_holdout", 22_000, "dynamic"), "rmse"])
    step22000_zero = float(indexed.loc[("temporal_holdout", 22_000, "zero"), "rmse"])
    step22000_static = float(indexed.loc[("temporal_holdout", 22_000, "static"), "rmse"])
    final_dynamic = float(indexed.loc[("temporal_holdout", switch_shock.FINAL_STEP, "dynamic"), "rmse"])
    final_zero = float(indexed.loc[("temporal_holdout", switch_shock.FINAL_STEP, "zero"), "rmse"])
    final_static = float(indexed.loc[("temporal_holdout", switch_shock.FINAL_STEP, "static"), "rmse"])
    checks = {
        "fit_oof_zero_improvement": float(indexed.loc[("fit_oof", "pooled", "dynamic"), "zero_improvement"]) >= 0.10,
        "fit_dynamic_vs_static_bootstrap": (
            float(increments.loc[("fit_oof", "pooled", "dynamic_minus_static"), "rmse_difference_high"]) < 0.0
        ),
        "fit_dynamic_vs_static_fold_wins": fold_wins >= 4,
        "full_decay_rate_interior": 0.05 <= float(full["decay_rate"]) <= 45.0,
        "fold_decay_positive": int((fold_parameters["decay_rate"].astype(float) >= 0.05).sum()) >= 4,
        "fold_amplitude_positive": int((fold_parameters["amplitude"].astype(float) > 0.0).sum()) == 5,
        "step22000_zero_regression": step22000_dynamic <= 1.02 * step22000_zero,
        "step22000_beats_static": step22000_dynamic < step22000_static,
        "final_zero_regression": final_dynamic <= 1.02 * final_zero,
        "final_beats_static": final_dynamic < final_static,
    }
    passed = all(checks.values())
    return {
        "candidate_id": CANDIDATE_ID,
        "passed": passed,
        "decision": (
            "PASS: shock-initiated transient identified; endpoint state remains unsolved"
            if passed
            else "FAIL: shock-initiated exponential transient rejected"
        ),
        "checks": checks,
        "fit_dynamic_vs_static_fold_wins": fold_wins,
        "full_parameters": {
            "amplitude": float(full["amplitude"]),
            "decay_rate": float(full["decay_rate"]),
            "half_life_phase1_fraction": float(np.log(2.0) / full["decay_rate"]),
        },
        "scope": "transition_diagnostic_only_not_endpoint_surrogate",
    }


def render_plot(fit_rows: pd.DataFrame, holdout: pd.DataFrame, parameters: pd.DataFrame, path: Path) -> None:
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Fit-step OOF residuals", "Mean residual trajectory", "Decay-rate stability"),
    )
    figure.add_trace(
        go.Scatter(
            x=fit_rows["common_residual"],
            y=fit_rows["predicted_dynamic"],
            mode="markers",
            marker={"color": fit_rows["global_step"], "colorscale": "RdYlGn_r", "size": 6},
            text=fit_rows["pair_id"],
            name="fit rows",
        ),
        row=1,
        col=1,
    )
    combined = pd.concat([fit_rows.assign(stage="fit OOF"), holdout.assign(stage="holdout")], ignore_index=True)
    trajectory = combined.groupby(["global_step", "stage"], as_index=False).agg(
        observed=("common_residual", "mean"),
        dynamic=("predicted_dynamic", "mean"),
        static=("predicted_static", "mean"),
    )
    for column, label, color in (
        ("observed", "observed", "#d95f0e"),
        ("dynamic", "dynamic", "#1f6f8b"),
        ("static", "static shock", "#718096"),
    ):
        figure.add_trace(
            go.Scatter(
                x=trajectory["global_step"],
                y=trajectory[column],
                mode="lines+markers",
                name=label,
                line={"color": color},
            ),
            row=1,
            col=2,
        )
    dynamic_parameters = parameters.loc[parameters["model"].eq("dynamic")]
    figure.add_trace(
        go.Bar(
            x=dynamic_parameters["outer_fold"].astype(str),
            y=dynamic_parameters["decay_rate"],
            marker_color="#1f6f8b",
            name="decay rate",
        ),
        row=1,
        col=3,
    )
    figure.update_xaxes(title_text="Observed common residual (BPB)", row=1, col=1)
    figure.update_yaxes(title_text="Predicted common residual (BPB)", row=1, col=1)
    figure.update_xaxes(title_text="Global step", row=1, col=2)
    figure.update_yaxes(title_text="Mean common residual (BPB)", row=1, col=2)
    figure.update_xaxes(title_text="Outer fold", row=1, col=3)
    figure.update_yaxes(title_text="Decay rate per phase-1 duration", row=1, col=3)
    figure.update_layout(title="Shock-initiated phase-1 transient", template="plotly_white", width=1500, height=500)
    figure.write_html(path, include_plotlyjs="cdn")


def render_report(
    protocol: dict[str, object],
    metrics: pd.DataFrame,
    incremental: pd.DataFrame,
    parameters: pd.DataFrame,
    decision: dict[str, object],
    path: Path,
) -> None:
    path.write_text(
        "\n".join(
            [
                "# Shock-initiated phase-1 transient",
                "",
                f"**Decision: {decision['decision']}**",
                "",
                f"Frozen protocol: `{protocol['protocol_sha256']}`.",
                "",
                "The initial state is SUR-070's blocked-OOF policy prediction of the observed gradient shock. "
                "A nonnegative BPB response amplitude and one exponential decay rate are fit only on steps "
                "19000--21000. Step 22000 and final are applied without refitting.",
                "",
                "## Metrics",
                "",
                metrics.to_markdown(index=False),
                "",
                "## Paired-bootstrap comparisons",
                "",
                incremental.to_markdown(index=False),
                "",
                "## Parameter stability",
                "",
                parameters.to_markdown(index=False),
                "",
                "## Interpretation boundary",
                "",
                "A pass identifies only an early transient. It cannot establish a persistent endpoint temporal "
                "state or repair the aggregate spine. All temporal-holdout outcomes were exposed in prior "
                "development summaries and are not confirmatory.",
                "",
            ]
        )
    )


def evaluate(output_dir: Path) -> None:
    protocol = require_frozen_protocol(output_dir)
    frame = load_frame()
    fit_rows, fold_parameters = out_of_fold_predictions(frame)
    holdout, full_parameters = full_fit_holdout(frame)
    parameters = pd.concat([fold_parameters, full_parameters], ignore_index=True)
    metrics = build_metrics(fit_rows, holdout)
    incremental = incremental_metrics(fit_rows, holdout)
    decision = decide(metrics, incremental, parameters, fit_rows)

    fit_rows.to_csv(output_dir / "fit_oof_predictions.csv", index=False)
    holdout.to_csv(output_dir / "temporal_holdout_predictions.csv", index=False)
    parameters.to_csv(output_dir / "parameters.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    incremental.to_csv(output_dir / "incremental_metrics.csv", index=False)
    write_json(output_dir / "decision.json", decision)
    render_plot(fit_rows, holdout, parameters, output_dir / "shock_initiated_transient.html")
    render_report(protocol, metrics, incremental, parameters, decision, output_dir / "report.md")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "preregister":
        freeze_protocol(args.output_dir)
        return
    evaluate(args.output_dir)


if __name__ == "__main__":
    main()
