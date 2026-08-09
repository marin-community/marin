# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
# ]
# ///
"""Test whether retained-mass phase effects equal an external token-budget effect.

This is an identification diagnostic, not a fitted phase model. It imports the
Programming-Languages BPB slope from the existing 1B--8B fixed-model,
fixed-simulated-epoch token ladder and predicts each phase-pair delta as

    delta_bpb = token_slope * log(retained_mass_asymmetric / retained_mass_tied).

The retained-mass coordinate is frozen from the exposed SUR-046 development
fit. No coefficient is fitted to phase-pair outcomes. The small mismatch
between nominal 80/20 fibers and the realized 3040/3814 phase boundary is
removed with a tied-only piecewise-linear response curve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    starcoder_wsd80_panel_20260728 as wsd80,
)

TARGET = "eval/paloma/dolma_100_programing_languages-llama3/bpb"
TOKEN_LADDER_DIR = (
    SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_fixed_model_token_scaling_20260728" / "results_20260730"
)
TOKEN_OBSERVATIONS = TOKEN_LADDER_DIR / "observations.csv"
MASS_DIAGNOSTIC_DIR = SCRIPT_DIR / "reference_outputs" / "retained_mass_artifact_diagnostic_20260731"
PAIR_DIAGNOSTICS = MASS_DIAGNOSTIC_DIR / "pair_diagnostics.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "effective_budget_equivalence_20260731"
ALL_METRICS_CSV = wsd80.SURFACE_DIR / "wsd80_all_bpb_metrics.csv"

PROTOCOL_VERSION = "effective-budget-equivalence-v1"
REFERENCE_SEED = 20260711
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260731
MAX_LEAVE_ONE_AGGREGATE_RELATIVE_SLOPE_CHANGE = 0.25
EXPORT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_inputs() -> tuple[Path, ...]:
    return (
        Path(__file__),
        TOKEN_OBSERVATIONS,
        PAIR_DIAGNOSTICS,
        wsd80.SURFACE_CSV,
        ALL_METRICS_CSV,
        Path(wsd80.__file__),
    )


def protocol_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": PROTOCOL_VERSION,
        "candidate": "WSD80-SUR-049 diagnostic only",
        "hypothesis": "retained mass is quantitatively equivalent to an independently measured token-budget multiplier",
        "target": TARGET,
        "prediction": "delta_bpb = shared_token_slope * log_retained_mass_ratio",
        "phase_parameters": "frozen target-specific SUR-046 full-fit retained-mass coordinate; no refit",
        "token_slope_rows": {
            "policy_class": "tied",
            "trainer_data_seed": REFERENCE_SEED,
            "simulated_epoch_subset_seed": REFERENCE_SEED,
            "token_budgets": [1_000_000_000, 2_000_000_000, 4_000_000_000, 8_000_000_000],
        },
        "token_slope_estimator": (
            "fit one BPB-versus-log-materialized-tokens slope per tied aggregate, "
            "then average the six slopes with equal aggregate weight"
        ),
        "aggregate_mismatch_correction": (
            "piecewise-linear interpolation of 1B tied-only WSD80 BPB; subtract "
            "F(a_realized_asymmetric)-F(a_realized_tied) from each nominally matched pair delta"
        ),
        "primary_pairs": "nominal aggregate within the tied token-ladder support [0.10, 0.35]",
        "secondary_pairs": "all nominally aggregate-matched WSD80 pairs",
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "resampling": "tied aggregate slopes and phase pairs independently with replacement",
        },
        "frozen_gate": {
            "all_six_token_slopes_negative": True,
            "maximum_leave_one_aggregate_relative_slope_change": MAX_LEAVE_ONE_AGGREGATE_RELATIVE_SLOPE_CHANGE,
            "rmse_improvement_ci95_lower_greater_than_zero": True,
            "calibration_slope_ci95_contains_one_and_excludes_zero": True,
            "absolute_bias_no_worse_than_zero_phase_null": True,
        },
        "interpretation": (
            "passing supports effective-budget equivalence on Programming Languages only; "
            "failing rejects this equivalence but does not prove retained mass is causally irrelevant"
        ),
        "data_use": (
            "all phase outcomes are exposed development evidence; the token slope is estimated "
            "without asymmetric-policy outcomes"
        ),
        "source_hashes": {str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in source_inputs()},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "protocol_hash": hashlib.sha256(encoded).hexdigest()}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def slope_with_intercept(x: np.ndarray, y: np.ndarray) -> float:
    centered = x - np.mean(x)
    denominator = float(centered @ centered)
    if denominator <= 0.0:
        raise ValueError("slope input has no variation")
    return float(centered @ (y - np.mean(y)) / denominator)


def load_token_slopes() -> tuple[pd.DataFrame, pd.DataFrame]:
    observations = pd.read_csv(TOKEN_OBSERVATIONS)
    tied = observations[
        observations["phase_contrast"].abs().lt(1e-12)
        & observations["trainer_data_seed"].eq(REFERENCE_SEED)
        & observations["simulated_epoch_subset_seed"].eq(REFERENCE_SEED)
    ].copy()
    expected_budgets = {1_000_000_000, 2_000_000_000, 4_000_000_000, 8_000_000_000}
    if set(tied["token_budget_requested"].astype(int)) != expected_budgets:
        raise ValueError("tied token-ladder rows do not cover the frozen four budgets")

    counts = tied.groupby("coordinate_index")["token_budget_requested"].nunique()
    if len(counts) != 6 or not counts.eq(4).all():
        raise ValueError("expected six tied aggregates with all four token rungs")

    tied["log_materialized_tokens"] = np.log(tied["materialized_tokens"] / 1_000_000_000)
    rows: list[dict[str, float | int]] = []
    for coordinate, block in tied.groupby("coordinate_index", sort=True):
        rows.append(
            {
                "coordinate_index": int(coordinate),
                "aggregate_starcoder": float(block["aggregate_starcoder_nominal"].iloc[0]),
                "slope_bpb_per_log_token": slope_with_intercept(
                    block["log_materialized_tokens"].to_numpy(dtype=float),
                    block["starcoder_bpb"].to_numpy(dtype=float),
                ),
            }
        )
    slopes = pd.DataFrame(rows)
    shared = float(slopes["slope_bpb_per_log_token"].mean())
    leave_one_out = np.asarray(
        [slopes.loc[slopes.index != index, "slope_bpb_per_log_token"].mean() for index in slopes.index],
        dtype=float,
    )
    slopes["shared_slope"] = shared
    slopes["leave_one_aggregate_out_slope"] = leave_one_out
    slopes["leave_one_relative_change"] = np.abs(leave_one_out - shared) / abs(shared)
    return tied, slopes


def tied_curve(panel: wsd80.Panel, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    starcoder_phase_0 = panel.weights[:, 0, 1]
    starcoder_phase_1 = panel.weights[:, 1, 1]
    tied = np.isclose(starcoder_phase_0, starcoder_phase_1, atol=1e-12)
    values = pd.DataFrame(
        {
            "aggregate": starcoder_phase_0[tied],
            "bpb": frame.loc[tied, TARGET].to_numpy(dtype=float),
        }
    )
    grouped = values.groupby("aggregate", as_index=False)["bpb"].mean().sort_values("aggregate")
    if len(grouped) < 6:
        raise ValueError("too few tied coordinates for aggregate mismatch correction")
    return grouped["aggregate"].to_numpy(dtype=float), grouped["bpb"].to_numpy(dtype=float)


def load_metric_panel() -> tuple[wsd80.Panel, pd.DataFrame]:
    panel = wsd80.load_surface()
    metrics = pd.read_csv(ALL_METRICS_CSV)
    merged = panel.frame.merge(metrics, on="wandb_run_id", how="left", validate="one_to_one")
    if len(merged) != len(panel.frame):
        raise ValueError("WSD80 metric join changed the panel row count")
    if merged[TARGET].isna().any():
        raise ValueError(f"missing complete target {TARGET}")
    return panel, merged


def load_phase_pairs(token_slopes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = pd.read_csv(PAIR_DIAGNOSTICS)
    pairs = pairs[pairs["panel"].eq("starcoder_wsd80") & pairs["target"].eq(TARGET)].copy()
    panel, frame = load_metric_panel()

    tied_rows = pairs["tied_row"].to_numpy(dtype=int)
    asymmetric_rows = pairs["asymmetric_row"].to_numpy(dtype=int)
    if np.max(np.concatenate([tied_rows, asymmetric_rows])) >= len(panel.y):
        raise ValueError("pair row index exceeds WSD80 panel")

    phase_0 = panel.weights[:, 0, 1]
    phase_1 = panel.weights[:, 1, 1]
    aggregate_nominal = wsd80.PHASE_0_FRACTION * phase_0 + wsd80.PHASE_1_FRACTION * phase_1
    aggregate_realized = wsd80.REALIZED_PHASE_0_FRACTION * phase_0 + wsd80.REALIZED_PHASE_1_FRACTION * phase_1
    curve_x, curve_y = tied_curve(panel, frame)
    asym_realized = aggregate_realized[asymmetric_rows]
    tied_realized = aggregate_realized[tied_rows]
    if np.any(asym_realized < curve_x[0]) or np.any(asym_realized > curve_x[-1]):
        raise ValueError("realized asymmetric aggregate lies outside the tied interpolation range")

    aggregate_correction = np.interp(asym_realized, curve_x, curve_y) - np.interp(tied_realized, curve_x, curve_y)
    shared_slope = float(token_slopes["shared_slope"].iloc[0])
    pairs["aggregate_starcoder_nominal"] = aggregate_nominal[asymmetric_rows]
    pairs["aggregate_starcoder_realized"] = asym_realized
    pairs["aggregate_mismatch_bpb"] = aggregate_correction
    pairs["observed_phase_delta"] = pairs["observed_delta"] - aggregate_correction
    pairs["predicted_effective_budget_delta"] = shared_slope * pairs["log_retained_mass_ratio"]

    support_min = float(token_slopes["aggregate_starcoder"].min())
    support_max = float(token_slopes["aggregate_starcoder"].max())
    primary = pairs[pairs["aggregate_starcoder_nominal"].between(support_min - 1e-12, support_max + 1e-12)].copy()
    if len(primary) < 30:
        raise ValueError("effective-budget primary subset has fewer than 30 pairs")
    return pairs, primary


def calibration_slope(predicted: np.ndarray, observed: np.ndarray) -> float:
    return slope_with_intercept(predicted, observed)


def summarize_pairs(frame: pd.DataFrame) -> dict[str, float | int]:
    observed = frame["observed_phase_delta"].to_numpy(dtype=float)
    predicted = frame["predicted_effective_budget_delta"].to_numpy(dtype=float)
    zero_rmse = float(np.sqrt(np.mean(observed**2)))
    model_rmse = float(np.sqrt(np.mean((predicted - observed) ** 2)))
    rank = spearmanr(predicted, observed)
    return {
        "pairs": len(frame),
        "zero_phase_rmse": zero_rmse,
        "effective_budget_rmse": model_rmse,
        "rmse_improvement": zero_rmse - model_rmse,
        "zero_phase_bias": float(-np.mean(observed)),
        "effective_budget_bias": float(np.mean(predicted - observed)),
        "calibration_intercept": float(np.mean(observed) - calibration_slope(predicted, observed) * np.mean(predicted)),
        "calibration_slope": calibration_slope(predicted, observed),
        "spearman": float(rank.statistic),
        "sign_accuracy": float(np.mean(np.sign(predicted) == np.sign(observed))),
        "mean_absolute_aggregate_correction": float(np.mean(np.abs(frame["aggregate_mismatch_bpb"]))),
        "max_absolute_aggregate_correction": float(np.max(np.abs(frame["aggregate_mismatch_bpb"]))),
    }


def bootstrap(
    primary: pd.DataFrame,
    token_slopes: pd.DataFrame,
) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    observed = primary["observed_phase_delta"].to_numpy(dtype=float)
    log_mass = primary["log_retained_mass_ratio"].to_numpy(dtype=float)
    slopes = token_slopes["slope_bpb_per_log_token"].to_numpy(dtype=float)
    rows: list[dict[str, float]] = []
    for _draw in range(BOOTSTRAP_DRAWS):
        slope = float(np.mean(rng.choice(slopes, size=len(slopes), replace=True)))
        indices = rng.integers(0, len(primary), size=len(primary))
        observed_draw = observed[indices]
        predicted_draw = slope * log_mass[indices]
        zero_rmse = float(np.sqrt(np.mean(observed_draw**2)))
        model_rmse = float(np.sqrt(np.mean((predicted_draw - observed_draw) ** 2)))
        rows.append(
            {
                "token_slope": slope,
                "rmse_improvement": zero_rmse - model_rmse,
                "calibration_slope": calibration_slope(predicted_draw, observed_draw),
                "effective_budget_bias": float(np.mean(predicted_draw - observed_draw)),
            }
        )
    return pd.DataFrame(rows)


def interval(values: pd.Series) -> tuple[float, float]:
    low, high = np.quantile(values.to_numpy(dtype=float), [0.025, 0.975])
    return float(low), float(high)


def gate_result(
    primary_summary: dict[str, float | int],
    token_slopes: pd.DataFrame,
    bootstraps: pd.DataFrame,
) -> dict[str, Any]:
    rmse_low, rmse_high = interval(bootstraps["rmse_improvement"])
    calibration_low, calibration_high = interval(bootstraps["calibration_slope"])
    checks = {
        "all_six_token_slopes_negative": bool((token_slopes["slope_bpb_per_log_token"] < 0.0).all()),
        "leave_one_aggregate_slope_stable": bool(
            token_slopes["leave_one_relative_change"].max() <= MAX_LEAVE_ONE_AGGREGATE_RELATIVE_SLOPE_CHANGE
        ),
        "rmse_improvement_ci95_lower_greater_than_zero": rmse_low > 0.0,
        "calibration_slope_ci95_contains_one_and_excludes_zero": (
            calibration_low > 0.0 and calibration_low <= 1.0 <= calibration_high
        ),
        "absolute_bias_no_worse_than_zero_phase_null": (
            abs(float(primary_summary["effective_budget_bias"])) <= abs(float(primary_summary["zero_phase_bias"]))
        ),
    }
    return {
        "decision": (
            "supports_effective_budget_equivalence" if all(checks.values()) else "rejects_effective_budget_equivalence"
        ),
        "checks": checks,
        "rmse_improvement_ci95": [rmse_low, rmse_high],
        "calibration_slope_ci95": [calibration_low, calibration_high],
        "token_slope_ci95": list(interval(bootstraps["token_slope"])),
    }


def render_pair_plot(output_dir: Path, primary: pd.DataFrame) -> None:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=primary["predicted_effective_budget_delta"],
            y=primary["observed_phase_delta"],
            mode="markers",
            marker={
                "color": primary["aggregate_starcoder_nominal"],
                "colorscale": "RdYlGn_r",
                "size": 8,
                "line": {"color": "#173247", "width": 0.5},
                "colorbar": {"title": "Nominal aggregate"},
            },
            customdata=np.column_stack(
                [
                    primary["aggregate_starcoder_nominal"],
                    primary["retained_mass_ratio"],
                    primary["aggregate_mismatch_bpb"],
                ]
            ),
            hovertemplate=(
                "Predicted %{x:+.5f} BPB<br>Observed %{y:+.5f} BPB"
                "<br>Aggregate %{customdata[0]:.3f}"
                "<br>Retained-mass ratio %{customdata[1]:.3f}"
                "<br>Aggregate correction %{customdata[2]:+.5f}<extra></extra>"
            ),
        )
    )
    bounds = np.asarray(
        [
            primary["predicted_effective_budget_delta"].min(),
            primary["predicted_effective_budget_delta"].max(),
            primary["observed_phase_delta"].min(),
            primary["observed_phase_delta"].max(),
        ],
        dtype=float,
    )
    low, high = float(bounds.min()), float(bounds.max())
    figure.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            line={"color": "#173247", "dash": "dash"},
            name="Calibrated equivalence",
            hoverinfo="skip",
        )
    )
    figure.update_layout(
        title="Effective-budget equivalence: frozen prediction versus phase delta",
        xaxis_title="Predicted BPB delta from token slope x log retained mass",
        yaxis_title="Observed phase BPB delta after aggregate correction",
        template="plotly_white",
        width=1050,
        height=760,
    )
    figure.write_html(
        output_dir / "effective_budget_predicted_vs_observed.html",
        include_plotlyjs=True,
        config=EXPORT_CONFIG,
    )


def write_report(
    output_dir: Path,
    protocol: dict[str, Any],
    token_slopes: pd.DataFrame,
    summaries: pd.DataFrame,
    gate: dict[str, Any],
) -> None:
    slope = float(token_slopes["shared_slope"].iloc[0])
    primary = summaries[summaries["subset"].eq("primary")].iloc[0]
    lines = [
        "# Effective-Budget Equivalence Diagnostic",
        "",
        f"- Protocol: `{protocol['protocol_hash']}`",
        f"- Decision: **{gate['decision']}**",
        f"- Imported token slope: `{slope:+.6f}` BPB per log materialized token",
        f"- Primary phase pairs: `{int(primary['pairs'])}`",
        "",
        "## Frozen prediction",
        "",
        r"\[",
        r"\Delta L_{\mathrm{budget}}=s_{\mathrm{token}}\log\left(M_{\mathrm{asym}}/M_{\mathrm{tied}}\right).",
        r"\]",
        "",
        "No coefficient is fitted to phase-pair outcomes. The retained-mass coordinate is frozen from",
        "the exposed SUR-046 development fit, and the target response slope is imported from tied rows",
        "of the orthogonal 1B--8B token ladder.",
        "",
        "## Pair metrics",
        "",
        "| subset | pairs | zero RMSE | budget RMSE | improvement | calibration slope | Spearman | sign accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries.itertuples(index=False):
        lines.append(
            f"| {row.subset} | {int(row.pairs)} | {row.zero_phase_rmse:.6f} | "
            f"{row.effective_budget_rmse:.6f} | {row.rmse_improvement:+.6f} | "
            f"{row.calibration_slope:.3f} | {row.spearman:.3f} | {row.sign_accuracy:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen gate",
            "",
        ]
    )
    for name, passed in gate["checks"].items():
        lines.append(f"- `{name}`: **{'PASS' if passed else 'FAIL'}**")
    lines.extend(
        [
            f"- RMSE-improvement 95% interval: `{gate['rmse_improvement_ci95'][0]:+.6f}` to "
            f"`{gate['rmse_improvement_ci95'][1]:+.6f}` BPB",
            f"- Calibration-slope 95% interval: `{gate['calibration_slope_ci95'][0]:.3f}` to "
            f"`{gate['calibration_slope_ci95'][1]:.3f}`",
            f"- Token-slope 95% interval: `{gate['token_slope_ci95'][0]:+.6f}` to "
            f"`{gate['token_slope_ci95'][1]:+.6f}`",
            "",
            "## Interpretation boundary",
            "",
            "A pass would support effective-budget equivalence only for the WSD80 Programming-Languages",
            "development panel. It would not establish retained mass as causal or supply a 39-bucket",
            "phase model. A failure rejects this parameter-free equivalence; it does not prove that",
            "retained mass is irrelevant.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = protocol_payload()
    write_json(args.output_dir / "protocol.json", protocol)
    if args.prepare_only:
        print(f"prepared protocol {protocol['protocol_hash']} in {args.output_dir}", flush=True)
        return

    tied_rows, token_slopes = load_token_slopes()
    all_pairs, primary_pairs = load_phase_pairs(token_slopes)
    summaries = pd.DataFrame(
        [
            {"subset": "primary", **summarize_pairs(primary_pairs)},
            {"subset": "all", **summarize_pairs(all_pairs)},
        ]
    )
    bootstraps = bootstrap(primary_pairs, token_slopes)
    gate = gate_result(summaries[summaries["subset"].eq("primary")].iloc[0].to_dict(), token_slopes, bootstraps)

    tied_rows.to_csv(args.output_dir / "token_ladder_tied_rows.csv", index=False)
    token_slopes.to_csv(args.output_dir / "token_slopes.csv", index=False)
    all_pairs.to_csv(args.output_dir / "pair_predictions.csv", index=False)
    summaries.to_csv(args.output_dir / "summary.csv", index=False)
    bootstraps.to_csv(args.output_dir / "bootstrap.csv", index=False)
    write_json(args.output_dir / "gate.json", gate)
    render_pair_plot(args.output_dir, primary_pairs)
    write_report(args.output_dir, protocol, token_slopes, summaries, gate)
    print(
        json.dumps(
            {
                "protocol_hash": protocol["protocol_hash"],
                "decision": gate["decision"],
                "primary_pairs": len(primary_pairs),
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
