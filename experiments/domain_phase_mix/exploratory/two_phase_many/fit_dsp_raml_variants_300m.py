# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "scipy", "scikit-learn", "tabulate"]
# ///
"""Compare DSP/RAML hybrid variants on 300M target-complete rows.

This is a focused follow-up to the RAML paper read. It tests whether replacing
DSP's exponential saturation feature with the repetition-aware effective
exposure curve improves the existing fixed-300M DSP fits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.fit_and_plot_grp_power_family_penalty_no_l2_60m_vs_300m import (
    _packet_from_frame,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.fit_dsp_canonical_variants_300m import (
    VARIANTS,
    FittedDSPModel,
    _fit_variant,
    _metrics,
    _optimize_model,
    _predict,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.fit_repetition_aware_data_filtering_law_300m import (
    _load_signal_targets,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "dsp_raml_variants_300m_20260521"

SELECTED_VARIANT_NAMES = (
    "dsp_phase_benefit_penalty_nnls",
    "dsp_effective_exposure_penalty_nnls",
    "dsp_phase_benefit_saturation_penalty_nnls",
    "dsp_saturation_penalty_split_nnls",
    "dsp_apple_repetition_per_domain_r1_phase_benefit_penalty_nnls",
    "dsp_raml_phi_shared_r1_benefit_penalty_nnls",
    "dsp_raml_phi_per_domain_r1_benefit_penalty_nnls",
    "dsp_raml_phi_per_domain_r1_effective_penalty_nnls",
)


def _target_packet(signal: pd.DataFrame, values: np.ndarray, target_name: str) -> Any:
    frame = signal.copy()
    frame["objective_metric"] = np.asarray(values, dtype=float)
    return _packet_from_frame(frame, name=f"dsp_raml_{target_name}").base


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _model_record(model: FittedDSPModel, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant": model.variant.name,
        "params": {key: _jsonable(value) for key, value in model.params.items()},
        "intercept": model.intercept,
        "benefit_coef": model.benefit_coef.tolist(),
        "penalty_coef": model.penalty_coef.tolist(),
        "metrics": {key: _jsonable(value) for key, value in metrics.items()},
    }


def _weight_frame(packet: Any, target_name: str, variant_name: str, weights: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for phase_idx, phase_name in enumerate(("phase_0", "phase_1")):
        for domain, weight in zip(packet.domain_names, weights[phase_idx], strict=True):
            rows.append(
                {
                    "target": target_name,
                    "variant": variant_name,
                    "phase": phase_name,
                    "domain": str(domain),
                    "weight": float(weight),
                }
            )
    return pd.DataFrame.from_records(rows)


def _prediction_frame(packet: Any, target_name: str, variant_name: str, model: FittedDSPModel) -> pd.DataFrame:
    predictions = _predict(model, packet.w, packet)
    frame = packet.frame[[packet.name_col]].copy()
    frame = frame.rename(columns={packet.name_col: "run_name"})
    frame["target"] = target_name
    frame["variant"] = variant_name
    frame["actual"] = packet.y
    frame["predicted_train"] = predictions
    frame["residual_train"] = predictions - packet.y
    return frame


def _baseline_row(packet: Any) -> tuple[str, float] | tuple[None, float]:
    run_names = packet.frame[packet.name_col].astype(str)
    exact = np.flatnonzero(run_names.to_numpy() == "baseline_proportional")
    if len(exact) == 1:
        idx = int(exact[0])
        return "baseline_proportional", float(packet.y[idx])
    contains = np.flatnonzero(run_names.str.contains("baseline_proportional", regex=False).to_numpy())
    if len(contains) == 1:
        idx = int(contains[0])
        return str(run_names.iloc[idx]), float(packet.y[idx])
    return None, float("nan")


def _fit_target(
    target_name: str, signal: pd.DataFrame, values: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    variants = {variant.name: variant for variant in VARIANTS}
    packet = _target_packet(signal, values, target_name)
    baseline_run, baseline_value = _baseline_row(packet)
    rows: list[dict[str, Any]] = []
    traces: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    weights_long: list[pd.DataFrame] = []
    model_records: list[dict[str, Any]] = []
    for variant_name in SELECTED_VARIANT_NAMES:
        print(f"Fitting {target_name} / {variant_name}", flush=True)
        variant = variants[variant_name]
        model, trace = _fit_variant(packet, variant)
        raw_result, optimum_weights = _optimize_model(model, packet)
        metric_row = _metrics(packet, model, raw_result, optimum_weights)
        metric_row["target"] = target_name
        metric_row["row_count"] = len(packet.y)
        metric_row["baseline_run_name"] = baseline_run
        metric_row["baseline_value"] = baseline_value
        metric_row["best_observed_value"] = float(np.min(packet.y))
        metric_row["predicted_improvement_vs_baseline"] = (
            float(baseline_value - raw_result.fun) if np.isfinite(baseline_value) else np.nan
        )
        rows.append(metric_row)
        trace = trace.copy()
        trace.insert(0, "target", target_name)
        trace.insert(1, "variant", variant_name)
        traces.append(trace)
        predictions.append(_prediction_frame(packet, target_name, variant_name, model))
        weights_long.append(_weight_frame(packet, target_name, variant_name, optimum_weights))
        model_records.append(_model_record(model, metric_row))
        print(
            f"  cv_rmse={metric_row['cv_rmse']:.6f} spearman={metric_row['oof_spearman']:.6f} "
            f"pred_gain={metric_row['predicted_improvement_vs_baseline']:.6f}",
            flush=True,
        )
    return (
        pd.DataFrame.from_records(rows),
        pd.concat(traces, ignore_index=True),
        pd.concat(predictions, ignore_index=True),
        model_records,
        pd.concat(weights_long, ignore_index=True),
    )


def _write_report(summary: pd.DataFrame) -> None:
    cols = [
        "target",
        "variant",
        "row_count",
        "m_dependent_params_per_domain",
        "total_param_count",
        "repetition_mode",
        "fitted_gamma",
        "fitted_gamma_benefit",
        "fitted_gamma_saturation",
        "fitted_gamma_penalty",
        "fitted_r1",
        "fitted_r1_min",
        "fitted_r1_median",
        "fitted_r1_max",
        "cv_rmse",
        "oof_spearman",
        "cv_foldmean_regret_at_1",
        "lower_tail_optimism",
        "baseline_value",
        "best_observed_value",
        "predicted_improvement_vs_baseline",
        "raw_nearest_observed_tv",
        "phase0_max_weight",
        "phase1_max_weight",
        "optimum_success",
    ]
    lines = [
        "# RAML-DSP Variant Check on 300M",
        "",
        "## Setup",
        "",
        "- Data: current 300M/6B raw metric matrix, target-complete rows only.",
        "- Targets: `eval/uncheatable_eval/bpb` and `-issue5416_aggregate`; lower is better for both rows.",
        "- Baselines: current canonical DSP, effective-exposure DSP, split phase DSP, Apple repetition-discount DSP, and RAML-phi DSP hybrids.",
        "",
        "## Results",
        "",
        summary[cols].sort_values(["target", "cv_rmse"]).to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Reading Notes",
        "",
        "- `predicted_improvement_vs_baseline` is baseline actual value minus the model raw optimum value, so positive is better under the lower-is-better orientation.",
        "- The RAML-phi variants replace \\(1-\\exp(-\\rho_i z_i)\\) with the repetition-aware effective exposure curve itself.",
        r"- Per-domain \(r_1\) is an extra domain-dependent parameter; the shared-\(r_1\) variant is the lower-capacity check.",
        "- Raw optima remain diagnostics, not launch recommendations, unless the mixture is non-degenerate and near enough to observed support.",
        "",
    ]
    best_by_target = summary.loc[summary.groupby("target")["cv_rmse"].idxmin()]
    lines.extend(["## Best By CV RMSE", "", best_by_target[cols].to_markdown(index=False, floatfmt=".6f"), ""])
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targets, _projection = _load_signal_targets()
    summaries: list[pd.DataFrame] = []
    traces: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    weights: list[pd.DataFrame] = []
    models: list[dict[str, Any]] = []
    for target_name, (signal, values) in targets.items():
        summary, trace, prediction, model_records, weight = _fit_target(target_name, signal, values)
        summaries.append(summary)
        traces.append(trace)
        predictions.append(prediction)
        weights.append(weight)
        models.extend(model_records)
    summary_frame = pd.concat(summaries, ignore_index=True)
    trace_frame = pd.concat(traces, ignore_index=True)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    weight_frame = pd.concat(weights, ignore_index=True)
    summary_frame.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    trace_frame.to_csv(OUTPUT_DIR / "tuning.csv", index=False)
    prediction_frame.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    weight_frame.to_csv(OUTPUT_DIR / "raw_optimum_weights_long.csv", index=False)
    (OUTPUT_DIR / "models.json").write_text(json.dumps(models, indent=2))
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary_frame.to_dict(orient="records"), indent=2))
    _write_report(summary_frame)
    print(f"Wrote {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
