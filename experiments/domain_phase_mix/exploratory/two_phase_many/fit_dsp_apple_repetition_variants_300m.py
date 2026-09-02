# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "scipy", "scikit-learn", "tabulate"]
# ///
"""Compare Apple-style repetition-aware DSP variants on the 300M swarm."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.fit_dsp_canonical_variants_300m import (
    VARIANTS,
    DSPVariant,
    _append_logbook,
    _fit_variant,
    _load_packet,
    _metrics,
    _observed_leaderboard_row,
    _old_grp_baseline_row,
    _optimize_model,
    _plot_predicted_vs_actual,
    _plot_raw_optimum,
    _predict,
    _write_weight_table,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "dsp_apple_repetition_variants_300m_20260514"

SELECTED_VARIANT_NAMES = (
    "dsp_phase_benefit_penalty_nnls",
    "dsp_effective_exposure_penalty_nnls",
    "dsp_phase_benefit_saturation_penalty_nnls",
    "dsp_apple_repetition_shared_r1_phase_benefit_penalty_nnls",
    "dsp_apple_repetition_per_domain_r1_phase_benefit_penalty_nnls",
)


def _selected_variants() -> tuple[DSPVariant, ...]:
    variant_by_name = {variant.name: variant for variant in VARIANTS}
    missing = [name for name in SELECTED_VARIANT_NAMES if name not in variant_by_name]
    if missing:
        raise ValueError(f"Missing DSP variants: {missing}")
    return tuple(variant_by_name[name] for name in SELECTED_VARIANT_NAMES)


def _write_predictions(packet: Any, model: Any, variant_dir: Path) -> None:
    prediction_frame = packet.frame[[packet.name_col]].copy()
    prediction_frame["actual_bpb"] = packet.y
    prediction_frame["predicted_bpb"] = _predict(model, packet.w, packet)
    prediction_frame["residual_bpb"] = prediction_frame["predicted_bpb"] - prediction_frame["actual_bpb"]
    prediction_frame.to_csv(variant_dir / "predictions.csv", index=False)


def _report(summary: pd.DataFrame, observed: pd.DataFrame) -> str:
    sorted_summary = summary.sort_values(["cv_rmse", "oof_spearman"], ascending=[True, False])
    columns = [
        "variant",
        "total_param_count",
        "repetition_mode",
        "fitted_r1",
        "fitted_r1_min",
        "fitted_r1_median",
        "fitted_r1_max",
        "fitted_gamma",
        "fitted_gamma_benefit",
        "fitted_gamma_saturation",
        "fitted_gamma_penalty",
        "train_rmse",
        "cv_rmse",
        "oof_spearman",
        "cv_foldmean_regret_at_1",
        "lower_tail_optimism",
        "raw_nearest_observed_tv",
        "phase0_max_weight",
        "phase1_max_weight",
    ]
    baseline = summary.set_index("variant").loc["dsp_phase_benefit_penalty_nnls"]
    effective = summary.set_index("variant").loc["dsp_effective_exposure_penalty_nnls"]
    shared = summary.set_index("variant").loc["dsp_apple_repetition_shared_r1_phase_benefit_penalty_nnls"]
    per_domain = summary.set_index("variant").loc["dsp_apple_repetition_per_domain_r1_phase_benefit_penalty_nnls"]
    split = summary.set_index("variant").loc["dsp_phase_benefit_saturation_penalty_nnls"]
    lines = [
        "# Apple-Style Repetition-Aware DSP Check on 300M",
        "",
        "## Model Change",
        "",
        "The Apple-style variants keep the DSP benefit/penalty head but replace the saturation exposure with",
        "a repetition-discounted physical exposure. For domain `i`, raw physical exposure is measured in epochs:",
        "",
        "$$r_i(w)=c_{0i}w_{0i}+c_{1i}w_{1i}.$$",
        "",
        "The saturation exposure is transformed by:",
        "",
        "$$\\phi(r_i;r_1)=\\begin{cases}r_i,& r_i\\le1\\\\1+r_1(1-e^{-(r_i-1)/r_1}),& r_i>1.\\end{cases}$$",
        "",
        "The benefit signal is:",
        "",
        "$$S_i(w)=\\left(1+\\gamma\\frac{c_{1i}w_{1i}}{r_i(w)+\\epsilon}\\right)\\left(1-e^{-\\rho_i\\phi(r_i(w);r_1)}\\right).$$",
        "",
        "The penalty intentionally remains on raw physical exposure:",
        "",
        "$$P_i(w)=\\operatorname{softplus}(\\log(1+r_i(w))-\\tau_i)^2.$$",
        "",
        "This tests Apple-style repetition discounting without assuming phase 1 is physically repeated faster.",
        "",
        "## Results",
        "",
        sorted_summary[columns].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Best Observed Rows by Prediction",
        "",
        observed.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation",
        "",
        f"- Canonical benefit-only DSP: CV RMSE {float(baseline['cv_rmse']):.6f}, OOF Spearman {float(baseline['oof_spearman']):.6f}.",
        f"- Tied effective-exposure DSP: CV RMSE {float(effective['cv_rmse']):.6f}, OOF Spearman {float(effective['oof_spearman']):.6f}.",
        f"- Split phase benefit/saturation/penalty DSP: CV RMSE {float(split['cv_rmse']):.6f}, OOF Spearman {float(split['oof_spearman']):.6f}.",
        f"- Apple shared-r1 DSP: CV RMSE {float(shared['cv_rmse']):.6f}, OOF Spearman {float(shared['oof_spearman']):.6f}, fitted r1 {float(shared['fitted_r1']):.6f}.",
        f"- Apple per-domain-r1 DSP: CV RMSE {float(per_domain['cv_rmse']):.6f}, OOF Spearman {float(per_domain['oof_spearman']):.6f}, median fitted r1 {float(per_domain['fitted_r1_median']):.6f}.",
        "- The per-domain-r1 form should only be preferred if it improves OOF fit and optimum geometry enough to justify 39 extra nonlinear parameters.",
        "",
    ]
    return "\n".join(lines)


def _append_repetition_logbook(summary: pd.DataFrame) -> None:
    path = Path(".agents/logbooks/reduced-bias-domain-grp.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    heading = "### 2026-05-14 - Apple-style repetition-aware DSP check"
    if path.exists() and heading in path.read_text():
        return
    by_name = summary.set_index("variant")
    shared = by_name.loc["dsp_apple_repetition_shared_r1_phase_benefit_penalty_nnls"]
    per_domain = by_name.loc["dsp_apple_repetition_per_domain_r1_phase_benefit_penalty_nnls"]
    entry = "\n".join(
        [
            f"\n{heading}",
            "- Hypothesis: explicit Apple-style repetition discounting may improve DSP by separating physical repeated exposure from learned saturation.",
            f"- Command: `uv run --with matplotlib --with scipy --with scikit-learn --with tabulate python {Path(__file__).as_posix()}`",
            "- Config: 300M/6B 242-row fit frame; compared canonical, effective-exposure, split-phase, shared-r1 Apple DSP, and per-domain-r1 Apple DSP.",
            f"- Result: shared-r1 cv_rmse={float(shared['cv_rmse']):.6f}, oof_spearman={float(shared['oof_spearman']):.6f}; per-domain-r1 cv_rmse={float(per_domain['cv_rmse']):.6f}, oof_spearman={float(per_domain['oof_spearman']):.6f}.",
            f"- Artifacts: `{OUTPUT_DIR}`.",
            "- Interpretation: see `report.md`.",
            "",
        ]
    )
    if path.exists():
        path.write_text(path.read_text() + entry)
    else:
        path.write_text("# Reduced-Bias Domain GRP: Research Logbook\n" + entry)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    packet = _load_packet()
    summary_rows: list[dict[str, Any]] = [_old_grp_baseline_row()]
    observed_rows: list[dict[str, Any]] = []
    tuning_frames: list[pd.DataFrame] = []
    print(f"Loaded {len(packet.y)} rows and {packet.m} domains", flush=True)

    for variant in _selected_variants():
        print(f"Fitting {variant.name}", flush=True)
        variant_dir = OUTPUT_DIR / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        model, tune_frame = _fit_variant(packet, variant)
        raw_result, weights = _optimize_model(model, packet)
        metrics = _metrics(packet, model, raw_result, weights)
        summary_rows.append(metrics)
        observed_rows.append(_observed_leaderboard_row(packet, model))
        tune_frame.insert(0, "variant", variant.name)
        tuning_frames.append(tune_frame)
        _write_weight_table(packet, variant_dir, weights)
        _write_predictions(packet, model, variant_dir)
        _plot_predicted_vs_actual(packet, model, variant_dir)
        _plot_raw_optimum(packet, variant, weights, variant_dir)
        model_params = {
            key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in model.params.items()
        }
        (variant_dir / "model.json").write_text(
            json.dumps(
                {
                    "variant": variant.name,
                    "params": model_params,
                    "intercept": model.intercept,
                    "benefit_coef": model.benefit_coef.tolist(),
                    "penalty_coef": model.penalty_coef.tolist(),
                    "metrics": metrics,
                },
                indent=2,
            )
        )
        print(
            f"  cv_rmse={metrics['cv_rmse']:.6f} oof_spearman={metrics['oof_spearman']:.6f} "
            f"raw_tv={metrics['raw_nearest_observed_tv']:.3f}",
            flush=True,
        )

    summary = pd.DataFrame.from_records(summary_rows)
    observed = pd.DataFrame.from_records(observed_rows)
    tuning = pd.concat(tuning_frames, ignore_index=True)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    observed.to_csv(OUTPUT_DIR / "predicted_observed_leaderboard.csv", index=False)
    tuning.to_csv(OUTPUT_DIR / "tuning.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary.to_dict(orient="records"), indent=2))
    (OUTPUT_DIR / "report.md").write_text(_report(summary, observed))
    _append_logbook(summary)
    _append_repetition_logbook(summary)
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote artifacts to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
