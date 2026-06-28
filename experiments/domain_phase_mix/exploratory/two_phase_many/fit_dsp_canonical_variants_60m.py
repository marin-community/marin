# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "scipy", "scikit-learn", "tabulate"]
# ///
"""Fit candidate DSP functional forms on the 60M fit swarm."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import fit_dsp_canonical_variants_300m as dsp
from experiments.domain_phase_mix.exploratory.two_phase_many.fit_and_plot_grp_power_family_penalty_no_l2_60m_vs_300m import (
    RUN_SET_60M,
    _build_fit_frame,
    _metric_frame,
    _packet_from_frame,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "dsp_canonical_variants_60m_20260510"
SCALE = "60m_1p2b"
DISPLAY_SCALE = "60M/1.2B"


def _load_packet() -> dsp.PacketData:
    fit_frame = _build_fit_frame(_metric_frame(), scale=SCALE, run_set=RUN_SET_60M)
    packet = _packet_from_frame(fit_frame, name="dsp_canonical_60m")
    return packet.base


def _report(summary: pd.DataFrame, observed: pd.DataFrame) -> str:
    keep = [
        "variant",
        "m_dependent_params_per_domain",
        "total_param_count",
        "retention_mode",
        "fitted_gamma",
        "fitted_gamma_benefit",
        "fitted_gamma_saturation",
        "fitted_gamma_penalty",
        "fitted_lam",
        "cv_rmse",
        "oof_spearman",
        "cv_foldmean_regret_at_1",
        "lower_tail_optimism",
        "raw_nearest_observed_tv",
        "raw_nearest_observed_value",
        "phase0_max_weight",
        "phase1_max_weight",
    ]
    table = summary[keep].copy()
    best_rmse = summary.loc[summary["cv_rmse"].astype(float).idxmin()]
    best_rank = summary.loc[summary["oof_spearman"].astype(float).idxmax()]
    lines = [
        f"# DSP Canonical Form Sweep on {DISPLAY_SCALE}",
        "",
        "## Setup",
        "",
        "Same DSP variants as the 300M sweep, fit on the clean 242-row 60M fit swarm.",
        "The sweep keeps M-dependent parameters to at most four per domain and uses only global scalars for phase effects.",
        "",
        "## Results",
        "",
        table.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Best Observed Rows By Prediction",
        "",
        observed.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation",
        "",
        f"- Best CV RMSE row: `{best_rmse['variant']}` with cv_rmse={float(best_rmse['cv_rmse']):.6f}.",
        f"- Best OOF Spearman row: `{best_rank['variant']}` with oof_spearman={float(best_rank['oof_spearman']):.6f}.",
        "- Use this as the cross-scale check for whether the split phase-mechanism form is stable enough to become canonical.",
        "",
    ]
    return "\n".join(lines)


def _append_logbook(summary: pd.DataFrame) -> None:
    path = Path(".agents/logbooks/reduced-bias-domain-grp.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    heading = "### 2026-05-10 - 60M DSP split phase-mechanism check"
    if path.exists() and heading in path.read_text():
        return
    best_rmse = summary.loc[summary["cv_rmse"].astype(float).idxmin()]
    best_rank = summary.loc[summary["oof_spearman"].astype(float).idxmax()]
    entry = "\n".join(
        [
            f"\n{heading}",
            "- Hypothesis: split phase-mechanism DSP should be checked at 60M before promoting the 300M winner.",
            f"- Command: `uv run --no-project --with matplotlib --with scipy --with scikit-learn --with tabulate --with pandas --with numpy python {Path(__file__).as_posix()}`",
            "- Config: 60M/1.2B 242-row fit frame, same DSP variants as the 300M split sweep.",
            f"- Result: best CV RMSE `{best_rmse['variant']}` cv_rmse={float(best_rmse['cv_rmse']):.6f}; best rank `{best_rank['variant']}` oof_spearman={float(best_rank['oof_spearman']):.6f}.",
            f"- Artifacts: `{OUTPUT_DIR}`.",
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
    summary_rows: list[dict[str, Any]] = []
    observed_rows: list[dict[str, Any]] = []
    tune_frames: list[pd.DataFrame] = []
    print(f"Loaded {len(packet.y)} rows and {packet.m} domains", flush=True)

    for variant in dsp.VARIANTS:
        print(f"Fitting {variant.name}", flush=True)
        variant_dir = OUTPUT_DIR / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        model, tune_frame = dsp._fit_variant(packet, variant)
        raw_result, weights = dsp._optimize_model(model, packet)
        metrics = dsp._metrics(packet, model, raw_result, weights)
        summary_rows.append(metrics)
        observed_rows.append(dsp._observed_leaderboard_row(packet, model))
        tune_frame.insert(0, "variant", variant.name)
        tune_frames.append(tune_frame)
        dsp._write_weight_table(packet, variant_dir, weights)
        dsp._plot_predicted_vs_actual(packet, model, variant_dir)
        dsp._plot_raw_optimum(packet, variant, weights, variant_dir)
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
    tune = pd.concat(tune_frames, ignore_index=True)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    observed.to_csv(OUTPUT_DIR / "predicted_observed_leaderboard.csv", index=False)
    tune.to_csv(OUTPUT_DIR / "tuning.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary.to_dict(orient="records"), indent=2))
    (OUTPUT_DIR / "report.md").write_text(_report(summary, observed))
    _append_logbook(summary)
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote artifacts to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
