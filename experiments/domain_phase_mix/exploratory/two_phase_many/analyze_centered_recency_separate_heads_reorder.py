# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Reorder the validated separate-heads frontier at fixed aggregate exposure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_centered_recency_anchor_ordering as anchor_order,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_centered_recency_optima as optimum_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_centered_recency_residual as centered,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "centered_recency_sepheads_reorder_20260710"
DEFAULT_BENCHMARK_DIR = pooled.REFERENCE_OUTPUTS / "centered_recency_residual_20260710"
SEPARATE_HEAD_DIR = pooled.REFERENCE_OUTPUTS / "sep_lf_kl_sweep_panel_20260706"
NOISE_SD = optimum_audit.NOISE_SD
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def separate_heads_path(objective: str) -> Path:
    short = "unch" if objective == "uncheatable" else "t9"
    return SEPARATE_HEAD_DIR / f"seplf_{short}_sep_kl0p1" / "proposed_mixture_weights.csv"


def weights_from_frame(path: Path, domains: list[str]) -> np.ndarray:
    frame = pd.read_csv(path).set_index("domain").reindex(domains)
    weights = frame[["phase_0_weight", "phase_1_weight"]].to_numpy(dtype=float).T
    if not np.all(np.isfinite(weights)) or not np.allclose(weights.sum(axis=1), 1.0):
        raise ValueError(f"Invalid separate-head weights in {path}")
    return weights


def analyze_objective(
    dataset: pooled.Dataset,
    objective: str,
    benchmark_dir: Path,
    order_kl_values: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    l2 = optimum_audit.selected_l2(benchmark_dir, dataset.name, centered.ResidualKind.TIED)
    nonnegative = centered.fit_full_candidate(dataset, centered.ResidualKind.TIED, l2)
    signed = anchor_order.fit_signed_residual(nonnegative, dataset, l2)
    incumbent = weights_from_frame(separate_heads_path(objective), dataset.domain_names)
    aggregate = nonnegative.alpha0 * incumbent[0] + nonnegative.alpha1 * incumbent[1]
    tied = np.stack([aggregate, aggregate])
    incumbent_prediction = float(nonnegative.predict(incumbent[None, :, :])[0])
    incumbent_signed_prediction = float(signed.predict(incumbent[None, :, :])[0])
    tied_prediction = float(nonnegative.predict(tied[None, :, :])[0])
    pair_noise_sd = np.sqrt(2.0) * NOISE_SD[objective]
    rows = []
    weight_rows = []
    for order_kl_reg in order_kl_values:
        print(f"Reordering separate-heads {objective}, order KL={order_kl_reg:g}", flush=True)
        candidate = anchor_order.optimize_order(
            nonnegative,
            aggregate,
            order_kl_reg,
            penalty_reference=incumbent,
        )
        candidate_prediction = float(nonnegative.predict(candidate[None, :, :])[0])
        candidate_signed_prediction = float(signed.predict(candidate[None, :, :])[0])
        gain = incumbent_prediction - candidate_prediction
        signed_gain = incumbent_signed_prediction - candidate_signed_prediction
        nearest_tv, nearest_observed = anchor_order.nearest_observed_tv(dataset, candidate)
        total_epochs = candidate[0] * dataset.c0 + candidate[1] * dataset.c1
        distance_to_incumbent = float(0.5 * np.abs(candidate - incumbent).sum(axis=1).mean())
        rows.append(
            {
                "dataset": dataset.name,
                "objective": objective,
                "residual_l2": l2,
                "order_kl_reg": order_kl_reg,
                "incumbent_prediction": incumbent_prediction,
                "candidate_prediction": candidate_prediction,
                "signed_incumbent_prediction": incumbent_signed_prediction,
                "signed_candidate_prediction": candidate_signed_prediction,
                "gain_vs_incumbent": gain,
                "signed_gain_vs_incumbent": signed_gain,
                "gain_vs_incumbent_diff_sd": gain / pair_noise_sd,
                "signed_gain_vs_incumbent_diff_sd": signed_gain / pair_noise_sd,
                "gain_vs_tied": tied_prediction - candidate_prediction,
                "phase_order_kl": anchor_order.phase_order_kl(
                    candidate,
                    aggregate,
                    nonnegative.alpha0,
                    nonnegative.alpha1,
                ),
                "kl_to_incumbent": anchor_order.phase_schedule_kl(
                    candidate,
                    incumbent,
                    nonnegative.alpha0,
                    nonnegative.alpha1,
                ),
                "candidate_phase_tv": float(0.5 * np.abs(candidate[0] - candidate[1]).sum()),
                "incumbent_phase_tv": float(0.5 * np.abs(incumbent[0] - incumbent[1]).sum()),
                "tv_to_incumbent": distance_to_incumbent,
                "nearest_observed_tv": nearest_tv,
                "nearest_observed": nearest_observed,
                "max_total_epoch": float(np.max(total_epochs)),
                "signed_coef_signal": float(signed.residual.coef[0]),
                "signed_coef_penalty": float(signed.residual.coef[1]),
                "passes_support_gate": nearest_tv <= 0.05,
                "passes_improvement_power_gate": gain >= 2.0 * pair_noise_sd,
                "passes_signed_direction_gate": signed_gain > 0.0,
                "passes_signed_power_gate": signed_gain >= pair_noise_sd,
            }
        )
        for phase in range(2):
            for domain, weight in zip(dataset.domain_names, candidate[phase], strict=True):
                weight_rows.append(
                    {
                        "dataset": dataset.name,
                        "objective": objective,
                        "order_kl_reg": order_kl_reg,
                        "phase": phase,
                        "domain": domain,
                        "weight": float(weight),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(weight_rows)


def write_plots(diagnostics: pd.DataFrame, output_dir: Path) -> None:
    for metric in (
        "gain_vs_incumbent",
        "signed_gain_vs_incumbent",
        "candidate_phase_tv",
        "tv_to_incumbent",
        "nearest_observed_tv",
    ):
        figure = px.line(
            diagnostics,
            x="order_kl_reg",
            y=metric,
            color="objective",
            markers=True,
            log_x=True,
            color_discrete_sequence=px.colors.diverging.RdYlGn_r,
            title=f"Fixed-aggregate reordering of separate-heads frontier: {metric}",
        )
        figure.write_html(
            output_dir / f"sepheads_reorder_{metric}.html",
            include_plotlyjs="cdn",
            config=PLOT_CONFIG,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument(
        "--order-kl-values",
        default="0.1,0.3,1,3,10,30,100,300,1000",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    order_kl_values = pooled.parse_float_list(args.order_kl_values)
    datasets, _external = centered.load_datasets()
    result_frames = []
    weight_frames = []
    for objective in ("uncheatable", "table9"):
        result, weights = analyze_objective(
            datasets[f"300m_{objective}"],
            objective,
            args.benchmark_dir,
            order_kl_values,
        )
        result_frames.append(result)
        weight_frames.append(weights)
    diagnostics = pd.concat(result_frames, ignore_index=True)
    weights = pd.concat(weight_frames, ignore_index=True)
    diagnostics.to_csv(args.output_dir / "sepheads_reorder_diagnostics.csv", index=False)
    weights.to_csv(args.output_dir / "sepheads_reorder_weights_long.csv", index=False)
    write_plots(diagnostics, args.output_dir)
    report = [
        "# Fixed-aggregate reordering of the validated separate-heads frontier",
        "",
        diagnostics.to_markdown(index=False),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(diagnostics.to_string(index=False))
    print(f"Wrote separate-heads reorder audit to {args.output_dir}")


if __name__ == "__main__":
    main()
