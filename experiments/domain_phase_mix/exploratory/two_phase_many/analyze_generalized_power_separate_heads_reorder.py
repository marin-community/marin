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
"""Reorder validated separate-heads aggregates with generalized recency DSP."""

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
    analyze_centered_recency_optima as centered_optimum,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_centered_recency_separate_heads_reorder as reorder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_calibrated_cumulative_recency as generalized,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_centered_recency_residual as centered,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "generalized_power_sepheads_reorder_20260710"
POWER_L2 = 0.001
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def analyze_objective(
    dataset: pooled.Dataset,
    objective: str,
    order_kl_values: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Optimize phase order while preserving a separate-heads aggregate."""
    config = generalized.ModelConfig(generalized.CalibrationKind.SHARED_POWER, POWER_L2)
    model = generalized.fit_model(
        dataset,
        np.arange(dataset.n),
        config,
        generalized.HEAD_L2_BY_DATASET[dataset.name],
        maxiter=100,
        coarse_top_k=3,
    )
    incumbent = reorder.weights_from_frame(reorder.separate_heads_path(objective), dataset.domain_names)
    aggregate = model.alpha0 * incumbent[0] + model.alpha1 * incumbent[1]
    tied = np.stack([aggregate, aggregate])
    incumbent_prediction = float(model.predict(incumbent[None, :, :])[0])
    tied_prediction = float(model.predict(tied[None, :, :])[0])
    pair_noise_sd = np.sqrt(2.0) * centered_optimum.NOISE_SD[objective]
    rows: list[dict[str, float | str | bool]] = []
    weight_rows: list[dict[str, float | str | int]] = []
    for order_kl_reg in order_kl_values:
        print(f"Reordering {objective} separate-heads aggregate, order KL={order_kl_reg:g}", flush=True)
        candidate = anchor_order.optimize_order(
            model,
            aggregate,
            order_kl_reg,
            penalty_reference=incumbent,
        )
        prediction = float(model.predict(candidate[None, :, :])[0])
        gain = incumbent_prediction - prediction
        max_total_epoch = float(np.max(candidate[0] * dataset.c0 + candidate[1] * dataset.c1))
        rows.append(
            {
                "dataset": dataset.name,
                "model": config.name,
                "order_kl_reg": order_kl_reg,
                "incumbent_prediction": incumbent_prediction,
                "tied_prediction": tied_prediction,
                "candidate_prediction": prediction,
                "gain_vs_incumbent": gain,
                "gain_vs_incumbent_diff_sd": gain / pair_noise_sd,
                "incumbent_ordering_gain_vs_tied": tied_prediction - incumbent_prediction,
                "candidate_ordering_gain_vs_tied": tied_prediction - prediction,
                "tv_to_incumbent": 0.5 * np.abs(candidate - incumbent).sum(axis=1).mean(),
                "candidate_phase_tv": float(0.5 * np.abs(candidate[0] - candidate[1]).sum()),
                "incumbent_phase_tv": float(0.5 * np.abs(incumbent[0] - incumbent[1]).sum()),
                "phase_schedule_kl": anchor_order.phase_schedule_kl(
                    candidate,
                    incumbent,
                    model.alpha0,
                    model.alpha1,
                ),
                "max_total_epoch": max_total_epoch,
                "passes_gain_gate": gain >= 2.0 * pair_noise_sd,
                "passes_epoch_gate": max_total_epoch <= 10.0,
            }
        )
        for phase in range(2):
            for domain, weight in zip(dataset.domain_names, candidate[phase], strict=True):
                weight_rows.append(
                    {
                        "dataset": dataset.name,
                        "model": config.name,
                        "order_kl_reg": order_kl_reg,
                        "phase": phase,
                        "domain": domain,
                        "weight": float(weight),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(weight_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--order-kl-values", default="0,0.1,0.3,1,3,10")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    order_kl_values = pooled.parse_float_list(args.order_kl_values)
    datasets, _external = centered.load_datasets()
    frames = []
    weight_frames = []
    for objective in ("uncheatable", "table9"):
        frame, weights = analyze_objective(
            datasets[f"300m_{objective}"],
            objective,
            order_kl_values,
        )
        frames.append(frame)
        weight_frames.append(weights)
    diagnostics = pd.concat(frames, ignore_index=True)
    weights = pd.concat(weight_frames, ignore_index=True)
    diagnostics.to_csv(args.output_dir / "reorder_diagnostics.csv", index=False)
    weights.to_csv(args.output_dir / "reorder_weights_long.csv", index=False)
    for metric in ("gain_vs_incumbent", "tv_to_incumbent", "max_total_epoch"):
        figure = px.line(
            diagnostics,
            x="order_kl_reg",
            y=metric,
            color="dataset",
            markers=True,
            color_discrete_sequence=px.colors.diverging.RdYlGn_r,
            title=f"Generalized recency reorder: {metric}",
        )
        figure.write_html(
            args.output_dir / f"reorder_{metric}.html",
            include_plotlyjs="cdn",
            config=PLOT_CONFIG,
        )
    report = [
        "# Generalized cumulative-recency reorder audit",
        "",
        "Every candidate preserves the validated separate-heads aggregate exposure exactly.",
        "",
        diagnostics.to_markdown(index=False),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(diagnostics.to_string(index=False))
    print(f"Wrote generalized recency reorder audit to {args.output_dir}")


if __name__ == "__main__":
    main()
