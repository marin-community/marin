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
"""Materialize centered-recency reorderings of the separate-heads frontier."""

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
    analyze_centered_recency_separate_heads_reorder as reorder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_centered_recency_residual as centered,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_per_component_dsp_kl_sweep_300m as per_component,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_two_phase_canonical_bowl_candidates_300m as bowl,
)

DEFAULT_AUDIT_DIR = pooled.REFERENCE_OUTPUTS / "centered_recency_sepheads_reorder_20260710"
DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "centered_recency_sepheads_reorder_panel_20260710"
DEFAULT_ORDER_KL = "1,3"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def candidate_key(objective: str, order_kl_reg: float) -> str:
    kl_name = f"{order_kl_reg:g}".replace(".", "p")
    return f"centrec_sep_{objective}_okl{kl_name}"


def candidate_weights(
    frame: pd.DataFrame,
    dataset: str,
    order_kl_reg: float,
    domains: list[str],
) -> np.ndarray:
    selected = frame.loc[frame["dataset"].eq(dataset) & np.isclose(frame["order_kl_reg"], order_kl_reg)]
    weights = (
        selected.pivot(index="phase", columns="domain", values="weight")
        .reindex(index=[0, 1], columns=domains)
        .to_numpy(dtype=float)
    )
    if not np.all(np.isfinite(weights)) or not np.allclose(weights.sum(axis=1), 1.0):
        raise ValueError(f"Invalid weights for {dataset} order KL={order_kl_reg:g}")
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--order-kl-values", default=DEFAULT_ORDER_KL)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    order_kl_values = pooled.parse_float_list(args.order_kl_values)
    diagnostics = pd.read_csv(args.audit_dir / "sepheads_reorder_diagnostics.csv")
    weight_rows = pd.read_csv(args.audit_dir / "sepheads_reorder_weights_long.csv")
    datasets, _external = centered.load_datasets()

    manifest_rows = []
    plot_rows = []
    for objective in ("uncheatable", "table9"):
        dataset_name = f"300m_{objective}"
        _packet, domains, natural, token_counts, target_budget, _folds = bowl.load_objective(objective)
        alpha0, alpha1 = centered.phase_fractions(datasets[dataset_name])
        incumbent = reorder.weights_from_frame(reorder.separate_heads_path(objective), list(domains))
        incumbent_aggregate = alpha0 * incumbent[0] + alpha1 * incumbent[1]
        for order_kl_reg in order_kl_values:
            selected = diagnostics.loc[
                diagnostics["dataset"].eq(dataset_name) & np.isclose(diagnostics["order_kl_reg"], order_kl_reg)
            ]
            if len(selected) != 1:
                raise ValueError(f"Expected one diagnostic row for {dataset_name}/{order_kl_reg:g}")
            diagnostic = selected.iloc[0]
            if not (
                bool(diagnostic["passes_improvement_power_gate"]) and bool(diagnostic["passes_signed_direction_gate"])
            ):
                raise ValueError(f"{dataset_name} order KL={order_kl_reg:g} failed improvement gates")
            weights = candidate_weights(weight_rows, dataset_name, order_kl_reg, list(domains))
            aggregate = alpha0 * weights[0] + alpha1 * weights[1]
            if not np.allclose(aggregate, incumbent_aggregate, atol=1e-9):
                raise ValueError("Reordered candidate changed incumbent aggregate exposure")
            key = candidate_key(objective, order_kl_reg)
            candidate_dir = args.output_dir / key
            candidate_dir.mkdir(parents=True, exist_ok=True)
            frame = per_component.mixture_frame(
                domains=list(domains),
                natural=np.asarray(natural, dtype=float),
                weights=weights,
                token_counts=np.asarray(token_counts, dtype=float),
                target_budget=int(target_budget),
            )
            weights_path = candidate_dir / "proposed_mixture_weights.csv"
            frame.to_csv(weights_path, index=False)
            manifest_rows.append(
                {
                    "candidate": key,
                    "objective": objective,
                    "model": "centered_tied_late",
                    "order_kl_reg": order_kl_reg,
                    "gain_vs_incumbent_300m": float(diagnostic["gain_vs_incumbent"]),
                    "signed_gain_vs_incumbent_300m": float(diagnostic["signed_gain_vs_incumbent"]),
                    "gain_diff_sd": float(diagnostic["gain_vs_incumbent_diff_sd"]),
                    "tv_to_incumbent": float(diagnostic["tv_to_incumbent"]),
                    "candidate_phase_tv": float(diagnostic["candidate_phase_tv"]),
                    "incumbent_phase_tv": float(diagnostic["incumbent_phase_tv"]),
                    "max_simulated_epoch_300m": float(frame["simulated_epochs"].max()),
                    "incumbent_source": str(reorder.separate_heads_path(objective).relative_to(REPO_ROOT)),
                    "weights_csv": str(weights_path.relative_to(args.output_dir)),
                }
            )
            for phase in (0, 1):
                for domain, candidate_weight, incumbent_weight in zip(
                    domains,
                    weights[phase],
                    incumbent[phase],
                    strict=True,
                ):
                    plot_rows.append(
                        {
                            "candidate": key,
                            "phase": phase,
                            "domain": domain,
                            "candidate_weight": float(candidate_weight),
                            "incumbent_weight": float(incumbent_weight),
                            "delta": float(candidate_weight - incumbent_weight),
                        }
                    )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(args.output_dir / "candidate_manifest.csv", index=False)
    plot_frame = pd.DataFrame(plot_rows)
    plot_frame.to_csv(args.output_dir / "candidate_weight_deltas.csv", index=False)
    figure = px.bar(
        plot_frame,
        x="domain",
        y="delta",
        color="candidate",
        facet_row="phase",
        barmode="group",
        color_discrete_sequence=px.colors.diverging.RdYlGn_r,
        title="Centered-recency phase edits relative to validated separate-heads schedules",
    )
    figure.update_layout(height=850, margin={"l": 60, "r": 40, "t": 90, "b": 220})
    figure.write_html(
        args.output_dir / "candidate_weight_deltas.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )
    report = [
        "# Centered-recency reorder validation panel",
        "",
        "Every candidate preserves its validated separate-heads incumbent's aggregate exposure exactly and changes "
        "only phase order.",
        "",
        manifest.to_markdown(index=False),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(manifest.to_string(index=False))
    print(f"Wrote centered-recency reorder panel to {args.output_dir}")


if __name__ == "__main__":
    main()
