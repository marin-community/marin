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
"""Materialize the supported centered-recency proposals and tied controls."""

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

DEFAULT_AUDIT_DIR = pooled.REFERENCE_OUTPUTS / "centered_recency_optima_20260710"
DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "centered_recency_validation_panel_20260710"
MODEL_NAME = centered.ResidualKind.TIED.value
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def candidate_key(objective: str, policy: str, kl_reg: float) -> str:
    kl_name = f"{kl_reg:g}".replace(".", "p")
    return f"centrec_{objective}_{policy}_kl{kl_name}"


def matrix_from_long(frame: pd.DataFrame, domains: list[str]) -> np.ndarray:
    matrix = (
        frame.pivot(index="phase", columns="domain", values="weight")
        .reindex(index=[0, 1], columns=domains)
        .to_numpy(dtype=float)
    )
    if matrix.shape != (2, len(domains)) or not np.all(np.isfinite(matrix)):
        raise ValueError("Audit weights are incomplete or non-finite")
    if np.min(matrix) < -1e-12 or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError(f"Invalid phase weights: minima={matrix.min()}, sums={matrix.sum(axis=1)}")
    return matrix


def materialize_candidate(
    *,
    output_dir: Path,
    objective: str,
    policy: str,
    kl_reg: float,
    weights: np.ndarray,
    domains: list[str],
    natural: np.ndarray,
    token_counts: np.ndarray,
    target_budget: int,
    diagnostics: pd.Series,
) -> dict[str, object]:
    key = candidate_key(objective, policy, kl_reg)
    frame = per_component.mixture_frame(
        domains=domains,
        natural=natural,
        weights=weights,
        token_counts=token_counts,
        target_budget=target_budget,
    )
    candidate_dir = output_dir / key
    candidate_dir.mkdir(parents=True, exist_ok=True)
    weights_path = candidate_dir / "proposed_mixture_weights.csv"
    frame.to_csv(weights_path, index=False)

    if policy == "two_phase":
        predicted_bpb = float(diagnostics["predicted_target"])
    else:
        predicted_bpb = float(diagnostics["predicted_target"] + diagnostics["predicted_ordering_margin_vs_tied"])
    return {
        "candidate": key,
        "objective": objective,
        "policy": policy,
        "pair_key": f"{objective}_kl{kl_reg:g}".replace(".", "p"),
        "model": MODEL_NAME,
        "residual_l2": float(diagnostics["residual_l2"]),
        "kl_reg": kl_reg,
        "predicted_bpb_300m": predicted_bpb,
        "predicted_ordering_gain": (
            float(diagnostics["predicted_ordering_margin_vs_tied"]) if policy == "two_phase" else 0.0
        ),
        "ordering_gain_diff_sd": float(diagnostics["ordering_margin_in_3e18_diff_sd"]) if policy == "two_phase" else 0.0,
        "tv_to_proportional": float(0.5 * np.abs(weights - natural[None, :]).sum(axis=1).mean()),
        "phase_tv": float(0.5 * np.abs(weights[0] - weights[1]).sum()),
        "max_weight": float(np.max(weights)),
        "max_simulated_epoch": float(frame["simulated_epochs"].max()),
        "nearest_observed_tv": float(diagnostics["nearest_observed_tv"]),
        "passes_all_primary_gates": bool(diagnostics["passes_all_primary_gates"]),
        "weights_csv": str(weights_path.relative_to(output_dir)),
    }


def write_plot(output_dir: Path, manifest: pd.DataFrame) -> None:
    rows = []
    for candidate in manifest["candidate"]:
        frame = pd.read_csv(output_dir / candidate / "proposed_mixture_weights.csv")
        for phase in (0, 1):
            for row in frame.itertuples(index=False):
                rows.append(
                    {
                        "candidate_phase": f"{candidate} / phase {phase}",
                        "domain": row.domain,
                        "epoch_multiplier": getattr(row, f"phase_{phase}_epoch_multiplier"),
                    }
                )
    plot_frame = pd.DataFrame(rows)
    matrix = plot_frame.pivot(index="candidate_phase", columns="domain", values="epoch_multiplier")
    figure = px.imshow(
        matrix,
        aspect="auto",
        color_continuous_scale="RdYlGn_r",
        labels={"color": "epoch multiplier"},
        title="Centered-recency proposals and exact aggregate-matched tied controls",
    )
    figure.update_layout(height=520, margin={"l": 260, "r": 40, "t": 80, "b": 180})
    figure.write_html(
        output_dir / "centered_recency_validation_panel.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kl-reg", type=float, default=10.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = pd.read_csv(args.audit_dir / "kl_path_diagnostics.csv")
    weight_rows = pd.read_csv(args.audit_dir / "kl_path_weights_long.csv")
    manifest_rows: list[dict[str, object]] = []
    for objective in ("uncheatable", "table9"):
        dataset_name = f"300m_{objective}"
        selected = diagnostics.loc[
            diagnostics["dataset"].eq(dataset_name)
            & diagnostics["model"].eq(MODEL_NAME)
            & np.isclose(diagnostics["kl_reg"], args.kl_reg)
        ]
        if len(selected) != 1:
            raise ValueError(f"Expected one diagnostic row for {dataset_name}, found {len(selected)}")
        diagnostic = selected.iloc[0]
        if not bool(diagnostic["passes_all_primary_gates"]):
            raise ValueError(f"{dataset_name} KL={args.kl_reg:g} did not pass all primary gates")

        _packet, domains, natural, token_counts, target_budget, _folds = bowl.load_objective(objective)
        selected_weights = weight_rows.loc[
            weight_rows["dataset"].eq(dataset_name)
            & weight_rows["model"].eq(MODEL_NAME)
            & np.isclose(weight_rows["kl_reg"], args.kl_reg)
        ]
        two_phase = matrix_from_long(selected_weights, list(domains))
        dataset = centered.load_datasets()[0][dataset_name]
        alpha0, alpha1 = centered.phase_fractions(dataset)
        aggregate = alpha0 * two_phase[0] + alpha1 * two_phase[1]
        tied = np.stack([aggregate, aggregate])
        for policy, weights in (("two_phase", two_phase), ("tied", tied)):
            manifest_rows.append(
                materialize_candidate(
                    output_dir=args.output_dir,
                    objective=objective,
                    policy=policy,
                    kl_reg=args.kl_reg,
                    weights=weights,
                    domains=list(domains),
                    natural=np.asarray(natural, dtype=float),
                    token_counts=np.asarray(token_counts, dtype=float),
                    target_budget=int(target_budget),
                    diagnostics=diagnostic,
                )
            )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(args.output_dir / "candidate_manifest.csv", index=False)
    write_plot(args.output_dir, manifest)
    report = [
        "# Centered-recency validation panel",
        "",
        "The free two-phase proposal and tied control have identical aggregate exposure. Their observed difference "
        "isolates phase ordering.",
        "",
        manifest.to_markdown(index=False),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(manifest.to_string(index=False))
    print(f"Wrote validation panel to {args.output_dir}")


if __name__ == "__main__":
    main()
