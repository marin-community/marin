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
"""Screen a centered late-phase residual on effective-exposure DSP.

The current cumulative-plus-recency model improves StarCoder and the 300M
objectives but replaces the learned per-bucket DSP timescales with rigid
exposure-derived scales, which fails on the 168-bucket production swarm. This
model instead keeps effective-exposure DSP plus its validated geometry terms
as the backbone and adds a shrinkable order-only residual.

For cumulative exposure ``E_i = e0_i + e1_i``, the phase-1 exposure of the
aggregate-matched tied schedule is

    tied_e1_i = E_i * c1_i / (c0_i + c1_i).

The residual features are differences between actual and tied late response:

    delta_S_i = S_i(e1_i) - S_i(tied_e1_i)
    delta_H_i = H_i(e1_i) - H_i(tied_e1_i).

They vanish exactly whenever both phases use the same mixture. Two heads are
screened: a two-scalar head tied to the backbone's bucket values, and a
per-bucket head with nonnegative coefficients. Ridge shrinkage nests the
backbone at zero correction.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_table9_phase_split_dsp_300m as phase_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_joint_phase_correspondence_dsp as joint,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_nested_coverage_dsp as coverage,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import (  # noqa: E402
    dsp_exact as dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.plot_separate_heads_starcoder_u_shape_fit import (  # noqa: E402, E501
    fit_separate_heads,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.starcoder_grp import (  # noqa: E402
    load_completed_two_phase_starcoder_packet,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "centered_recency_residual_20260710"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
STARCODER_NAME = "starcoder_code_bpb"
BASELINE_CONFIG = coverage.FitConfig(
    "effective_exposure_geometry",
    True,
    "effective_exposure",
    (0, 1),
)


class ResidualKind(StrEnum):
    """Order-only residual parameterizations."""

    TIED = "centered_tied_late"
    DOMAIN = "centered_domain_late"


@dataclass(frozen=True)
class ResidualHead:
    """Nonnegative correction to a fixed DSP backbone."""

    kind: ResidualKind
    coef: np.ndarray
    l2: float


@dataclass(frozen=True)
class FittedCandidate:
    """DSP backbone plus a centered order-only residual."""

    backbone: coverage.CoverageModel
    residual: ResidualHead | None
    c0: np.ndarray
    c1: np.ndarray
    alpha0: float
    alpha1: float

    def predict(self, weights: np.ndarray) -> np.ndarray:
        prediction = coverage.predict(self.backbone, weights, self.alpha0, self.alpha1)
        if self.residual is None:
            return np.asarray(prediction, dtype=float)
        design = residual_design(
            self.backbone.base,
            weights,
            self.c0,
            self.c1,
            self.residual.kind,
        )
        return np.asarray(prediction + design @ self.residual.coef, dtype=float)


def starcoder_dataset() -> pooled.Dataset:
    packet = load_completed_two_phase_starcoder_packet()
    return pooled.Dataset(
        name=STARCODER_NAME,
        frame=packet.frame.copy(),
        y=np.asarray(packet.y, dtype=float),
        weights=np.asarray(packet.w, dtype=float),
        c0=np.asarray(packet.c0, dtype=float),
        c1=np.asarray(packet.c1, dtype=float),
        domain_names=list(packet.domain_names),
    )


def packet_from_dataset(dataset: pooled.Dataset, indices: np.ndarray) -> dsp.PacketData:
    name_col = "run_id" if "run_id" in dataset.frame.columns else "run_name"
    if name_col not in dataset.frame.columns:
        name_col = dataset.frame.columns[0]
    return dsp.PacketData(
        frame=dataset.frame.iloc[indices].reset_index(drop=True),
        name_col=name_col,
        y=dataset.y[indices],
        w=dataset.weights[indices],
        m=dataset.m,
        c0=dataset.c0,
        c1=dataset.c1,
        domain_names=list(dataset.domain_names),
    )


def phase_fractions(dataset: pooled.Dataset) -> tuple[float, float]:
    ratio = float(np.median(dataset.c0 / dataset.c1))
    alpha0 = ratio / (1.0 + ratio)
    return alpha0, 1.0 - alpha0


def fit_backbone(dataset: pooled.Dataset, indices: np.ndarray) -> coverage.CoverageModel:
    """Fit the established effective-exposure backbone inside one fold."""
    if dataset.name == STARCODER_NAME:
        model, _tuning = phase_dsp.fit_variant_with_l2(
            packet_from_dataset(dataset, indices),
            "effective_exposure",
            0.01,
            maxiter=40,
            coarse_top_k=3,
            basin_hopping_iters=0,
        )
        return coverage.CoverageModel(base=model, coverage_coef=np.asarray([], dtype=float))
    return coverage.fit_model(
        dataset,
        indices,
        BASELINE_CONFIG,
        linear_reg=coverage.dataset_linear_reg(dataset),
        maxiter=0 if dataset.name == "production_uncheatable" else 8,
        coarse_top_k=1,
    )


def late_response(
    model: dsp.FittedDSPModel,
    exposure: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the backbone's per-bucket response shape on late exposure."""
    rho = np.asarray(model.params["rho"], dtype=float)
    tau = np.asarray(model.params["tau"], dtype=float)
    signal = 1.0 - np.exp(-rho[None, :] * exposure)
    penalty = dsp.softplus(np.log1p(exposure) - tau[None, :]) ** 2
    return signal, penalty


def residual_design(
    model: dsp.FittedDSPModel,
    weights: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    kind: ResidualKind,
) -> np.ndarray:
    """Build late-response differences relative to the exact tied schedule."""
    e0 = weights[:, 0, :] * c0[None, :]
    e1 = weights[:, 1, :] * c1[None, :]
    total = e0 + e1
    tied_e1 = total * (c1 / (c0 + c1))[None, :]
    signal, penalty = late_response(model, e1)
    tied_signal, tied_penalty = late_response(model, tied_e1)
    delta_signal = signal - tied_signal
    delta_penalty = penalty - tied_penalty
    if kind is ResidualKind.DOMAIN:
        return np.hstack([-delta_signal, delta_penalty])
    return np.column_stack(
        [
            -(delta_signal @ model.benefit_coef),
            delta_penalty @ model.penalty_coef,
        ]
    )


def fit_residual(
    backbone: coverage.CoverageModel,
    dataset: pooled.Dataset,
    indices: np.ndarray,
    kind: ResidualKind,
    l2: float,
    alpha0: float,
    alpha1: float,
) -> ResidualHead:
    """Fit a zero-at-tied nonnegative ridge correction without an intercept."""
    design = residual_design(
        backbone.base,
        dataset.weights[indices],
        dataset.c0,
        dataset.c1,
        kind,
    )
    base_prediction = coverage.predict(backbone, dataset.weights[indices], alpha0, alpha1)
    target = dataset.y[indices] - base_prediction
    if l2 > 0.0:
        design = np.vstack([design, np.sqrt(l2) * np.eye(design.shape[1])])
        target = np.concatenate([target, np.zeros(design.shape[1], dtype=float)])
    coef, _residual = nnls(design, target, maxiter=20 * design.shape[1])
    return ResidualHead(kind=kind, coef=np.asarray(coef, dtype=float), l2=l2)


def folds_for(dataset: pooled.Dataset, seed: int, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if "phase_correspondence_key" in dataset.frame.columns:
        return joint.grouped_folds(dataset.frame, seed, n_splits)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(train, test) for train, test in splitter.split(np.arange(dataset.n))]


def model_label(kind: ResidualKind | None, l2: float | None) -> str:
    if kind is None:
        return "effective_exposure_backbone"
    return f"{kind.value}_l2_{l2:g}"


def benchmark_seed(
    dataset: pooled.Dataset,
    seed: int,
    n_splits: int,
    l2_values: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all residual configurations using one outer-fold partition."""
    folds = folds_for(dataset, seed, n_splits)
    labels = [model_label(None, None)] + [model_label(kind, l2) for kind in ResidualKind for l2 in l2_values]
    predictions = {label: np.zeros(dataset.n, dtype=float) for label in labels}
    parameter_rows: list[dict[str, Any]] = []
    for fold_id, (train_indices, test_indices) in enumerate(folds):
        print(
            f"{dataset.name}: seed={seed} fold={fold_id + 1}/{n_splits}",
            flush=True,
        )
        alpha0, alpha1 = phase_fractions(dataset)
        backbone = fit_backbone(dataset, train_indices)
        predictions[model_label(None, None)][test_indices] = coverage.predict(
            backbone, dataset.weights[test_indices], alpha0, alpha1
        )
        for kind in ResidualKind:
            for l2 in l2_values:
                residual = fit_residual(
                    backbone,
                    dataset,
                    train_indices,
                    kind,
                    l2,
                    alpha0,
                    alpha1,
                )
                candidate = FittedCandidate(
                    backbone=backbone,
                    residual=residual,
                    c0=dataset.c0,
                    c1=dataset.c1,
                    alpha0=alpha0,
                    alpha1=alpha1,
                )
                label = model_label(kind, l2)
                predictions[label][test_indices] = candidate.predict(dataset.weights[test_indices])
                parameter_rows.append(
                    {
                        "dataset": dataset.name,
                        "seed": seed,
                        "fold": fold_id,
                        "model": label,
                        "l2": l2,
                        "coef_norm": float(np.linalg.norm(residual.coef)),
                        "coef_max": float(np.max(residual.coef)),
                        "nonzero_coef": int(np.sum(residual.coef > 1e-12)),
                    }
                )
    metric_rows: list[dict[str, Any]] = []
    for label, prediction in predictions.items():
        row = asdict(pooled.metrics(dataset, label, seed, prediction, folds))
        if label == model_label(None, None):
            row["nominal_param_count"] = 4 * dataset.m + 4
        elif label.startswith(ResidualKind.TIED.value):
            row["nominal_param_count"] = 4 * dataset.m + 6
        else:
            row["nominal_param_count"] = 6 * dataset.m + 4
        metric_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(parameter_rows)


def load_datasets() -> tuple[dict[str, pooled.Dataset], dict[str, pooled.Dataset]]:
    frame = pd.read_csv(joint.PACKET)
    domains = pooled.load_300m_dataset("table9").domain_names
    frame = joint.attach_single_phase_weights(frame, joint.ONE_PHASE_SOURCE, domains)
    datasets: dict[str, pooled.Dataset] = {STARCODER_NAME: starcoder_dataset()}
    external: dict[str, pooled.Dataset] = {}
    for objective, target in joint.TARGET_COLUMNS.items():
        name = f"300m_{objective}"
        datasets[name] = joint.dataset_from_frame(
            objective,
            frame.loc[frame["split"].eq("train") | frame["policy_family"].eq("single_phase")].copy(),
            target,
        )
        external[name] = joint.dataset_from_frame(
            objective,
            frame.loc[frame["split"].eq("heldout") & frame["policy_family"].eq("two_phase")].copy(),
            target,
        )
    production = pooled.load_production_dataset()
    datasets[production.name] = production
    return datasets, external


def selected_configs(summary: pd.DataFrame) -> pd.DataFrame:
    candidates = summary.loc[summary["model"].str.startswith(tuple(kind.value for kind in ResidualKind))]
    selected_indices = candidates.groupby(
        ["dataset", candidates["model"].str.extract(r"^(centered_[a-z_]+)_l2_")[0]],
        sort=False,
    )["oof_rmse_mean"].idxmin()
    return candidates.loc[selected_indices].sort_values(["dataset", "model"])


def fit_full_candidate(
    dataset: pooled.Dataset,
    kind: ResidualKind,
    l2: float,
) -> FittedCandidate:
    indices = np.arange(dataset.n)
    alpha0, alpha1 = phase_fractions(dataset)
    backbone = fit_backbone(dataset, indices)
    residual = fit_residual(backbone, dataset, indices, kind, l2, alpha0, alpha1)
    return FittedCandidate(
        backbone=backbone,
        residual=residual,
        c0=dataset.c0,
        c1=dataset.c1,
        alpha0=alpha0,
        alpha1=alpha1,
    )


def external_metrics(
    datasets: dict[str, pooled.Dataset],
    external: dict[str, pooled.Dataset],
    selected: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_name, external_dataset in external.items():
        dataset = datasets[dataset_name]
        backbone = fit_full_candidate(dataset, ResidualKind.TIED, 1.0).backbone
        base_prediction = coverage.predict(
            backbone,
            external_dataset.weights,
            *phase_fractions(dataset),
        )
        base_row = joint.external_metrics(model_label(None, None), external_dataset.y, base_prediction)
        base_row["dataset"] = dataset_name
        rows.append(base_row)
        for selected_row in selected.loc[selected["dataset"].eq(dataset_name)].itertuples():
            kind = (
                ResidualKind.TIED if str(selected_row.model).startswith(ResidualKind.TIED.value) else ResidualKind.DOMAIN
            )
            l2 = float(str(selected_row.model).rsplit("_", maxsplit=1)[-1])
            candidate = fit_full_candidate(dataset, kind, l2)
            row = joint.external_metrics(
                str(selected_row.model),
                external_dataset.y,
                candidate.predict(external_dataset.weights),
            )
            row["dataset"] = dataset_name
            rows.append(row)
    return pd.DataFrame(rows)


def starcoder_slice_metrics(
    dataset: pooled.Dataset,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    packet = load_completed_two_phase_starcoder_packet()
    mask = packet.frame["phase_0_starcoder"].lt(1e-10).to_numpy(dtype=bool)
    indices = np.flatnonzero(mask)
    order = np.argsort(packet.frame.iloc[indices]["phase_1_starcoder"].to_numpy())
    indices = indices[order]
    weights = dataset.weights[indices]
    targets = dataset.y[indices]
    rows: list[dict[str, Any]] = []
    separate = fit_separate_heads(packet)
    predictions: dict[str, np.ndarray] = {
        "separate_heads": separate.predict(weights),
    }
    backbone = fit_full_candidate(dataset, ResidualKind.TIED, 1.0).backbone
    predictions[model_label(None, None)] = coverage.predict(backbone, weights, *phase_fractions(dataset))
    for selected_row in selected.loc[selected["dataset"].eq(STARCODER_NAME)].itertuples():
        kind = ResidualKind.TIED if str(selected_row.model).startswith(ResidualKind.TIED.value) else ResidualKind.DOMAIN
        l2 = float(str(selected_row.model).rsplit("_", maxsplit=1)[-1])
        candidate = fit_full_candidate(dataset, kind, l2)
        predictions[str(selected_row.model)] = candidate.predict(weights)
    phase1 = packet.frame.iloc[indices]["phase_1_starcoder"].to_numpy(dtype=float)
    for label, prediction in predictions.items():
        minimum = int(np.argmin(prediction))
        rows.append(
            {
                "model": label,
                "slice_rows": len(indices),
                "slice_rmse": float(np.sqrt(np.mean((prediction - targets) ** 2))),
                "slice_spearman": float(spearmanr(targets, prediction).statistic),
                "predicted_min_phase1_starcoder_weight": float(phase1[minimum]),
                "predicted_min_bpb": float(prediction[minimum]),
            }
        )
    return pd.DataFrame(rows)


def checkpoint_paths(output_dir: Path, dataset: str, seed: int) -> tuple[Path, Path]:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return (
        checkpoint_dir / f"{dataset}__seed_{seed}__metrics.csv",
        checkpoint_dir / f"{dataset}__seed_{seed}__parameters.csv",
    )


def write_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    selected = selected_configs(summary)
    baseline = summary.loc[summary["model"].eq(model_label(None, None))]
    display = pd.concat([baseline, selected], ignore_index=True)
    long = display.melt(
        id_vars=["dataset", "model"],
        value_vars=[
            "oof_rmse_mean",
            "oof_spearman_mean",
            "fold_mean_regret_at_1_mean",
            "lower_tail_optimism_mean",
        ],
        var_name="metric",
        value_name="value",
    )
    figure = px.bar(
        long,
        x="model",
        y="value",
        color="model",
        facet_row="dataset",
        facet_col="metric",
        color_discrete_sequence=px.colors.diverging.RdYlGn_r,
        title="Centered order-only recency residual",
    )
    figure.update_layout(height=1150, showlegend=False)
    figure.update_xaxes(tickangle=-25)
    figure.write_html(
        output_dir / "centered_recency_residual_comparison.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--datasets",
        default=f"{STARCODER_NAME},300m_uncheatable,300m_table9,production_uncheatable",
    )
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--l2-values", default="0.01,0.1,1,10,100,1000,10000")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets, external = load_datasets()
    selected_names = [part.strip() for part in args.datasets.split(",") if part.strip()]
    unknown = sorted(set(selected_names).difference(datasets))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    seeds = pooled.parse_int_list(args.seeds)
    l2_values = pooled.parse_float_list(args.l2_values)
    metric_frames = []
    parameter_frames = []
    for dataset_name in selected_names:
        dataset = datasets[dataset_name]
        for seed in seeds:
            metric_path, parameter_path = checkpoint_paths(args.output_dir, dataset_name, seed)
            if metric_path.exists() and parameter_path.exists():
                print(f"Loading checkpoint {dataset_name}/seed={seed}", flush=True)
                metrics = pd.read_csv(metric_path)
                parameters = pd.read_csv(parameter_path)
            else:
                metrics, parameters = benchmark_seed(dataset, seed, args.n_splits, l2_values)
                metrics.to_csv(metric_path, index=False)
                parameters.to_csv(parameter_path, index=False)
            metric_frames.append(metrics)
            parameter_frames.append(parameters)
    raw = pd.concat(metric_frames, ignore_index=True)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    summary = pooled.summarize(raw)
    selected = selected_configs(summary)
    external_summary = external_metrics(datasets, external, selected)
    slice_summary = starcoder_slice_metrics(datasets[STARCODER_NAME], selected)

    raw.to_csv(args.output_dir / "cv_metrics_by_seed.csv", index=False)
    parameters.to_csv(args.output_dir / "fold_parameter_diagnostics.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    selected.to_csv(args.output_dir / "selected_configs.csv", index=False)
    external_summary.to_csv(args.output_dir / "external_two_phase_summary.csv", index=False)
    slice_summary.to_csv(args.output_dir / "starcoder_slice_summary.csv", index=False)
    write_plot(summary, args.output_dir)
    report = [
        "# Centered recency residual screen",
        "",
        "## Selected configurations",
        "",
        selected.to_markdown(index=False),
        "",
        "## External two-phase interventions",
        "",
        external_summary.to_markdown(index=False),
        "",
        "## StarCoder phase-0 Nemotron slice",
        "",
        slice_summary.to_markdown(index=False),
        "",
        "The residual is exactly zero for aggregate-matched tied schedules. It is accepted only if it beats the "
        "separate-heads proposal baseline while preserving the effective-exposure production guardrail.",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(selected.to_string(index=False))
    print(external_summary.to_string(index=False))
    print(slice_summary.to_string(index=False))
    print(f"Wrote centered-recency benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
