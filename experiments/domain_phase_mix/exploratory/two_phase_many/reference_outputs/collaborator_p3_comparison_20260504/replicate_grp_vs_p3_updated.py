#!/usr/bin/env python3
"""Replicate the collaborator GRP-vs-P3 comparison on the current 300M matrix.

The gist notebook compares:

* GRP no-L2, using the shipped standalone GRP implementation and included
  nonlinear parameters, with only the linear NNLS head refit to an IRT target.
* P3, a simplified ridge model over combined phase exposure plus two
  phase-specific concentration penalties, optimized by bootstrap Frank-Wolfe.

The current raw matrix has two row-name differences relative to the older GRP
packet. This script runs the head-to-head comparison on the clean row-name
intersection so both models see exactly the same target rows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[5]
RAW_MATRIX = (
    REPO_ROOT
    / "experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/raw_metric_matrix_300m"
    / "raw_metric_matrix_300m.csv"
)
VARIABLE_NOISE = (
    REPO_ROOT
    / "experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/raw_metric_matrix_300m"
    / "noise_baseline_run00097_variable_subset_300m.csv"
)
EPOCH_METADATA = REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/two_phase_many_epoch_metadata.csv"
GRP_PACKET = (
    REPO_ROOT
    / "experiments/domain_phase_mix/exploratory/two_phase_many/collaborator_scaling_data_packet_20260430"
)
GRP_CODE = GRP_PACKET / "standalone_code"
GRP_DATA = GRP_PACKET / "data/grp_no_l2"

MMLU_KEEP = {"lm_eval/mmlu_5shot/bpb", "lm_eval/mmlu_sl_verb_5shot/bpb"}
AGG_DROP = {
    "eval/bpb",
    "eval/macro_bpb",
    "eval/paloma/bpb",
    "eval/paloma/macro_bpb",
    "eval/uncheatable_eval/bpb",
    "eval/uncheatable_eval/macro_bpb",
}
TASK_DROP = {"teacher_forced/gsm8k_5shot_answer_hash/bpb", "mcq_smooth/sciq_5shot/bpb"}

ETA_GRID = (0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0)
A_GRID = tuple(np.linspace(0.5, 2.0, 8))
P_GRID = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
ALPHA_GRID = tuple(np.logspace(0, 4, 9))
BOOTSTRAP_COUNT = 200
FRANK_WOLFE_STEPS = 300
EPS = 1e-12


@dataclass(frozen=True)
class IrtTarget:
    """IRT target construction results."""

    aggregate: np.ndarray
    task_columns: list[str]
    task_signs: np.ndarray
    k_horn: int
    noise_share: np.ndarray
    factor_loadings: np.ndarray
    factor_noise: np.ndarray


@dataclass(frozen=True)
class P3Fit:
    """Fitted P3 model and bootstrap optimum."""

    eta: float
    a: float
    p: float
    alpha: float
    cv_r2: float
    in_sample_r2: float
    p0: np.ndarray
    p1: np.ndarray
    coefficient_std: np.ndarray
    design_mean: np.ndarray
    design_std: np.ndarray
    active_mask: np.ndarray


def _keep_metric(column: str) -> bool:
    if not column.endswith("/bpb"):
        return False
    if column in AGG_DROP or column in TASK_DROP:
        return False
    if not column.startswith(("eval/uncheatable_eval/", "lm_eval/", "mcq_smooth/", "teacher_forced/")):
        return False
    return not (column.startswith("lm_eval/mmlu_") and column not in MMLU_KEEP)


def selected_task_columns(raw: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Return task metrics and signs using collaborator notebook rules."""
    raw_columns = set(raw.columns)
    task_columns: list[str] = []
    task_signs: list[float] = []
    for candidate in [column for column in raw.columns if _keep_metric(column)]:
        base = candidate.removesuffix("/bpb")
        if base.startswith(("lm_eval/", "mcq_smooth/")):
            alt = base + "/choice_logprob"
            if alt in raw_columns:
                task_columns.append(alt)
                task_signs.append(1.0)
                continue
        task_columns.append(candidate)
        task_signs.append(-1.0)
    return task_columns, np.asarray(task_signs, dtype=np.float64)


def nonnegative_factor_irt(z: np.ndarray, noise_share: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Fit the same anchored nonnegative factor model as the collaborator gist."""
    row_count, item_count = z.shape
    real_eigenvalues = np.sort(np.linalg.eigvalsh(np.corrcoef(z.T)))[::-1]
    rng = np.random.default_rng(42)
    random_eigenvalues = np.empty((300, item_count), dtype=np.float64)
    for idx in range(random_eigenvalues.shape[0]):
        random_z = rng.standard_normal((row_count, item_count))
        random_eigenvalues[idx] = np.sort(np.linalg.eigvalsh(np.corrcoef(random_z.T)))[::-1]
    q95 = np.percentile(random_eigenvalues, 95, axis=0)
    k_horn = max(1, int((real_eigenvalues > q95).sum()))

    psi_anchor = np.where(np.isnan(noise_share), np.nan, np.clip(noise_share, 1e-3, 0.999))
    psi_fixed = ~np.isnan(psi_anchor)
    fit_rng = np.random.default_rng(0)
    loadings = np.abs(fit_rng.normal(scale=0.1, size=(item_count, k_horn)))
    psi = np.where(psi_fixed, psi_anchor, 1.0)

    for _ in range(3000):
        weighted_loadings = loadings / psi[:, None]
        posterior_cov = np.linalg.inv(np.eye(k_horn) + loadings.T @ weighted_loadings)
        theta = z @ weighted_loadings @ posterior_cov
        second_moment = row_count * posterior_cov + theta.T @ theta
        z_theta = z.T @ theta
        next_loadings = z_theta @ np.linalg.inv(second_moment)
        next_loadings = np.clip(next_loadings, 0.0, None)
        free_psi = (
            (z**2).mean(axis=0)
            - 2 * (z_theta * next_loadings).sum(axis=1) / row_count
            + ((next_loadings @ second_moment) * next_loadings).sum(axis=1) / row_count
        )
        free_psi = np.clip(free_psi, 1e-6, None)
        next_psi = np.where(psi_fixed, psi_anchor, free_psi)
        if np.max(np.abs(next_loadings - loadings)) < 1e-7:
            loadings = next_loadings
            psi = next_psi
            break
        loadings = next_loadings
        psi = next_psi

    weighted_loadings = loadings / psi[:, None]
    posterior_cov = np.linalg.inv(np.eye(k_horn) + loadings.T @ weighted_loadings)
    projection = (weighted_loadings @ posterior_cov).mean(axis=1)
    return k_horn, loadings, psi, projection


def build_irt_target(raw: pd.DataFrame, noise: pd.DataFrame) -> IrtTarget:
    """Construct the collaborator's IRT aggregate target."""
    task_columns, task_signs = selected_task_columns(raw)
    matrix = raw[task_columns].to_numpy(dtype=np.float64) * task_signs[None, :]
    if np.isnan(matrix).any():
        missing = [column for column in task_columns if raw[column].isna().any()]
        raise ValueError(f"Missing task metrics in raw matrix: {missing[:8]}")
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    if np.any(stds <= 1e-12):
        zero = [column for column, std in zip(task_columns, stds, strict=True) if std <= 1e-12]
        raise ValueError(f"Zero-variance task metrics: {zero[:8]}")
    z = (matrix - means) / stds

    noise_columns = set(noise.columns)
    has_noise = np.asarray([column in noise_columns for column in task_columns], dtype=bool)
    present = [column for column in task_columns if column in noise_columns]
    noise_share = np.full(len(task_columns), np.nan, dtype=np.float64)
    if present:
        present_signs = task_signs[has_noise]
        noise_matrix = noise[present].to_numpy(dtype=np.float64) * present_signs[None, :]
        noise_share[has_noise] = (
            noise_matrix.std(axis=0, ddof=1) / matrix[:, has_noise].std(axis=0, ddof=1)
        ) ** 2

    k_horn, loadings, psi, projection = nonnegative_factor_irt(z, noise_share)
    aggregate = z @ projection
    return IrtTarget(
        aggregate=aggregate,
        task_columns=task_columns,
        task_signs=task_signs,
        k_horn=k_horn,
        noise_share=noise_share,
        factor_loadings=loadings,
        factor_noise=psi,
    )


def ridge_inner_cv(design: np.ndarray, target: np.ndarray, fold_seed: int, alphas: tuple[float, ...]) -> tuple[float, float]:
    """Choose ridge alpha by 5-fold CV for one P3 nonlinear setting."""
    active = design.std(axis=0) > 1e-10
    if not active.any():
        return -np.inf, float("nan")
    active_design = design[:, active]
    mean = active_design.mean(axis=0)
    std = active_design.std(axis=0)
    standardized = (active_design - mean) / std
    feature_count = standardized.shape[1]
    rng = np.random.default_rng(fold_seed)
    indices = rng.permutation(len(target))
    folds = np.array_split(indices, 5)
    best_r2 = -np.inf
    best_alpha = float("nan")
    for alpha in alphas:
        predictions = np.zeros(len(target), dtype=np.float64)
        for fold_idx, test_indices in enumerate(folds):
            train_indices = np.concatenate([folds[idx] for idx in range(5) if idx != fold_idx])
            train_x = standardized[train_indices]
            train_y = target[train_indices]
            centered = train_y - train_y.mean()
            coef = np.linalg.solve(train_x.T @ train_x + alpha * np.eye(feature_count), train_x.T @ centered)
            predictions[test_indices] = standardized[test_indices] @ coef + train_y.mean()
        r2 = 1.0 - ((target - predictions) ** 2).sum() / ((target - target.mean()) ** 2).sum()
        if r2 > best_r2:
            best_r2 = float(r2)
            best_alpha = float(alpha)
    return best_r2, best_alpha


def build_p3_design(w0: np.ndarray, w1: np.ndarray, c0: np.ndarray, c1: np.ndarray, eta: float, a: float, p: float) -> np.ndarray:
    """Build P3 design: combined exposure signal plus phase penalties."""
    combined_exposure = np.maximum(w0 + eta * w1, 1e-4)
    signal = np.power(combined_exposure, a)
    normalized_epochs_0 = np.maximum(w0 * c0[None, :], 1e-4)
    normalized_epochs_1 = np.maximum(w1 * c1[None, :], 1e-4)
    penalty_0 = np.power(normalized_epochs_0, p).sum(axis=1, keepdims=True)
    penalty_1 = np.power(normalized_epochs_1, p).sum(axis=1, keepdims=True)
    return np.column_stack([signal, -penalty_0, -penalty_1])


def fit_p3(raw: pd.DataFrame, domains: list[str], c0: np.ndarray, c1: np.ndarray, target: np.ndarray) -> P3Fit:
    """Fit P3 exactly following the collaborator comparison notebook."""
    w0 = raw[[f"phase_0_{domain}" for domain in domains]].to_numpy(dtype=np.float64)
    w1 = raw[[f"phase_1_{domain}" for domain in domains]].to_numpy(dtype=np.float64)
    w0 = w0 / w0.sum(axis=1, keepdims=True)
    w1 = w1 / w1.sum(axis=1, keepdims=True)

    best_r2 = -np.inf
    best_combo: tuple[float, float, float, float] | None = None
    for eta in ETA_GRID:
        for a_value in A_GRID:
            for p_value in P_GRID:
                design = build_p3_design(w0, w1, c0, c1, eta, float(a_value), p_value)
                r2, alpha = ridge_inner_cv(design, target, fold_seed=42, alphas=ALPHA_GRID)
                if r2 > best_r2:
                    best_r2 = r2
                    best_combo = (eta, float(a_value), p_value, alpha)
    if best_combo is None:
        raise RuntimeError("P3 grid search failed")

    eta, a_value, p_value, alpha = best_combo
    full_design = build_p3_design(w0, w1, c0, c1, eta, a_value, p_value)
    active = full_design.std(axis=0) > 1e-10
    design = full_design[:, active]
    design_mean = design.mean(axis=0)
    design_std = design.std(axis=0)
    standardized = (design - design_mean) / design_std
    centered = target - target.mean()
    coefficient_std = np.linalg.solve(
        standardized.T @ standardized + alpha * np.eye(standardized.shape[1]),
        standardized.T @ centered,
    )
    fitted = standardized @ coefficient_std + target.mean()
    in_sample_r2 = float(1.0 - ((target - fitted) ** 2).sum() / ((target - target.mean()) ** 2).sum())
    p0, p1 = bootstrap_frank_wolfe_optimum(
        standardized=standardized,
        target=target,
        alpha=alpha,
        design_std=design_std,
        domains=domains,
        c0=c0,
        c1=c1,
        eta=eta,
        a_value=a_value,
        p_value=p_value,
    )
    return P3Fit(
        eta=eta,
        a=a_value,
        p=p_value,
        alpha=alpha,
        cv_r2=float(best_r2),
        in_sample_r2=in_sample_r2,
        p0=p0,
        p1=p1,
        coefficient_std=coefficient_std,
        design_mean=design_mean,
        design_std=design_std,
        active_mask=active,
    )


def bootstrap_frank_wolfe_optimum(
    *,
    standardized: np.ndarray,
    target: np.ndarray,
    alpha: float,
    design_std: np.ndarray,
    domains: list[str],
    c0: np.ndarray,
    c1: np.ndarray,
    eta: float,
    a_value: float,
    p_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean of bootstrap Frank-Wolfe simplex optima."""
    domain_count = len(domains)
    rng = np.random.default_rng(7)
    bootstrap_count = BOOTSTRAP_COUNT
    feature_count = standardized.shape[1]
    boot_coefs = np.zeros((bootstrap_count, feature_count), dtype=np.float64)
    for boot_idx in range(bootstrap_count):
        sample = rng.integers(0, len(target), len(target))
        sample_x = standardized[sample]
        sample_y = target[sample]
        centered = sample_y - sample_y.mean()
        boot_coefs[boot_idx] = np.linalg.solve(
            sample_x.T @ sample_x + alpha * np.eye(feature_count),
            sample_x.T @ centered,
        )

    natural_coefs = boot_coefs / design_std[None, :]
    beta_signal = natural_coefs[:, :domain_count]
    beta_penalty_0 = natural_coefs[:, domain_count]
    beta_penalty_1 = natural_coefs[:, domain_count + 1]

    p0_optima = np.empty((bootstrap_count, domain_count), dtype=np.float64)
    p1_optima = np.empty((bootstrap_count, domain_count), dtype=np.float64)
    for boot_idx in range(bootstrap_count):
        signal_coef = beta_signal[boot_idx]
        penalty_coef_0 = beta_penalty_0[boot_idx]
        penalty_coef_1 = beta_penalty_1[boot_idx]
        w0 = np.full(domain_count, 1.0 / domain_count, dtype=np.float64)
        w1 = np.full(domain_count, 1.0 / domain_count, dtype=np.float64)
        for step in range(FRANK_WOLFE_STEPS):
            exposure = np.maximum(w0 + eta * w1, EPS)
            signal_grad = a_value * np.power(exposure, a_value - 1.0)
            epochs_0 = np.maximum(c0 * w0, EPS)
            epochs_1 = np.maximum(c1 * w1, EPS)
            penalty_grad_0 = penalty_coef_0 * p_value * np.power(epochs_0, p_value - 1.0)
            penalty_grad_1 = penalty_coef_1 * p_value * np.power(epochs_1, p_value - 1.0)
            gradient_0 = signal_coef * signal_grad - c0 * penalty_grad_0
            gradient_1 = signal_coef * eta * signal_grad - c1 * penalty_grad_1
            gamma = 2.0 / (step + 2.0)
            w0 *= 1.0 - gamma
            w1 *= 1.0 - gamma
            w0[int(np.argmax(gradient_0))] += gamma
            w1[int(np.argmax(gradient_1))] += gamma
        p0_optima[boot_idx] = w0
        p1_optima[boot_idx] = w1
    return p0_optima.mean(axis=0), p1_optima.mean(axis=0)


def prepare_grp_data_dir(raw: pd.DataFrame, domains: list[str], target: np.ndarray) -> tempfile.TemporaryDirectory[str]:
    """Create a GRP-compatible temporary packet from the current raw matrix."""
    temp_context = tempfile.TemporaryDirectory(prefix="grp_irt_replicate_")
    temp_dir = Path(temp_context.name)
    phase_columns = [f"phase_{phase_idx}_{domain}" for phase_idx in (0, 1) for domain in domains]
    metadata_columns = [
        column
        for column in ("run_id", "run_name", "source_experiment", "status")
        if column in raw.columns
    ]
    frame = raw[metadata_columns + phase_columns].copy()
    frame["irt_neg"] = -target
    frame.to_csv(temp_dir / "two_phase_many.csv", index=False)
    shutil.copy2(EPOCH_METADATA, temp_dir / "two_phase_many_epoch_metadata.csv")
    shutil.copy2(GRP_DATA / "grp_power_family_penalty_no_l2_retune_best.csv", temp_dir / "grp_power_family_penalty_no_l2_retune_best.csv")
    shutil.copy2(GRP_DATA / "grp_penalty_calibration_variants_best.csv", temp_dir / "grp_penalty_calibration_variants_best.csv")
    return temp_context


def fit_grp(
    raw: pd.DataFrame,
    domains: list[str],
    c0: np.ndarray,
    c1: np.ndarray,
    target: np.ndarray,
    *,
    retune: bool,
    retune_method: str,
    retune_coarse_top_k: int,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """Fit GRP to negative target, optionally retuning nonlinear params."""
    if str(GRP_CODE) not in sys.path:
        sys.path.insert(0, str(GRP_CODE))
    import grp_no_l2_exact as grp

    with prepare_grp_data_dir(raw, domains, target) as temp_name:
        temp_dir = Path(temp_name)
        packet = grp.load_packet(temp_dir, target="irt_neg")
        if retune:
            starts = grp.start_bank(temp_dir)
            coarse_frame, best_metrics, refine_frame = grp.refine_rows(
                packet,
                starts,
                coarse_top_k=retune_coarse_top_k,
                method=retune_method,
            )
            coarse_frame.to_csv(ROOT / "grp_vs_p3_all242_grp_retune_coarse.csv", index=False)
            refine_frame.sort_values("objective").to_csv(ROOT / "grp_vs_p3_all242_grp_retune_refine.csv", index=False)
            params = {key: float(best_metrics[key]) for key in grp.param_keys() if key != "reg"}
            params["reg"] = grp.REG_FIXED
            retune_summary: dict[str, object] = {
                "retuned": True,
                "retune_method": retune_method,
                "retune_coarse_top_k": int(retune_coarse_top_k),
                "best_objective": float(best_metrics["objective"]),
                "best_cv_rmse": float(best_metrics["cv_rmse"]),
                "best_cv_depopt_best8": float(best_metrics["cv_depopt_best8"]),
                "best_cv_rawopt_nearest_tv": float(best_metrics["cv_rawopt_nearest_tv"]),
                "params": params,
            }
        else:
            params = grp.included_no_l2_best_params(temp_dir)
            retune_summary = {"retuned": False, "params": params}
        model = grp.build_model(packet, params)
        model.fit(packet.base.w, packet.base.y)
        predictions = model.predict(packet.base.w)
        y = packet.base.y
        in_sample_r2 = float(1.0 - ((y - predictions) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        _, phase0, phase1 = grp.optimize_model(packet, model, n_random=5, seed=0)

        grp_domains = packet.base.domain_names
        alignment = np.asarray([grp_domains.index(domain) for domain in domains], dtype=int)
        phase0 = phase0[alignment]
        phase1 = phase1[alignment]

    _ = c0, c1
    retune_summary["in_sample_r2_on_negative_irt"] = in_sample_r2
    return retune_summary, phase0, phase1


def domain_metadata(domains: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load phase epoch multipliers aligned to domains."""
    metadata = pd.read_csv(EPOCH_METADATA).set_index("domain_name")
    missing = [domain for domain in domains if domain not in metadata.index]
    if missing:
        raise ValueError(f"Missing epoch metadata for domains: {missing}")
    c0 = metadata.loc[domains, "phase_0_epoch_multiplier"].to_numpy(dtype=np.float64)
    c1 = metadata.loc[domains, "phase_1_epoch_multiplier"].to_numpy(dtype=np.float64)
    return c0, c1


def write_plot(path: Path, domains: list[str], c0: np.ndarray, c1: np.ndarray, grp_p0: np.ndarray, grp_p1: np.ndarray, p3: P3Fit) -> None:
    """Write side-by-side GRP/P3 mixture plot."""
    total_grp = grp_p0 + grp_p1
    total_p3 = p3.p0 + p3.p1
    epochs_grp = grp_p0 * c0 + grp_p1 * c1
    epochs_p3 = p3.p0 * c0 + p3.p1 * c1
    order = np.argsort(-(total_grp + total_p3))
    names = [domains[idx][:44] for idx in order]

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.18,
        subplot_titles=("weight space: phase 0 + phase 1", "epoch space: total epochs"),
    )
    fig.add_trace(
        go.Bar(x=total_grp[order], y=names, orientation="h", name="GRP", marker_color="rgba(120,120,120,0.75)"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=total_p3[order], y=names, orientation="h", name="P3", marker_color="#1877F2"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=epochs_grp[order], y=names, orientation="h", name="GRP epochs", marker_color="rgba(120,120,120,0.75)", showlegend=False),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(x=epochs_p3[order], y=names, orientation="h", name="P3 epochs", marker_color="#1877F2", showlegend=False),
        row=1,
        col=2,
    )
    fig.add_vline(x=8.0, line_dash="dot", line_color="black", row=1, col=2)
    fig.update_layout(
        template="plotly_white",
        height=max(700, 27 * len(domains)),
        width=1550,
        barmode="group",
        title=(
            "Optimal data mixture: GRP vs P3 on updated 300M IRT aggregate<br>"
            "<sup>Both models fit on the 240-row updated/GRP-packet intersection. Dotted epoch line = 8.</sup>"
        ),
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
        margin=dict(l=180, r=40, t=90, b=90),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(220,220,220,0.5)")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.write_html(path)
    try:
        fig.write_image(path.with_suffix(".png"), scale=2)
    except Exception as exc:
        print(f"Skipping PNG export: {exc}")


def summary_tables(
    domains: list[str],
    c0: np.ndarray,
    c1: np.ndarray,
    grp_p0: np.ndarray,
    grp_p1: np.ndarray,
    p3: P3Fit,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Build weight CSVs and agreement metrics."""
    total_grp = grp_p0 + grp_p1
    total_p3 = p3.p0 + p3.p1
    epochs_grp = grp_p0 * c0 + grp_p1 * c1
    epochs_p3 = p3.p0 * c0 + p3.p1 * c1
    rows = []
    for idx, domain in enumerate(domains):
        rows.append(
            {
                "domain": domain,
                "grp_phase0_weight": grp_p0[idx],
                "grp_phase1_weight": grp_p1[idx],
                "grp_total_weight": total_grp[idx],
                "grp_total_epochs": epochs_grp[idx],
                "p3_phase0_weight": p3.p0[idx],
                "p3_phase1_weight": p3.p1[idx],
                "p3_total_weight": total_p3[idx],
                "p3_total_epochs": epochs_p3[idx],
            }
        )
    weights = pd.DataFrame(rows)
    top = weights.assign(combined_attention=weights["grp_total_weight"] + weights["p3_total_weight"]).sort_values(
        "combined_attention", ascending=False
    )
    metrics = {
        "pearson_total_weight": float(np.corrcoef(total_grp, total_p3)[0, 1]),
        "pearson_phase0_weight": float(np.corrcoef(grp_p0, p3.p0)[0, 1]),
        "pearson_phase1_weight": float(np.corrcoef(grp_p1, p3.p1)[0, 1]),
        "pearson_total_epochs": float(np.corrcoef(epochs_grp, epochs_p3)[0, 1]),
        "l1_total_weight": float(np.abs(total_grp - total_p3).sum()),
    }
    return weights, top.head(20), metrics


def write_report(
    *,
    path: Path,
    row_audit: dict[str, object],
    irt: IrtTarget,
    grp_summary: dict[str, object],
    p3: P3Fit,
    agreement: dict[str, float],
    top: pd.DataFrame,
) -> None:
    """Write a concise replication report."""
    report = f"""# GRP vs P3 Replication on Updated 300M Data

## What Was Run

This reproduces the collaborator gist `grp_vs_p3_comparison.py` against the current 300M raw metric matrix. The raw gist is a Marimo notebook; export mostly worked locally, but the final `_table` display cell failed under `marimo export`. I therefore reproduced the notebook logic in `replicate_grp_vs_p3_updated.py`.

## Data Alignment

- Updated raw signal rows: `{row_audit["raw_rows"]}`.
- Rows used for this head-to-head comparison: `{row_audit["used_rows"]}` rows.
- Row mode: `{row_audit["row_mode"]}`.
- Older GRP packet rows: `{row_audit["grp_packet_rows"]}`.
- Clean old-packet intersection: `{row_audit["common_rows"]}` rows.
- Updated-only rows: `{", ".join(row_audit["raw_only"])}`.
- GRP-packet-only rows: `{", ".join(row_audit["grp_only"])}`.

For `row_mode=all_signal`, the GRP packet is rebuilt directly from the current raw matrix, so the renamed Olmix row and `baseline_stratified` are included instead of relying on the older packet's row list.

## IRT Target

The target is the collaborator's task aggregate:

- Start from BPB-like columns under `eval/uncheatable_eval/`, `lm_eval/`, `mcq_smooth/`, and `teacher_forced/`.
- Drop aggregate BPBs and two known-bad/special columns: `teacher_forced/gsm8k_5shot_answer_hash/bpb`, `mcq_smooth/sciq_5shot/bpb`.
- For MCQ-style `lm_eval/` and `mcq_smooth/` tasks, use `choice_logprob` instead of BPB when available.
- Orient every item as higher-is-better, z-score by the swarm, estimate variable-subset noise shares, and fit a nonnegative anchored factor model.
- Horn parallel analysis selected `{irt.k_horn}` factor(s) on the aligned data.
- Selected task/proxy item count: `{len(irt.task_columns)}`.

## Models

**GRP no-L2.** Uses the standalone `grp_no_l2_exact.py` implementation from the collaborator packet. The GRP target is `-IRT` because the GRP optimizer minimizes its target. In this run, GRP nonlinear parameters were `{"retuned" if grp_summary["retuned"] else "not retuned"}`.

**P3.** Uses:

```text
yhat = alpha0
     + sum_d beta_d * (w0_d + eta * w1_d)^a
     - gamma0 * sum_d (c0_d * w0_d)^p
     - gamma1 * sum_d (c1_d * w1_d)^p
```

`eta`, `a`, and `p` are chosen by the collaborator's nested 5x5 CV grid; the linear head is ridge; the reported optimum is the mean over 200 bootstrap Frank-Wolfe simplex argmaxes.

## Fit Results

- GRP in-sample R2 on `-IRT`: `{float(grp_summary["in_sample_r2_on_negative_irt"]):.4f}`.
- GRP retuned: `{grp_summary["retuned"]}`.
- P3 nested-CV R2 on `IRT`: `{p3.cv_r2:.4f}`.
- P3 full-data in-sample R2 on `IRT`: `{p3.in_sample_r2:.4f}`.
- P3 selected hyperparameters: `eta={p3.eta:g}`, `a={p3.a:.6g}`, `p={p3.p:g}`, `ridge_alpha={p3.alpha:g}`.

## Optimum Agreement

- Pearson total weight: `{agreement["pearson_total_weight"]:+.3f}`.
- Pearson phase-0 weight: `{agreement["pearson_phase0_weight"]:+.3f}`.
- Pearson phase-1 weight: `{agreement["pearson_phase1_weight"]:+.3f}`.
- Pearson total epochs: `{agreement["pearson_total_epochs"]:+.3f}`.
- L1 distance between total-weight recommendations: `{agreement["l1_total_weight"]:.3f}`.

## Top Domains

{top[["domain", "grp_total_weight", "p3_total_weight", "grp_total_epochs", "p3_total_epochs"]].to_markdown(index=False)}

## Interpretation

P3 is not a generic regression; it is a deliberately simplified GRP-style law. It keeps the phase-1 exposure multiplier and an explicit concentration penalty, but drops retained exposure, CC-pair aggregation, family totals, per-family curvature, per-family thresholds, and NNLS sign constraints. The useful question is therefore not whether it is more expressive than GRP; it is whether the simpler inductive bias is better matched to this task-IRT target.

If GRP is retuned, this is the stronger comparison requested by Calvin: P3 tuned on IRT versus GRP nonlinear body retuned on IRT, both fit on the same current rows.
"""
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse replication flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--row-mode",
        choices=("all-signal", "intersection"),
        default="all-signal",
        help="Use all current signal rows or only the old GRP packet intersection.",
    )
    parser.add_argument("--retune-grp", action="store_true", help="Retune GRP nonlinear parameters on the IRT target.")
    parser.add_argument("--retune-method", default="Powell", help="Scipy optimizer used by GRP nonlinear retune.")
    parser.add_argument("--retune-coarse-top-k", type=int, default=3, help="Number of GRP coarse starts to refine.")
    parser.add_argument("--output-prefix", default=None, help="Output file prefix under this artifact directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_all = pd.read_csv(RAW_MATRIX)
    raw_all = raw_all[(raw_all["status"] == "completed") & (raw_all["row_kind"] == "signal")].copy()
    noise = pd.read_csv(VARIABLE_NOISE)
    grp_frame = pd.read_csv(GRP_DATA / "two_phase_many.csv", usecols=["run_name"])

    raw_names = set(raw_all["run_name"])
    grp_names = set(grp_frame["run_name"])
    common_names = sorted(raw_names & grp_names)
    row_audit = {
        "raw_rows": int(len(raw_all)),
        "grp_packet_rows": int(len(grp_frame)),
        "common_rows": int(len(common_names)),
        "raw_only": sorted(raw_names - grp_names),
        "grp_only": sorted(grp_names - raw_names),
        "row_mode": args.row_mode,
    }
    if args.row_mode == "intersection":
        raw = raw_all[raw_all["run_name"].isin(common_names)].sort_values("run_name").reset_index(drop=True)
    else:
        raw = raw_all.sort_values("run_name").reset_index(drop=True)
    row_audit["used_rows"] = int(len(raw))

    domains = sorted(column.removeprefix("phase_0_") for column in raw.columns if column.startswith("phase_0_"))
    c0, c1 = domain_metadata(domains)
    irt = build_irt_target(raw, noise)
    grp_summary, grp_p0, grp_p1 = fit_grp(
        raw,
        domains,
        c0,
        c1,
        irt.aggregate,
        retune=args.retune_grp,
        retune_method=args.retune_method,
        retune_coarse_top_k=args.retune_coarse_top_k,
    )
    p3 = fit_p3(raw, domains, c0, c1, irt.aggregate)
    weights, top, agreement = summary_tables(domains, c0, c1, grp_p0, grp_p1, p3)

    ROOT.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    if prefix is None:
        prefix = "grp_vs_p3_all242_retuned" if args.row_mode == "all-signal" and args.retune_grp else "grp_vs_p3_updated"
    weights.to_csv(ROOT / f"{prefix}_weights.csv", index=False)
    top.to_csv(ROOT / f"{prefix}_top_domains.csv", index=False)
    pd.DataFrame(
        {
            "task_metric": irt.task_columns,
            "sign": irt.task_signs,
            "noise_share_variable_subset": irt.noise_share,
            "factor_noise": irt.factor_noise,
        }
    ).to_csv(ROOT / f"{prefix}_irt_items.csv", index=False)
    summary = {
        "row_audit": row_audit,
        "irt": {
            "k_horn": irt.k_horn,
            "task_metric_count": len(irt.task_columns),
            "aggregate_mean": float(irt.aggregate.mean()),
            "aggregate_std": float(irt.aggregate.std(ddof=1)),
        },
        "grp": grp_summary,
        "p3": {
            "eta": p3.eta,
            "a": p3.a,
            "p": p3.p,
            "ridge_alpha": p3.alpha,
            "nested_cv_r2": p3.cv_r2,
            "in_sample_r2": p3.in_sample_r2,
        },
        "agreement": agreement,
    }
    (ROOT / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_plot(ROOT / f"{prefix}_comparison.html", domains, c0, c1, grp_p0, grp_p1, p3)
    write_report(
        path=ROOT / f"{prefix}_report.md",
        row_audit=row_audit,
        irt=irt,
        grp_summary=grp_summary,
        p3=p3,
        agreement=agreement,
        top=top,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
