# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce Helw150's 300M aggregate-metric GRP pipeline locally.

The source gist is a Marimo notebook:
https://gist.github.com/Helw150/de1cd95cf1617d5d812fd6dad6d17d17

This script ports the notebook's aggregate construction and 235-parameter
per-phase/domain epoch-loss model to a reusable command-line workflow.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jax import jit, value_and_grad
from jax.flatten_util import ravel_pytree
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from sklearn.decomposition import FactorAnalysis

REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = (
    REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "collaborator_grug_v4_aggregate_repro_20260525"
)
GRUG_DASHBOARD_WEIGHTS = (
    REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "grug_moe_mix_dashboard_20260517/grug_moe_mix_weights_long.csv"
)
GIST_RAW_URL = (
    "https://gist.githubusercontent.com/Helw150/de1cd95cf1617d5d812fd6dad6d17d17/raw/"
    "b3a662b757992e969a6025a736712481cbf53ee7/grp_pipeline_300m.py"
)

DATASETS = {
    "old_packet": {
        "raw": (
            REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/"
            "collaborator_scaling_data_packet_20260430/data/raw_metric_matrix_300m/raw_metric_matrix_300m.csv"
        ),
        "noise": (
            REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/"
            "collaborator_scaling_data_packet_20260430/data/raw_metric_matrix_300m/"
            "noise_baseline_run00097_300m.csv"
        ),
        "metadata": (
            REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/two_phase_many_epoch_metadata.csv"
        ),
    },
    "current_registry": {
        "raw": (
            REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/"
            "metric_registry/raw_metric_matrix_300m/raw_metric_matrix_300m.csv"
        ),
        "noise": (
            REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/"
            "metric_registry/raw_metric_matrix_300m/noise_baseline_run00097_300m.csv"
        ),
        "metadata": (
            REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/two_phase_many_epoch_metadata.csv"
        ),
    },
}

MMLU_KEEP = {"lm_eval/mmlu_sl_verb_5shot/bpb"}
AGG_DROP = {
    "eval/bpb",
    "eval/macro_bpb",
    "eval/paloma/bpb",
    "eval/paloma/macro_bpb",
    "eval/uncheatable_eval/bpb",
    "eval/uncheatable_eval/macro_bpb",
}
TASK_DROP = {
    "teacher_forced/gsm8k_5shot_answer_hash/bpb",
    "lm_eval/arc_easy/bpb",
    "lm_eval/piqa/bpb",
    "mcq_smooth/swag_0shot/bpb",
}
EXTRA_TASKS = [("lm_eval/socialiqa_5shot/choice_logprob", 1.0)]
CAPABILITY_PREFIX_EXCLUDE = ("eval/paloma/", "eval/uncheatable_eval/")


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    raw_path: Path
    noise_path: Path
    metadata_path: Path
    drop_incomplete_task_cols: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="old_packet")
    parser.add_argument("--dataset-name")
    parser.add_argument("--raw-csv", type=Path)
    parser.add_argument("--noise-csv", type=Path)
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--drop-incomplete-task-cols", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=30)
    parser.add_argument("--fw-iters", type=int, default=300)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--skip-lcb", action="store_true")
    parser.add_argument("--skip-pareto-blend", action="store_true")
    parser.add_argument("--download-gist", action="store_true")
    return parser.parse_args()


def dataset_bundle(args: argparse.Namespace) -> DatasetBundle:
    if args.raw_csv is not None:
        name = args.dataset_name or args.raw_csv.parent.name
        return DatasetBundle(
            name=name,
            raw_path=args.raw_csv,
            noise_path=args.noise_csv or args.raw_csv.parent / "noise_baseline_run00097_300m.csv",
            metadata_path=args.metadata_csv
            or REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/two_phase_many_epoch_metadata.csv",
            drop_incomplete_task_cols=args.drop_incomplete_task_cols,
        )
    paths = DATASETS[args.dataset]
    return DatasetBundle(
        name=args.dataset,
        raw_path=paths["raw"],
        noise_path=paths["noise"],
        metadata_path=paths["metadata"],
        drop_incomplete_task_cols=args.drop_incomplete_task_cols,
    )


def download_gist(output_dir: Path) -> None:
    output_path = output_dir / "source_gist_grp_pipeline_300m.py"
    subprocess.run(["curl", "-fsSL", GIST_RAW_URL, "-o", str(output_path)], check=True)


def keep_bpb(column: str) -> bool:
    if not column.endswith("/bpb"):
        return False
    if column in AGG_DROP or column in TASK_DROP:
        return False
    if not column.startswith(
        (
            "eval/paloma/",
            "eval/uncheatable_eval/",
            "lm_eval/",
            "mcq_smooth/",
            "teacher_forced/",
        )
    ):
        return False
    if column.startswith("lm_eval/mmlu_") and column not in MMLU_KEEP:
        return False
    return True


def selected_tasks(raw: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    raw_cols = set(raw.columns)
    task_cols: list[str] = []
    task_signs: list[float] = []
    for column in [name for name in raw.columns if keep_bpb(name)]:
        base = column.removesuffix("/bpb")
        choice_logprob = base + "/choice_logprob"
        if base.startswith(("lm_eval/", "mcq_smooth/")) and choice_logprob in raw_cols:
            task_cols.append(choice_logprob)
            task_signs.append(1.0)
        else:
            task_cols.append(column)
            task_signs.append(-1.0)
    for column, sign in EXTRA_TASKS:
        if column in raw_cols and column not in task_cols:
            task_cols.append(column)
            task_signs.append(sign)
    return task_cols, np.asarray(task_signs, dtype=np.float64)


def load_data(bundle: DatasetBundle) -> dict[str, object]:
    raw = pd.read_csv(bundle.raw_path)
    raw = raw[raw["status"].eq("completed")].reset_index(drop=True)
    task_cols_all, task_signs_all = selected_tasks(raw)
    missing_counts = raw[task_cols_all].isna().sum().astype(int)
    incomplete_task_cols = [column for column in task_cols_all if int(missing_counts[column]) > 0]
    if incomplete_task_cols and not bundle.drop_incomplete_task_cols:
        joined = "\n".join(f"- {column}: {int(missing_counts[column])}" for column in incomplete_task_cols)
        raise ValueError(
            "Selected task columns are incomplete. Re-run with --drop-incomplete-task-cols or use "
            f"a complete input matrix.\n{joined}"
        )
    keep_mask = np.array([column not in incomplete_task_cols for column in task_cols_all], dtype=bool)
    task_cols = [column for column, keep in zip(task_cols_all, keep_mask, strict=True) if keep]
    task_signs = task_signs_all[keep_mask]
    x = raw[task_cols].to_numpy(dtype=np.float64) * task_signs[None, :]
    z = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-12)

    domains = sorted(column.removeprefix("phase_0_") for column in raw.columns if column.startswith("phase_0_"))
    w0 = raw[[f"phase_0_{domain}" for domain in domains]].to_numpy(dtype=np.float64)
    w1 = raw[[f"phase_1_{domain}" for domain in domains]].to_numpy(dtype=np.float64)
    w0 = w0 / w0.sum(axis=1, keepdims=True)
    w1 = w1 / w1.sum(axis=1, keepdims=True)

    metadata = pd.read_csv(bundle.metadata_path)
    metadata_by_domain = metadata.set_index("domain_name").to_dict(orient="index")
    c0 = np.asarray([metadata_by_domain[domain]["phase_0_epoch_multiplier"] for domain in domains])
    c1 = np.asarray([metadata_by_domain[domain]["phase_1_epoch_multiplier"] for domain in domains])

    noise_share = np.full(len(task_cols), np.nan)
    if bundle.noise_path.exists():
        noise = pd.read_csv(bundle.noise_path)
        for index, column in enumerate(task_cols):
            if column not in noise.columns:
                continue
            values = noise[column].to_numpy(dtype=np.float64) * task_signs[index]
            sd_swarm = x[:, index].std(ddof=1)
            sd_noise = values.std(ddof=1)
            if sd_swarm > 0:
                noise_share[index] = (sd_noise / sd_swarm) ** 2

    return {
        "raw": raw,
        "task_cols_all": task_cols_all,
        "task_cols": task_cols,
        "task_signs": task_signs,
        "incomplete_task_cols": incomplete_task_cols,
        "x": x,
        "z": z,
        "domains": domains,
        "w0": w0,
        "w1": w1,
        "c0": c0,
        "c1": c1,
        "noise_share": noise_share,
    }


def aggregate_targets(z: np.ndarray, noise_share: np.ndarray) -> dict[str, object]:
    y_mean = z.mean(axis=1)

    ns = np.where(np.isnan(noise_share), np.nanmedian(noise_share), noise_share)
    ns = np.clip(ns, 1e-3, 1.0)
    weights = 1.0 / ns
    weights = weights / weights.sum()
    y_invnoise = (z * weights[None, :]).sum(axis=1)

    u, s, _ = np.linalg.svd(z - z.mean(axis=0), full_matrices=False)
    y_pc1 = u[:, 0] * s[0]
    if np.corrcoef(y_pc1, y_mean)[0, 1] < 0:
        y_pc1 = -y_pc1

    factor_count = 5
    fa = FactorAnalysis(n_components=factor_count, rotation="varimax", random_state=0)
    factor_scores = fa.fit_transform(z)
    factor_loadings = fa.components_.T.copy()
    for index in range(factor_count):
        if np.corrcoef(factor_scores[:, index], y_mean)[0, 1] < 0:
            factor_scores[:, index] = -factor_scores[:, index]
            factor_loadings[:, index] = -factor_loadings[:, index]
    y_factor = factor_scores.mean(axis=1)

    return {
        "y_mean": y_mean,
        "y_invnoise": y_invnoise,
        "y_pc1": y_pc1,
        "y_factor": y_factor,
        "factor_scores": factor_scores,
        "factor_loadings": factor_loadings,
        "invnoise_weights": weights,
        "target_correlations": {
            "corr_mean_invnoise": float(np.corrcoef(y_mean, y_invnoise)[0, 1]),
            "corr_mean_pc1": float(np.corrcoef(y_mean, y_pc1)[0, 1]),
            "corr_mean_factor": float(np.corrcoef(y_mean, y_factor)[0, 1]),
            "corr_pc1_factor": float(np.corrcoef(y_pc1, y_factor)[0, 1]),
        },
    }


def softplus_inverse(value: float) -> float:
    return float(np.log(np.exp(value) - 1.0))


def theta_init(domain_count: int) -> dict[str, jax.Array]:
    return {
        "c": jnp.array(0.0),
        "log_A0": jnp.full(domain_count, softplus_inverse(0.5)),
        "log_k0": jnp.full(domain_count, softplus_inverse(0.5)),
        "log_B0": jnp.full(domain_count, softplus_inverse(0.1)),
        "log_A1": jnp.full(domain_count, softplus_inverse(0.5)),
        "log_k1": jnp.full(domain_count, softplus_inverse(0.5)),
        "log_B1": jnp.full(domain_count, softplus_inverse(0.1)),
    }


def predictor(c0: np.ndarray, c1: np.ndarray):
    c0_jax = jnp.asarray(c0)
    c1_jax = jnp.asarray(c1)

    def predict(theta: dict[str, jax.Array], w0: jax.Array, w1: jax.Array) -> jax.Array:
        c = theta["c"]
        a0 = jax.nn.softplus(theta["log_A0"])
        k0 = jax.nn.softplus(theta["log_k0"])
        b0 = jax.nn.softplus(theta["log_B0"])
        a1 = jax.nn.softplus(theta["log_A1"])
        k1 = jax.nn.softplus(theta["log_k1"])
        b1 = jax.nn.softplus(theta["log_B1"])

        t0 = w0 * c0_jax[None, :]
        t1 = w1 * c1_jax[None, :]
        sp0 = jax.nn.softplus(k0[None, :] * (t0 - 1.0 / k0[None, :])) / k0[None, :]
        sp1 = jax.nn.softplus(k1[None, :] * (t1 - 1.0 / k1[None, :])) / k1[None, :]
        ell0 = a0[None, :] * jnp.exp(-k0[None, :] * t0) + b0[None, :] * sp0
        ell1 = a1[None, :] * jnp.exp(-k1[None, :] * t1) + b1[None, :] * sp1
        return -(c + ell0.sum(axis=1) + ell1.sum(axis=1))

    return predict


def fit_full(
    predict,
    initial_theta: dict[str, jax.Array],
    w0: np.ndarray,
    w1: np.ndarray,
    y: np.ndarray,
    maxiter: int,
) -> dict[str, jax.Array]:
    theta_flat_init, unravel = ravel_pytree(initial_theta)

    @jit
    def loss_vg(theta_flat, w0_batch, w1_batch, y_batch):
        def loss(theta_flat_inner):
            return jnp.mean((predict(unravel(theta_flat_inner), w0_batch, w1_batch) - y_batch) ** 2)

        return value_and_grad(loss)(theta_flat)

    def scipy_loss(theta_flat: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = loss_vg(jnp.asarray(theta_flat), jnp.asarray(w0), jnp.asarray(w1), jnp.asarray(y))
        return float(value), np.asarray(grad, dtype=np.float64)

    result = minimize(
        scipy_loss,
        np.asarray(theta_flat_init, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "ftol": 1e-9},
    )
    return unravel(jnp.asarray(result.x))


def fit_with_cv(
    predict,
    initial_theta: dict[str, jax.Array],
    w0: np.ndarray,
    w1: np.ndarray,
    y: np.ndarray,
    maxiter: int,
    folds: int,
    seed: int = 0,
) -> tuple[dict[str, jax.Array], np.ndarray, float]:
    n = len(y)
    rng = np.random.default_rng(seed)
    fold_indices = np.array_split(rng.permutation(n), folds)
    oof = np.zeros(n)
    for fold_index in range(folds):
        test = fold_indices[fold_index]
        train = np.concatenate([fold_indices[index] for index in range(folds) if index != fold_index])
        print(f"fit fold {fold_index + 1}/{folds}: train={len(train)} test={len(test)}", flush=True)
        fold_theta = fit_full(predict, initial_theta, w0[train], w1[train], y[train], maxiter=maxiter)
        oof[test] = np.asarray(predict(fold_theta, jnp.asarray(w0[test]), jnp.asarray(w1[test])))
    print("fit full-data theta", flush=True)
    full_theta = fit_full(predict, initial_theta, w0, w1, y, maxiter=maxiter)
    ss_res = float(((y - oof) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2_cv = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return full_theta, oof, r2_cv


def theta_param_summary(theta: dict[str, jax.Array]) -> dict[str, object]:
    summary: dict[str, object] = {"c": float(theta["c"])}
    for phase in (0, 1):
        for name in ("A", "k", "B"):
            values = np.asarray(jax.nn.softplus(theta[f"log_{name}{phase}"]), dtype=np.float64)
            summary[f"phase_{phase}_{name}"] = {
                "min": float(values.min()),
                "median": float(np.median(values)),
                "max": float(values.max()),
            }
    return summary


def proportional_weights(c0: np.ndarray, c1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w0 = (1.0 / c0) / (1.0 / c0).sum()
    w1 = (1.0 / c1) / (1.0 / c1).sum()
    return w0, w1


def lcb_optimize(
    predict,
    initial_theta: dict[str, jax.Array],
    w0_swarm: np.ndarray,
    w1_swarm: np.ndarray,
    y: np.ndarray,
    bootstrap_count: int,
    fw_iters: int,
    kappa: float,
    maxiter: int,
) -> tuple[np.ndarray, np.ndarray]:
    n, domain_count = w0_swarm.shape
    rng = np.random.default_rng(0)
    thetas = []
    for boot_index in range(bootstrap_count):
        print(f"bootstrap fit {boot_index + 1}/{bootstrap_count}", flush=True)
        sample = rng.integers(0, n, n)
        thetas.append(fit_full(predict, initial_theta, w0_swarm[sample], w1_swarm[sample], y[sample], maxiter=maxiter))

    @jit
    def lcb(w0, w1):
        preds = jnp.stack([predict(theta, w0[None, :], w1[None, :])[0] for theta in thetas])
        return preds.mean() - kappa * preds.std()

    @jit
    def lcb_grad(w0, w1):
        return value_and_grad(lcb, argnums=(0, 1))(w0, w1)

    w0 = jnp.full(domain_count, 1.0 / domain_count)
    w1 = jnp.full(domain_count, 1.0 / domain_count)
    for iteration in range(fw_iters):
        _, (g0, g1) = lcb_grad(w0, w1)
        v0 = jnp.zeros(domain_count).at[jnp.argmax(g0)].set(1.0)
        v1 = jnp.zeros(domain_count).at[jnp.argmax(g1)].set(1.0)
        gamma = 2.0 / (iteration + 2)
        w0 = (1.0 - gamma) * w0 + gamma * v0
        w1 = (1.0 - gamma) * w1 + gamma * v1
    return np.asarray(w0), np.asarray(w1)


def pareto_blend(
    predict,
    initial_theta: dict[str, jax.Array],
    w0_swarm: np.ndarray,
    w1_swarm: np.ndarray,
    z: np.ndarray,
    task_cols: list[str],
    w0_prop: np.ndarray,
    w1_prop: np.ndarray,
    w0_lcb: np.ndarray,
    w1_lcb: np.ndarray,
    maxiter: int,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    per_task_theta = []
    for task_index in range(z.shape[1]):
        print(f"per-task fit {task_index + 1}/{z.shape[1]}", flush=True)
        per_task_theta.append(
            fit_full(
                predict,
                initial_theta,
                w0_swarm,
                w1_swarm,
                z[:, task_index],
                maxiter=maxiter,
            )
        )

    def eval_per_task(w0_eval: np.ndarray, w1_eval: np.ndarray) -> np.ndarray:
        w0_jax = jnp.asarray(w0_eval)[None, :]
        w1_jax = jnp.asarray(w1_eval)[None, :]
        return np.asarray([float(predict(theta, w0_jax, w1_jax)[0]) for theta in per_task_theta])

    base_pred = eval_per_task(w0_prop, w1_prop)
    opt_pred = eval_per_task(w0_lcb, w1_lcb)
    full_delta = opt_pred - base_pred
    cap_mask = np.asarray([not task.startswith(CAPABILITY_PREFIX_EXCLUDE) for task in task_cols], dtype=bool)
    alphas = np.linspace(0.0, 1.0, 51)
    scan_rows = []
    for alpha in alphas:
        w0_blend = alpha * w0_lcb + (1.0 - alpha) * w0_prop
        w1_blend = alpha * w1_lcb + (1.0 - alpha) * w1_prop
        delta = eval_per_task(w0_blend, w1_blend) - base_pred
        cap_delta = delta[cap_mask]
        scan_rows.append(
            {
                "alpha": float(alpha),
                "n_no_regress_capability": int((cap_delta >= -1e-9).sum()),
                "n_capability": int(cap_mask.sum()),
                "mean_delta_capability": float(cap_delta.mean()),
                "min_delta_capability": float(cap_delta.min()),
            }
        )
    scan = pd.DataFrame(scan_rows)
    safe = scan[scan["n_no_regress_capability"].eq(scan["n_capability"])]
    if not safe.empty:
        alpha_star = float(safe["alpha"].max())
    else:
        best_index = scan["n_no_regress_capability"].idxmax()
        alpha_star = float(scan.loc[best_index, "alpha"])

    task_deltas = pd.DataFrame(
        {
            "task": task_cols,
            "capability_task": cap_mask,
            "base_pred": base_pred,
            "lcb_pred": opt_pred,
            "lcb_delta": full_delta,
        }
    ).sort_values("lcb_delta")
    return alpha_star, scan, task_deltas


def write_factor_loadings_plot(task_cols: list[str], factor_loadings: np.ndarray, output_path: Path) -> None:
    dominant = np.argmax(np.abs(factor_loadings), axis=1)
    peak = factor_loadings[np.arange(len(task_cols)), dominant]
    order = np.lexsort((-peak, dominant))
    names = [task_cols[index].removesuffix("/bpb").removesuffix("/choice_logprob") for index in order]
    vmax = float(np.max(np.abs(factor_loadings)))
    fig = go.Figure(
        go.Heatmap(
            z=factor_loadings[order],
            x=[f"F{index + 1}" for index in range(factor_loadings.shape[1])],
            y=names,
            colorscale="RdBu",
            zmid=0,
            zmin=-vmax,
            zmax=vmax,
            colorbar={"title": "loading"},
        )
    )
    fig.update_layout(
        template="plotly_white",
        title={"text": f"Varimax-rotated factor loadings (K={factor_loadings.shape[1]})", "x": 0.5},
        height=max(500, 24 * len(task_cols)),
        width=850,
        margin={"l": 350, "r": 30, "t": 70, "b": 40},
        yaxis={"autorange": "reversed", "tickfont": {"size": 10}},
        xaxis={"side": "top"},
    )
    fig.write_html(output_path, include_plotlyjs="cdn")


def write_mixture_plot(
    domains: list[str],
    c0: np.ndarray,
    c1: np.ndarray,
    mixtures: list[tuple[str, np.ndarray, np.ndarray, str]],
    output_path: Path,
) -> None:
    epoch_totals = [w0 * c0 + w1 * c1 for _, w0, w1, _ in mixtures]
    order = np.argsort(-np.sum(epoch_totals, axis=0))
    names = [domains[index][:50] for index in order]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("weight", "epochs"))
    for label, w0, w1, color in mixtures:
        darker = color.replace("0.85", "1.0")
        fig.add_trace(
            go.Bar(x=w0[order], y=names, orientation="h", name=f"{label} phase 0", marker={"color": color}),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(x=w1[order], y=names, orientation="h", name=f"{label} phase 1", marker={"color": darker}),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=(w0 * c0)[order],
                y=names,
                orientation="h",
                name=f"{label} phase 0",
                showlegend=False,
                marker={"color": color},
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=(w1 * c1)[order],
                y=names,
                orientation="h",
                name=f"{label} phase 1",
                showlegend=False,
                marker={"color": darker},
            ),
            row=1,
            col=2,
        )
    fig.update_layout(
        template="plotly_white",
        barmode="group",
        height=max(650, 24 * len(domains)),
        width=1500,
        margin={"l": 310, "r": 30, "t": 70, "b": 70},
        title={"text": "Reproduced aggregate optimum", "x": 0.5},
        legend={"orientation": "h", "yanchor": "top", "y": -0.05, "xanchor": "center", "x": 0.5},
    )
    fig.write_html(output_path, include_plotlyjs="cdn")


def compare_dashboard_v4(mixture_df: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    if not GRUG_DASHBOARD_WEIGHTS.exists():
        return {}
    dashboard = pd.read_csv(GRUG_DASHBOARD_WEIGHTS)
    v4 = (
        dashboard[dashboard["track"].eq("grug_moe_mix_v4") & dashboard["hidden_dim"].eq(512)]
        .pivot(index="domain", columns="phase", values="weight")
        .fillna(0.0)
    )
    comparisons = {}
    detail_frames = []
    for label, group in mixture_df.groupby("label"):
        if label == "proportional":
            continue
        group = group.set_index("domain")
        domains = sorted(set(group.index) | set(v4.index))
        left = group.reindex(domains).fillna(0.0)
        right = v4.reindex(domains).fillna(0.0)
        flat_left = np.concatenate([left["w0"].to_numpy(float), left["w1"].to_numpy(float)])
        flat_right = np.concatenate([right["phase_0"].to_numpy(float), right["phase_1"].to_numpy(float)])
        phase0_tv = 0.5 * float(np.abs(left["w0"] - right["phase_0"]).sum())
        phase1_tv = 0.5 * float(np.abs(left["w1"] - right["phase_1"]).sum())
        detail = pd.DataFrame(
            {
                "label": label,
                "domain": domains,
                "repro_w0": left["w0"].to_numpy(float),
                "repro_w1": left["w1"].to_numpy(float),
                "v4_w0": right["phase_0"].to_numpy(float),
                "v4_w1": right["phase_1"].to_numpy(float),
            }
        )
        detail["repro_total"] = detail["repro_w0"] + detail["repro_w1"]
        detail["v4_total"] = detail["v4_w0"] + detail["v4_w1"]
        detail["abs_total_diff"] = (detail["repro_total"] - detail["v4_total"]).abs()
        detail_frames.append(detail.sort_values("abs_total_diff", ascending=False))
        comparisons[label] = {
            "flat_phase_weight_correlation": float(np.corrcoef(flat_left, flat_right)[0, 1]),
            "phase0_tv": phase0_tv,
            "phase1_tv": phase1_tv,
            "mean_phase_tv": 0.5 * (phase0_tv + phase1_tv),
            "max_total_weight_abs_diff": float(detail["abs_total_diff"].max()),
        }
    if detail_frames:
        pd.concat(detail_frames, ignore_index=True).to_csv(
            output_dir / "reproduced_vs_dashboard_v4_weights.csv",
            index=False,
        )
    return comparisons


def summarize_mixture(
    label: str,
    domains: list[str],
    w0: np.ndarray,
    w1: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "label": label,
            "domain": domains,
            "w0": w0,
            "w1": w1,
            "total_weight": w0 + w1,
            "phase0_epochs": w0 * c0,
            "phase1_epochs": w1 * c1,
            "total_epochs": w0 * c0 + w1 * c1,
        }
    )
    return out.sort_values("total_epochs", ascending=False)


def write_markdown_report(output_dir: Path, summary: dict[str, object], top_mixture: pd.DataFrame) -> None:
    lines = [
        "# Collaborator Grug v4 Aggregate Reproduction",
        "",
        "Source gist: https://gist.github.com/Helw150/de1cd95cf1617d5d812fd6dad6d17d17",
        "",
        "## Summary",
        "",
        f"- Dataset: `{summary['dataset']}`.",
        f"- Raw rows used: `{summary['n_rows']}`.",
        f"- Gist-selected task columns before completeness filtering: `{summary['n_task_cols_all']}`.",
        f"- Task columns used: `{summary['n_task_cols_used']}`.",
        f"- Incomplete selected task columns dropped: `{summary['n_incomplete_task_cols']}`.",
        f"- Domains: `{summary['n_domains']}`.",
        f"- Target correlations: `{json.dumps(summary['target_correlations'], sort_keys=True)}`.",
        f"- CV R2 for 235-parameter epoch-loss model: `{summary['r2_cv']:.4f}`.",
        f"- Pareto-blend alpha*: `{summary['alpha_star']}`.",
        "",
        "## Top LCB/Optimized Domains",
        "",
    ]
    for _, row in top_mixture.head(12).iterrows():
        lines.append(
            f"- `{row['domain']}`: w0 `{row['w0']:.4f}`, w1 `{row['w1']:.4f}`, " f"epochs `{row['total_epochs']:.3f}`."
        )
    if summary.get("dashboard_v4_comparison"):
        lines.extend(["", "## Comparison To Dashboard v4", ""])
        for label, comparison in summary["dashboard_v4_comparison"].items():
            lines.append(
                f"- `{label}`: flattened phase-weight correlation "
                f"`{comparison['flat_phase_weight_correlation']:.3f}`, "
                f"mean per-phase TV `{comparison['mean_phase_tv']:.3f}`."
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            (
                "- The current registry matrix is not notebook-compatible without filtering because many selected "
                "Paloma, Uncheatable, HellaSwag, and MMLU-SL columns are missing for 142 or 148 rows."
            ),
            (
                "- The older collaborator packet currently has 242 completed rows and all 41 gist-selected task "
                "columns complete."
            ),
            (
                "- The model being fit is not our canonical DSP form; it is the collaborator notebook's "
                "per-phase/domain epoch-loss form with 235 parameters."
            ),
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    bundle = dataset_bundle(args)
    output_dir = args.output_dir / bundle.name
    if bundle.drop_incomplete_task_cols:
        output_dir = output_dir.with_name(output_dir.name + "_drop_incomplete")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.download_gist:
        download_gist(output_dir)

    print(f"loading dataset {bundle.name}", flush=True)
    data = load_data(bundle)
    task_cols = data["task_cols"]
    domains = data["domains"]
    w0 = data["w0"]
    w1 = data["w1"]
    c0 = data["c0"]
    c1 = data["c1"]
    z = data["z"]
    noise_share = data["noise_share"]
    assert isinstance(task_cols, list)
    assert isinstance(domains, list)
    assert isinstance(w0, np.ndarray)
    assert isinstance(w1, np.ndarray)
    assert isinstance(c0, np.ndarray)
    assert isinstance(c1, np.ndarray)
    assert isinstance(z, np.ndarray)
    assert isinstance(noise_share, np.ndarray)

    print(f"rows={len(w0)} tasks={len(task_cols)} domains={len(domains)}", flush=True)
    aggregate = aggregate_targets(z, noise_share)
    y_factor = aggregate["y_factor"]
    assert isinstance(y_factor, np.ndarray)

    pd.DataFrame({"task": task_cols, "sign": data["task_signs"], "noise_share": noise_share}).to_csv(
        output_dir / "selected_tasks.csv",
        index=False,
    )
    pd.DataFrame(aggregate["factor_loadings"], index=task_cols).to_csv(output_dir / "factor_loadings.csv")
    write_factor_loadings_plot(task_cols, aggregate["factor_loadings"], output_dir / "factor_loadings.html")
    raw = data["raw"]
    assert isinstance(raw, pd.DataFrame)
    pd.DataFrame(
        {
            "run_name": raw["run_name"],
            "run_id": raw["run_id"],
            "y_mean": aggregate["y_mean"],
            "y_invnoise": aggregate["y_invnoise"],
            "y_pc1": aggregate["y_pc1"],
            "y_factor": y_factor,
        }
    ).to_csv(output_dir / "aggregate_scores.csv", index=False)

    predict = predictor(c0, c1)
    initial_theta = theta_init(len(domains))
    theta_fit, oof, r2_cv = fit_with_cv(
        predict,
        initial_theta,
        w0,
        w1,
        y_factor,
        maxiter=args.maxiter,
        folds=args.cv_folds,
    )
    pd.DataFrame({"run_name": raw["run_name"], "run_id": raw["run_id"], "y_factor": y_factor, "oof": oof}).to_csv(
        output_dir / "oof_predictions.csv",
        index=False,
    )

    w0_prop, w1_prop = proportional_weights(c0, c1)
    mixture_tables = [summarize_mixture("proportional", domains, w0_prop, w1_prop, c0, c1)]
    optimized_table = pd.DataFrame()
    alpha_star = float("nan")
    if not args.skip_lcb:
        w0_lcb, w1_lcb = lcb_optimize(
            predict,
            initial_theta,
            w0,
            w1,
            y_factor,
            bootstrap_count=args.bootstrap,
            fw_iters=args.fw_iters,
            kappa=args.kappa,
            maxiter=args.maxiter,
        )
        optimized_table = summarize_mixture("lcb_optimum", domains, w0_lcb, w1_lcb, c0, c1)
        mixture_tables.append(optimized_table)
        if not args.skip_pareto_blend:
            alpha_star, alpha_scan, task_deltas = pareto_blend(
                predict,
                initial_theta,
                w0,
                w1,
                z,
                task_cols,
                w0_prop,
                w1_prop,
                w0_lcb,
                w1_lcb,
                maxiter=args.maxiter,
            )
            alpha_scan.to_csv(output_dir / "pareto_alpha_scan.csv", index=False)
            task_deltas.to_csv(output_dir / "per_task_lcb_deltas.csv", index=False)
            w0_blend = alpha_star * w0_lcb + (1.0 - alpha_star) * w0_prop
            w1_blend = alpha_star * w1_lcb + (1.0 - alpha_star) * w1_prop
            blend_table = summarize_mixture("pareto_blend", domains, w0_blend, w1_blend, c0, c1)
            mixture_tables.append(blend_table)
        write_mixture_plot(
            domains,
            c0,
            c1,
            [
                ("proportional", w0_prop, w1_prop, "rgba(24,119,242,0.85)"),
                *([("Pareto blend", w0_blend, w1_blend, "rgba(44,160,44,0.85)")] if not args.skip_pareto_blend else []),
                ("LCB optimum", w0_lcb, w1_lcb, "rgba(240,112,26,0.85)"),
            ],
            output_dir / "mixture_plot.html",
        )
    mixture_df = pd.concat(mixture_tables, ignore_index=True)
    mixture_df.to_csv(output_dir / "mixture_weights.csv", index=False)
    dashboard_v4_comparison = compare_dashboard_v4(mixture_df, output_dir)

    summary = {
        "dataset": bundle.name,
        "raw_path": str(bundle.raw_path),
        "noise_path": str(bundle.noise_path),
        "metadata_path": str(bundle.metadata_path),
        "n_rows": len(raw),
        "n_task_cols_all": len(data["task_cols_all"]),
        "n_task_cols_used": len(task_cols),
        "n_incomplete_task_cols": len(data["incomplete_task_cols"]),
        "incomplete_task_cols": data["incomplete_task_cols"],
        "n_domains": len(domains),
        "drop_incomplete_task_cols": bool(bundle.drop_incomplete_task_cols),
        "maxiter": int(args.maxiter),
        "cv_folds": int(args.cv_folds),
        "bootstrap": int(args.bootstrap),
        "fw_iters": int(args.fw_iters),
        "kappa": float(args.kappa),
        "alpha_star": alpha_star,
        "dashboard_v4_comparison": dashboard_v4_comparison,
        "target_correlations": aggregate["target_correlations"],
        "r2_cv": float(r2_cv),
        "theta_summary": theta_param_summary(theta_fit),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    top_table = optimized_table if not optimized_table.empty else mixture_tables[0]
    write_markdown_report(output_dir, summary, top_table)
    print(
        textwrap.dedent(
            f"""
        wrote {output_dir}
        rows={summary['n_rows']} tasks_used={summary['n_task_cols_used']} domains={summary['n_domains']}
        corr_mean_factor={summary['target_correlations']['corr_mean_factor']:.3f}
        r2_cv={summary['r2_cv']:.4f}
    """
        ).strip()
    )


if __name__ == "__main__":
    main()
