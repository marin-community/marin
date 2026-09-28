# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
# ]
# ///
"""Benchmark and optimize Hellinger kernel ridge on the Delphi 3e18 swarm.

The content kernel represents each phase by the token-weighted distribution over
a frozen Luxical embedding partition:

    h_t(w) = sum_i w_i^(t) V_i

and compares policies with the squared Hellinger distance

    d_H^2(w, w') = sum_t pi_t [1 - <sqrt(h_t(w)), sqrt(h_t(w'))>].

The weight-space ablation replaces V with the identity. Kernel bandwidth and
ridge strength are selected by nested CV using only the fitting swarm. Existing
3e18 heldouts are evaluated only after the fit is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold, KFold

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from swarm39_harness_20260725 import (  # noqa: E402
    CANONICAL,
    TABLE9,
    UNCHEATABLE,
    Panel,
    grouped_splits,
    load_scale,
    metric_row,
)

from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import (  # noqa: E402
    TARGET_BUDGET_DOLMA3_COMMON_CRAWL,
    TOP_LEVEL_DOMAIN_TOKEN_COUNTS,
)

REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "hellinger_krr_delphi_3e18_20260727"
DEFAULT_HISTOGRAM_DIR = DEFAULT_OUTPUT_DIR / "input" / "domain_histograms"
DEFAULT_LOOKUP = DEFAULT_OUTPUT_DIR / "input" / "basis" / "lookup_5000_to_1000.npy"
ONE_PHASE_FIT = CANONICAL / "delphi_3e18_one_phase_fit.csv"

GAMMA_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)
RIDGE_ALPHAS = tuple(float(value) for value in np.logspace(-3, 2, 6))
NESTED_FOLDS = 5
TOP_K_AVERAGE = 64
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
EXPECTED_LOOKUP_SHA256 = "a1aa004f38d4240ba94e56b03c8c252e513a572c2766d387d986e17ada7413c3"


@dataclass(frozen=True)
class KernelFit:
    """A fitted Hellinger RBF kernel ridge surrogate."""

    kernel_space: str
    target: str
    gamma: float
    ridge_alpha: float
    target_mean: float
    dual: np.ndarray
    train_sqrt_features: np.ndarray
    train_phase_histograms: np.ndarray
    basis: np.ndarray
    phase_fractions: np.ndarray
    oof_predictions: np.ndarray
    oof_rmse: float
    cv_sse: float
    gram_inverse: np.ndarray


@dataclass(frozen=True)
class Candidate:
    """One optimized policy and its frozen-model diagnostics."""

    target: str
    candidate_kind: str
    predicted_bpb: float
    nearest_fit_hellinger_sq: float
    posterior_std: float
    max_weight: float
    max_simulated_epoch: float
    phase_tv: float
    phase_0_weights: np.ndarray
    phase_1_weights: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--histogram-dir", type=Path, default=DEFAULT_HISTOGRAM_DIR)
    parser.add_argument("--lookup", type=Path, default=DEFAULT_LOOKUP)
    parser.add_argument("--candidate-count", type=int, default=60_000)
    parser.add_argument("--optimizer-starts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-one-phase", action="store_true")
    parser.add_argument("--skip-optimize", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_embedding_basis(
    buckets: tuple[str, ...],
    histogram_dir: Path,
    lookup_path: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return the 39 by 1000 token-weighted content histogram matrix."""
    metadata_path = histogram_dir / "_meta.json"
    metadata = json.loads(metadata_path.read_text())
    lookup_hash = sha256(lookup_path)
    assert lookup_hash == EXPECTED_LOOKUP_SHA256, f"unexpected lookup hash: {lookup_hash}"
    assert metadata["basis"]["view_sha256"]["1000"] == lookup_hash
    lookup = np.load(lookup_path)
    assert lookup.shape == (5000,)
    assert int(lookup.min()) == 0 and int(lookup.max()) == 999

    rows = []
    for bucket in buckets:
        domain = metadata["domains"].get(bucket)
        assert domain is not None, f"missing embedding histogram for {bucket}"
        frame = pd.read_parquet(histogram_dir / domain["parquet"])
        assert set(frame["domain"].unique()) == {bucket}
        fine = np.bincount(
            frame["cluster_id"].to_numpy(int),
            weights=frame["token_count"].to_numpy(float),
            minlength=5000,
        )
        coarse = np.bincount(lookup, weights=fine, minlength=1000).astype(float)
        assert coarse.sum() > 0.0
        rows.append(coarse / coarse.sum())
    basis = np.stack(rows)
    assert basis.shape == (len(buckets), 1000)
    assert np.max(np.abs(basis.sum(axis=1) - 1.0)) < 1e-12
    provenance = {
        "metadata_sha256": sha256(metadata_path),
        "lookup_sha256": lookup_hash,
        "embedder": metadata["basis"]["embedder"],
        "sample_size_per_bucket": metadata["sampling"]["sample_size"],
        "bucket_count": len(buckets),
        "embedding_cells": basis.shape[1],
    }
    return basis, provenance


def one_phase_fit_panel(template: Panel) -> Panel:
    """Load the independent 280-row tied-policy fit in the shared coordinate system."""
    frame = pd.read_csv(ONE_PHASE_FIT)
    phase0 = frame[[f"phase_0_weight::{bucket}" for bucket in template.buckets]].to_numpy(float)
    phase1 = frame[[f"phase_1_weight::{bucket}" for bucket in template.buckets]].to_numpy(float)
    assert np.max(np.abs(phase0 - phase1)) < 1e-12
    assert np.max(np.abs(phase0.sum(axis=1) - 1.0)) < 1e-10
    return Panel(
        scale=template.scale,
        split="fit_one_phase",
        alpha=template.alpha,
        buckets=template.buckets,
        c0=template.c0,
        c1=template.c1,
        family_index=template.family_index,
        family_names=template.family_names,
        phase0=phase0,
        phase1=phase1,
        targets={
            UNCHEATABLE: frame[UNCHEATABLE].to_numpy(float),
            TABLE9: frame[TABLE9].to_numpy(float),
        },
        series=frame["training_series"].astype(str).to_numpy(),
        policy_class=frame["policy_class"].astype(str).to_numpy(),
        group=frame["group_id"].astype(str).to_numpy(),
        row_id=frame["row_id"].astype(str).to_numpy(),
    )


def remove_fit_aliases(fit: Panel, query: Panel, tolerance: float = 1e-10) -> Panel:
    """Remove query coordinates that are exact policies in the fitting panel."""
    keep = np.ones(len(query), dtype=bool)
    for start in range(0, len(query), 128):
        stop = min(start + 128, len(query))
        distance = np.abs(query.phase0[start:stop, None] - fit.phase0[None]).sum(axis=2) + np.abs(
            query.phase1[start:stop, None] - fit.phase1[None]
        ).sum(axis=2)
        keep[start:stop] = distance.min(axis=1) > tolerance
    return query.subset(keep)


def phase_histograms(panel: Panel, basis: np.ndarray) -> np.ndarray:
    """Compose per-phase bucket policies with a row-stochastic content basis."""
    return np.stack([panel.phase0 @ basis, panel.phase1 @ basis], axis=1)


def sqrt_features(histograms: np.ndarray, phase_fractions: np.ndarray) -> np.ndarray:
    """Embed phase histograms on the Hellinger sphere."""
    assert histograms.ndim == 3
    assert histograms.shape[1] == len(phase_fractions)
    weighted = histograms * phase_fractions[None, :, None]
    features = np.sqrt(np.clip(weighted, 0.0, None)).reshape(len(histograms), -1)
    norms = (features**2).sum(axis=1)
    assert np.max(np.abs(norms - 1.0)) < 1e-6
    return features


def squared_hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pairwise squared Hellinger distance between unit Hellinger features."""
    return np.clip(1.0 - left @ right.T, 0.0, 1.0)


def folds_from_groups(groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rows = len(groups)
    if len(np.unique(groups)) >= n_splits and len(np.unique(groups)) < rows:
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(np.zeros((rows, 1)), groups=groups))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros((rows, 1))))


def solve_kernel(
    kernel_train: np.ndarray,
    target_train: np.ndarray,
    kernel_query_train: np.ndarray,
    ridge_alpha: float,
) -> np.ndarray:
    target_mean = float(target_train.mean())
    dual = np.linalg.solve(
        kernel_train + ridge_alpha * np.eye(len(kernel_train)),
        target_train - target_mean,
    )
    return target_mean + kernel_query_train @ dual


def select_hyperparameters(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> tuple[float, float, float]:
    """Select kernel bandwidth and ridge alpha by inner grouped CV."""
    distance = squared_hellinger(features, features)
    off_diagonal = distance[~np.eye(len(distance), dtype=bool)]
    positive = off_diagonal[off_diagonal > 1e-14]
    median = float(np.median(positive))
    gammas = np.asarray(GAMMA_FACTORS) / max(median, 1e-12)
    folds = folds_from_groups(groups, NESTED_FOLDS, seed)
    best: tuple[float, float, float] | None = None
    for gamma in gammas:
        kernel = np.exp(-gamma * distance)
        for ridge_alpha in RIDGE_ALPHAS:
            sse = 0.0
            for train_index, test_index in folds:
                predicted = solve_kernel(
                    kernel[np.ix_(train_index, train_index)],
                    target[train_index],
                    kernel[np.ix_(test_index, train_index)],
                    ridge_alpha,
                )
                sse += float(np.sum((predicted - target[test_index]) ** 2))
            candidate = (sse, float(gamma), float(ridge_alpha))
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return best[1], best[2], best[0]


def fit_kernel_model(
    panel: Panel,
    basis: np.ndarray,
    kernel_space: str,
    target_name: str,
    seed: int,
) -> KernelFit:
    """Fit a frozen KRR model and obtain leakage-free nested OOF predictions."""
    target = panel.targets[target_name]
    usable = np.isfinite(target)
    panel = panel.subset(usable)
    target = panel.targets[target_name]
    model_basis = basis if kernel_space == "content" else np.eye(len(panel.buckets))
    histograms = phase_histograms(panel, model_basis)
    phase_fractions = np.asarray([panel.alpha, 1.0 - panel.alpha])
    features = sqrt_features(histograms, phase_fractions)
    oof = np.empty(len(panel))
    outer_splits = grouped_splits(panel, NESTED_FOLDS, seed)
    for fold_index, (train_mask, test_mask) in enumerate(outer_splits):
        train_index = np.flatnonzero(train_mask)
        test_index = np.flatnonzero(test_mask)
        gamma, ridge_alpha, _ = select_hyperparameters(
            features[train_index],
            target[train_index],
            panel.group[train_index],
            seed + fold_index + 1,
        )
        distance_train = squared_hellinger(features[train_index], features[train_index])
        distance_test = squared_hellinger(features[test_index], features[train_index])
        oof[test_index] = solve_kernel(
            np.exp(-gamma * distance_train),
            target[train_index],
            np.exp(-gamma * distance_test),
            ridge_alpha,
        )
    gamma, ridge_alpha, cv_sse = select_hyperparameters(features, target, panel.group, seed)
    distance = squared_hellinger(features, features)
    kernel = np.exp(-gamma * distance)
    target_mean = float(target.mean())
    system = kernel + ridge_alpha * np.eye(len(panel))
    dual = np.linalg.solve(system, target - target_mean)
    gram_inverse = np.linalg.inv(system)
    return KernelFit(
        kernel_space=kernel_space,
        target=target_name,
        gamma=gamma,
        ridge_alpha=ridge_alpha,
        target_mean=target_mean,
        dual=dual,
        train_sqrt_features=features,
        train_phase_histograms=histograms,
        basis=model_basis,
        phase_fractions=phase_fractions,
        oof_predictions=oof,
        oof_rmse=float(np.sqrt(np.mean((oof - target) ** 2))),
        cv_sse=cv_sse,
        gram_inverse=gram_inverse,
    )


def predict_weights(
    fit: KernelFit,
    phase0: np.ndarray,
    phase1: np.ndarray,
    with_uncertainty: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Predict policies and return mean, nearest-fit distance, and optional GP std."""
    phase0 = np.atleast_2d(phase0)
    phase1 = np.atleast_2d(phase1)
    histograms = np.stack([phase0 @ fit.basis, phase1 @ fit.basis], axis=1)
    features = sqrt_features(histograms, fit.phase_fractions)
    distance = squared_hellinger(features, fit.train_sqrt_features)
    kernel = np.exp(-fit.gamma * distance)
    predicted = fit.target_mean + kernel @ fit.dual
    posterior_std = None
    if with_uncertainty:
        projected = kernel @ fit.gram_inverse
        variance = np.clip(1.0 - np.sum(projected * kernel, axis=1), 0.0, None)
        posterior_std = np.sqrt(variance)
    return predicted, distance.min(axis=1), posterior_std


def heldout_metrics(
    fit: KernelFit,
    panel: Panel,
    target_name: str,
    policy_class: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a frozen fit on coordinate-disjoint policy-matched heldouts."""
    mask = (panel.policy_class == policy_class) & np.isfinite(panel.targets[target_name])
    query = panel.subset(mask)
    predicted, support, posterior_std = predict_weights(fit, query.phase0, query.phase1, with_uncertainty=True)
    assert posterior_std is not None
    prediction_rows = pd.DataFrame(
        {
            "row_id": query.row_id,
            "series": query.series,
            "policy_class": query.policy_class,
            "target": target_name,
            "kernel_space": fit.kernel_space,
            "observed": query.targets[target_name],
            "predicted": predicted,
            "residual": predicted - query.targets[target_name],
            "nearest_fit_hellinger_sq": support,
            "posterior_std": posterior_std,
        }
    )
    rows = [
        {
            "split": "heldout",
            "stratum_type": "pooled",
            "stratum": policy_class,
            **metric_row(query.targets[target_name], predicted),
        }
    ]
    for series in sorted(set(query.series.tolist())):
        series_mask = query.series == series
        if int(series_mask.sum()) < 8:
            continue
        rows.append(
            {
                "split": "heldout",
                "stratum_type": "series",
                "stratum": str(series),
                **metric_row(query.targets[target_name][series_mask], predicted[series_mask]),
            }
        )
    return pd.DataFrame(rows), prediction_rows


def fit_loo_support_radius(fit: KernelFit, quantile: float) -> float:
    distance = squared_hellinger(fit.train_sqrt_features, fit.train_sqrt_features)
    np.fill_diagonal(distance, np.inf)
    return float(np.quantile(distance.min(axis=1), quantile))


def sample_candidate_bank(
    panel: Panel,
    target_name: str,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample local Dirichlet proposals and convex in-support mixtures."""
    rng = np.random.default_rng(seed)
    target = panel.targets[target_name]
    best_rows = np.argsort(target)[:20]
    centers = np.concatenate(
        [
            np.stack([panel.proportional, panel.proportional])[None],
            np.stack([panel.phase0[best_rows], panel.phase1[best_rows]], axis=1),
        ],
        axis=0,
    )
    random_count = max(0, count - len(panel) - 1)
    dirichlet_count = round(0.75 * random_count)
    convex_count = random_count - dirichlet_count

    center_index = rng.integers(0, len(centers), size=dirichlet_count)
    concentration = np.exp(rng.uniform(np.log(10.0), np.log(500.0), size=dirichlet_count))
    dirichlet = np.empty((dirichlet_count, 2, len(panel.buckets)), dtype=np.float32)
    for phase in range(2):
        shape = centers[center_index, phase] * concentration[:, None] + 0.05
        samples = rng.gamma(shape, 1.0)
        dirichlet[:, phase] = samples / samples.sum(axis=1, keepdims=True)

    convex = np.empty((convex_count, 2, len(panel.buckets)), dtype=np.float32)
    if convex_count:
        source = rng.integers(0, len(panel), size=(convex_count, 4))
        coefficients = rng.dirichlet(np.ones(4), size=convex_count)
        for phase, values in enumerate((panel.phase0, panel.phase1)):
            convex[:, phase] = np.sum(values[source] * coefficients[:, :, None], axis=1)

    phase0 = np.concatenate(
        [
            panel.phase0.astype(np.float32),
            panel.proportional[None].astype(np.float32),
            dirichlet[:, 0],
            convex[:, 0],
        ]
    )
    phase1 = np.concatenate(
        [
            panel.phase1.astype(np.float32),
            panel.proportional[None].astype(np.float32),
            dirichlet[:, 1],
            convex[:, 1],
        ]
    )
    kind = np.concatenate(
        [
            np.full(len(panel), "fit_policy"),
            np.asarray(["proportional"]),
            np.full(dirichlet_count, "local_dirichlet"),
            np.full(convex_count, "fit_convex_hull"),
        ]
    )
    return phase0, phase1, kind


def evaluate_bank(
    fit: KernelFit,
    phase0: np.ndarray,
    phase1: np.ndarray,
    chunk_size: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.empty(len(phase0))
    support = np.empty(len(phase0))
    for start in range(0, len(phase0), chunk_size):
        stop = min(start + chunk_size, len(phase0))
        predicted[start:stop], support[start:stop], _ = predict_weights(
            fit,
            phase0[start:stop],
            phase1[start:stop],
        )
    return predicted, support


def weights_to_free_logits(weights: np.ndarray) -> np.ndarray:
    clipped = np.clip(weights, 1e-12, None)
    return np.log(clipped[:, :-1] / clipped[:, -1:]).reshape(-1)


def free_logits_to_weights(logits: np.ndarray, bucket_count: int) -> np.ndarray:
    free = logits.reshape(2, bucket_count - 1)
    full = np.concatenate([free, np.zeros((2, 1))], axis=1)
    full -= full.max(axis=1, keepdims=True)
    exponential = np.exp(full)
    return exponential / exponential.sum(axis=1, keepdims=True)


def prediction_and_gradient(fit: KernelFit, logits: np.ndarray) -> tuple[float, np.ndarray]:
    """KRR prediction and analytic gradient in two simplex logit charts."""
    bucket_count = fit.basis.shape[0]
    weights = free_logits_to_weights(logits, bucket_count)
    histograms = weights @ fit.basis
    histograms = np.clip(histograms, 1e-15, None)
    phase_sqrt = np.sqrt(histograms)
    feature = np.sqrt(fit.phase_fractions)[:, None] * phase_sqrt
    flat_feature = feature.reshape(1, -1)
    distance = squared_hellinger(flat_feature, fit.train_sqrt_features)[0]
    kernel = np.exp(-fit.gamma * distance)
    predicted = fit.target_mean + float(kernel @ fit.dual)

    weighted_dual = fit.dual * kernel
    gradient_weights = np.empty_like(weights)
    train_phase_sqrt = np.sqrt(np.clip(fit.train_phase_histograms, 0.0, None))
    for phase in range(2):
        signal = weighted_dual @ train_phase_sqrt[:, phase]
        histogram_gradient = signal / phase_sqrt[phase]
        gradient_weights[phase] = 0.5 * fit.gamma * fit.phase_fractions[phase] * (fit.basis @ histogram_gradient)
    gradient_logits = weights * (gradient_weights - np.sum(gradient_weights * weights, axis=1, keepdims=True))
    return predicted, gradient_logits[:, :-1].reshape(-1)


def optimize_raw(
    fit: KernelFit,
    starts: np.ndarray,
    max_starts: int,
) -> tuple[np.ndarray, str]:
    """Optimize the unconstrained KRR mean over the product of phase simplices."""
    best_result = None
    for weights in starts[:max_starts]:
        result = minimize(
            lambda value: prediction_and_gradient(fit, value),
            weights_to_free_logits(weights),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 700, "ftol": 1e-12, "gtol": 1e-8},
        )
        if best_result is None or float(result.fun) < float(best_result.fun):
            best_result = result
    assert best_result is not None
    return free_logits_to_weights(best_result.x, fit.basis.shape[0]), str(best_result.message)


def candidate_from_weights(
    panel: Panel,
    fit: KernelFit,
    target_name: str,
    kind: str,
    weights: np.ndarray,
) -> Candidate:
    weights = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    weights /= weights.sum(axis=1, keepdims=True)
    predicted, support, posterior_std = predict_weights(
        fit,
        weights[0],
        weights[1],
        with_uncertainty=True,
    )
    assert posterior_std is not None
    aggregate = panel.alpha * weights[0] + (1.0 - panel.alpha) * weights[1]
    epochs = (
        TARGET_BUDGET_DOLMA3_COMMON_CRAWL
        * aggregate
        / np.asarray([TOP_LEVEL_DOMAIN_TOKEN_COUNTS[bucket] for bucket in panel.buckets])
    )
    return Candidate(
        target=target_name,
        candidate_kind=kind,
        predicted_bpb=float(predicted[0]),
        nearest_fit_hellinger_sq=float(support[0]),
        posterior_std=float(posterior_std[0]),
        max_weight=float(weights.max()),
        max_simulated_epoch=float(epochs.max()),
        phase_tv=float(0.5 * np.abs(weights[1] - weights[0]).sum()),
        phase_0_weights=weights[0],
        phase_1_weights=weights[1],
    )


def optimize_candidates(
    panel: Panel,
    fit: KernelFit,
    target_name: str,
    candidate_count: int,
    optimizer_starts: int,
    seed: int,
) -> tuple[list[Candidate], pd.DataFrame]:
    """Build raw, top-64-average, and support-radius proposal paths."""
    phase0, phase1, bank_kind = sample_candidate_bank(panel, target_name, candidate_count, seed)
    predicted, support = evaluate_bank(fit, phase0, phase1)
    order = np.argsort(predicted)
    starts = np.stack([phase0[order[:optimizer_starts]], phase1[order[:optimizer_starts]]], axis=1)
    raw_weights, optimizer_status = optimize_raw(fit, starts, optimizer_starts)

    top = order[: min(TOP_K_AVERAGE, len(order))]
    stable_weights = np.stack(
        [
            phase0[top].mean(axis=0),
            phase1[top].mean(axis=0),
        ]
    )
    radii = {
        f"hard_support_q{round(100 * quantile):02d}": fit_loo_support_radius(fit, quantile)
        for quantile in (0.50, 0.75, 0.90, 0.95, 1.00)
    }
    candidates = [
        candidate_from_weights(panel, fit, target_name, "raw_continuous", raw_weights),
        candidate_from_weights(panel, fit, target_name, f"top_{TOP_K_AVERAGE}_average", stable_weights),
    ]
    for kind, radius in radii.items():
        feasible = np.flatnonzero(support <= radius)
        assert len(feasible), f"no bank candidates inside {kind}"
        selected = feasible[np.argmin(predicted[feasible])]
        weights = np.stack([phase0[selected], phase1[selected]])
        candidates.append(candidate_from_weights(panel, fit, target_name, kind, weights))

    bank_summary = pd.DataFrame(
        {
            "bank_kind": bank_kind,
            "predicted_bpb": predicted,
            "nearest_fit_hellinger_sq": support,
        }
    )
    bank_summary.attrs["optimizer_status"] = optimizer_status
    return candidates, bank_summary


def candidate_health(panel: Panel, fit: KernelFit, candidate: Candidate) -> dict[str, object]:
    """Apply training-derived support and coordinate plausibility gates."""
    fit_max_weight = np.maximum(panel.phase0.max(axis=1), panel.phase1.max(axis=1))
    fit_max_epoch = panel.epochs.max(axis=1)
    support_p95 = fit_loo_support_radius(fit, 0.95)
    return {
        "support_p95": support_p95,
        "fit_max_weight": float(fit_max_weight.max()),
        "fit_max_epoch": float(fit_max_epoch.max()),
        "fit_max_phase_tv": float(panel.phase_tv.max()),
        "inside_support_p95": candidate.nearest_fit_hellinger_sq <= support_p95,
        "within_fit_max_weight": candidate.max_weight <= float(fit_max_weight.max()) + 1e-9,
        "within_fit_max_epoch": candidate.max_simulated_epoch <= float(fit_max_epoch.max()) + 1e-9,
        "within_fit_max_phase_tv": candidate.phase_tv <= float(panel.phase_tv.max()) + 1e-9,
    }


def selected_candidate(panel: Panel, fit: KernelFit, candidates: list[Candidate]) -> tuple[Candidate, str]:
    """Select raw if healthy, otherwise use the collaborator's stable top-64 proposal."""
    raw = next(candidate for candidate in candidates if candidate.candidate_kind == "raw_continuous")
    raw_health = candidate_health(panel, fit, raw)
    raw_ok = all(
        bool(raw_health[key])
        for key in (
            "inside_support_p95",
            "within_fit_max_weight",
            "within_fit_max_epoch",
            "within_fit_max_phase_tv",
        )
    )
    if raw_ok:
        return raw, "raw optimum passed all training-derived gates"
    stable = next(candidate for candidate in candidates if candidate.candidate_kind == f"top_{TOP_K_AVERAGE}_average")
    stable_health = candidate_health(panel, fit, stable)
    stable_ok = all(
        bool(stable_health[key])
        for key in (
            "inside_support_p95",
            "within_fit_max_weight",
            "within_fit_max_epoch",
            "within_fit_max_phase_tv",
        )
    )
    if stable_ok:
        return stable, "raw optimum failed a training-derived gate; used frozen top-64 averaging"
    fallback = next(candidate for candidate in candidates if candidate.candidate_kind == "hard_support_q95")
    return fallback, "raw and top-64 proposals failed training-derived gates; used hard q95 support candidate"


def mixture_frame(panel: Panel, candidate: Candidate) -> pd.DataFrame:
    weights = np.stack([candidate.phase_0_weights, candidate.phase_1_weights])
    aggregate = panel.alpha * weights[0] + (1.0 - panel.alpha) * weights[1]
    available = np.asarray([TOP_LEVEL_DOMAIN_TOKEN_COUNTS[bucket] for bucket in panel.buckets], dtype=float)
    proportional = available / available.sum()
    simulated_epochs = TARGET_BUDGET_DOLMA3_COMMON_CRAWL * aggregate / available
    frame = pd.DataFrame(
        {
            "domain": panel.buckets,
            "proportional": proportional,
            "phase_0_weight": weights[0],
            "phase_1_weight": weights[1],
            "aggregate_weight": aggregate,
            "available_tokens": available,
            "simulated_epochs": simulated_epochs,
            "phase_0_epoch_multiplier": weights[0] / proportional,
            "phase_1_epoch_multiplier": weights[1] / proportional,
            "phase_0_delta": weights[0] - proportional,
            "phase_1_delta": weights[1] - proportional,
        }
    )
    frame["max_abs_delta"] = frame[["phase_0_delta", "phase_1_delta"]].abs().max(axis=1)
    return frame


def write_candidate_artifacts(
    output_dir: Path,
    panel: Panel,
    fits: dict[str, KernelFit],
    candidate_sets: dict[str, list[Candidate]],
) -> pd.DataFrame:
    rows = []
    validation_rows = []
    for target_name, candidates in candidate_sets.items():
        fit = fits[target_name]
        selected, reason = selected_candidate(panel, fit, candidates)
        for candidate in candidates:
            health = candidate_health(panel, fit, candidate)
            rows.append(
                {
                    "target": target_name,
                    "candidate_kind": candidate.candidate_kind,
                    "selected_for_validation": candidate is selected,
                    "selection_reason": reason if candidate is selected else "",
                    "predicted_bpb": candidate.predicted_bpb,
                    "nearest_fit_hellinger_sq": candidate.nearest_fit_hellinger_sq,
                    "posterior_std": candidate.posterior_std,
                    "max_weight": candidate.max_weight,
                    "max_simulated_epoch": candidate.max_simulated_epoch,
                    "phase_tv": candidate.phase_tv,
                    **health,
                }
            )
            candidate_dir = output_dir / "candidates" / target_name / candidate.candidate_kind
            candidate_dir.mkdir(parents=True, exist_ok=True)
            mixture_frame(panel, candidate).to_csv(candidate_dir / "proposed_mixture_weights.csv", index=False)
        selected_dir = output_dir / "selected" / target_name
        selected_dir.mkdir(parents=True, exist_ok=True)
        mixture_frame(panel, selected).to_csv(selected_dir / "proposed_mixture_weights.csv", index=False)
        target_slug = "unch" if target_name == UNCHEATABLE else "t9"
        weights = np.stack([selected.phase_0_weights, selected.phase_1_weights])
        available = np.asarray([TOP_LEVEL_DOMAIN_TOKEN_COUNTS[bucket] for bucket in panel.buckets])
        nominal_aggregate = 0.8 * weights[0] + 0.2 * weights[1]
        nominal_epochs = TARGET_BUDGET_DOLMA3_COMMON_CRAWL * nominal_aggregate / available
        proportional = available / available.sum()
        mean_phase_tv_to_proportional = float(
            np.mean([0.5 * np.abs(weights[phase] - proportional).sum() for phase in range(2)])
        )
        validation_row: dict[str, object] = {
            "run_order": len(validation_rows),
            "run_id": 2_026_072_700 + len(validation_rows),
            "run_name": f"hkrr_{target_slug}_{selected.candidate_kind}",
            "source_experiment": "hellinger_krr_delphi_3e18_20260727",
            "panel_source": "hellinger_krr_validation",
            "proposal_target": target_name,
            "candidate_kind": selected.candidate_kind,
            "predicted_bpb": selected.predicted_bpb,
            "nearest_fit_hellinger_sq": selected.nearest_fit_hellinger_sq,
            "posterior_std": selected.posterior_std,
            "max_weight": selected.max_weight,
            "max_simulated_epoch": selected.max_simulated_epoch,
            "nominal_0p8_max_simulated_epoch": float(nominal_epochs.max()),
            "phase_tv": selected.phase_tv,
            "mean_phase_tv_to_proportional": mean_phase_tv_to_proportional,
            "fit_phase_0_fraction": panel.alpha,
            "fit_phase_1_fraction": 1.0 - panel.alpha,
            "data_seed": 2_026_072_700 + len(validation_rows),
            "trainer_seed": 0,
        }
        for phase, phase_weights in enumerate((selected.phase_0_weights, selected.phase_1_weights)):
            validation_row.update(
                {
                    f"phase_{phase}_{bucket}": float(weight)
                    for bucket, weight in zip(panel.buckets, phase_weights, strict=True)
                }
            )
        validation_rows.append(validation_row)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "candidate_manifest.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(output_dir / "validation_panel.csv", index=False)
    return manifest


def plot_calibration(predictions: pd.DataFrame, output_path: Path) -> None:
    model_order = ["weight", "content"]
    targets = [UNCHEATABLE, TABLE9]
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"{target.replace('_', ' ')} · {kernel}" for target in targets for kernel in model_order],
    )
    colors = {"single_phase_tied": "#e76f51", "two_phase": "#2a9d8f"}
    for row, target_name in enumerate(targets, start=1):
        for column, kernel_space in enumerate(model_order, start=1):
            subset = predictions[(predictions["target"] == target_name) & (predictions["kernel_space"] == kernel_space)]
            for policy_class in ("single_phase_tied", "two_phase"):
                policy = subset[subset["policy_class"] == policy_class]
                figure.add_trace(
                    go.Scatter(
                        x=policy["observed"],
                        y=policy["predicted"],
                        mode="markers",
                        name=policy_class.replace("_", " "),
                        legendgroup=policy_class,
                        showlegend=row == 1 and column == 1,
                        marker={"color": colors[policy_class], "size": 6, "opacity": 0.65},
                        customdata=np.stack([policy["row_id"], policy["series"]], axis=1),
                        hovertemplate=(
                            "%{customdata[0]}<br>%{customdata[1]}"
                            "<br>observed=%{x:.6f}<br>predicted=%{y:.6f}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=column,
                )
            if len(subset):
                lower = float(min(subset["observed"].min(), subset["predicted"].min()))
                upper = float(max(subset["observed"].max(), subset["predicted"].max()))
                figure.add_trace(
                    go.Scatter(
                        x=[lower, upper],
                        y=[lower, upper],
                        mode="lines",
                        line={"color": "#264653", "dash": "dash"},
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=column,
                )
    figure.update_layout(
        title="Delphi 3e18 Hellinger KRR heldout calibration",
        template="plotly_white",
        width=1300,
        height=1000,
    )
    figure.update_xaxes(title_text="Observed BPB")
    figure.update_yaxes(title_text="Predicted BPB")
    figure.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def plot_candidate_paths(manifest: pd.DataFrame, output_path: Path) -> None:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Uncheatable", "Table-9 macro"),
    )
    for column, target_name in enumerate((UNCHEATABLE, TABLE9), start=1):
        subset = manifest[manifest["target"] == target_name]
        figure.add_trace(
            go.Scatter(
                x=subset["nearest_fit_hellinger_sq"],
                y=subset["predicted_bpb"],
                mode="markers+text",
                text=subset["candidate_kind"],
                textposition="top center",
                marker={
                    "size": np.where(subset["selected_for_validation"], 16, 10),
                    "color": subset["phase_tv"],
                    "colorscale": "RdYlGn_r",
                    "showscale": column == 2,
                    "colorbar": {"title": "Phase TV"},
                },
                customdata=np.stack(
                    [
                        subset["max_weight"],
                        subset["max_simulated_epoch"],
                        subset["posterior_std"],
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "%{text}<br>predicted=%{y:.6f}<br>distance=%{x:.5f}"
                    "<br>max weight=%{customdata[0]:.3f}"
                    "<br>max epochs=%{customdata[1]:.2f}"
                    "<br>posterior std=%{customdata[2]:.4f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=column,
        )
    figure.update_layout(
        title="KRR raw and support-aware proposal audit",
        template="plotly_white",
        width=1400,
        height=650,
    )
    figure.update_xaxes(title_text="Nearest fit squared Hellinger distance")
    figure.update_yaxes(title_text="Predicted BPB")
    figure.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def fit_summary_row(fit: KernelFit, panel: Panel, policy_class: str) -> dict[str, object]:
    target = panel.targets[fit.target]
    metrics = metric_row(target, fit.oof_predictions)
    return {
        "kernel_space": fit.kernel_space,
        "target": fit.target,
        "policy_class": policy_class,
        "gamma": fit.gamma,
        "ridge_alpha": fit.ridge_alpha,
        "oof_rmse_selected": fit.oof_rmse,
        "cv_sse": fit.cv_sse,
        **metrics,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    two_phase_fit, all_heldouts = load_scale("delphi_3e18")
    content_basis, basis_provenance = load_embedding_basis(
        two_phase_fit.buckets,
        args.histogram_dir,
        args.lookup,
    )
    panels = {"two_phase": two_phase_fit}
    if not args.skip_one_phase:
        panels["single_phase_tied"] = one_phase_fit_panel(two_phase_fit)

    fit_rows = []
    metric_rows = []
    prediction_frames = []
    content_two_phase_fits: dict[str, KernelFit] = {}
    for policy_class, panel in panels.items():
        policy_heldout = all_heldouts.subset(all_heldouts.policy_class == policy_class)
        policy_heldout = remove_fit_aliases(panel, policy_heldout)
        for kernel_space in ("weight", "content"):
            for target_name in (UNCHEATABLE, TABLE9):
                fit = fit_kernel_model(panel, content_basis, kernel_space, target_name, args.seed)
                fit_rows.append(fit_summary_row(fit, panel, policy_class))
                metrics, predictions = heldout_metrics(
                    fit,
                    policy_heldout,
                    target_name,
                    policy_class,
                )
                metrics.insert(0, "policy_class", policy_class)
                metrics.insert(0, "target", target_name)
                metrics.insert(0, "kernel_space", kernel_space)
                metric_rows.append(metrics)
                prediction_frames.append(predictions)
                if policy_class == "two_phase" and kernel_space == "content":
                    content_two_phase_fits[target_name] = fit

    fit_summary = pd.DataFrame(fit_rows)
    heldout_metrics_frame = pd.concat(metric_rows, ignore_index=True)
    predictions_frame = pd.concat(prediction_frames, ignore_index=True)
    fit_summary.to_csv(args.output_dir / "fit_oof_metrics.csv", index=False)
    heldout_metrics_frame.to_csv(args.output_dir / "heldout_metrics.csv", index=False)
    predictions_frame.to_csv(args.output_dir / "heldout_predictions.csv", index=False)
    plot_calibration(predictions_frame, args.output_dir / "heldout_calibration.html")

    candidate_manifest = pd.DataFrame()
    if not args.skip_optimize:
        candidate_sets = {}
        for target_index, target_name in enumerate((UNCHEATABLE, TABLE9)):
            candidates, bank = optimize_candidates(
                two_phase_fit,
                content_two_phase_fits[target_name],
                target_name,
                args.candidate_count,
                args.optimizer_starts,
                args.seed + 1000 * target_index,
            )
            candidate_sets[target_name] = candidates
            bank.groupby("bank_kind").agg(
                count=("predicted_bpb", "size"),
                predicted_min=("predicted_bpb", "min"),
                predicted_median=("predicted_bpb", "median"),
                support_median=("nearest_fit_hellinger_sq", "median"),
                support_max=("nearest_fit_hellinger_sq", "max"),
            ).reset_index().to_csv(args.output_dir / f"{target_name}_candidate_bank_summary.csv", index=False)
        candidate_manifest = write_candidate_artifacts(
            args.output_dir,
            two_phase_fit,
            content_two_phase_fits,
            candidate_sets,
        )
        plot_candidate_paths(candidate_manifest, args.output_dir / "candidate_support_paths.html")

    run_provenance = {
        "fit_data": {
            "two_phase_sha256": sha256(CANONICAL / "delphi_3e18_two_phase_fit.csv"),
            "one_phase_sha256": sha256(ONE_PHASE_FIT),
        },
        "basis": basis_provenance,
        "kernel": {
            "distance": "phase-fraction-weighted squared Hellinger",
            "gamma_factors": GAMMA_FACTORS,
            "ridge_alphas": RIDGE_ALPHAS,
            "nested_folds": NESTED_FOLDS,
        },
        "proposal": {
            "candidate_count": args.candidate_count,
            "optimizer_starts": args.optimizer_starts,
            "top_k_average": TOP_K_AVERAGE,
            "seed": args.seed,
        },
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(run_provenance, indent=2) + "\n")

    print("\nOOF summary")
    print(
        fit_summary[
            ["policy_class", "target", "kernel_space", "rmse", "spearman", "regret_at_1", "gamma", "ridge_alpha"]
        ].to_string(index=False)
    )
    pooled = heldout_metrics_frame[heldout_metrics_frame["stratum_type"] == "pooled"]
    print("\nHeldout summary")
    print(
        pooled[
            [
                "policy_class",
                "target",
                "kernel_space",
                "n",
                "rmse",
                "spearman",
                "calibration_slope",
                "regret_at_1",
                "optimism_over_0p05",
            ]
        ].to_string(index=False)
    )
    if len(candidate_manifest):
        print("\nCandidate audit")
        print(
            candidate_manifest[
                [
                    "target",
                    "candidate_kind",
                    "selected_for_validation",
                    "predicted_bpb",
                    "nearest_fit_hellinger_sq",
                    "inside_support_p95",
                    "max_weight",
                    "max_simulated_epoch",
                    "phase_tv",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
