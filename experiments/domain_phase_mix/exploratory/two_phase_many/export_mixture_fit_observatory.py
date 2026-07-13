# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Export the multi-swarm mixture-fit observatory data bundle.

The existing 300M debugger remains the source of truth for its carefully
accounted fit/heldout split and grouped OOF predictions. This exporter wraps
that bundle, fits the same surrogate family on the two StarCoder surfaces and
the production Grug-MoE swarm, and emits semantic parameter records for the
Fit Explorer.

Every expensive swarm/model result is cached independently. Re-running this
script skips complete fits and regenerates only stale outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.special import softplus
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_original_separate_heads_policy_ablation_300m as separate_heads,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_nested_coverage_dsp as coverage_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_debugger_300m as legacy_exporter,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search import (  # noqa: E402
    generic_family_penalty_calibration as grp_calibration,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search import (  # noqa: E402
    starcoder_grp,
)

SCRIPT_DIR = Path(__file__).resolve().parent
APP_DATA = SCRIPT_DIR / "mixture_fit_debugger/src/generated/dashboard_data.json"
LEGACY_DATA = SCRIPT_DIR / "reference_outputs/mixture_fit_debugger_300m_v1/dashboard_data.json"
CACHE_DIR = SCRIPT_DIR / "reference_outputs/mixture_fit_observatory_cache_20260713"
COSINE_DATA = SCRIPT_DIR.parent / "paper_plots/data/two_phase_starcoder_combined_143_from_wandb.csv"
WSD80_DATA = SCRIPT_DIR / "reference_outputs/starcoder_wsd80_surface_analysis_20260711/wsd80_observed_metrics.csv"
PRODUCTION_DATA = SCRIPT_DIR / (
    "reference_outputs/grug_moe_production_swarm_results_20260704/production_swarm_840_wide.csv"
)
PRODUCTION_MODEL = SCRIPT_DIR / (
    "reference_outputs/grug_moe_production_swarm_effective_exposure_dsp_uncheatable_20260705/model.json"
)
STARCODER_TARGET_COLUMN = "eval/paloma/dolma_100_programing_languages/bpb"
STARCODER_DOMAINS = ["nemotron_full", "starcoder"]
MODEL_IDS = ("canonical", "effective_exposure", "effective_exposure_geometry", "separate_heads", "grp")
MODEL_LABELS = {
    "canonical": "Canonical DSP",
    "effective_exposure": "Effective-exposure DSP",
    "effective_exposure_geometry": "Eff-exp DSP + geometry",
    "separate_heads": "Separate heads",
    "grp": "GRP (regularized)",
}
MODEL_DESCRIPTIONS = {
    "canonical": "Phase-1 share changes benefit, while overexposure uses raw total exposure.",
    "effective_exposure": "A shared phase-1 multiplier changes both saturation and overexposure exposure.",
    "effective_exposure_geometry": "Effective-exposure DSP plus phase divergence and concentration features.",
    "separate_heads": "Independent phase-specific asymmetric exposure bowls.",
    "grp": "Retained exposure, grouped response features, and explicit overexposure penalties.",
}
SEPARATE_L2_GRID = (0.03, 0.1, 0.3, 1.0, 1.5, 3.0)
PRODUCTION_GRP_L2_GRID = (0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 3.0)
GRP_SHAPE_PARAMS = legacy_exporter.GRP_SHAPE_PARAMS
CACHE_VERSION = "mixture-fit-observatory-v2"


@dataclass(frozen=True)
class FittedResult:
    model: Any
    prediction: np.ndarray
    full_prediction: np.ndarray
    fit_detail: dict[str, Any]
    nike_swoosh: dict[str, Any] | None = None


class Predictable(Protocol):
    def predict(self, weights: np.ndarray) -> np.ndarray: ...


class UngroupedGRP:
    """GRP ablation with no semantic family or pair structure."""

    def __init__(
        self,
        dataset: pooled.Dataset,
        *,
        exponent: float,
        eta: float,
        retention_lambda: float,
        threshold: float,
        l2: float,
    ):
        self.dataset = dataset
        self.exponent = float(exponent)
        self.eta = float(eta)
        self.retention_lambda = float(retention_lambda)
        self.threshold = float(threshold)
        self.l2 = float(l2)
        self.intercept: float | None = None
        self.signal_coef: np.ndarray | None = None
        self.penalty_coef: np.ndarray | None = None

    def _design(self, weights: np.ndarray) -> np.ndarray:
        p0 = weights[:, 0, :]
        p1 = weights[:, 1, :]
        e0 = p0 * self.dataset.c0[None, :]
        e1 = p1 * self.dataset.c1[None, :]
        exposure = np.exp(-self.retention_lambda * (1.0 - p1)) * e0 + self.eta * e1
        signal = np.maximum(exposure, 1e-12) ** self.exponent
        penalty = softplus(np.log1p(exposure) - self.threshold) ** 2
        return np.hstack([-signal, penalty])

    def fit(self, indices: np.ndarray) -> UngroupedGRP:
        design = self._design(self.dataset.weights[indices])
        target = self.dataset.y[indices]
        design_mean = design.mean(axis=0, keepdims=True)
        target_mean = float(target.mean())
        centered = design - design_mean
        centered_target = target - target_mean
        if self.l2 > 0.0:
            centered = np.vstack([centered, np.sqrt(self.l2) * np.eye(design.shape[1])])
            centered_target = np.concatenate([centered_target, np.zeros(design.shape[1])])
        coef, _residual = nnls(centered, centered_target, maxiter=20 * design.shape[1])
        self.intercept = target_mean - float((design_mean @ coef).item())
        self.signal_coef = np.asarray(coef[: self.dataset.m], dtype=float)
        self.penalty_coef = np.asarray(coef[self.dataset.m :], dtype=float)
        return self

    def predict(self, weights: np.ndarray) -> np.ndarray:
        if self.intercept is None or self.signal_coef is None or self.penalty_coef is None:
            raise RuntimeError("Ungrouped GRP must be fit before prediction")
        design = self._design(weights)
        coef = np.concatenate([self.signal_coef, self.penalty_coef])
        return np.asarray(self.intercept + design @ coef, dtype=float)


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def file_fingerprint(paths: list[Path], payload: dict[str, Any]) -> str:
    inputs = {
        str(path.relative_to(REPO_ROOT)): {"size": path.stat().st_size, "mtimeNs": path.stat().st_mtime_ns}
        for path in paths
    }
    return hashlib.sha256(
        json.dumps({"version": CACHE_VERSION, "inputs": inputs, **payload}, sort_keys=True).encode()
    ).hexdigest()


def parameter(
    key: str,
    symbol: str,
    value: float,
    role: str,
    *,
    scope: str = "global",
    domain_id: str | None = None,
    group_label: str | None = None,
    transformed_value: float | None = None,
    transformed_label: str | None = None,
    unit: str | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "symbol": symbol,
        "value": safe_float(value),
        "role": role,
        "scope": scope,
        "domainId": domain_id,
        "groupLabel": group_label,
        "transformedValue": safe_float(transformed_value),
        "transformedLabel": transformed_label,
        "unit": unit,
    }


def metric_summary(observed: np.ndarray, prediction: np.ndarray) -> dict[str, float | int | None]:
    valid = np.isfinite(observed) & np.isfinite(prediction)
    if valid.sum() < 3:
        return {"n": int(valid.sum()), "rmse": None, "mae": None, "spearman": None}
    residual = prediction[valid] - observed[valid]
    return {
        "n": int(valid.sum()),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "spearman": float(spearmanr(observed[valid], prediction[valid]).statistic),
    }


def phase_fractions(dataset: pooled.Dataset) -> tuple[float, float]:
    return coverage_dsp.phase_fractions(dataset)


def natural_weights(dataset: pooled.Dataset, alpha0: float) -> np.ndarray:
    phase0_tokens = alpha0 / np.maximum(dataset.c0, 1e-12)
    phase1_tokens = (1.0 - alpha0) / np.maximum(dataset.c1, 1e-12)
    token_proxy = 0.5 * (phase0_tokens + phase1_tokens)
    return token_proxy / token_proxy.sum()


def target_budget(dataset: pooled.Dataset, alpha0: float, known_budget: float | None = None) -> float:
    if known_budget is not None:
        return float(known_budget)
    natural = natural_weights(dataset, alpha0)
    implied = dataset.c0 * natural / max(alpha0, 1e-12)
    scale = float(np.median(implied))
    return scale


def folds(dataset: pooled.Dataset, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if dataset.name.startswith("300m_"):
        return pooled.dataset_folds(dataset, seed, n_splits=5)
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    return [(train, test) for train, test in splitter.split(np.arange(dataset.n))]


def dsp_fit(dataset: pooled.Dataset, indices: np.ndarray, model_id: str) -> coverage_dsp.CoverageModel:
    config = coverage_dsp.FitConfig(
        name=model_id,
        use_coverage=model_id == "effective_exposure_geometry",
        variant_name="canonical" if model_id == "canonical" else "effective_exposure",
    )
    if dataset.name.startswith("production"):
        linear_reg, maxiter, top_k = 1e-6, 0, 3
    elif dataset.name.startswith("starcoder"):
        linear_reg, maxiter, top_k = 0.01, 24, 3
    else:
        linear_reg, maxiter, top_k = legacy_exporter.DSP_LINEAR_REG, legacy_exporter.DSP_MAXITER, 3
    return coverage_dsp.fit_model(
        dataset,
        indices,
        config,
        linear_reg=linear_reg,
        maxiter=maxiter,
        coarse_top_k=top_k,
    )


def dsp_predict(model: coverage_dsp.CoverageModel, dataset: pooled.Dataset, weights: np.ndarray) -> np.ndarray:
    alpha0, alpha1 = phase_fractions(dataset)
    return coverage_dsp.predict(model, weights, alpha0, alpha1)


def select_separate_l2(dataset: pooled.Dataset) -> tuple[float, list[dict[str, float]]]:
    if dataset.name == "300m_uncheatable":
        return 1.0, []
    if dataset.name == "300m_table9":
        return 1.5, []
    packet = coverage_dsp.packet(dataset, np.arange(dataset.n))
    rows = []
    for l2 in SEPARATE_L2_GRID:
        prediction = np.full(dataset.n, np.nan, dtype=float)
        for train, test in folds(dataset, seed=0):
            model = separate_heads.fit_separate_heads(packet, train, l2)
            prediction[test] = separate_heads.predict_separate_heads(model, packet, dataset.weights[test])
        rows.append({"l2": l2, "oofRmse": metric_summary(dataset.y, prediction)["rmse"]})
    selected = min(rows, key=lambda row: (float(row["oofRmse"]), row["l2"]))
    return float(selected["l2"]), rows


def separate_fit(dataset: pooled.Dataset, indices: np.ndarray, l2: float) -> separate_heads.SeparateHeadsModel:
    packet = coverage_dsp.packet(dataset, np.arange(dataset.n))
    return separate_heads.fit_separate_heads(packet, indices, l2)


def separate_predict(
    model: separate_heads.SeparateHeadsModel,
    dataset: pooled.Dataset,
    weights: np.ndarray,
) -> np.ndarray:
    packet = coverage_dsp.packet(dataset, np.arange(dataset.n))
    return separate_heads.predict_separate_heads(model, packet, weights)


def starcoder_grp_packet(dataset: pooled.Dataset) -> dsp.PacketData:
    return coverage_dsp.packet(dataset, np.arange(dataset.n))


def starcoder_grp_fit(dataset: pooled.Dataset, indices: np.ndarray, params: dict[str, float] | None = None):
    packet = starcoder_grp_packet(dataset)
    subset = dsp.PacketData(
        frame=packet.frame.iloc[indices].reset_index(drop=True),
        name_col=packet.name_col,
        y=packet.y[indices],
        w=packet.w[indices],
        m=packet.m,
        c0=packet.c0,
        c1=packet.c1,
        domain_names=list(packet.domain_names),
    )
    return starcoder_grp.fit_starcoder_grp(subset, params=params, seed=0)


def production_grp_params(l2: float) -> dict[str, float]:
    exponent = float(np.mean([GRP_SHAPE_PARAMS[f"a_{family}"] for family in ("broad_text", "tech_code", "reasoning")]))
    threshold = float(
        np.median([GRP_SHAPE_PARAMS[f"tau_{family}"] for family in ("broad_text", "tech_code", "reasoning")])
    )
    return {
        "exponent": exponent,
        "eta": float(GRP_SHAPE_PARAMS["eta"]),
        "retention_lambda": float(GRP_SHAPE_PARAMS["lam"]),
        "threshold": threshold,
        "l2": float(l2),
    }


def select_production_grp_l2(dataset: pooled.Dataset) -> tuple[float, list[dict[str, float]]]:
    rows = []
    for l2 in PRODUCTION_GRP_L2_GRID:
        params = production_grp_params(l2)
        prediction = np.full(dataset.n, np.nan, dtype=float)
        for train, test in folds(dataset, seed=0):
            model = UngroupedGRP(dataset, **params).fit(train)
            prediction[test] = model.predict(dataset.weights[test])
        rows.append({"l2": l2, "oofRmse": metric_summary(dataset.y, prediction)["rmse"]})
    selected = min(rows, key=lambda row: (float(row["oofRmse"]), row["l2"]))
    return float(selected["l2"]), rows


def grp_300m_fit(dataset: pooled.Dataset, indices: np.ndarray, l2: float):
    packet = legacy_exporter.grp_packet(dataset)
    return grp_calibration.build_penalty_calibration_surrogate(
        packet,
        params=legacy_exporter.grp_params(l2),
        variant_name=legacy_exporter.GRP_VARIANT,
    ).fit(dataset.weights[indices], dataset.y[indices])


def fit_one_model(
    dataset: pooled.Dataset,
    model_id: str,
    seeds: tuple[int, ...],
    *,
    legacy_model_summary: dict[str, Any] | None = None,
) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any]]:
    all_indices = np.arange(dataset.n)
    tuning: dict[str, Any] = {}
    if model_id in {"canonical", "effective_exposure", "effective_exposure_geometry"}:
        full_model = dsp_fit(dataset, all_indices, model_id)

        def fold_predict(train: np.ndarray, test: np.ndarray) -> np.ndarray:
            return dsp_predict(dsp_fit(dataset, train, model_id), dataset, dataset.weights[test])

        full_prediction = dsp_predict(full_model, dataset, dataset.weights)
    elif model_id == "separate_heads":
        l2, sweep = select_separate_l2(dataset)
        tuning = {"l2": l2, "l2Sweep": sweep}
        full_model = separate_fit(dataset, all_indices, l2)

        def fold_predict(train: np.ndarray, test: np.ndarray) -> np.ndarray:
            return separate_predict(separate_fit(dataset, train, l2), dataset, dataset.weights[test])

        full_prediction = separate_predict(full_model, dataset, dataset.weights)
    elif dataset.name.startswith("300m_"):
        if legacy_model_summary is None:
            raise ValueError("300M GRP requires the legacy selected-L2 summary")
        l2 = float(legacy_model_summary["l2"])
        tuning = {"l2": l2, "l2Sweep": legacy_model_summary.get("l2Sweep", [])}
        full_model = grp_300m_fit(dataset, all_indices, l2)

        def fold_predict(train: np.ndarray, test: np.ndarray) -> np.ndarray:
            return grp_300m_fit(dataset, train, l2).predict(dataset.weights[test])

        full_prediction = full_model.predict(dataset.weights)
    elif dataset.name.startswith("starcoder"):
        params, full_model = starcoder_grp_fit(dataset, all_indices)
        tuning = {"shapeParameters": params, "oofShapeProtocol": "Full-fit shape; fold-refit linear head"}

        def fold_predict(train: np.ndarray, test: np.ndarray) -> np.ndarray:
            _params, model = starcoder_grp_fit(dataset, train, params=params)
            return model.predict(dataset.weights[test])

        full_prediction = full_model.predict(dataset.weights)
    elif dataset.name == "production_uncheatable":
        l2, sweep = select_production_grp_l2(dataset)
        params = production_grp_params(l2)
        tuning = {
            "l2": l2,
            "l2Sweep": sweep,
            "shapeParameters": params,
            "ablation": "No family or pair grouping; 300M GRP shape transferred and ridge retuned.",
        }
        full_model = UngroupedGRP(dataset, **params).fit(all_indices)

        def fold_predict(train: np.ndarray, test: np.ndarray) -> np.ndarray:
            return UngroupedGRP(dataset, **params).fit(train).predict(dataset.weights[test])

        full_prediction = full_model.predict(dataset.weights)
    else:
        raise ValueError(f"Unsupported model {model_id!r} for {dataset.name!r}")

    seed_predictions = []
    for seed in seeds:
        oof = np.full(dataset.n, np.nan, dtype=float)
        for train, test in folds(dataset, seed):
            oof[test] = fold_predict(train, test)
        if not np.isfinite(oof).all():
            raise ValueError(f"Incomplete OOF prediction for {dataset.name}/{model_id}/seed={seed}")
        seed_predictions.append(oof)
    prediction = np.mean(seed_predictions, axis=0)
    return full_model, prediction, full_prediction, tuning


def dsp_parameters(
    model: coverage_dsp.CoverageModel,
    dataset: pooled.Dataset,
    model_id: str,
) -> list[dict[str, Any]]:
    base = model.base
    records = [
        parameter(
            "intercept",
            "b_0",
            base.intercept,
            "Loss level after centering all response features.",
            unit="BPB",
        )
    ]
    for key, value in base.params.items():
        if isinstance(value, np.ndarray):
            continue
        if key == "gamma":
            role = (
                "Relative phase-1 premium on the benefit term."
                if model_id == "canonical"
                else "Phase-1 epoch value relative to a phase-0 epoch in effective exposure."
            )
        else:
            role = "Global nonlinear phase-response parameter."
        records.append(parameter(key, key.replace("_", " "), float(value), role))

    rho = np.asarray(base.params["rho"], dtype=float)
    tau = np.asarray(base.params.get("tau", np.zeros(dataset.m)), dtype=float)
    for index, domain in enumerate(dataset.domain_names):
        records.extend(
            [
                parameter(
                    f"rho:{domain}",
                    "rho",
                    rho[index],
                    "Saturation rate of useful exposure; larger values saturate sooner.",
                    scope="domain",
                    domain_id=domain,
                    transformed_value=float(np.log(2.0) / max(rho[index], 1e-12)),
                    transformed_label="Half-saturation exposure",
                    unit="effective epochs",
                ),
                parameter(
                    f"tau:{domain}",
                    "tau",
                    tau[index],
                    "Log-exposure threshold where the soft overexposure penalty turns on.",
                    scope="domain",
                    domain_id=domain,
                    transformed_value=float(np.expm1(tau[index])),
                    transformed_label="Penalty-onset exposure",
                    unit="effective epochs",
                ),
                parameter(
                    f"benefit:{domain}",
                    "a",
                    base.benefit_coef[index],
                    "Maximum fitted BPB reduction supplied by this bucket's saturation feature.",
                    scope="domain",
                    domain_id=domain,
                    unit="BPB",
                ),
                parameter(
                    f"penalty:{domain}",
                    "p",
                    base.penalty_coef[index],
                    "Strength of this bucket's overexposure penalty.",
                    scope="domain",
                    domain_id=domain,
                    unit="BPB",
                ),
            ]
        )
    if model_id == "effective_exposure_geometry":
        for key, symbol, value, role in (
            (
                "geometry:phase_tv",
                "theta_TV",
                model.coverage_coef[0],
                "Global cost assigned to total-variation distance between phase mixtures.",
            ),
            (
                "geometry:aggregate_hhi",
                "theta_agg",
                model.coverage_coef[1],
                "Global cost assigned to concentration of aggregate exposure.",
            ),
            (
                "geometry:phase1_hhi",
                "theta_1",
                model.coverage_coef[2],
                "Global cost assigned to concentration in the late phase.",
            ),
        ):
            records.append(parameter(key, symbol, value, role, unit="BPB"))
    return records


def separate_parameters(
    model: separate_heads.SeparateHeadsModel,
    dataset: pooled.Dataset,
) -> list[dict[str, Any]]:
    m = dataset.m
    coef = np.asarray(model.coefficients, dtype=float)
    records = [
        parameter(
            "intercept",
            "b_0",
            model.intercept,
            "Loss level after centering the phase-head features.",
            unit="BPB",
        ),
        parameter("l2", "lambda_L2", model.l2, "Ridge shrinkage applied to all phase-head coefficients."),
    ]
    blocks = (
        ("phase0_under", "a0-", coef[:m], "Phase-0 underexposure curvature."),
        ("phase0_over", "a0+", coef[m : 2 * m], "Phase-0 overexposure curvature."),
        ("phase1_under", "a1-", coef[2 * m : 3 * m], "Phase-1 underexposure curvature."),
        ("phase1_over", "a1+", coef[3 * m :], "Phase-1 overexposure curvature."),
    )
    for index, domain in enumerate(dataset.domain_names):
        for phase, mu in ((0, model.mu0[index]), (1, model.mu1[index])):
            records.append(
                parameter(
                    f"mu{phase}:{domain}",
                    f"mu_{phase}",
                    mu,
                    f"Center of the phase-{phase} asymmetric exposure bowl.",
                    scope="domain",
                    domain_id=domain,
                    transformed_value=float(np.expm1(mu)),
                    transformed_label="Preferred exposure",
                    unit="epochs",
                )
            )
        for key, symbol, values, role in blocks:
            records.append(
                parameter(
                    f"{key}:{domain}",
                    symbol,
                    values[index],
                    role,
                    scope="domain",
                    domain_id=domain,
                    unit="BPB / log-epoch squared",
                )
            )
    return records


def grp_300m_parameters(model: Any, dataset: pooled.Dataset, l2: float) -> list[dict[str, Any]]:
    packet = model.packet
    records = [
        parameter("intercept", "b_0", model.intercept_, "Loss level after centering all GRP features.", unit="BPB"),
    ]
    for key, value in model.params.items():
        transformed = None
        transformed_label = None
        role = "GRP nonlinear shape parameter."
        if key.startswith("a_"):
            role = "Power-law response exponent; smaller values imply faster diminishing returns."
        elif key.startswith("tau_"):
            transformed = float(np.expm1(value))
            transformed_label = "Penalty-onset exposure"
            role = "Log-exposure threshold where the family overexposure penalty turns on."
        elif key == "eta":
            role = "Phase-1 epoch value relative to one retained phase-0 epoch."
        elif key == "lam":
            role = "Forgetting rate applied to phase-0 exposure as phase-1 mass moves away from a bucket."
        elif key == "beta":
            role = "Discount applied to low-quality Common Crawl within a paired topic."
        elif key == "reg":
            role = "Ridge shrinkage applied to all linear GRP coefficients."
        records.append(
            parameter(
                key,
                key.replace("_", " "),
                float(value),
                role,
                transformed_value=transformed,
                transformed_label=transformed_label,
                unit="effective epochs" if transformed is not None else None,
            )
        )
    parts = model.components()
    for coefficient, domain_index in zip(parts["singleton_coef"], packet.singletons, strict=True):
        domain = dataset.domain_names[domain_index]
        records.append(
            parameter(
                f"signal:{domain}",
                "beta_signal",
                coefficient,
                "BPB reduction coefficient on this singleton bucket's power-law signal.",
                scope="domain",
                domain_id=domain,
                unit="BPB",
            )
        )
    for coefficient, (high, low), topic in zip(parts["pair_coef"], packet.pairs, packet.pair_topics, strict=True):
        records.append(
            parameter(
                f"pair:{topic}",
                "beta_pair",
                coefficient,
                "Joint signal coefficient for the high/low-quality Common Crawl topic pair.",
                scope="group",
                group_label=f"CC pair · {topic}",
                unit="BPB",
            )
        )
        for domain_index in (high, low):
            records.append(
                parameter(
                    f"pair-member:{topic}:{dataset.domain_names[domain_index]}",
                    "beta_pair",
                    coefficient,
                    "Shared pair coefficient; shown on both member buckets for inspection.",
                    scope="domain",
                    domain_id=dataset.domain_names[domain_index],
                    group_label=f"CC pair · {topic}",
                    unit="BPB",
                )
            )
    for family, coefficient in parts["family_coef"].items():
        records.append(
            parameter(
                f"family-signal:{family}",
                "beta_family",
                coefficient,
                "Signal coefficient on total retained exposure within this family.",
                scope="group",
                group_label=family.replace("_", " ").title(),
                unit="BPB",
            )
        )
    for family, coefficient in parts["family_group_penalty_coef"].items():
        records.append(
            parameter(
                f"family-penalty:{family}",
                "beta_penalty",
                coefficient,
                "Strength of the summed within-family overexposure penalty.",
                scope="group",
                group_label=family.replace("_", " ").title(),
                unit="BPB",
            )
        )
    if not math.isclose(float(model.params["reg"]), l2):
        raise ValueError("GRP model and selected ridge differ")
    return records


def starcoder_grp_parameters(model: Any, params: dict[str, float], dataset: pooled.Dataset) -> list[dict[str, Any]]:
    if model.intercept_ is None or model.coef_ is None:
        raise RuntimeError("StarCoder GRP is not fit")
    records = [parameter("intercept", "b_0", model.intercept_, "Loss level after centering GRP features.", unit="BPB")]
    roles = {
        "alpha": "Scale inside the log-satiation signal for both corpora.",
        "eta": "Phase-1 epoch value relative to one retained phase-0 epoch.",
        "lam": "Forgetting rate applied to phase-0 exposure.",
        "tau": "Log-exposure threshold for the aggregate overexposure penalty.",
        "reg": "Ridge shrinkage applied to the three linear coefficients.",
    }
    for key, value in params.items():
        records.append(
            parameter(
                key,
                key,
                value,
                roles[key],
                transformed_value=float(np.expm1(value)) if key == "tau" else None,
                transformed_label="Penalty-onset exposure" if key == "tau" else None,
                unit="effective epochs" if key == "tau" else None,
            )
        )
    for index, domain in enumerate(dataset.domain_names):
        records.append(
            parameter(
                f"signal:{domain}",
                "beta_signal",
                model.coef_[index],
                "BPB reduction coefficient on this corpus's retained-exposure signal.",
                scope="domain",
                domain_id=domain,
                unit="BPB",
            )
        )
    records.append(
        parameter(
            "penalty",
            "beta_penalty",
            model.coef_[2],
            "Strength of the summed two-corpus overexposure penalty.",
            unit="BPB",
        )
    )
    return records


def production_grp_parameters(model: UngroupedGRP, dataset: pooled.Dataset) -> list[dict[str, Any]]:
    if model.intercept is None or model.signal_coef is None or model.penalty_coef is None:
        raise RuntimeError("Production GRP ablation is not fit")
    records = [
        parameter("intercept", "b_0", model.intercept, "Loss level after centering ungrouped GRP features.", unit="BPB"),
        parameter(
            "a",
            "a",
            model.exponent,
            "Shared power-law exponent; semantic family-specific exponents are ablated.",
        ),
        parameter("eta", "eta", model.eta, "Phase-1 epoch value relative to one retained phase-0 epoch."),
        parameter("lambda", "lambda", model.retention_lambda, "Shared phase-0 forgetting rate."),
        parameter(
            "tau",
            "tau",
            model.threshold,
            "Shared log-exposure penalty threshold; semantic family thresholds are ablated.",
            transformed_value=float(np.expm1(model.threshold)),
            transformed_label="Penalty-onset exposure",
            unit="effective epochs",
        ),
        parameter("l2", "lambda_L2", model.l2, "Ridge shrinkage selected by production-swarm OOF RMSE."),
    ]
    for index, domain in enumerate(dataset.domain_names):
        records.extend(
            [
                parameter(
                    f"signal:{domain}",
                    "beta_signal",
                    model.signal_coef[index],
                    "BPB reduction coefficient for this ungrouped bucket.",
                    scope="domain",
                    domain_id=domain,
                    unit="BPB",
                ),
                parameter(
                    f"penalty:{domain}",
                    "beta_penalty",
                    model.penalty_coef[index],
                    "Overexposure-penalty coefficient for this ungrouped bucket.",
                    scope="domain",
                    domain_id=domain,
                    unit="BPB",
                ),
            ]
        )
    return records


def parameter_records(
    model: Any,
    dataset: pooled.Dataset,
    model_id: str,
    tuning: dict[str, Any],
) -> list[dict[str, Any]]:
    if model_id in {"canonical", "effective_exposure", "effective_exposure_geometry"}:
        return dsp_parameters(model, dataset, model_id)
    if model_id == "separate_heads":
        return separate_parameters(model, dataset)
    if dataset.name.startswith("300m_"):
        return grp_300m_parameters(model, dataset, float(tuning["l2"]))
    if dataset.name.startswith("starcoder"):
        return starcoder_grp_parameters(model, tuning["shapeParameters"], dataset)
    if dataset.name == "production_uncheatable":
        return production_grp_parameters(model, dataset)
    raise ValueError(f"Unsupported parameter extraction for {dataset.name}/{model_id}")


def model_caveats(dataset: pooled.Dataset, model_id: str) -> list[str]:
    caveats: list[str] = []
    if dataset.name == "production_uncheatable" and model_id == "grp":
        caveats.append(
            "Ungrouped GRP ablation: the production partition has no a priori family or pair grouping. "
            "The 300M GRP response shape is transferred, family features are removed, and ridge is retuned."
        )
    if dataset.name.startswith("starcoder") and model_id == "grp":
        caveats.append("Two-family StarCoder GRP; broad and code each supply one retained-exposure signal.")
    if model_id == "effective_exposure_geometry":
        caveats.append("The three global geometry coefficients are nonnegative and fit jointly with the DSP head.")
    return caveats


def fit_detail(
    dataset: pooled.Dataset,
    model_id: str,
    model: Any,
    oof_prediction: np.ndarray,
    full_prediction: np.ndarray,
    tuning: dict[str, Any],
    *,
    protocol: str,
) -> dict[str, Any]:
    parameters = parameter_records(model, dataset, model_id, tuning)
    return {
        "modelId": model_id,
        "modelLabel": MODEL_LABELS[model_id],
        "description": MODEL_DESCRIPTIONS[model_id],
        "parameterCount": len(parameters),
        "parameters": parameters,
        "diagnostics": {
            "oof": metric_summary(dataset.y, oof_prediction),
            "train": metric_summary(dataset.y, full_prediction),
        },
        "tuning": tuning,
        "protocol": protocol,
        "caveats": model_caveats(dataset, model_id),
    }


def subset_dataset(dataset: pooled.Dataset, indices: np.ndarray, suffix: str) -> pooled.Dataset:
    return pooled.Dataset(
        name=f"{dataset.name}_{suffix}",
        frame=dataset.frame.iloc[indices].reset_index(drop=True),
        y=np.asarray(dataset.y[indices], dtype=float),
        weights=np.asarray(dataset.weights[indices], dtype=float),
        c0=np.asarray(dataset.c0, dtype=float),
        c1=np.asarray(dataset.c1, dtype=float),
        domain_names=list(dataset.domain_names),
    )


def fit_full_model(dataset: pooled.Dataset, model_id: str, tuning: dict[str, Any]) -> Any:
    indices = np.arange(dataset.n)
    if model_id in {"canonical", "effective_exposure", "effective_exposure_geometry"}:
        return dsp_fit(dataset, indices, model_id)
    if model_id == "separate_heads":
        l2 = float(tuning.get("l2", select_separate_l2(dataset)[0]))
        return separate_fit(dataset, indices, l2)
    if dataset.name.startswith("starcoder"):
        _params, model = starcoder_grp_fit(dataset, indices)
        return model
    raise ValueError(f"Nike-swoosh fit is unsupported for {dataset.name}/{model_id}")


def predict_full_model(model: Any, dataset: pooled.Dataset, model_id: str, weights: np.ndarray) -> np.ndarray:
    if model_id in {"canonical", "effective_exposure", "effective_exposure_geometry"}:
        return dsp_predict(model, dataset, weights)
    if model_id == "separate_heads":
        return separate_predict(model, dataset, weights)
    if dataset.name.startswith("starcoder"):
        return model.predict(weights)
    raise ValueError(f"Nike-swoosh prediction is unsupported for {dataset.name}/{model_id}")


def nike_swoosh_diagnostic(
    dataset: pooled.Dataset,
    model_id: str,
    full_model: Any,
    tuning: dict[str, Any],
) -> dict[str, Any] | None:
    if not dataset.name.startswith("starcoder"):
        return None
    code_index = dataset.domain_names.index("starcoder")
    slice_indices = np.flatnonzero(np.isclose(dataset.weights[:, 0, code_index], 0.0, atol=1e-10))
    if len(slice_indices) < 8:
        raise ValueError(f"{dataset.name} has only {len(slice_indices)} p0=0 points")
    slice_data = subset_dataset(dataset, slice_indices, "p0_zero_slice")
    slice_tuning = dict(tuning)
    if model_id == "separate_heads":
        slice_l2, slice_sweep = select_separate_l2(slice_data)
        slice_tuning = {"l2": slice_l2, "l2Sweep": slice_sweep}
    slice_model = fit_full_model(slice_data, model_id, slice_tuning)
    grid = np.linspace(0.0, 1.0, 161)
    grid_weights = np.zeros((len(grid), 2, 2), dtype=float)
    grid_weights[:, 0, 0] = 1.0
    grid_weights[:, 1, 0] = 1.0 - grid
    grid_weights[:, 1, 1] = grid
    slice_prediction = predict_full_model(slice_model, slice_data, model_id, grid_weights)
    overall_prediction = predict_full_model(full_model, dataset, model_id, grid_weights)
    observed_order = np.argsort(dataset.weights[slice_indices, 1, code_index])
    observed_indices = slice_indices[observed_order]
    return {
        "sliceDefinition": "Phase-0 StarCoder weight = 0; phase-1 StarCoder weight varies.",
        "xLabel": "Phase-1 StarCoder weight",
        "yLabel": "Dolma 100 Programming Languages BPB",
        "observed": {
            "x": dataset.weights[observed_indices, 1, code_index].tolist(),
            "y": dataset.y[observed_indices].tolist(),
            "rowIds": [f"fit:{dataset.name}:{index}" for index in observed_indices],
        },
        "grid": grid.tolist(),
        "sliceFit": {
            "label": f"Fit only on p0=0 slice (n={len(slice_indices)})",
            "prediction": slice_prediction.tolist(),
            "minimumX": float(grid[int(np.argmin(slice_prediction))]),
            "minimumY": float(np.min(slice_prediction)),
        },
        "overallFit": {
            "label": f"Fit on full surface (n={dataset.n}), evaluated on p0=0",
            "prediction": overall_prediction.tolist(),
            "minimumX": float(grid[int(np.argmin(overall_prediction))]),
            "minimumY": float(np.min(overall_prediction)),
        },
    }


def load_cosine_starcoder() -> pooled.Dataset:
    frame = pd.read_csv(COSINE_DATA)
    frame = frame.loc[frame["status"].eq("completed") & frame[STARCODER_TARGET_COLUMN].notna()].reset_index(drop=True)
    weights = np.stack(
        [
            frame[["phase_0_nemotron_full", "phase_0_starcoder"]].to_numpy(dtype=float),
            frame[["phase_1_nemotron_full", "phase_1_starcoder"]].to_numpy(dtype=float),
        ],
        axis=1,
    )
    c0 = np.asarray(
        [
            np.median(
                frame.loc[frame["phase_0_nemotron_full"] > 0, "phase_0_nemotron_epochs"]
                / frame.loc[frame["phase_0_nemotron_full"] > 0, "phase_0_nemotron_full"]
            ),
            np.median(
                frame.loc[frame["phase_0_starcoder"] > 0, "phase_0_starcoder_epochs"]
                / frame.loc[frame["phase_0_starcoder"] > 0, "phase_0_starcoder"]
            ),
        ],
        dtype=float,
    )
    c1 = np.asarray(
        [
            np.median(
                frame.loc[frame["phase_1_nemotron_full"] > 0, "phase_1_nemotron_epochs"]
                / frame.loc[frame["phase_1_nemotron_full"] > 0, "phase_1_nemotron_full"]
            ),
            np.median(
                frame.loc[frame["phase_1_starcoder"] > 0, "phase_1_starcoder_epochs"]
                / frame.loc[frame["phase_1_starcoder"] > 0, "phase_1_starcoder"]
            ),
        ],
        dtype=float,
    )
    return pooled.Dataset(
        name="starcoder_cosine_50_50",
        frame=frame,
        y=frame[STARCODER_TARGET_COLUMN].to_numpy(dtype=float),
        weights=weights,
        c0=c0,
        c1=c1,
        domain_names=list(STARCODER_DOMAINS),
    )


def load_wsd80_starcoder(cosine: pooled.Dataset) -> pooled.Dataset:
    frame = pd.read_csv(WSD80_DATA)
    p0 = frame["phase_0_starcoder"].to_numpy(dtype=float)
    p1 = frame["phase_1_starcoder"].to_numpy(dtype=float)
    weights = np.stack(
        [np.column_stack([1.0 - p0, p0]), np.column_stack([1.0 - p1, p1])],
        axis=1,
    )
    return pooled.Dataset(
        name="starcoder_wsd_80_20",
        frame=frame,
        y=frame["wsd80_bpb"].to_numpy(dtype=float),
        weights=weights,
        c0=np.asarray(cosine.c0 * (0.8 / 0.5), dtype=float),
        c1=np.asarray(cosine.c1 * (0.2 / 0.5), dtype=float),
        domain_names=list(STARCODER_DOMAINS),
    )


def display_domain_name(domain: str) -> str:
    if domain == "nemotron_full":
        return "Nemotron broad"
    if domain == "starcoder":
        return "StarCoder"
    return legacy_exporter.display_domain_name(domain)


def domain_group(domain: str) -> str:
    if domain in STARCODER_DOMAINS:
        return "Broad / code"
    return legacy_exporter.domain_group(domain)


def domain_records(
    dataset: pooled.Dataset,
    alpha0: float,
    *,
    known_budget: float | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray, float]:
    natural = natural_weights(dataset, alpha0)
    budget = target_budget(dataset, alpha0, known_budget)
    token_counts = alpha0 * budget * natural / np.maximum(dataset.c0, 1e-12)
    records = [
        {
            "id": domain,
            "label": display_domain_name(domain),
            "group": domain_group(domain),
            "proportionalWeight": float(natural[index]),
            "tokenCount": float(token_counts[index]),
            "phase0EpochFactor": float(dataset.c0[index]),
            "phase1EpochFactor": float(dataset.c1[index]),
        }
        for index, domain in enumerate(dataset.domain_names)
    ]
    return records, natural, budget


def row_name(dataset: pooled.Dataset, index: int) -> str:
    row = dataset.frame.iloc[index]
    for column in ("run_name", "wandb_run_name", "candidate_name", "wandb_run_id", "run_id"):
        if column in row and pd.notna(row[column]):
            return str(row[column])
    return f"{dataset.name}_{index:04d}"


def row_url(dataset: pooled.Dataset, index: int) -> str | None:
    row = dataset.frame.iloc[index]
    if "wandb_url" in row and pd.notna(row["wandb_url"]):
        return str(row["wandb_url"])
    if "wandb_run_id" in row and pd.notna(row["wandb_run_id"]):
        return f"https://wandb.ai/marin-community/marin/runs/{row['wandb_run_id']}"
    return None


def row_records(
    dataset: pooled.Dataset,
    target_id: str,
    natural: np.ndarray,
    alpha0: float,
    alpha1: float,
) -> list[dict[str, Any]]:
    records = []
    for index in range(dataset.n):
        phase0 = dataset.weights[index, 0]
        phase1 = dataset.weights[index, 1]
        aggregate = alpha0 * phase0 + alpha1 * phase1
        phase0_epochs = phase0 * dataset.c0
        phase1_epochs = phase1 * dataset.c1
        total_epochs = phase0_epochs + phase1_epochs
        phase_tv = 0.5 * float(np.abs(phase0 - phase1).sum())
        aggregate_tv = 0.5 * float(np.abs(aggregate - natural).sum())
        aggregate_kl = float(np.sum(aggregate * np.log(np.maximum(aggregate, 1e-12) / np.maximum(natural, 1e-12))))
        record_id = f"fit:{dataset.name}:{index}"
        records.append(
            {
                "id": record_id,
                "name": row_name(dataset, index),
                "split": "fit",
                "policyFamily": "two_phase",
                "phaseFamily": "two_phase",
                "phaseStructure": "two independent phase weights",
                "panel": dataset.name,
                "method": "observed swarm design",
                "sourceExperiment": dataset.name,
                "wandbUrl": row_url(dataset, index),
                "interventionType": None,
                "targetDomain": None,
                "directionType": None,
                "directionId": None,
                "isSharedAlias": False,
                "pairedRow": None,
                "candidateTarget": None,
                "observed": {target_id: float(dataset.y[index])},
                "phase0": phase0.tolist(),
                "phase1": phase1.tolist(),
                "aggregate": aggregate.tolist(),
                "phase0Epochs": phase0_epochs.tolist(),
                "phase1Epochs": phase1_epochs.tolist(),
                "totalEpochs": total_epochs.tolist(),
                "diagnostics": {
                    "phaseTv": phase_tv,
                    "aggregateTvToProportional": aggregate_tv,
                    "aggregateKlToProportional": aggregate_kl,
                    "maxEpoch": float(np.max(total_epochs)),
                    "nearestFitId": record_id,
                    "supportDistance": 0.0,
                },
            }
        )
    return records


def empty_metric_summary() -> dict[str, int | None]:
    return {"n": 0, "rmse": None, "mae": None, "spearman": None}


def model_diagnostics(dataset: pooled.Dataset, prediction: np.ndarray) -> dict[str, Any]:
    return {
        "fitOof": metric_summary(dataset.y, prediction),
        "heldout": empty_metric_summary(),
        "heldoutSinglePhase": empty_metric_summary(),
        "heldoutTwoPhase": empty_metric_summary(),
    }


def nearest_row(weights: np.ndarray, target: np.ndarray) -> int:
    phase0_distance = np.abs(weights[:, 0] - target[None, :]).sum(axis=1)
    phase1_distance = np.abs(weights[:, 1] - target[None, :]).sum(axis=1)
    return int(np.argmin(phase0_distance + phase1_distance))


def generic_baselines(
    dataset: pooled.Dataset,
    target_id: str,
    rows: list[dict[str, Any]],
    natural: np.ndarray,
) -> dict[str, list[dict[str, str]]]:
    options: list[dict[str, str]] = []
    natural_index = nearest_row(dataset.weights, natural)
    options.append({"id": rows[natural_index]["id"], "label": "Nearest proportional policy"})
    tied = np.max(np.abs(dataset.weights[:, 0] - dataset.weights[:, 1]), axis=1) < 1e-9
    if tied.any():
        tied_indices = np.flatnonzero(tied)
        tied_best = int(tied_indices[int(np.argmin(dataset.y[tied_indices]))])
        options.append({"id": rows[tied_best]["id"], "label": "Empirical constant-mixture frontier"})
    best = int(np.argmin(dataset.y))
    options.append({"id": rows[best]["id"], "label": "Empirical two-phase frontier"})
    if dataset.name.startswith("starcoder"):
        code_index = dataset.domain_names.index("starcoder")
        boundary = np.isclose(dataset.weights[:, 0, code_index], 0.0, atol=1e-10)
        boundary_indices = np.flatnonzero(boundary)
        boundary_best = int(boundary_indices[int(np.argmin(dataset.y[boundary_indices]))])
        options.append({"id": rows[boundary_best]["id"], "label": "Best p0=0 boundary point"})
    return {target_id: options}


def cache_path(swarm_id: str, target_id: str, model_id: str) -> Path:
    return CACHE_DIR / swarm_id / target_id / f"{model_id}.json"


def cached_swarm_fit(
    swarm_id: str,
    target_id: str,
    dataset: pooled.Dataset,
    model_id: str,
    source_paths: list[Path],
    *,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    fingerprint = file_fingerprint(
        [Path(__file__), *source_paths],
        {"swarm": swarm_id, "target": target_id, "model": model_id, "seeds": list(seeds)},
    )
    path = cache_path(swarm_id, target_id, model_id)
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get("fingerprint") == fingerprint:
            print(f"cache hit: {swarm_id}/{target_id}/{model_id}", flush=True)
            return cached
    print(f"fitting: {swarm_id}/{target_id}/{model_id}", flush=True)
    model, prediction, full_prediction, tuning = fit_one_model(dataset, model_id, seeds)
    detail = fit_detail(
        dataset,
        model_id,
        model,
        prediction,
        full_prediction,
        tuning,
        protocol=f"Five-fold OOF averaged over seeds {list(seeds)}; full model refit on all {dataset.n} rows.",
    )
    result = {
        "fingerprint": fingerprint,
        "prediction": prediction.tolist(),
        "fullFitPrediction": full_prediction.tolist(),
        "fitDetail": detail,
        "nikeSwoosh": nike_swoosh_diagnostic(dataset, model_id, model, tuning),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
    return result


def build_generic_swarm(
    *,
    swarm_id: str,
    label: str,
    description: str,
    dataset: pooled.Dataset,
    target_id: str,
    target_label: str,
    metric_column: str,
    source_paths: list[Path],
    known_budget: float | None = None,
) -> dict[str, Any]:
    alpha0, alpha1 = phase_fractions(dataset)
    domains, natural, budget = domain_records(dataset, alpha0, known_budget=known_budget)
    rows = row_records(dataset, target_id, natural, alpha0, alpha1)
    predictions: dict[str, Any] = {target_id: {}}
    diagnostics: dict[str, Any] = {target_id: {}}
    fits: dict[str, Any] = {target_id: {}}
    nike_swoosh: dict[str, Any] = {target_id: {}}
    for model_id in MODEL_IDS:
        result = cached_swarm_fit(
            swarm_id,
            target_id,
            dataset,
            model_id,
            source_paths,
            seeds=(0,),
        )
        predictions[target_id][model_id] = {
            "prediction": result["prediction"],
            "fullFitPrediction": result["fullFitPrediction"],
        }
        diagnostics[target_id][model_id] = model_diagnostics(
            dataset,
            np.asarray(result["prediction"], dtype=float),
        )
        fits[target_id][model_id] = result["fitDetail"]
        if result["nikeSwoosh"] is not None:
            nike_swoosh[target_id][model_id] = result["nikeSwoosh"]
    noise_scale = float(np.std(dataset.y, ddof=1))
    return {
        "id": swarm_id,
        "label": label,
        "description": description,
        "dataset": {
            "label": label,
            "fitDesignCount": dataset.n,
            "rawFitObservationCount": dataset.n,
            "heldoutCount": 0,
            "noiseReferenceCount": 0,
            "supplementalCandidateCount": 0,
            "phaseFractions": [alpha0, alpha1],
            "targetBudget": float(budget),
            "oofSeeds": [0],
            "fitProtocol": "Five-fold random OOF; full fit on all observed surface rows.",
        },
        "domains": domains,
        "targets": {
            target_id: {
                "id": target_id,
                "label": target_label,
                "metricColumn": metric_column,
                "lowerIsBetter": True,
                "noiseReference": {
                    "n": dataset.n,
                    "mean": float(np.mean(dataset.y)),
                    "standardDeviation": noise_scale,
                    "differenceStandardDeviation": noise_scale,
                },
                "noiseLabel": "Target SD used only as a visual scale; no repeat-noise estimate is available.",
            }
        },
        "rows": rows,
        "predictions": predictions,
        "diagnostics": diagnostics,
        "baselines": generic_baselines(dataset, target_id, rows, natural),
        "fits": fits,
        "nikeSwoosh": nike_swoosh,
        "provenance": {
            "sources": [str(path.relative_to(REPO_ROOT)) for path in source_paths],
            "exporter": str(Path(__file__).relative_to(REPO_ROOT)),
        },
    }


def load_legacy_bundle() -> dict[str, Any]:
    if LEGACY_DATA.exists():
        bundle = json.loads(LEGACY_DATA.read_text())
    elif APP_DATA.exists():
        bundle = json.loads(APP_DATA.read_text())
        if bundle.get("schemaVersion") != 1:
            raise ValueError(f"{APP_DATA} is already v2, but the preserved v1 bundle is missing")
        LEGACY_DATA.parent.mkdir(parents=True, exist_ok=True)
        LEGACY_DATA.write_bytes(APP_DATA.read_bytes())
    else:
        raise FileNotFoundError("Generate the 300M debugger bundle before the multi-swarm observatory")
    if bundle.get("schemaVersion") != 1:
        raise ValueError(f"Expected v1 300M bundle, got schema {bundle.get('schemaVersion')}")
    return bundle


def cached_300m_fit_detail(
    legacy: dict[str, Any],
    target_id: str,
    dataset: pooled.Dataset,
    model_id: str,
) -> dict[str, Any]:
    tuning = dict(legacy["models"][model_id]["protocol"]["targetParameters"][target_id])
    fingerprint = file_fingerprint(
        [Path(__file__), LEGACY_DATA],
        {"swarm": "300m", "target": target_id, "model": model_id, "tuning": tuning},
    )
    path = cache_path("300m", target_id, model_id)
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get("fingerprint") == fingerprint:
            print(f"cache hit: 300m/{target_id}/{model_id}", flush=True)
            return cached["fitDetail"]
    indices = np.arange(dataset.n)
    if model_id in {"canonical", "effective_exposure", "effective_exposure_geometry"}:
        model = dsp_fit(dataset, indices, model_id)
        full_prediction = dsp_predict(model, dataset, dataset.weights)
    elif model_id == "separate_heads":
        model = separate_fit(dataset, indices, float(tuning["l2"]))
        full_prediction = separate_predict(model, dataset, dataset.weights)
    else:
        model = grp_300m_fit(dataset, indices, float(tuning["l2"]))
        full_prediction = model.predict(dataset.weights)
    old_prediction = np.asarray(legacy["predictions"][target_id][model_id]["prediction"][: dataset.n], dtype=float)
    detail = fit_detail(
        dataset,
        model_id,
        model,
        old_prediction,
        full_prediction,
        tuning,
        protocol=(
            "Existing three-seed, five-fold panel-stratified grouped OOF; " "full model refit on 280 collapsed designs."
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fingerprint": fingerprint, "fitDetail": detail}, separators=(",", ":")) + "\n")
    return detail


def build_300m_swarm(legacy: dict[str, Any]) -> dict[str, Any]:
    fits: dict[str, Any] = {}
    for target_id in ("uncheatable", "table9"):
        dataset = pooled.load_300m_dataset(target_id)
        legacy_names = [row["name"] for row in legacy["rows"][: dataset.n]]
        dataset_names = dataset.frame["run_name"].astype(str).tolist()
        if legacy_names != dataset_names:
            raise ValueError(f"300M row order differs for {target_id}")
        fits[target_id] = {
            model_id: cached_300m_fit_detail(legacy, target_id, dataset, model_id) for model_id in MODEL_IDS
        }
    targets = dict(legacy["targets"])
    for target in targets.values():
        target["noiseLabel"] = "Difference SD from the 11 proportional observations used by the fit panel."
    return {
        "id": "300m",
        "label": legacy["dataset"]["label"],
        "description": "300M / 6B-token Dolma 3 + Dolmino panel with fit, heldout, repeat, and candidate checkpoints.",
        "dataset": legacy["dataset"],
        "domains": legacy["domains"],
        "targets": targets,
        "rows": legacy["rows"],
        "predictions": legacy["predictions"],
        "diagnostics": legacy["diagnostics"],
        "baselines": legacy["baselines"],
        "fits": fits,
        "nikeSwoosh": {},
        "provenance": legacy["provenance"],
    }


def model_catalog() -> dict[str, Any]:
    return {
        model_id: {
            "id": model_id,
            "label": MODEL_LABELS[model_id],
            "description": MODEL_DESCRIPTIONS[model_id],
        }
        for model_id in MODEL_IDS
    }


def write_bundle(output_json: Path) -> dict[str, Any]:
    legacy = load_legacy_bundle()
    cosine = load_cosine_starcoder()
    wsd80 = load_wsd80_starcoder(cosine)
    production = pooled.load_production_dataset()
    production_metadata = json.loads(PRODUCTION_MODEL.read_text())["metrics"]
    swarms = {
        "300m": build_300m_swarm(legacy),
        "starcoder_cosine": build_generic_swarm(
            swarm_id="starcoder_cosine",
            label="StarCoder 50/50 cosine surface",
            description=(
                "Dense two-domain, two-phase surface under equal cosine-schedule phases; "
                "target is Dolma 100 Programming Languages BPB."
            ),
            dataset=cosine,
            target_id="starcoder_bpb",
            target_label="Dolma 100 Programming Languages BPB",
            metric_column=STARCODER_TARGET_COLUMN,
            source_paths=[COSINE_DATA],
        ),
        "starcoder_wsd80": build_generic_swarm(
            swarm_id="starcoder_wsd80",
            label="StarCoder 80/20 WSD surface",
            description=(
                "Pruned two-domain StarCoder surface under an 80/20 warmup-stable-decay schedule; "
                "target is Dolma 100 Programming Languages BPB."
            ),
            dataset=wsd80,
            target_id="starcoder_bpb",
            target_label="Dolma 100 Programming Languages BPB",
            metric_column="wsd80_bpb",
            source_paths=[WSD80_DATA, COSINE_DATA],
        ),
        "production": build_generic_swarm(
            swarm_id="production",
            label="Production Grug-MoE swarm",
            description=(
                "840 sampled two-phase mixtures over 168 production buckets; "
                "GRP is intentionally evaluated without semantic family or pair grouping."
            ),
            dataset=production,
            target_id="uncheatable",
            target_label="Uncheatable eval BPB",
            metric_column="eval/uncheatable_eval/bpb",
            source_paths=[PRODUCTION_DATA, PRODUCTION_MODEL],
            known_budget=float(production_metadata["production_experiment_budget_tokens"]),
        ),
    }
    bundle = {
        "schemaVersion": 2,
        "generatedAt": datetime.now(UTC).isoformat(),
        "models": model_catalog(),
        "swarms": swarms,
        "provenance": {
            "exporter": str(Path(__file__).relative_to(REPO_ROOT)),
            "cacheVersion": CACHE_VERSION,
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(bundle, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Wrote {output_json} ({output_json.stat().st_size / 1_000_000:.2f} MB)", flush=True)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=APP_DATA)
    args = parser.parse_args()
    write_bundle(args.output_json)


if __name__ == "__main__":
    main()
