#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "scipy"]
# ///
"""Explore stronger S2-style structural laws without breaking scaling structure.

The candidates here preserve the fixed-mixture scaling-law property:

    L(w, N, D) = E(w) + A(w) u_N + B(w) u_D + C(w) u_ND

Only the mixture-dependent functions E/A/B/C change. This script is intended as
an exploratory local sprint, not a packet entry point.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.stats import kendalltau, spearmanr

SESSION2_SCRIPT = (
    Path("experiments/domain_phase_mix/exploratory/two_phase_many")
    / "chatgpt_pro_hybrid_data_mixing_packet_v30_local_artifacts"
    / "reference_outputs/session10_candidate_revalidation_20260424"
    / "session2_structural/run_joint_mixture_scale_law.py"
)
DEFAULT_PACKET_ROOT = Path(
    "experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v30"
)
DEFAULT_OUT_DIR = (
    Path("experiments/domain_phase_mix/exploratory/two_phase_many")
    / "chatgpt_pro_hybrid_data_mixing_packet_v30_local_artifacts"
    / "reference_outputs/s2_structural_upgrade_sprint_20260424"
)
FAMILIES = ("broad_text", "tech_code", "reasoning")
TARGETS = (
    ("60m_1p2b", "60M/1.2B"),
    ("300m_6b", "100M/6B"),
    ("520m_10p4b", "340M/10.4B"),
    ("1_2b_24b", "900M/24B"),
)


@dataclass(frozen=True)
class VariantSpec:
    """Configuration for one structural-law variant."""

    name: str
    anchor_kind: str
    head_a: str
    head_b: str
    head_c: str
    exponents: tuple[float, float, float, float] = (0.20, 0.25, 0.30, 0.65)
    pair_weight: float = 6.0
    ridge_anchor: float = 1e-4
    ridge_scale: float = 1e-5
    family_beta: tuple[float, ...] | None = None
    explicit_safety_strength: float = 0.0
    donor_constant_count: int = 9


@dataclass
class SafetyWrappedModel:
    """Base structural model plus a fixed nonnegative mixture-only safety term."""

    name: str
    base_model: object
    feature_factory: EnhancedFeatureFactory
    strength: float

    @property
    def model_id(self) -> str:
        return self.name

    @property
    def total_constant_count(self) -> int:
        return int(getattr(self.base_model, "total_constant_count", 0) + 1)

    def predict_custom(
        self, weights: np.ndarray, model_size: np.ndarray | float, tokens: np.ndarray | float
    ) -> np.ndarray:
        base = self.base_model.predict_custom(weights, model_size, tokens)
        return base + self.strength * self.feature_factory.explicit_safety_penalty(weights)

    def predict_all(self) -> np.ndarray:
        data = self.base_model.data
        return self.predict_custom(data.W, data.N, data.D)


def load_module(path: Path) -> ModuleType:
    """Load the Session 2 artifact as a module."""
    spec = importlib.util.spec_from_file_location("session2_structural_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["session2_structural_module"] = module
    spec.loader.exec_module(module)
    return module


def safe_corr(fn, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return float("nan")
    value = fn(y_true, y_pred).statistic
    return float(value) if np.isfinite(value) else float("nan")


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    resid = y_pred - y_true
    actual_std = float(np.std(y_true, ddof=0))
    pred_std = float(np.std(y_pred, ddof=0))
    if len(y_true) > 1 and actual_std > 0.0 and pred_std > 0.0:
        slope, intercept = np.polyfit(y_true, y_pred, 1)
        std_ratio = pred_std / actual_std
    else:
        slope = intercept = std_ratio = float("nan")
    low_n = min(len(y_true), max(3, math.ceil(0.15 * len(y_true))))
    low_idx = np.argsort(y_pred)[:low_n]
    return {
        "rows": len(y_true),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "bias": float(np.mean(resid)),
        "spearman": safe_corr(spearmanr, y_true, y_pred),
        "kendall": safe_corr(kendalltau, y_true, y_pred),
        "slope": float(slope),
        "intercept": float(intercept),
        "std_ratio": float(std_ratio),
        "regret_at_1": float(y_true[int(np.argmin(y_pred))] - np.min(y_true)),
        "top5_overlap": float(
            len(set(np.argsort(y_true)[: min(5, len(y_true))]).intersection(np.argsort(y_pred)[: min(5, len(y_true))]))
            / min(5, len(y_true))
        ),
        "lower_tail_optimism": float(np.mean(np.maximum(y_true[low_idx] - y_pred[low_idx], 0.0))),
        "low_tail_rmse": float(np.sqrt(np.mean((y_true[low_idx] - y_pred[low_idx]) ** 2))),
    }


class EnhancedFeatureFactory:
    """Feature factory that adds stronger mixture-only anchors and heads."""

    def __init__(self, base_factory: object, module: ModuleType):
        self.base = base_factory
        self.module = module
        self.data = base_factory.data
        self.domains = base_factory.domains
        self.family_names = base_factory.family_names
        self.family_masks = base_factory.family_masks
        self.selected_domain_idx = base_factory.selected_domain_idx

    def family_shares(self, weights: np.ndarray) -> np.ndarray:
        return self.base.family_shares(weights)

    def grp_features(self, weights: np.ndarray) -> tuple[np.ndarray, list[str]]:
        return self.base.grp_features(weights)

    def safety_features(self, weights: np.ndarray) -> tuple[np.ndarray, list[str]]:
        weights = np.asarray(weights, dtype=float)
        p0 = weights[:, 0, :]
        p1 = weights[:, 1, :]
        family = self.family_shares(weights).reshape(weights.shape[0], 2, len(self.family_names))
        entropy0 = self.module.entropy_rows(p0)
        entropy1 = self.module.entropy_rows(p1)
        cols = [
            np.max(p0, axis=1),
            np.max(p1, axis=1),
            np.max(family[:, 0, :], axis=1),
            np.max(family[:, 1, :], axis=1),
            np.maximum(0.0, 2.2 - entropy0) ** 2,
            np.maximum(0.0, 2.2 - entropy1) ** 2,
            family[:, 0, self.family_names.index("tech_code")] ** 2,
            family[:, 1, self.family_names.index("tech_code")] ** 2,
            family[:, 0, self.family_names.index("reasoning")] ** 2,
            family[:, 1, self.family_names.index("reasoning")] ** 2,
        ]
        names = [
            "safety:p0_max_domain",
            "safety:p1_max_domain",
            "safety:p0_max_family",
            "safety:p1_max_family",
            "safety:p0_low_entropy_sq",
            "safety:p1_low_entropy_sq",
            "safety:p0_tech_sq",
            "safety:p1_tech_sq",
            "safety:p0_reasoning_sq",
            "safety:p1_reasoning_sq",
        ]
        return np.column_stack(cols), names

    def explicit_safety_penalty(self, weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=float)
        p0 = weights[:, 0, :]
        p1 = weights[:, 1, :]
        family = self.family_shares(weights).reshape(weights.shape[0], 2, len(self.family_names))
        entropy0 = self.module.entropy_rows(p0)
        entropy1 = self.module.entropy_rows(p1)
        p1_tech = family[:, 1, self.family_names.index("tech_code")]
        p0_tech = family[:, 0, self.family_names.index("tech_code")]
        p1_reasoning = family[:, 1, self.family_names.index("reasoning")]
        return (
            np.maximum(0.0, np.max(p1, axis=1) - 0.35) ** 2
            + 0.5 * np.maximum(0.0, np.max(p0, axis=1) - 0.35) ** 2
            + np.maximum(0.0, np.max(family[:, 1, :], axis=1) - 0.75) ** 2
            + 0.5 * np.maximum(0.0, np.max(family[:, 0, :], axis=1) - 0.80) ** 2
            + 0.25 * np.maximum(0.0, 2.0 - entropy1) ** 2
            + 0.10 * np.maximum(0.0, 2.0 - entropy0) ** 2
            + 0.25 * np.maximum(0.0, p1_tech - 0.65) ** 2
            + 0.10 * np.maximum(0.0, p0_tech - 0.65) ** 2
            + 0.25 * np.maximum(0.0, p1_reasoning - 0.20) ** 2
        )

    def anchor_features(self, weights: np.ndarray, kind: str) -> tuple[np.ndarray, list[str]]:
        if kind in {"grp", "grp_famsqrt", "grp_selectedsqrt"}:
            return self.base.anchor_features(weights, kind)
        if kind == "grp_famsqrt_safety":
            x, names = self.base.anchor_features(weights, "grp_famsqrt")
            safety, safety_names = self.safety_features(weights)
            return np.column_stack([x, safety]), names + safety_names
        if kind == "grp_strong":
            x, names = self.base.anchor_features(weights, "grp_famsqrt")
            selected = np.sqrt(np.clip(weights[:, :, self.selected_domain_idx], 0.0, None) + 1e-9).reshape(
                weights.shape[0], -1
            )
            selected_names = [
                f"sqrt_selected_p{phase}:{self.domains[domain_index]}"
                for phase in range(2)
                for domain_index in self.selected_domain_idx
            ]
            safety, safety_names = self.safety_features(weights)
            return np.column_stack([x, selected, safety]), names + selected_names + safety_names
        raise ValueError(f"unknown anchor kind: {kind}")

    def scale_head_features(self, weights: np.ndarray, mode: str = "family") -> np.ndarray:
        if mode in {"constant", "family", "family_quality", "domain_raw", "domain_sqrt"}:
            return self.base.scale_head_features(weights, mode)
        if mode == "family_safety":
            family = self.base.scale_head_features(weights, "family")
            safety, _ = self.safety_features(weights)
            return np.column_stack([family, safety])
        if mode == "family_quality_safety":
            family_quality = self.base.scale_head_features(weights, "family_quality")
            safety, _ = self.safety_features(weights)
            return np.column_stack([family_quality, safety])
        if mode == "selected_domain_sqrt":
            selected = np.sqrt(np.clip(weights[:, :, self.selected_domain_idx], 0.0, None) + 1e-9).reshape(
                weights.shape[0], -1
            )
            return np.column_stack([np.ones(weights.shape[0]), selected])
        raise ValueError(f"unknown scale head mode: {mode}")


class FamilyBetaPowerLaw:
    """Two-stage law with per-B-head fixed beta exponents."""

    def __init__(
        self,
        model_id: str,
        data: object,
        feature_factory: EnhancedFeatureFactory,
        anchor_kind: str,
        exponents: tuple[float, float, float, float],
        beta_values: tuple[float, ...],
        pair_weight: float,
        ridge_anchor: float,
        ridge_scale: float,
        donor_constant_count: int,
        head_a: str,
        head_b: str,
        head_c: str,
    ):
        self.model_id = model_id
        self.data = data
        self.ff = feature_factory
        self.anchor_kind = anchor_kind
        self.alpha, self.beta, self.gamma, self.delta = map(float, exponents)
        self.beta_values = np.asarray(beta_values, dtype=float)
        self.pair_weight = float(pair_weight)
        self.ridge_anchor = float(ridge_anchor)
        self.ridge_scale = float(ridge_scale)
        self.donor_constant_count = int(donor_constant_count)
        self.head_A = head_a
        self.head_B = head_b
        self.head_C = head_c

    def scale_terms(
        self, model_size: np.ndarray, tokens: np.ndarray, b_width: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model_size = np.asarray(model_size, dtype=float)
        tokens = np.asarray(tokens, dtype=float)
        u_n = (model_size / self.data.N0) ** (-self.alpha) - 1.0
        beta_values = self.beta_values
        if len(beta_values) != b_width:
            raise ValueError(f"beta_values length {len(beta_values)} does not match B head width {b_width}")
        u_d = (tokens[:, None] / self.data.D0) ** (-beta_values[None, :]) - 1.0
        u_nd = (model_size / self.data.N0) ** (-self.gamma) * (tokens / self.data.D0) ** (-self.delta) - 1.0
        return u_n, u_d, u_nd

    def make_pairs(self, mask: np.ndarray) -> list[tuple[int, int]]:
        frame = pd.DataFrame(
            {
                "idx": np.arange(len(mask)),
                "scale": self.data.scale_names_by_row,
                "mix": self.data.mixture_ids,
                "D": self.data.D,
            }
        )
        pairs = []
        for _, group in frame[mask].groupby(["scale", "mix"]):
            if group["D"].nunique() < 2:
                continue
            rows = group.sort_values("D")["idx"].to_numpy()
            for left in range(len(rows)):
                for right in range(left + 1, len(rows)):
                    pairs.append((int(rows[left]), int(rows[right])))
        return pairs

    def _design(self, weights: np.ndarray, model_size: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        head_a = self.ff.scale_head_features(weights, self.head_A)
        head_b = self.ff.scale_head_features(weights, self.head_B)
        head_c = self.ff.scale_head_features(weights, self.head_C)
        u_n, u_d, u_nd = self.scale_terms(model_size, tokens, head_b.shape[1])
        return np.column_stack([head_a * u_n[:, None], head_b * u_d, head_c * u_nd[:, None]])

    def fit(self, train_mask: np.ndarray) -> FamilyBetaPowerLaw:
        self.train_mask = np.asarray(train_mask, dtype=bool).copy()
        x_anchor_all, anchor_names = self.ff.anchor_features(self.data.W, self.anchor_kind)
        anchor_rows = self.train_mask & (self.data.scale_names_by_row == "300m_6b") & np.isclose(self.data.mu, 1.0)
        coef_anchor, stats_anchor = self.module_ridge_fit(x_anchor_all[anchor_rows], self.data.y[anchor_rows])
        anchor_pred_all = self.module_ridge_predict(x_anchor_all, coef_anchor, stats_anchor)
        design_all = self._design(self.data.W, self.data.N, self.data.D)
        y_resid = self.data.y - anchor_pred_all

        rows = np.flatnonzero(self.train_mask)
        matrix = design_all[rows]
        target = y_resid[rows]
        pair_rows = []
        pair_targets = []
        for left, right in self.make_pairs(self.train_mask):
            pair_rows.append(math.sqrt(self.pair_weight) * (design_all[left] - design_all[right]))
            pair_targets.append(math.sqrt(self.pair_weight) * (y_resid[left] - y_resid[right]))
        if pair_rows:
            matrix = np.vstack([matrix, np.asarray(pair_rows)])
            target = np.concatenate([target, np.asarray(pair_targets, dtype=float)])
        if self.ridge_scale > 0.0:
            matrix = np.vstack([matrix, math.sqrt(self.ridge_scale) * np.eye(design_all.shape[1])])
            target = np.concatenate([target, np.zeros(design_all.shape[1], dtype=float)])
        result = lsq_linear(matrix, target, bounds=(0.0, np.inf), max_iter=2000, lsmr_tol="auto")
        self.anchor_coef = coef_anchor
        self.anchor_stats = stats_anchor
        self.anchor_names = anchor_names
        self.scale_coef = np.clip(result.x, 0.0, np.inf)
        self.anchor_feature_count = x_anchor_all.shape[1] + 1
        self.scale_param_count = len(self.scale_coef)
        self.total_constant_count = self.anchor_feature_count + self.scale_param_count + 4 + len(self.beta_values)
        self.total_constant_count += self.donor_constant_count
        self.scale_fit_success = bool(result.success)
        return self

    def module_ridge_fit(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        return self.ff.module.ridge_fit(x, y, ridge=self.ridge_anchor)

    def module_ridge_predict(self, x: np.ndarray, coef: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
        return self.ff.module.ridge_predict_from_stats(x, coef, stats)

    def predict_custom(
        self, weights: np.ndarray, model_size: np.ndarray | float, tokens: np.ndarray | float
    ) -> np.ndarray:
        weights = np.asarray(weights, dtype=float)
        n = weights.shape[0]
        model_array = np.full(n, float(model_size)) if np.ndim(model_size) == 0 else np.asarray(model_size, dtype=float)
        token_array = np.full(n, float(tokens)) if np.ndim(tokens) == 0 else np.asarray(tokens, dtype=float)
        x_anchor, _ = self.ff.anchor_features(weights, self.anchor_kind)
        anchor = self.module_ridge_predict(x_anchor, self.anchor_coef, self.anchor_stats)
        return anchor + self._design(weights, model_array, token_array) @ self.scale_coef

    def predict_all(self) -> np.ndarray:
        return self.predict_custom(self.data.W, self.data.N, self.data.D)


def fit_variant(
    module: ModuleType, data: object, feature_factory: EnhancedFeatureFactory, train_mask: np.ndarray, spec: VariantSpec
) -> object:
    if spec.family_beta is None:
        model = module.TwoStagePowerLaw(
            spec.name,
            data,
            feature_factory,
            spec.anchor_kind,
            spec.exponents,
            pair_weight=spec.pair_weight,
            ridge_anchor=spec.ridge_anchor,
            ridge_scale=spec.ridge_scale,
            donor_constant_count=spec.donor_constant_count,
            head_A=spec.head_a,
            head_B=spec.head_b,
            head_C=spec.head_c,
        ).fit(train_mask)
    else:
        model = FamilyBetaPowerLaw(
            spec.name,
            data,
            feature_factory,
            spec.anchor_kind,
            spec.exponents,
            spec.family_beta,
            spec.pair_weight,
            spec.ridge_anchor,
            spec.ridge_scale,
            spec.donor_constant_count,
            spec.head_a,
            spec.head_b,
            spec.head_c,
        ).fit(train_mask)
    if spec.explicit_safety_strength > 0.0:
        model = SafetyWrappedModel(spec.name, model, feature_factory, spec.explicit_safety_strength)
    return model


def variants() -> list[VariantSpec]:
    family_beta_small = (0.25, 0.25, 0.22, 0.34, 0.25, 0.22, 0.34)
    family_beta_alt = (0.25, 0.22, 0.25, 0.36, 0.22, 0.25, 0.36)
    return [
        VariantSpec("s2_base", "grp_famsqrt", "constant", "family", "constant"),
        VariantSpec("s2_beta036", "grp_famsqrt", "constant", "family", "constant", exponents=(0.20, 0.36, 0.30, 0.65)),
        VariantSpec("s2_pair20", "grp_famsqrt", "constant", "family", "constant", pair_weight=20.0),
        VariantSpec("s2_anchor_safety", "grp_famsqrt_safety", "constant", "family", "constant"),
        VariantSpec(
            "s2_anchor_safety_beta036",
            "grp_famsqrt_safety",
            "constant",
            "family",
            "constant",
            exponents=(0.20, 0.36, 0.30, 0.65),
        ),
        VariantSpec("s2_anchor_safety_pair20", "grp_famsqrt_safety", "constant", "family", "constant", pair_weight=20.0),
        VariantSpec("s2_anchor_strong", "grp_strong", "constant", "family", "constant", ridge_anchor=3e-4),
        VariantSpec("s2_rich_family_heads", "grp_famsqrt", "family", "family_quality", "family", ridge_scale=3e-5),
        VariantSpec(
            "s2_strong_rich_heads",
            "grp_strong",
            "family",
            "family_quality_safety",
            "family",
            ridge_anchor=3e-4,
            ridge_scale=1e-4,
        ),
        VariantSpec("s2_domain_b_head", "grp_famsqrt", "family", "domain_sqrt", "constant", ridge_scale=1e-3),
        VariantSpec("s2_selected_head", "grp_famsqrt", "family", "selected_domain_sqrt", "constant", ridge_scale=5e-4),
        VariantSpec(
            "s2_base_safety010", "grp_famsqrt", "constant", "family", "constant", explicit_safety_strength=0.010
        ),
        VariantSpec(
            "s2_base_safety020", "grp_famsqrt", "constant", "family", "constant", explicit_safety_strength=0.020
        ),
        VariantSpec(
            "s2_base_safety050", "grp_famsqrt", "constant", "family", "constant", explicit_safety_strength=0.050
        ),
        VariantSpec(
            "s2_base_safety100", "grp_famsqrt", "constant", "family", "constant", explicit_safety_strength=0.100
        ),
        VariantSpec(
            "s2_base_safety200", "grp_famsqrt", "constant", "family", "constant", explicit_safety_strength=0.200
        ),
        VariantSpec(
            "s2_family_beta_small",
            "grp_famsqrt",
            "constant",
            "family",
            "constant",
            family_beta=family_beta_small,
        ),
        VariantSpec(
            "s2_strong_family_beta",
            "grp_strong",
            "family",
            "family",
            "constant",
            ridge_anchor=3e-4,
            ridge_scale=5e-5,
            family_beta=family_beta_alt,
        ),
    ]


def evaluate_subsets(data: object, models: dict[str, object], predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    subsets = {
        "seed7_holdout": data.seed7_holdout,
        "fixed_340m_10p4b": data.fixed340,
        "random_supplement": data.random_supplement,
        "all_900m_24b_seed7_fit": data.all900_holdout,
    }
    rows = []
    for name, pred in predictions.items():
        params = int(getattr(models[name], "total_constant_count", getattr(models[name], "param_count", -1)))
        for subset, mask in subsets.items():
            metrics = metric_dict(data.y[mask], pred[mask])
            metrics.update(model=name, subset=subset, params=params)
            rows.append(metrics)
    return pd.DataFrame(rows)


def all900_protocol(
    module: ModuleType,
    data: object,
    feature_factory: EnhancedFeatureFactory,
    specs: list[VariantSpec],
) -> pd.DataFrame:
    rows = []
    for spec in specs:
        model = fit_variant(module, data, feature_factory, data.all900_train, spec)
        pred = np.asarray(model.predict_all(), dtype=float)
        metrics = metric_dict(data.y[data.all900_holdout], pred[data.all900_holdout])
        metrics.update(
            model=spec.name,
            subset="all_900m_24b",
            train_regime="all_non_900m_train",
            params=int(getattr(model, "total_constant_count", getattr(model, "param_count", -1))),
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def fixed_drop_summary(data: object, predictions: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixed_idx = np.flatnonzero(data.fixed340)
    frame = pd.DataFrame(
        {
            "mixture_id": data.mixture_ids[fixed_idx].astype(str),
            "idx": fixed_idx,
            "mu": data.mu[fixed_idx],
        }
    )
    rows = []
    for name, pred in predictions.items():
        for mixture_id, group in frame.groupby("mixture_id", sort=False):
            by_mu = {float(row.mu): int(row.idx) for row in group.itertuples()}
            for start, end, label in [(0.5, 1.0, "0.5->1"), (0.5, 2.0, "0.5->2"), (1.0, 2.0, "1->2")]:
                if start not in by_mu or end not in by_mu:
                    continue
                left = by_mu[start]
                right = by_mu[end]
                actual_drop = float(data.y[left] - data.y[right])
                predicted_drop = float(pred[left] - pred[right])
                rows.append(
                    {
                        "model": name,
                        "mixture_id": mixture_id,
                        "drop": label,
                        "actual_drop": actual_drop,
                        "predicted_drop": predicted_drop,
                        "drop_error": predicted_drop - actual_drop,
                        "drop_ratio": predicted_drop / actual_drop if actual_drop else np.nan,
                    }
                )
    details = pd.DataFrame(rows)
    summary = (
        details.groupby(["model", "drop"], sort=False)
        .agg(
            n=("mixture_id", "size"),
            actual_drop=("actual_drop", "mean"),
            predicted_drop=("predicted_drop", "mean"),
            drop_ratio=("drop_ratio", "mean"),
            drop_rmse=("drop_error", lambda values: float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))),
        )
        .reset_index()
    )
    return details, summary


def family_summary(weights: np.ndarray, feature_factory: EnhancedFeatureFactory) -> dict[str, float]:
    weights = np.asarray(weights, dtype=float)
    family = feature_factory.family_shares(weights[None, :, :]).reshape(2, len(FAMILIES))
    p0 = weights[0]
    p1 = weights[1]
    out = {
        "p0_support_inv_l2": float(1.0 / np.sum(p0**2)),
        "p1_support_inv_l2": float(1.0 / np.sum(p1**2)),
        "p0_max_domain": float(np.max(p0)),
        "p1_max_domain": float(np.max(p1)),
    }
    for phase_index, phase_name in enumerate(("p0", "p1")):
        for family_index, family_name in enumerate(FAMILIES):
            out[f"{phase_name}_{family_name}_share"] = float(family[phase_index, family_index])
    return out


def random_simplex(rng: np.random.Generator, count: int, dim: int, alpha: float) -> np.ndarray:
    return rng.dirichlet(np.full(dim, alpha, dtype=float), size=count)


def candidate_weights(data: object, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    dim = data.W.shape[-1]
    raw_parts = [data.W[data.seed7_train]]
    for alpha in (0.03, 0.10, 0.30, 1.00, 3.00):
        p0 = random_simplex(rng, 1200, dim, alpha)
        p1 = random_simplex(rng, 1200, dim, alpha)
        raw_parts.append(np.stack([p0, p1], axis=1))
    raw = np.concatenate(raw_parts, axis=0)

    top_rows = np.flatnonzero(data.seed7_train)
    top_rows = top_rows[np.argsort(data.y[top_rows])[:8]]
    top_weights = data.W[top_rows]
    hull = []
    for alpha in (0.3, 1.0, 3.0):
        coeff = rng.dirichlet(np.full(len(top_weights), alpha), size=2000)
        hull.append(np.einsum("nk,kpj->npj", coeff, top_weights))
    return raw, np.concatenate(hull, axis=0)


def target_nd(data: object, scale_name: str) -> tuple[float, float]:
    rows = (data.scale_names_by_row == scale_name) & np.isclose(data.mu, 1.0)
    if not np.any(rows):
        raise ValueError(f"No mu=1 rows for {scale_name}")
    return float(np.median(data.N[rows])), float(np.median(data.D[rows]))


def optimum_diagnostics(
    data: object,
    feature_factory: EnhancedFeatureFactory,
    models: dict[str, object],
    rng: np.random.Generator,
) -> pd.DataFrame:
    raw, hull = candidate_weights(data, rng)
    candidate_sets = {
        "raw_random_search": raw,
        "top8_actual_hull_random_search": hull,
    }
    rows = []
    for model_name, model in models.items():
        for scale_name, display in TARGETS:
            model_size, tokens = target_nd(data, scale_name)
            for opt_kind, candidates in candidate_sets.items():
                pred = np.asarray(model.predict_custom(candidates, model_size, tokens), dtype=float)
                best = int(np.argmin(pred))
                best_weights = candidates[best]
                summary = family_summary(best_weights, feature_factory)
                summary.update(
                    model=model_name,
                    target_scale=scale_name,
                    target_display=display,
                    opt_kind=opt_kind,
                    candidate_count=len(candidates),
                    predicted_bpb=float(pred[best]),
                    hard_corner_flag=bool(summary["p0_max_domain"] > 0.60 or summary["p1_max_domain"] > 0.60),
                    phase0_family_collapse_flag=bool(max(summary[f"p0_{family}_share"] for family in FAMILIES) > 0.90),
                    phase1_family_collapse_flag=bool(max(summary[f"p1_{family}_share"] for family in FAMILIES) > 0.90),
                    phase1_tech_collapse_flag=bool(summary["p1_tech_code_share"] > 0.85),
                )
                rows.append(summary)
    return pd.DataFrame(rows)


def predictions_frame(data: object, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "registry_run_key": data.registry_keys.astype(str),
            "mixture_id": data.mixture_ids.astype(str),
            "scale": data.scale_names_by_row.astype(str),
            "mu": data.mu.astype(float),
            "actual": data.y.astype(float),
            "seed7_holdout": data.seed7_holdout,
            "fixed_340m_10p4b": data.fixed340,
            "all_900m_24b": data.all900_holdout,
        }
    )
    for name, pred in predictions.items():
        frame[name] = pred
        frame[f"{name}_residual"] = pred - data.y
    return frame


def plot_predicted_vs_actual(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    models: list[str],
    subset_col: str,
    subset_name: str,
    out_path: Path,
) -> None:
    subset = predictions[predictions[subset_col]].copy()
    if subset.empty:
        return
    cols = ["actual", *models]
    lo = float(subset[cols].to_numpy(dtype=float).min() - 0.008)
    hi = float(subset[cols].to_numpy(dtype=float).max() + 0.008)
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.3), constrained_layout=True)
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("RdYlGn_r")
    scatter = None
    for ax, model in zip(axes, models, strict=False):
        row = metrics[(metrics["model"] == model) & (metrics["subset"] == subset_name)]
        title_metrics = "metrics unavailable"
        if not row.empty:
            first = row.iloc[0]
            title_metrics = f"RMSE={first.rmse:.4f}, Sp={first.spearman:.3f}, slope={first.slope:.3f}"
        scatter = ax.scatter(
            subset["actual"],
            subset[model],
            c=subset["mu"],
            cmap=cmap,
            edgecolor="black",
            linewidth=0.4,
            s=46,
        )
        ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.0)
        ax.set_title(f"{model}\n{title_metrics}")
        ax.set_xlabel("Actual BPB")
        ax.set_ylabel("Predicted BPB")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.25)
    if scatter is not None:
        fig.colorbar(scatter, ax=axes.ravel().tolist(), label="target-budget multiplier")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def select_plot_models(metrics: pd.DataFrame, count: int = 5) -> list[str]:
    fixed = metrics[metrics["subset"] == "fixed_340m_10p4b"].sort_values("rmse")
    names = ["s2_base"]
    for name in fixed["model"]:
        if name not in names:
            names.append(str(name))
        if len(names) >= count:
            break
    return names


def write_report(
    out_dir: Path,
    metrics: pd.DataFrame,
    all900: pd.DataFrame,
    drops: pd.DataFrame,
    optima: pd.DataFrame,
    plot_models: list[str],
) -> None:
    fixed = metrics[metrics["subset"] == "fixed_340m_10p4b"].sort_values("rmse")
    holdout = metrics[metrics["subset"] == "seed7_holdout"].sort_values("rmse")
    all900_sorted = all900.sort_values("rmse")
    raw_340 = optima[(optima["target_scale"] == "520m_10p4b") & (optima["opt_kind"] == "raw_random_search")]
    report = [
        "# S2 Structural Upgrade Sprint 2026-04-24",
        "",
        (
            "All candidates preserve the S2 fixed-mixture scaling-law skeleton. The sprint only changes "
            "mixture-dependent anchor/head capacity or adds mixture-only safety penalties."
        ),
        "",
        "## Best Fixed-340M Metrics",
        "",
        fixed[
            ["model", "params", "rmse", "bias", "spearman", "slope", "std_ratio", "regret_at_1", "lower_tail_optimism"]
        ]
        .head(12)
        .to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Best Seed-7 Holdout Metrics",
        "",
        holdout[
            ["model", "params", "rmse", "bias", "spearman", "slope", "std_ratio", "regret_at_1", "lower_tail_optimism"]
        ]
        .head(12)
        .to_markdown(index=False, floatfmt=".6f"),
        "",
        "## All-900M Leave-Scale-Out Metrics",
        "",
        all900_sorted[
            ["model", "params", "rmse", "bias", "spearman", "slope", "std_ratio", "regret_at_1", "lower_tail_optimism"]
        ]
        .head(12)
        .to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Fixed-340M Drop Summary",
        "",
        drops.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Raw 340M Optimum Diagnostics",
        "",
        raw_340[
            [
                "model",
                "predicted_bpb",
                "hard_corner_flag",
                "phase0_family_collapse_flag",
                "phase1_family_collapse_flag",
                "phase1_tech_collapse_flag",
                "p0_support_inv_l2",
                "p1_support_inv_l2",
                "p0_broad_text_share",
                "p1_broad_text_share",
                "p0_tech_code_share",
                "p1_tech_code_share",
                "p0_reasoning_share",
                "p1_reasoning_share",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Plot Models",
        "",
        ", ".join(plot_models),
    ]
    (out_dir / "REPORT.md").write_text("\n".join(report) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-root", type=Path, default=DEFAULT_PACKET_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    module = load_module(SESSION2_SCRIPT)
    data = module.load_packet(args.packet_root)
    feature_factory = EnhancedFeatureFactory(module.FeatureFactory(data), module)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "plots").mkdir(parents=True, exist_ok=True)

    specs = variants()
    models = {}
    predictions = {}
    for spec in specs:
        model = fit_variant(module, data, feature_factory, data.seed7_train, spec)
        models[spec.name] = model
        predictions[spec.name] = np.asarray(model.predict_all(), dtype=float)

    metrics = evaluate_subsets(data, models, predictions)
    all900 = all900_protocol(module, data, feature_factory, specs)
    drop_details, drop_summary = fixed_drop_summary(data, predictions)
    pred_frame = predictions_frame(data, predictions)
    optima = optimum_diagnostics(data, feature_factory, models, np.random.default_rng(20260424))
    plot_models = select_plot_models(metrics)

    metrics.to_csv(args.out_dir / "metrics_seed7_fit.csv", index=False)
    all900.to_csv(args.out_dir / "all900_protocol_metrics.csv", index=False)
    drop_details.to_csv(args.out_dir / "fixed340_drop_pairs.csv", index=False)
    drop_summary.to_csv(args.out_dir / "fixed340_drop_summary.csv", index=False)
    pred_frame.to_csv(args.out_dir / "predictions_seed7_fit.csv", index=False)
    optima.to_csv(args.out_dir / "optimum_diagnostics.csv", index=False)
    with (args.out_dir / "variant_specs.json").open("w") as handle:
        json.dump([spec.__dict__ for spec in specs], handle, indent=2)

    plot_predicted_vs_actual(
        pred_frame,
        metrics,
        plot_models,
        "fixed_340m_10p4b",
        "fixed_340m_10p4b",
        args.out_dir / "plots/fixed340_pred_vs_actual_top_variants.png",
    )
    plot_predicted_vs_actual(
        pred_frame,
        metrics,
        plot_models,
        "all_900m_24b",
        "all_900m_24b_seed7_fit",
        args.out_dir / "plots/all900_pred_vs_actual_seed7_fit_top_variants.png",
    )
    write_report(args.out_dir, metrics, all900, drop_summary, optima, plot_models)

    print("Best fixed-340M:")
    print(
        metrics[metrics["subset"] == "fixed_340m_10p4b"]
        .sort_values("rmse")[["model", "params", "rmse", "spearman", "slope", "std_ratio"]]
        .head(8)
        .to_string(index=False)
    )
    print("\nAll-900M protocol:")
    print(
        all900.sort_values("rmse")[["model", "params", "rmse", "spearman", "slope", "std_ratio"]]
        .head(8)
        .to_string(index=False)
    )
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    main()
