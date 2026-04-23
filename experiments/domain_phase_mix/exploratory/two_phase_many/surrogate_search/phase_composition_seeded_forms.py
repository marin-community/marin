# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two seeded phase-composition surrogates recovered from the second Yixin packet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    MANY_DOMAIN_TARGET,
    PacketData,
    ParameterCount,
    load_two_phase_many_packet,
)

jax.config.update("jax_enable_x64", True)

EPS = 1e-9
INNER_Z_CLIP = 3.0
OUTER_EXP_CLIP = 6.0

RAW_FEATURE_NAMES = (
    "phase_sum__dolmino_stack_edu_fim",
    "phase_sum__dolma3_stack_edu",
    "phase_sum__dolmino_stem_heavy_crawl",
    "phase_alr__phase_1_dolma3_arxiv__over__phase_1_dolmino_synth_thinking",
    "phase_gap__dolma3_stack_edu",
    "phase_sum__dolmino_synth_code",
    "phase0_phase1_epoch_prod_total",
    "phase_epochs__phase_1_dolmino_stack_edu_fim",
    "phase_epoch_sum__dolmino_stack_edu_fim",
    "phase_ilr__32",
    "phase_alr__phase_0_dolma3_cc_entertainment_high__over__phase_1_dolmino_synth_thinking",
    "phase_gap__dolmino_stack_edu_fim",
    "phase_alr__phase_1_dolma3_cc_history_and_geography_high__over__phase_1_dolmino_synth_thinking",
    "phase_epoch_sum__dolma3_stack_edu",
    "phase_epochs__phase_1_dolma3_stack_edu",
    "phase_ilr__66",
    "phase_sum__dolma3_cc_history_and_geography_low",
)
RAW_FEATURE_INDEX = {name: idx for idx, name in enumerate(RAW_FEATURE_NAMES)}

BALANCED_SEEDED_TERMS = (
    "expneg_phase_sum__dolmino_stack_edu_fim",
    "expneg_phase_sum__dolma3_stack_edu",
    "exp(expneg_phase_sum__dolmino_stem_heavy_crawl)",
    "negrelu_phase_alr__phase_1_dolma3_arxiv__over__phase_1_dolmino_synth_thinking",
    "exp_phase_gap__dolma3_stack_edu",
    "invsq1p_phase_sum__dolmino_synth_code",
    "expneg_phase_sum__dolmino_stem_heavy_crawl",
    "expneg_phase0_phase1_epoch_prod_total",
    "phase_epochs__phase_1_dolmino_stack_edu_fim",
)

HIGH_SPEARMAN_TERMS = (
    "exp_phase_epoch_sum__dolmino_stack_edu_fim",
    "exp_phase_ilr__32",
    "exp_phase_sum__dolma3_stack_edu",
    "expneg_phase0_phase1_epoch_prod_total",
    "expneg_phase_alr__phase_0_dolma3_cc_entertainment_high__over__phase_1_dolmino_synth_thinking",
    "expneg_phase_sum__dolmino_stack_edu_fim",
    "invsq1p_phase_gap__dolmino_stack_edu_fim",
    "negrelu_phase_alr__phase_1_dolma3_cc_history_and_geography_high__over__phase_1_dolmino_synth_thinking",
    "negrelu_phase_epoch_sum__dolma3_stack_edu",
    "negrelu_phase_epochs__phase_1_dolma3_stack_edu",
    "negrelu_phase_ilr__66",
    "phase_epochs__phase_1_dolmino_stack_edu_fim",
    "phase_sum__dolma3_cc_history_and_geography_low",
    "phase_sum__dolmino_stem_heavy_crawl",
)


@dataclass(frozen=True)
class SeededVariantSpec:
    """Static spec for one collaborator variant."""

    key: str
    display_name: str
    selected_terms: tuple[str, ...]


BALANCED_SEEDED_VARIANT = SeededVariantSpec(
    key="balanced_seeded",
    display_name="Balanced Seeded",
    selected_terms=BALANCED_SEEDED_TERMS,
)
HIGH_SPEARMAN_VARIANT = SeededVariantSpec(
    key="high_spearman",
    display_name="High Spearman",
    selected_terms=HIGH_SPEARMAN_TERMS,
)
SEEDED_VARIANTS = (BALANCED_SEEDED_VARIANT, HIGH_SPEARMAN_VARIANT)


@dataclass(frozen=True)
class OutlierFilterSummary:
    """Summary of the target outlier filter."""

    lower: float
    upper: float
    kept_rows: int
    dropped_rows: int


def filter_target_outliers(y: np.ndarray) -> tuple[np.ndarray, OutlierFilterSummary]:
    """Filter 3-IQR target outliers exactly as in the collaborator scripts."""
    mask = np.isfinite(y)
    q1, q3 = np.quantile(y[mask], [0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 3.0 * iqr
    upper = q3 + 3.0 * iqr
    keep = mask & (y >= lower) & (y <= upper)
    return keep, OutlierFilterSummary(
        lower=float(lower),
        upper=float(upper),
        kept_rows=int(np.sum(keep)),
        dropped_rows=int(len(y) - np.sum(keep)),
    )


def helmert_basis(n: int) -> np.ndarray:
    """Return the normalized Helmert basis used in the collaborator script."""
    basis = np.zeros((n - 1, n), dtype=float)
    for i in range(1, n):
        basis[i - 1, :i] = 1.0 / i
        basis[i - 1, i] = -1.0
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    return basis


def load_phase_composition_seeded_packet(target: str = MANY_DOMAIN_TARGET) -> PacketData:
    """Load the many-domain packet used by the collaborator scripts."""
    return load_two_phase_many_packet(target=target)


def _safe_std(values: np.ndarray) -> np.ndarray:
    std = np.asarray(values, dtype=float).std(axis=0)
    return np.where(std == 0.0, 1.0, std)


class PhaseCompositionSeededSurrogate:
    """Seeded linear surrogate from the second collaborator packet."""

    def __init__(self, data: PacketData, variant: SeededVariantSpec):
        self.data = data
        self.variant = variant
        self._domain_index = {domain_name: idx for idx, domain_name in enumerate(data.domain_names)}
        self._stack_fim_idx = self._domain_index["dolmino_stack_edu_fim"]
        self._stack_edu_idx = self._domain_index["dolma3_stack_edu"]
        self._stem_idx = self._domain_index["dolmino_stem_heavy_crawl"]
        self._arxiv_idx = self._domain_index["dolma3_arxiv"]
        self._thinking_idx = self._domain_index["dolmino_synth_thinking"]
        self._code_idx = self._domain_index["dolmino_synth_code"]
        self._ent_high_idx = self._domain_index["dolma3_cc/entertainment_high"]
        self._hist_high_idx = self._domain_index["dolma3_cc/history_and_geography_high"]
        self._hist_low_idx = self._domain_index["dolma3_cc/history_and_geography_low"]
        basis = helmert_basis(2 * data.m)
        self._ilr_basis32 = np.asarray(basis[32], dtype=float)
        self._ilr_basis66 = np.asarray(basis[66], dtype=float)
        self.raw_feature_mean_: np.ndarray | None = None
        self.raw_feature_std_: np.ndarray | None = None
        self.term_mean_: np.ndarray | None = None
        self.term_std_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def parameter_count(self) -> ParameterCount:
        """Return the standardized parameter-count breakdown."""
        return ParameterCount(
            linear_coefficients=len(self.variant.selected_terms),
            intercept=1,
            global_shape_parameters=0,
        )

    def _raw_feature_array(self, weights: np.ndarray) -> np.ndarray:
        arr = np.asarray(weights, dtype=float)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        p0 = arr[:, 0, :]
        p1 = arr[:, 1, :]
        x0 = 0.5 * p0
        x1 = 0.5 * p1
        e0 = p0 * self.data.c0[None, :]
        e1 = p1 * self.data.c1[None, :]

        comp = np.concatenate([x0, x1], axis=1)
        log_comp = np.log(comp + EPS)
        clr = log_comp - log_comp.mean(axis=1, keepdims=True)
        ilr32 = clr @ self._ilr_basis32
        ilr66 = clr @ self._ilr_basis66

        raw = np.empty((arr.shape[0], len(RAW_FEATURE_NAMES)), dtype=float)
        raw[:, RAW_FEATURE_INDEX["phase_sum__dolmino_stack_edu_fim"]] = (
            x0[:, self._stack_fim_idx] + x1[:, self._stack_fim_idx]
        )
        raw[:, RAW_FEATURE_INDEX["phase_sum__dolma3_stack_edu"]] = (
            x0[:, self._stack_edu_idx] + x1[:, self._stack_edu_idx]
        )
        raw[:, RAW_FEATURE_INDEX["phase_sum__dolmino_stem_heavy_crawl"]] = x0[:, self._stem_idx] + x1[:, self._stem_idx]
        raw[:, RAW_FEATURE_INDEX["phase_alr__phase_1_dolma3_arxiv__over__phase_1_dolmino_synth_thinking"]] = np.log(
            (x1[:, self._arxiv_idx] + EPS) / (x1[:, self._thinking_idx] + EPS)
        )
        raw[:, RAW_FEATURE_INDEX["phase_gap__dolma3_stack_edu"]] = (
            x1[:, self._stack_edu_idx] - x0[:, self._stack_edu_idx]
        )
        raw[:, RAW_FEATURE_INDEX["phase_sum__dolmino_synth_code"]] = x0[:, self._code_idx] + x1[:, self._code_idx]
        raw[:, RAW_FEATURE_INDEX["phase0_phase1_epoch_prod_total"]] = e0.sum(axis=1) * e1.sum(axis=1)
        raw[:, RAW_FEATURE_INDEX["phase_epochs__phase_1_dolmino_stack_edu_fim"]] = e1[:, self._stack_fim_idx]
        raw[:, RAW_FEATURE_INDEX["phase_epoch_sum__dolmino_stack_edu_fim"]] = (
            e0[:, self._stack_fim_idx] + e1[:, self._stack_fim_idx]
        )
        raw[:, RAW_FEATURE_INDEX["phase_ilr__32"]] = ilr32
        raw[
            :, RAW_FEATURE_INDEX["phase_alr__phase_0_dolma3_cc_entertainment_high__over__phase_1_dolmino_synth_thinking"]
        ] = np.log((x0[:, self._ent_high_idx] + EPS) / (x1[:, self._thinking_idx] + EPS))
        raw[:, RAW_FEATURE_INDEX["phase_gap__dolmino_stack_edu_fim"]] = (
            x1[:, self._stack_fim_idx] - x0[:, self._stack_fim_idx]
        )
        raw[
            :,
            RAW_FEATURE_INDEX[
                "phase_alr__phase_1_dolma3_cc_history_and_geography_high__over__phase_1_dolmino_synth_thinking"
            ],
        ] = np.log((x1[:, self._hist_high_idx] + EPS) / (x1[:, self._thinking_idx] + EPS))
        raw[:, RAW_FEATURE_INDEX["phase_epoch_sum__dolma3_stack_edu"]] = (
            e0[:, self._stack_edu_idx] + e1[:, self._stack_edu_idx]
        )
        raw[:, RAW_FEATURE_INDEX["phase_epochs__phase_1_dolma3_stack_edu"]] = e1[:, self._stack_edu_idx]
        raw[:, RAW_FEATURE_INDEX["phase_ilr__66"]] = ilr66
        raw[:, RAW_FEATURE_INDEX["phase_sum__dolma3_cc_history_and_geography_low"]] = (
            x0[:, self._hist_low_idx] + x1[:, self._hist_low_idx]
        )
        return raw

    @staticmethod
    def _selected_term_array_from_raw_z(raw_z: np.ndarray, variant: SeededVariantSpec) -> np.ndarray:
        terms = np.empty((raw_z.shape[0], len(variant.selected_terms)), dtype=float)
        if variant.key == BALANCED_SEEDED_VARIANT.key:
            stem_expneg = np.exp(
                np.clip(-raw_z[:, RAW_FEATURE_INDEX["phase_sum__dolmino_stem_heavy_crawl"]], -INNER_Z_CLIP, INNER_Z_CLIP)
            )
            terms[:, 0] = np.exp(
                np.clip(-raw_z[:, RAW_FEATURE_INDEX["phase_sum__dolmino_stack_edu_fim"]], -INNER_Z_CLIP, INNER_Z_CLIP)
            )
            terms[:, 1] = np.exp(
                np.clip(-raw_z[:, RAW_FEATURE_INDEX["phase_sum__dolma3_stack_edu"]], -INNER_Z_CLIP, INNER_Z_CLIP)
            )
            terms[:, 2] = np.exp(np.clip(stem_expneg, -OUTER_EXP_CLIP, OUTER_EXP_CLIP))
            terms[:, 3] = np.maximum(
                -raw_z[:, RAW_FEATURE_INDEX["phase_alr__phase_1_dolma3_arxiv__over__phase_1_dolmino_synth_thinking"]],
                0.0,
            )
            terms[:, 4] = np.exp(
                np.clip(raw_z[:, RAW_FEATURE_INDEX["phase_gap__dolma3_stack_edu"]], -INNER_Z_CLIP, INNER_Z_CLIP)
            )
            synth_code_z = raw_z[:, RAW_FEATURE_INDEX["phase_sum__dolmino_synth_code"]]
            terms[:, 5] = 1.0 / np.sqrt(1.0 + synth_code_z**2)
            terms[:, 6] = stem_expneg
            terms[:, 7] = np.exp(
                np.clip(-raw_z[:, RAW_FEATURE_INDEX["phase0_phase1_epoch_prod_total"]], -INNER_Z_CLIP, INNER_Z_CLIP)
            )
            terms[:, 8] = raw_z[:, RAW_FEATURE_INDEX["phase_epochs__phase_1_dolmino_stack_edu_fim"]]
            return terms

        terms[:, 0] = np.exp(
            np.clip(raw_z[:, RAW_FEATURE_INDEX["phase_epoch_sum__dolmino_stack_edu_fim"]], -INNER_Z_CLIP, INNER_Z_CLIP)
        )
        terms[:, 1] = np.exp(np.clip(raw_z[:, RAW_FEATURE_INDEX["phase_ilr__32"]], -INNER_Z_CLIP, INNER_Z_CLIP))
        terms[:, 2] = np.exp(
            np.clip(raw_z[:, RAW_FEATURE_INDEX["phase_sum__dolma3_stack_edu"]], -INNER_Z_CLIP, INNER_Z_CLIP)
        )
        terms[:, 3] = np.exp(
            np.clip(-raw_z[:, RAW_FEATURE_INDEX["phase0_phase1_epoch_prod_total"]], -INNER_Z_CLIP, INNER_Z_CLIP)
        )
        terms[:, 4] = np.exp(
            np.clip(
                -raw_z[
                    :,
                    RAW_FEATURE_INDEX[
                        "phase_alr__phase_0_dolma3_cc_entertainment_high__over__phase_1_dolmino_synth_thinking"
                    ],
                ],
                -INNER_Z_CLIP,
                INNER_Z_CLIP,
            )
        )
        terms[:, 5] = np.exp(
            np.clip(-raw_z[:, RAW_FEATURE_INDEX["phase_sum__dolmino_stack_edu_fim"]], -INNER_Z_CLIP, INNER_Z_CLIP)
        )
        gap_fim_z = raw_z[:, RAW_FEATURE_INDEX["phase_gap__dolmino_stack_edu_fim"]]
        terms[:, 6] = 1.0 / np.sqrt(1.0 + gap_fim_z**2)
        terms[:, 7] = np.maximum(
            -raw_z[
                :,
                RAW_FEATURE_INDEX[
                    "phase_alr__phase_1_dolma3_cc_history_and_geography_high__over__phase_1_dolmino_synth_thinking"
                ],
            ],
            0.0,
        )
        terms[:, 8] = np.maximum(-raw_z[:, RAW_FEATURE_INDEX["phase_epoch_sum__dolma3_stack_edu"]], 0.0)
        terms[:, 9] = np.maximum(-raw_z[:, RAW_FEATURE_INDEX["phase_epochs__phase_1_dolma3_stack_edu"]], 0.0)
        terms[:, 10] = np.maximum(-raw_z[:, RAW_FEATURE_INDEX["phase_ilr__66"]], 0.0)
        terms[:, 11] = raw_z[:, RAW_FEATURE_INDEX["phase_epochs__phase_1_dolmino_stack_edu_fim"]]
        terms[:, 12] = raw_z[:, RAW_FEATURE_INDEX["phase_sum__dolma3_cc_history_and_geography_low"]]
        terms[:, 13] = raw_z[:, RAW_FEATURE_INDEX["phase_sum__dolmino_stem_heavy_crawl"]]
        return terms

    def _selected_term_array(
        self,
        weights: np.ndarray,
        *,
        raw_feature_mean: np.ndarray,
        raw_feature_std: np.ndarray,
    ) -> np.ndarray:
        raw = self._raw_feature_array(weights)
        raw_z = (raw - raw_feature_mean[None, :]) / raw_feature_std[None, :]
        return self._selected_term_array_from_raw_z(raw_z, self.variant)

    def fit(self, weights: np.ndarray, y: np.ndarray) -> PhaseCompositionSeededSurrogate:
        """Fit the selected-term full model."""
        raw = self._raw_feature_array(weights)
        self.raw_feature_mean_ = raw.mean(axis=0)
        self.raw_feature_std_ = _safe_std(raw)
        terms = self._selected_term_array(
            weights,
            raw_feature_mean=self.raw_feature_mean_,
            raw_feature_std=self.raw_feature_std_,
        )
        self.term_mean_ = terms.mean(axis=0)
        self.term_std_ = _safe_std(terms)
        design = (terms - self.term_mean_[None, :]) / self.term_std_[None, :]
        fit = LinearRegression().fit(design, np.asarray(y, dtype=float))
        self.coef_ = np.asarray(fit.coef_, dtype=float)
        self.intercept_ = float(fit.intercept_)
        return self

    def predict(self, weights: np.ndarray) -> np.ndarray:
        """Predict the target for one or more two-phase schedules."""
        if self.raw_feature_mean_ is None or self.raw_feature_std_ is None:
            raise RuntimeError("Model must be fit before prediction")
        if self.term_mean_ is None or self.term_std_ is None or self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model must be fit before prediction")
        terms = self._selected_term_array(
            weights,
            raw_feature_mean=self.raw_feature_mean_,
            raw_feature_std=self.raw_feature_std_,
        )
        design = (terms - self.term_mean_[None, :]) / self.term_std_[None, :]
        return np.asarray(self.intercept_ + design @ self.coef_, dtype=float)

    def coefficient_table(self) -> dict[str, float]:
        """Return the fitted term coefficients."""
        if self.coef_ is None:
            raise RuntimeError("Model must be fit before coefficients are available")
        return {name: float(coef) for name, coef in zip(self.variant.selected_terms, self.coef_, strict=True)}

    def fit_summary(self, weights: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Return train metrics for the fitted model."""
        pred = self.predict(weights)
        rho = spearmanr(y, pred).statistic
        return {
            "train_r2": float(r2_score(y, pred)),
            "train_spearman": float(0.0 if rho is None or not np.isfinite(rho) else rho),
        }


def reproduction_cv_summary(
    data: PacketData,
    variant: SeededVariantSpec,
    *,
    n_splits: int | None = None,
) -> tuple[dict[str, Any], PhaseCompositionSeededSurrogate]:
    """Reproduce the collaborator's reported CV and full-fit coefficients."""
    y_full = np.asarray(data.y, dtype=float)
    keep, filt = filter_target_outliers(y_full)
    weights = np.asarray(data.w[keep], dtype=float)
    y = np.asarray(y_full[keep], dtype=float)

    model = PhaseCompositionSeededSurrogate(data, variant).fit(weights, y)
    terms = model._selected_term_array(
        weights,
        raw_feature_mean=model.raw_feature_mean_,
        raw_feature_std=model.raw_feature_std_,
    )
    splits = min(5, max(2, len(y) // 8)) if n_splits is None else n_splits
    cv = KFold(n_splits=splits, shuffle=True, random_state=0)
    fold_rows: list[dict[str, float]] = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(terms), start=1):
        x_train = terms[train_idx]
        x_test = terms[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        mu = x_train.mean(axis=0)
        sd = np.where(x_train.std(axis=0) == 0.0, 1.0, x_train.std(axis=0))
        pred = LinearRegression().fit((x_train - mu) / sd, y_train).predict((x_test - mu) / sd)
        rho = spearmanr(y_test, pred).statistic
        fold_rows.append(
            {
                "fold": float(fold),
                "r2": float(r2_score(y_test, pred)),
                "spearman": float(0.0 if rho is None or not np.isfinite(rho) else rho),
            }
        )

    payload = {
        "variant_key": variant.key,
        "variant_name": variant.display_name,
        "target": MANY_DOMAIN_TARGET,
        "rows": len(y),
        "outlier_filter": {
            "lower": filt.lower,
            "upper": filt.upper,
            "kept_rows": filt.kept_rows,
            "dropped_rows": filt.dropped_rows,
        },
        "selected_terms": list(variant.selected_terms),
        "cv_r2_mean": float(np.mean([row["r2"] for row in fold_rows])),
        "cv_spearman_mean": float(np.mean([row["spearman"] for row in fold_rows])),
        "fold_metrics": fold_rows,
        "full_fit_intercept": float(model.intercept_),
        "full_fit_coefficients": model.coefficient_table(),
        **model.fit_summary(weights, y),
    }
    return payload, model


def optimize_phase_composition_seeded_model(
    data: PacketData,
    model: PhaseCompositionSeededSurrogate,
    *,
    n_random: int = 20,
    seed: int = 0,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Optimize the fitted seeded model over the two phase simplices."""
    if model.raw_feature_mean_ is None or model.raw_feature_std_ is None:
        raise RuntimeError("Model must be fit before optimization")
    if model.term_mean_ is None or model.term_std_ is None or model.coef_ is None or model.intercept_ is None:
        raise RuntimeError("Model must be fit before optimization")

    n_domains = data.m
    rng = np.random.default_rng(seed)
    raw_mu = jnp.asarray(model.raw_feature_mean_, dtype=jnp.float64)
    raw_sd = jnp.asarray(model.raw_feature_std_, dtype=jnp.float64)
    term_mu = jnp.asarray(model.term_mean_, dtype=jnp.float64)
    term_sd = jnp.asarray(model.term_std_, dtype=jnp.float64)
    coef = jnp.asarray(model.coef_, dtype=jnp.float64)
    intercept = jnp.asarray(model.intercept_, dtype=jnp.float64)
    c0 = jnp.asarray(data.c0, dtype=jnp.float64)
    c1 = jnp.asarray(data.c1, dtype=jnp.float64)
    ilr_basis32 = jnp.asarray(model._ilr_basis32, dtype=jnp.float64)
    ilr_basis66 = jnp.asarray(model._ilr_basis66, dtype=jnp.float64)

    idx = model._domain_index
    stack_fim_idx = idx["dolmino_stack_edu_fim"]
    stack_edu_idx = idx["dolma3_stack_edu"]
    stem_idx = idx["dolmino_stem_heavy_crawl"]
    arxiv_idx = idx["dolma3_arxiv"]
    thinking_idx = idx["dolmino_synth_thinking"]
    code_idx = idx["dolmino_synth_code"]
    ent_high_idx = idx["dolma3_cc/entertainment_high"]
    hist_high_idx = idx["dolma3_cc/history_and_geography_high"]
    hist_low_idx = idx["dolma3_cc/history_and_geography_low"]
    variant = model.variant

    def _raw_features_from_logits(z: jnp.ndarray) -> jnp.ndarray:
        p0 = jax.nn.softmax(z[:n_domains])
        p1 = jax.nn.softmax(z[n_domains:])
        x0 = 0.5 * p0
        x1 = 0.5 * p1
        e0 = p0 * c0
        e1 = p1 * c1
        comp = jnp.concatenate([x0, x1], axis=0)
        clr = jnp.log(comp + EPS)
        clr = clr - jnp.mean(clr)
        ilr32 = jnp.dot(clr, ilr_basis32)
        ilr66 = jnp.dot(clr, ilr_basis66)
        return jnp.array(
            [
                x0[stack_fim_idx] + x1[stack_fim_idx],
                x0[stack_edu_idx] + x1[stack_edu_idx],
                x0[stem_idx] + x1[stem_idx],
                jnp.log((x1[arxiv_idx] + EPS) / (x1[thinking_idx] + EPS)),
                x1[stack_edu_idx] - x0[stack_edu_idx],
                x0[code_idx] + x1[code_idx],
                jnp.sum(e0) * jnp.sum(e1),
                e1[stack_fim_idx],
                e0[stack_fim_idx] + e1[stack_fim_idx],
                ilr32,
                jnp.log((x0[ent_high_idx] + EPS) / (x1[thinking_idx] + EPS)),
                x1[stack_fim_idx] - x0[stack_fim_idx],
                jnp.log((x1[hist_high_idx] + EPS) / (x1[thinking_idx] + EPS)),
                e0[stack_edu_idx] + e1[stack_edu_idx],
                e1[stack_edu_idx],
                ilr66,
                x0[hist_low_idx] + x1[hist_low_idx],
            ],
            dtype=jnp.float64,
        )

    def _selected_terms_from_logits(z: jnp.ndarray) -> jnp.ndarray:
        raw = _raw_features_from_logits(z)
        raw_z = (raw - raw_mu) / raw_sd
        if variant.key == BALANCED_SEEDED_VARIANT.key:
            stem_expneg = jnp.exp(
                jnp.clip(-raw_z[RAW_FEATURE_INDEX["phase_sum__dolmino_stem_heavy_crawl"]], -INNER_Z_CLIP, INNER_Z_CLIP)
            )
            return jnp.array(
                [
                    jnp.exp(
                        jnp.clip(
                            -raw_z[RAW_FEATURE_INDEX["phase_sum__dolmino_stack_edu_fim"]], -INNER_Z_CLIP, INNER_Z_CLIP
                        )
                    ),
                    jnp.exp(
                        jnp.clip(-raw_z[RAW_FEATURE_INDEX["phase_sum__dolma3_stack_edu"]], -INNER_Z_CLIP, INNER_Z_CLIP)
                    ),
                    jnp.exp(jnp.clip(stem_expneg, -OUTER_EXP_CLIP, OUTER_EXP_CLIP)),
                    jnp.maximum(
                        -raw_z[
                            RAW_FEATURE_INDEX["phase_alr__phase_1_dolma3_arxiv__over__phase_1_dolmino_synth_thinking"]
                        ],
                        0.0,
                    ),
                    jnp.exp(
                        jnp.clip(raw_z[RAW_FEATURE_INDEX["phase_gap__dolma3_stack_edu"]], -INNER_Z_CLIP, INNER_Z_CLIP)
                    ),
                    1.0 / jnp.sqrt(1.0 + raw_z[RAW_FEATURE_INDEX["phase_sum__dolmino_synth_code"]] ** 2),
                    stem_expneg,
                    jnp.exp(
                        jnp.clip(
                            -raw_z[RAW_FEATURE_INDEX["phase0_phase1_epoch_prod_total"]], -INNER_Z_CLIP, INNER_Z_CLIP
                        )
                    ),
                    raw_z[RAW_FEATURE_INDEX["phase_epochs__phase_1_dolmino_stack_edu_fim"]],
                ],
                dtype=jnp.float64,
            )

        return jnp.array(
            [
                jnp.exp(
                    jnp.clip(
                        raw_z[RAW_FEATURE_INDEX["phase_epoch_sum__dolmino_stack_edu_fim"]], -INNER_Z_CLIP, INNER_Z_CLIP
                    )
                ),
                jnp.exp(jnp.clip(raw_z[RAW_FEATURE_INDEX["phase_ilr__32"]], -INNER_Z_CLIP, INNER_Z_CLIP)),
                jnp.exp(jnp.clip(raw_z[RAW_FEATURE_INDEX["phase_sum__dolma3_stack_edu"]], -INNER_Z_CLIP, INNER_Z_CLIP)),
                jnp.exp(
                    jnp.clip(-raw_z[RAW_FEATURE_INDEX["phase0_phase1_epoch_prod_total"]], -INNER_Z_CLIP, INNER_Z_CLIP)
                ),
                jnp.exp(
                    jnp.clip(
                        -raw_z[
                            RAW_FEATURE_INDEX[
                                "phase_alr__phase_0_dolma3_cc_entertainment_high__over__phase_1_dolmino_synth_thinking"
                            ]
                        ],
                        -INNER_Z_CLIP,
                        INNER_Z_CLIP,
                    )
                ),
                jnp.exp(
                    jnp.clip(-raw_z[RAW_FEATURE_INDEX["phase_sum__dolmino_stack_edu_fim"]], -INNER_Z_CLIP, INNER_Z_CLIP)
                ),
                1.0 / jnp.sqrt(1.0 + raw_z[RAW_FEATURE_INDEX["phase_gap__dolmino_stack_edu_fim"]] ** 2),
                jnp.maximum(
                    -raw_z[
                        RAW_FEATURE_INDEX[
                            "phase_alr__phase_1_dolma3_cc_history_and_geography_high__over__phase_1_dolmino_synth_thinking"
                        ]
                    ],
                    0.0,
                ),
                jnp.maximum(-raw_z[RAW_FEATURE_INDEX["phase_epoch_sum__dolma3_stack_edu"]], 0.0),
                jnp.maximum(-raw_z[RAW_FEATURE_INDEX["phase_epochs__phase_1_dolma3_stack_edu"]], 0.0),
                jnp.maximum(-raw_z[RAW_FEATURE_INDEX["phase_ilr__66"]], 0.0),
                raw_z[RAW_FEATURE_INDEX["phase_epochs__phase_1_dolmino_stack_edu_fim"]],
                raw_z[RAW_FEATURE_INDEX["phase_sum__dolma3_cc_history_and_geography_low"]],
                raw_z[RAW_FEATURE_INDEX["phase_sum__dolmino_stem_heavy_crawl"]],
            ],
            dtype=jnp.float64,
        )

    def _objective(z: jnp.ndarray) -> jnp.ndarray:
        terms = _selected_terms_from_logits(z)
        design = (terms - term_mu) / term_sd
        return intercept + jnp.dot(design, coef)

    value_and_grad = jax.jit(jax.value_and_grad(_objective))

    def scipy_value_grad(z: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = value_and_grad(jnp.asarray(z, dtype=jnp.float64))
        return float(value), np.asarray(grad, dtype=float)

    starts = [
        np.zeros(2 * n_domains, dtype=float),
        *[
            np.concatenate(
                [
                    np.log(rng.dirichlet(np.ones(n_domains))),
                    np.log(rng.dirichlet(np.ones(n_domains))),
                ]
            )
            for _ in range(n_random)
        ],
    ]
    best_result = None
    best_value = float("inf")
    for start in starts:
        result = minimize(scipy_value_grad, start, jac=True, method="L-BFGS-B", options={"maxiter": 800})
        if result.fun < best_value:
            best_result = result
            best_value = float(result.fun)

    if best_result is None:
        raise RuntimeError("Optimization failed")

    logits0 = best_result.x[:n_domains]
    logits1 = best_result.x[n_domains:]
    p0 = np.exp(logits0 - np.max(logits0))
    p0 /= np.sum(p0)
    p1 = np.exp(logits1 - np.max(logits1))
    p1 /= np.sum(p1)
    return best_result, p0, p1
