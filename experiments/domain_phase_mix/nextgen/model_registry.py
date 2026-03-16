# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Model registry and fit/propose adapters for next-gen mixture loops.

The registry allows adding both parametric models and policy-artifact models
without modifying the orchestration DAG. Add a model adapter, include its name
in ``LoopConfig.model_names``, and re-submit.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from experiments.domain_phase_mix.exploratory.general_scaling_models import (
    DatasetSpec,
    GENERAL_MODELS,
    GeneralModelSpec,
)
from experiments.domain_phase_mix.nextgen.dataset_metadata import resolve_dataset_epoch_metadata
from experiments.domain_phase_mix.nextgen.contracts import Candidate, LoopConfig, PolicyArtifactRef
from experiments.domain_phase_mix.nextgen.utils import stable_hash

logger = logging.getLogger(__name__)

_PHASE_DOMAIN_COL = re.compile(r"^phase_(\d+)_(.+)$")
_MODEL_REGISTRY: dict[str, ModelAdapter] = {}
_DEFAULT_REGISTRY_INITIALIZED = False


class ModelAdapter(Protocol):
    """Adapter interface used by the next-gen fit/propose stage."""

    @property
    def name(self) -> str: ...

    def applicable(self, spec: DatasetSpec) -> bool: ...

    def fit_and_propose(
        self,
        *,
        loop: LoopConfig,
        spec: DatasetSpec,
        training_setup: dict[str, Any],
    ) -> tuple[Candidate, int | None]: ...


@dataclass(frozen=True)
class GeneralScalingModelAdapter:
    """Adapter wrapper around ``GENERAL_MODELS`` entries."""

    model_spec: GeneralModelSpec

    @property
    def name(self) -> str:
        return self.model_spec.name

    def applicable(self, spec: DatasetSpec) -> bool:
        return self.model_spec.applicable(spec)

    def fit_and_propose(
        self,
        *,
        loop: LoopConfig,
        spec: DatasetSpec,
        training_setup: dict[str, Any],
    ) -> tuple[Candidate, int | None]:
        predict_fn, info = self.model_spec.fit_fn(spec)
        candidate = _propose_top1_candidate(
            loop=loop,
            model_name=self.name,
            predict_fn=predict_fn,
            spec=spec,
            training_setup=training_setup,
        )
        n_params = int(info.get("n_params")) if isinstance(info, dict) and info.get("n_params") else None
        return candidate, n_params


@dataclass(frozen=True)
class PolicyArtifactModelAdapter:
    """Adapter for externally-trained offline RL policies.

    This model does not fit on local data. It emits a policy candidate that is
    evaluated through the same validation flow as parametric candidates.
    """

    model_name: str
    policy_uri: str
    policy_format: str = "json"
    predicted_objective: float | None = None

    @property
    def name(self) -> str:
        return self.model_name

    def applicable(self, spec: DatasetSpec) -> bool:
        return spec.R > 0

    def fit_and_propose(
        self,
        *,
        loop: LoopConfig,
        spec: DatasetSpec,
        training_setup: dict[str, Any],
    ) -> tuple[Candidate, int | None]:
        fallback_pred = float(np.nanmean(spec.y))
        pred = self.predicted_objective if self.predicted_objective is not None else fallback_pred
        payload = {
            "kind": "policy",
            "policy_ref": {"uri": self.policy_uri, "format": self.policy_format},
            "objective_metric": loop.objective_metric,
            "training_setup": training_setup,
        }
        candidate_id = stable_hash(payload, prefix="cand")
        return (
            Candidate(
                candidate_id=candidate_id,
                model_name=self.name,
                kind="policy",
                phase_weights=None,
                policy_ref=PolicyArtifactRef(uri=self.policy_uri, format=self.policy_format),
                predicted_objective=float(pred),
            ),
            None,
        )


@dataclass(frozen=True)
class ModelFitArtifact:
    """Serializable fit summary for one model."""

    model_name: str
    n_params: int | None
    predicted_objective: float | None
    candidate_id: str | None
    error: str | None = None


@dataclass(frozen=True)
class FitAndProposeResult:
    """Output from fitting models and proposing candidates."""

    candidates: list[Candidate]
    model_fits: list[ModelFitArtifact]


def available_model_names() -> set[str]:
    """Return all registered model names."""
    _ensure_default_registry()
    return set(_MODEL_REGISTRY.keys())


def register_model_adapter(adapter: ModelAdapter, *, overwrite: bool = False) -> None:
    """Register a model adapter for fit/propose execution."""
    if adapter.name in _MODEL_REGISTRY and not overwrite:
        raise ValueError(f"Model adapter '{adapter.name}' is already registered")
    _MODEL_REGISTRY[adapter.name] = adapter


def register_policy_artifact_model(
    *,
    model_name: str,
    policy_uri: str,
    policy_format: str = "json",
    predicted_objective: float | None = None,
    overwrite: bool = False,
) -> None:
    """Register a policy model adapter for offline-RL policy evaluation."""
    register_model_adapter(
        PolicyArtifactModelAdapter(
            model_name=model_name,
            policy_uri=policy_uri,
            policy_format=policy_format,
            predicted_objective=predicted_objective,
        ),
        overwrite=overwrite,
    )


def _ensure_default_registry() -> None:
    global _DEFAULT_REGISTRY_INITIALIZED
    if _DEFAULT_REGISTRY_INITIALIZED:
        return
    for spec in GENERAL_MODELS:
        register_model_adapter(GeneralScalingModelAdapter(model_spec=spec), overwrite=True)
    _DEFAULT_REGISTRY_INITIALIZED = True


def _extract_phase_domain_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    phases: set[int] = set()
    domains_by_phase: dict[int, list[str]] = {}

    for col in df.columns:
        match = _PHASE_DOMAIN_COL.match(col)
        if not match:
            continue
        phase_idx = int(match.group(1))
        domain = match.group(2)
        if domain.endswith("_epochs"):
            continue
        phases.add(phase_idx)
        domains_by_phase.setdefault(phase_idx, [])
        if domain not in domains_by_phase[phase_idx]:
            domains_by_phase[phase_idx].append(domain)

    phase_ids = sorted(phases)
    if not phase_ids:
        raise ValueError("No phase/domain columns found in merged run dataframe")

    phase_names = [f"phase_{idx}" for idx in phase_ids]
    domains = domains_by_phase[phase_ids[0]]
    return phase_names, domains


def _build_weights_tensor(df: pd.DataFrame, phase_names: list[str], domains: list[str]) -> np.ndarray:
    weights = np.zeros((len(df), len(phase_names), len(domains)), dtype=float)
    for p_idx, phase_name in enumerate(phase_names):
        for d_idx, domain_name in enumerate(domains):
            col = f"{phase_name}_{domain_name}"
            if col not in df.columns:
                continue
            weights[:, p_idx, d_idx] = df[col].to_numpy(dtype=float)
    return weights


def _build_dataset_spec(df: pd.DataFrame, objective_metric: str, loop: LoopConfig) -> DatasetSpec:
    if objective_metric not in df.columns:
        raise ValueError(f"Objective metric '{objective_metric}' missing from run table")

    model_df = df[df[objective_metric].notna()].copy()
    if model_df.empty:
        raise ValueError(f"No rows with non-null objective metric '{objective_metric}'")

    phase_names, domains = _extract_phase_domain_columns(model_df)
    weights = _build_weights_tensor(model_df, phase_names, domains)
    y = model_df[objective_metric].to_numpy(dtype=float)

    epoch_multipliers, small_domains = resolve_dataset_epoch_metadata(
        loop=loop,
        phase_names=phase_names,
        domain_names=domains,
    )

    return DatasetSpec(
        weights=weights,
        y=y,
        epoch_multipliers=epoch_multipliers,
        domain_names=domains,
        phase_names=phase_names,
        small_domains=small_domains,
        name="nextgen_merged_runs",
    )


def _sample_simplex_points(rng: np.random.Generator, n_points: int, n_dims: int) -> np.ndarray:
    raw = rng.exponential(1.0, size=(n_points, n_dims))
    return raw / raw.sum(axis=1, keepdims=True)


def _sample_top1_point(
    *,
    loop: LoopConfig,
    predict_fn,
    spec: DatasetSpec,
    model_name: str,
) -> tuple[np.ndarray, float]:
    rng_seed = int(stable_hash({"loop": loop.name, "model": model_name, "seed": loop.candidate_search_seed})[:8], 16)
    rng = np.random.default_rng(rng_seed)
    n_points = max(loop.candidate_search_points, 512)

    points = np.zeros((n_points, spec.N, spec.M), dtype=float)
    for phase_idx in range(spec.N):
        points[:, phase_idx, :] = _sample_simplex_points(rng, n_points, spec.M)

    pred = np.asarray(predict_fn(points), dtype=float)
    if pred.ndim != 1 or pred.shape[0] != n_points:
        raise ValueError(f"Model '{model_name}' produced invalid prediction shape: {pred.shape}")

    finite_mask = np.isfinite(pred)
    if not finite_mask.any():
        raise ValueError(f"Model '{model_name}' produced no finite predictions")

    finite_idx = np.where(finite_mask)[0]
    best_idx = int(finite_idx[np.argmin(pred[finite_mask])])
    best = points[best_idx]
    best_pred = float(pred[best_idx])
    return best, best_pred


def _optimize_top1_point_two_domain(
    *,
    loop: LoopConfig,
    predict_fn,
    spec: DatasetSpec,
    initial_point: np.ndarray,
    initial_pred: float,
    model_name: str,
) -> tuple[np.ndarray, float]:
    if spec.M != 2:
        logger.info(
            "Optimization method scipy_minimize currently supports M=2 only for model '%s'; falling back to sampling",
            model_name,
        )
        return initial_point, initial_pred

    small_domain_idx = spec.small_domains[0] if spec.small_domains else 1
    if small_domain_idx not in (0, 1):
        small_domain_idx = 1
    other_domain_idx = 1 - small_domain_idx

    def _point_from_small_weights(small_weights: np.ndarray) -> np.ndarray:
        point = np.zeros((spec.N, spec.M), dtype=float)
        point[:, small_domain_idx] = small_weights
        point[:, other_domain_idx] = 1.0 - small_weights
        return point

    def _objective(small_weights: np.ndarray) -> float:
        clipped = np.clip(small_weights, 0.0, 1.0)
        point = _point_from_small_weights(clipped)
        value = np.asarray(predict_fn(point[None, :, :]), dtype=float)
        if value.shape != (1,) or not np.isfinite(value[0]):
            return float("inf")
        return float(value[0])

    rng_seed = int(stable_hash({"loop": loop.name, "model": model_name, "opt_seed": loop.candidate_search_seed})[:8], 16)
    rng = np.random.default_rng(rng_seed)
    n_restarts = max(loop.candidate_opt_restarts, 1)
    starts = [initial_point[:, small_domain_idx]]
    starts.extend(rng.uniform(0.0, 1.0, size=(n_restarts, spec.N)))

    best_point = initial_point.copy()
    best_pred = initial_pred
    for x0 in starts:
        try:
            result = minimize(
                _objective,
                x0=np.asarray(x0, dtype=float),
                method="L-BFGS-B",
                bounds=[(0.0, 1.0)] * spec.N,
                options={"maxiter": max(loop.candidate_opt_maxiter, 1)},
            )
        except Exception:
            logger.exception("scipy_minimize failed for model '%s' on one restart", model_name)
            continue

        candidate_small = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
        candidate_pred = _objective(candidate_small)
        if np.isfinite(candidate_pred) and candidate_pred < best_pred:
            best_pred = candidate_pred
            best_point = _point_from_small_weights(candidate_small)

    return best_point, best_pred


def _build_phase_weights(point: np.ndarray, spec: DatasetSpec) -> dict[str, dict[str, float]]:
    return {
        spec.phase_names[p_idx]: {spec.domain_names[d_idx]: float(point[p_idx, d_idx]) for d_idx in range(spec.M)}
        for p_idx in range(spec.N)
    }


def _propose_top1_candidate(
    *,
    loop: LoopConfig,
    model_name: str,
    predict_fn,
    spec: DatasetSpec,
    training_setup: dict[str, Any],
) -> Candidate:
    best_point, best_pred = _sample_top1_point(
        loop=loop,
        predict_fn=predict_fn,
        spec=spec,
        model_name=model_name,
    )

    if loop.candidate_opt_method == "scipy_minimize":
        best_point, best_pred = _optimize_top1_point_two_domain(
            loop=loop,
            predict_fn=predict_fn,
            spec=spec,
            initial_point=best_point,
            initial_pred=best_pred,
            model_name=model_name,
        )

    phase_weights = _build_phase_weights(best_point, spec)

    candidate_payload = {
        "kind": "schedule",
        "phase_weights": phase_weights,
        "objective_metric": loop.objective_metric,
        "training_setup": training_setup,
    }

    candidate_id = stable_hash(candidate_payload, prefix="cand")
    return Candidate(
        candidate_id=candidate_id,
        model_name=model_name,
        kind="schedule",
        phase_weights=phase_weights,
        policy_ref=None,
        predicted_objective=best_pred,
    )


def dedupe_candidates(candidates: list[Candidate]) -> tuple[list[Candidate], dict[str, str]]:
    """Dedupe candidates while preserving model ownership mapping.

    Returns:
        deduped candidates and mapping of model_name -> owning candidate_id
    """
    by_candidate_id: dict[str, Candidate] = {}
    model_to_candidate: dict[str, str] = {}

    for candidate in candidates:
        if candidate.candidate_id not in by_candidate_id:
            by_candidate_id[candidate.candidate_id] = candidate
        owner = by_candidate_id[candidate.candidate_id]
        model_to_candidate[candidate.model_name] = owner.candidate_id

    deduped = sorted(by_candidate_id.values(), key=lambda c: c.candidate_id)
    return deduped, model_to_candidate


def fit_and_propose(
    run_df: pd.DataFrame,
    *,
    loop: LoopConfig,
    training_setup: dict[str, Any],
) -> FitAndProposeResult:
    """Fit configured models and propose top-1 candidate per model."""
    _ensure_default_registry()
    spec = _build_dataset_spec(run_df, loop.objective_metric, loop)

    requested = list(loop.model_names)

    model_fits: list[ModelFitArtifact] = []
    raw_candidates: list[Candidate] = []

    for model_name in requested:
        adapter = _MODEL_REGISTRY.get(model_name)
        if adapter is None:
            model_fits.append(
                ModelFitArtifact(
                    model_name=model_name,
                    n_params=None,
                    predicted_objective=None,
                    candidate_id=None,
                    error="model_not_registered",
                )
            )
            continue

        if not adapter.applicable(spec):
            model_fits.append(
                ModelFitArtifact(
                    model_name=model_name,
                    n_params=None,
                    predicted_objective=None,
                    candidate_id=None,
                    error="model_not_applicable",
                )
            )
            continue

        try:
            candidate, n_params = adapter.fit_and_propose(
                loop=loop,
                spec=spec,
                training_setup=training_setup,
            )
            raw_candidates.append(candidate)
            model_fits.append(
                ModelFitArtifact(
                    model_name=model_name,
                    n_params=n_params,
                    predicted_objective=candidate.predicted_objective,
                    candidate_id=candidate.candidate_id,
                )
            )
        except Exception as exc:
            logger.exception("Model fit/propose failed for %s", model_name)
            model_fits.append(
                ModelFitArtifact(
                    model_name=model_name,
                    n_params=None,
                    predicted_objective=None,
                    candidate_id=None,
                    error=str(exc),
                )
            )

    deduped_candidates, model_to_candidate = dedupe_candidates(raw_candidates)

    # Rewrite model fit candidate IDs to owning dedup candidate IDs.
    normalized_model_fits: list[ModelFitArtifact] = []
    for fit in model_fits:
        if fit.candidate_id is None:
            normalized_model_fits.append(fit)
            continue
        owner_id = model_to_candidate.get(fit.model_name, fit.candidate_id)
        normalized_model_fits.append(
            ModelFitArtifact(
                model_name=fit.model_name,
                n_params=fit.n_params,
                predicted_objective=fit.predicted_objective,
                candidate_id=owner_id,
                error=fit.error,
            )
        )

    return FitAndProposeResult(candidates=deduped_candidates, model_fits=normalized_model_fits)
