# Copyright 2025 The Marin Authors
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

from experiments.domain_phase_mix.exploratory.general_scaling_models import (
    DatasetSpec,
    GENERAL_MODELS,
    GeneralModelSpec,
)
from experiments.domain_phase_mix.nextgen.contracts import Candidate, LoopConfig, PolicyArtifactRef
from experiments.domain_phase_mix.nextgen.utils import stable_hash

logger = logging.getLogger(__name__)

_PHASE_DOMAIN_COL = re.compile(r"^phase_(\d+)_(.+)$")
_MODEL_REGISTRY: dict[str, ModelAdapter] = {}
_DEFAULT_REGISTRY_INITIALIZED = False


class ModelAdapter(Protocol):
    """Adapter interface used by the next-gen fit/propose stage."""

    @property
    def name(self) -> str:
        ...

    def applicable(self, spec: DatasetSpec) -> bool:
        ...

    def fit_and_propose(
        self,
        *,
        loop: LoopConfig,
        spec: DatasetSpec,
        training_setup: dict[str, Any],
    ) -> tuple[Candidate, int | None]:
        ...


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


def _default_small_domains(domains: list[str]) -> list[int] | None:
    # Prefer keeping explicit behavior stable for StarCoder-style setups.
    for i, name in enumerate(domains):
        lower = name.lower()
        if "starcoder" in lower or "rare" in lower or "small" in lower:
            return [i]
    return None


def _build_dataset_spec(df: pd.DataFrame, objective_metric: str) -> DatasetSpec:
    if objective_metric not in df.columns:
        raise ValueError(f"Objective metric '{objective_metric}' missing from run table")

    model_df = df[df[objective_metric].notna()].copy()
    if model_df.empty:
        raise ValueError(f"No rows with non-null objective metric '{objective_metric}'")

    phase_names, domains = _extract_phase_domain_columns(model_df)
    weights = _build_weights_tensor(model_df, phase_names, domains)
    y = model_df[objective_metric].to_numpy(dtype=float)

    # V1 default for generic imports: epoch multipliers of 1.0 per (phase, domain).
    epoch_multipliers = np.ones((len(phase_names), len(domains)), dtype=float)

    return DatasetSpec(
        weights=weights,
        y=y,
        epoch_multipliers=epoch_multipliers,
        domain_names=domains,
        phase_names=phase_names,
        small_domains=_default_small_domains(domains),
        name="nextgen_merged_runs",
    )


def _sample_simplex_points(rng: np.random.Generator, n_points: int, n_dims: int) -> np.ndarray:
    raw = rng.exponential(1.0, size=(n_points, n_dims))
    return raw / raw.sum(axis=1, keepdims=True)


def _propose_top1_candidate(
    *,
    loop: LoopConfig,
    model_name: str,
    predict_fn,
    spec: DatasetSpec,
    training_setup: dict[str, Any],
) -> Candidate:
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

    phase_weights = {
        spec.phase_names[p_idx]: {
            spec.domain_names[d_idx]: float(best[p_idx, d_idx]) for d_idx in range(spec.M)
        }
        for p_idx in range(spec.N)
    }

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
    spec = _build_dataset_spec(run_df, loop.objective_metric)

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
