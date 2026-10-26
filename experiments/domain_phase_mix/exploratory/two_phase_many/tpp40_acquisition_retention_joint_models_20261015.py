# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "threadpoolctl"]
# ///
"""Explicit two-bucket acquisition/retention features and unchanged private J2.

The kinetic menu and masses are caller inputs. A local SQLite cache commits each
successful policy independently and retains numerical failures. No data loader,
default kinetic grid, semantic taxonomy, or scientific execution is included.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any
from uuid import uuid4

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_causal_acquisition_retention_20261015 as law
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_joint_estimation_20261014 as frozen
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_rate_joint_models_20261015 as control_union

MODEL = "causal_acquisition_retention"
MODEL_NAMES = ("aggregate_mariner", MODEL)
TestGrid = frozen.TestGrid
mariner = frozen.mariner


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


SOURCE_HASHES = {
    str(Path(path).resolve()): digest(Path(path))
    for path in (
        __file__,
        law.__file__,
        frozen.__file__,
        control_union.__file__,
        frozen.temporal.__file__,
        mariner.__file__,
    )
}
RUNTIME = {
    "python": sys.version,
    **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "threadpoolctl")},
    "sqlite": sqlite3.sqlite_version,
}


def encoded(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_key(value: Any) -> str:
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def verify_sources() -> None:
    for name, expected in SOURCE_HASHES.items():
        if digest(Path(name)) != expected:
            raise ValueError("Loaded feature/search sources changed; restart with a new registered identity")


@dataclass(frozen=True)
class FeatureConfig:
    """Two singleton groups with externally fixed masses and exposure metadata."""

    scales: tuple[float, float]
    clock: str
    masses: tuple[float, float]
    relative_tolerance: float
    max_panels: int

    def __post_init__(self) -> None:
        if self.clock not in ("equal_group", "global_pool"):
            raise ValueError("Declare one of the two registered clock hypotheses")
        if len(self.scales) != 2 or len(self.masses) != 2:
            raise ValueError("This adapter is restricted to two singleton groups")
        if not np.isfinite(self.scales).all() or np.any(np.asarray(self.scales) <= 0):
            raise ValueError("Require explicit positive finite exposure scales")
        if not np.isfinite(self.masses).all() or np.any(np.asarray(self.masses) < 0):
            raise ValueError("Require explicit nonnegative finite masses")
        law.Quadrature(self.relative_tolerance, self.max_panels)

    def to_json(self) -> dict[str, Any]:
        return {
            "scales": list(self.scales),
            "clock": self.clock,
            "masses": list(self.masses),
            "groups": ["first", "second"],
            "relative_tolerance": self.relative_tolerance,
            "max_panels": self.max_panels,
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> FeatureConfig:
        if value["groups"] != ["first", "second"]:
            raise ValueError("Only two fixed singleton groups are supported")
        return cls(
            tuple(value["scales"]),
            value["clock"],
            tuple(value["masses"]),
            value["relative_tolerance"],
            value["max_panels"],
        )


def native_tied(early: np.ndarray, late: np.ndarray, alpha: float) -> bool:
    """Recognize only roundoff-scale equality of both physical phase fractions."""
    total = early + late
    first = np.divide(early, total, out=np.full_like(total, alpha), where=total > 0)
    second = np.divide(late, total, out=np.full_like(total, 1 - alpha), where=total > 0)
    tolerance = 64 * np.finfo(float).eps
    return bool(
        np.all(abs(first - alpha) <= tolerance * alpha) and np.all(abs(second - (1 - alpha)) <= tolerance * (1 - alpha))
    )


def policy_feature(
    early: np.ndarray, late: np.ndarray, alpha: float, rate: float, power: float, mu: float, config: FeatureConfig
) -> dict[str, Any]:
    """Evaluate the law on physical epochs, including off-simplex query points."""
    total = early + late
    if mu == 0 or native_tied(early, late, alpha):
        hazard = (rate * total) ** power
        return {
            "status": "complete",
            "benefit": (-np.expm1(-hazard)).tolist(),
            "hazard": hazard.tolist(),
            "exact_native": True,
            "evaluations": 0,
            "panels": 0,
            "relative_error_estimate": 0.0,
        }
    parameters = law.Law(rate, power, mu, ("first", "second"), np.asarray(config.masses))
    control = law.Quadrature(config.relative_tolerance, config.max_panels)
    path = law.integrate(np.stack((early, late)), parameters, control)
    tied = law.integrate(total[None], parameters, control)
    hazard = path.survival_hazard(parameters) + tied.deficit(parameters)
    benefit = -np.expm1(-hazard)
    if not np.isfinite(hazard).all() or not np.isfinite(benefit).all():
        raise FloatingPointError("Nonfinite acquisition/retention feature")
    return {
        "status": "complete",
        "benefit": benefit.tolist(),
        "hazard": hazard.tolist(),
        "exact_native": False,
        "evaluations": path.evaluations + tied.evaluations,
        "panels": path.panels + tied.panels,
        "relative_error_estimate": max(path.maximum_relative_error_estimate, tied.maximum_relative_error_estimate),
    }


class BenefitCache:
    """Transactional, content-addressed per-policy cache without outcome inputs."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path, timeout=120)) as connection, connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("CREATE TABLE IF NOT EXISTS contexts (key TEXT PRIMARY KEY, identity TEXT NOT NULL)")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS features (context TEXT NOT NULL, policy TEXT NOT NULL, "
                "request TEXT NOT NULL, response TEXT NOT NULL, response_sha256 TEXT NOT NULL, "
                "PRIMARY KEY (context, policy))"
            )

    def features(
        self,
        early: np.ndarray,
        late: np.ndarray,
        alpha: float,
        rate: float,
        power: float,
        mu: float,
        config: FeatureConfig,
    ) -> list[dict[str, Any]]:
        verify_sources()
        context = {
            "schema": 1,
            "config": config.to_json(),
            "alpha": alpha,
            "rate": rate,
            "power": power,
            "mu": mu,
            "sources": SOURCE_HASHES,
            "runtime": RUNTIME,
            "tied_rule": "symmetric phase fractions within 64 machine eps relative to each phase duration",
        }
        context_key, context_json = content_key(context), encoded(context)
        result = []
        with closing(sqlite3.connect(self.path, timeout=120)) as connection:
            with connection:
                connection.execute("INSERT OR IGNORE INTO contexts VALUES (?, ?)", (context_key, context_json))
            if (
                connection.execute("SELECT identity FROM contexts WHERE key=?", (context_key,)).fetchone()[0]
                != context_json
            ):
                raise ValueError("Feature context collision or corruption")
            for first, second in zip(early, late, strict=True):
                request = {"early": first.tolist(), "late": second.tolist()}
                request_json, policy_key = encoded(request), content_key(request)
                record = connection.execute(
                    "SELECT request,response,response_sha256 FROM features WHERE context=? AND policy=?",
                    (context_key, policy_key),
                ).fetchone()
                if record is None:
                    try:
                        response = policy_feature(first, second, alpha, rate, power, mu, config)
                    except ArithmeticError as error:
                        response = {
                            "status": "numerical_failure",
                            "exception": type(error).__name__,
                            "message": str(error),
                        }
                    response_json = encoded(response)
                    response_sha = hashlib.sha256(response_json.encode()).hexdigest()
                    with connection:
                        connection.execute(
                            "INSERT OR IGNORE INTO features VALUES (?, ?, ?, ?, ?)",
                            (context_key, policy_key, request_json, response_json, response_sha),
                        )
                    record = connection.execute(
                        "SELECT request,response,response_sha256 FROM features WHERE context=? AND policy=?",
                        (context_key, policy_key),
                    ).fetchone()
                    if record != (request_json, response_json, response_sha):
                        raise ArithmeticError("Concurrent feature computations disagree; preserve the cache for audit")
                if record[0] != request_json or hashlib.sha256(record[1].encode()).hexdigest() != record[2]:
                    raise ValueError("Feature cache corruption")
                response = json.loads(record[1])
                if response["status"] != "complete":
                    raise ArithmeticError(f"Cached numerical failure: {response}")
                result.append(response)
        return result


def design_and_offset(
    early: np.ndarray,
    late: np.ndarray,
    alpha: float,
    shape: dict[str, float],
    model: str,
    parameters: dict[str, float],
    config: FeatureConfig,
    cache: BenefitCache | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace benefit only; keep native physical harm and exact aggregate law."""
    verify_sources()
    early, late = np.asarray(early, float), np.asarray(late, float)
    if early.ndim != 2 or early.shape != late.shape or early.shape[1] != 2:
        raise ValueError("Require aligned two-bucket physical epoch arrays")
    if not np.isfinite(early + late).all() or np.any(early < 0) or np.any(late < 0) or not 0 < alpha < 1:
        raise ValueError("Require finite nonnegative epochs and an interior phase fraction")
    if model not in MODEL_NAMES:
        raise ValueError("Unsupported acquisition/retention model")
    if model == "aggregate_mariner" or parameters["mu"] == 0:
        return mariner.design_matrix(early + late, shape), np.zeros(len(early))
    mu = parameters["mu"]
    if not np.isfinite(mu) or mu < 0:
        raise ValueError("Require a finite nonnegative kinetic strength")
    if cache is None:
        records = [
            policy_feature(a, b, alpha, shape["rate"], shape["power"], mu, config)
            for a, b in zip(early, late, strict=True)
        ]
    else:
        records = cache.features(early, late, alpha, shape["rate"], shape["power"], mu, config)
    benefit = np.asarray([item["benefit"] for item in records])
    return np.hstack((-benefit, mariner.harm(early + late, shape["threshold"]))), np.zeros(len(early))


def isolated_adapter(config: FeatureConfig, strengths: tuple[float, ...], cache: BenefitCache | None) -> ModuleType:
    if (
        not strengths
        or strengths[0] != 0
        or len(set(strengths)) != len(strengths)
        or not np.isfinite(strengths).all()
        or min(strengths) < 0
    ):
        raise ValueError("Supply a finite unique nonnegative kinetic menu with the exact null first")
    name = f"{__name__}._union_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, control_union.__file__)
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[name] = adapter
    spec.loader.exec_module(adapter)

    def parameters(model: str) -> tuple[dict[str, float], ...]:
        if model not in MODEL_NAMES:
            raise ValueError("Unsupported model")
        return ({},) if model == "aggregate_mariner" else tuple({"mu": value} for value in strengths)

    def design(early, late, alpha, shape, model, settings):
        return design_and_offset(early, late, alpha, shape, model, settings, config, cache)

    adapter.MODEL_NAMES = MODEL_NAMES
    adapter.parameter_grid = parameters
    adapter.design_and_offset = design
    adapter._BACKEND.parameter_grid = parameters
    adapter._BACKEND.design_and_offset = design
    return adapter


@dataclass(frozen=True)
class JointFit:
    selected: frozen.JointFit
    config: FeatureConfig

    @property
    def model(self) -> str:
        return self.selected.model

    @property
    def alpha(self) -> float:
        return self.selected.alpha

    @property
    def shape(self) -> dict[str, float]:
        return self.selected.shape

    @property
    def parameters(self) -> dict[str, float]:
        return self.selected.parameters

    @property
    def ridge(self) -> float:
        return self.selected.ridge

    @property
    def kappa(self) -> float | None:
        return self.selected.kappa

    @property
    def known_floor(self) -> float | None:
        return self.selected.known_floor

    @property
    def head(self) -> mariner.Head:
        return self.selected.head

    @property
    def inner_rmse(self) -> float:
        return self.selected.inner_rmse

    @property
    def search_space(self) -> dict[str, Any]:
        return self.selected.search_space

    @property
    def selection_profile(self) -> tuple[dict[str, Any], ...]:
        return self.selected.selection_profile

    def predict(self, early: np.ndarray, late: np.ndarray, cache: BenefitCache | None = None) -> np.ndarray:
        matrix, offset = design_and_offset(
            early, late, self.alpha, self.shape, self.model, self.parameters, self.config, cache
        )
        return frozen.temporal.predict_head(self.head, matrix, offset)

    def to_json(self) -> dict[str, Any]:
        payload = self.selected.to_json()
        payload["feature_config"] = self.config.to_json()
        payload["feature_source_hashes"] = dict(SOURCE_HASHES)
        payload["feature_runtime"] = dict(RUNTIME)
        payload["selection_contract"] = (
            "Unchanged J2 with the complete aggregate rescored at mu=0 and retained in final union"
        )
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> JointFit:
        verify_sources()
        if payload["feature_source_hashes"] != SOURCE_HASHES:
            raise ValueError("Serialized acquisition/retention source identity differs")
        if payload["feature_runtime"] != RUNTIME:
            raise ValueError("Serialized acquisition/retention runtime identity differs")
        return cls(frozen.JointFit.from_json(payload), FeatureConfig.from_json(payload["feature_config"]))


def fit_joint(
    early: np.ndarray,
    late: np.ndarray,
    observed: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    model: str,
    *,
    config: FeatureConfig,
    strengths: tuple[float, ...],
    cache: BenefitCache | None,
    controls: tuple[JointFit, ...] = (),
    shapes: tuple[dict[str, float], ...] | None = None,
    test_grid: TestGrid | None = None,
) -> JointFit:
    """Fit unchanged J2; explicit reduced grids are reserved for toy diagnostics."""
    if model not in MODEL_NAMES or len(controls) != (model == MODEL):
        raise ValueError("Fit the aggregate first, then the temporal family with that complete control")
    adapter = isolated_adapter(config, strengths, cache)
    control = None
    if controls:
        aggregate = controls[0]
        if aggregate.model != "aggregate_mariner" or aggregate.parameters != {} or aggregate.config != config:
            raise ValueError("Require a matching native aggregate with the same feature configuration")
        control = adapter.JointFit(replace(aggregate.selected, parameters={"mu": 0.0}))
    result = adapter._BACKEND.fit_joint(
        early,
        late,
        observed,
        labels,
        alpha,
        model,
        shapes=shapes,
        test_grid=test_grid,
        matched_null_fit=None if control is None else control.selected,
    )
    canonical = frozen.JointFit.from_json(result.to_json())
    if control is not None:
        result = adapter.retain_controls(adapter.JointFit(canonical), (control,), early, late, observed, labels)
        search = dict(result.search_space)
        search["complete_aggregate_source"] = controls[0].to_json()
        canonical = replace(frozen.JointFit.from_json(result.to_json()), search_space=search)
    return JointFit(canonical, config)


def fit_known_floor(
    early: np.ndarray,
    late: np.ndarray,
    observed: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    model: str,
    *,
    floor: float,
    config: FeatureConfig,
    strengths: tuple[float, ...],
    cache: BenefitCache | None,
    shapes: tuple[dict[str, float], ...] | None = None,
    test_grid: TestGrid | None = None,
    matched_null_fit: JointFit | None = None,
) -> JointFit:
    """Expose the unchanged known-floor diagnostic, without adding a kinetic menu."""
    if matched_null_fit is not None and matched_null_fit.config != config:
        raise ValueError("Known-floor control must share the exact feature configuration")
    adapter = isolated_adapter(config, strengths, cache)
    result = adapter._BACKEND.fit_known_floor(
        early,
        late,
        observed,
        labels,
        alpha,
        model,
        floor=floor,
        shapes=shapes,
        test_grid=test_grid,
        matched_null_fit=None if matched_null_fit is None else matched_null_fit.selected,
    )
    return JointFit(frozen.JointFit.from_json(result.to_json()), config)
