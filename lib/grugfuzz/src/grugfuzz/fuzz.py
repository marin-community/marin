"""Lightweight fuzzing helpers for deterministic test case generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

Constraint = Callable[[dict[str, Any]], bool]
Sampler = Callable[[dict[str, Any], np.random.Generator], Any]


@dataclass(frozen=True)
class IntRange:
    """Inclusive integer range."""

    low: int
    high: int


@dataclass(frozen=True)
class FloatRange:
    """Uniform float range."""

    low: float
    high: float


@dataclass(frozen=True)
class Choice:
    """Discrete choice set."""

    options: tuple[Any, ...]


@dataclass(frozen=True)
class FuzzSpace:
    """Generic fuzz space for sampling dict-like configs.

    specs: mapping of field name -> spec (range/choice/callable/literal)
    constraints: predicates that must all hold for a sample to be accepted
    """

    specs: dict[str, Any]
    constraints: tuple[Constraint, ...] = ()

    def sample(
        self,
        rng: np.random.Generator,
        *,
        overrides: dict[str, Any] | None = None,
        max_tries: int = 1000,
    ) -> dict[str, Any]:
        """Sample a config satisfying constraints."""
        overrides = {} if overrides is None else dict(overrides)
        for _ in range(max_tries):
            sample: dict[str, Any] = dict(overrides)
            for key, spec in self.specs.items():
                if key in sample:
                    continue
                sample[key] = _sample_value(spec, sample, rng)
            if all(constraint(sample) for constraint in self.constraints):
                return sample
        raise RuntimeError("Failed to sample a valid config from FuzzSpace")


def sample_fuzz_cases(
    space: FuzzSpace,
    *,
    seed: int,
    count: int,
    overrides: dict[str, Any] | None = None,
    max_tries: int = 1000,
) -> tuple[tuple[int, dict[str, Any]], ...]:
    """Sample deterministic fuzz cases as (case_seed, sample) tuples."""
    rng = np.random.default_rng(seed)
    cases: list[tuple[int, dict[str, Any]]] = []
    for _ in range(count):
        case_seed = int(rng.integers(0, 2**31 - 1))
        case_rng = np.random.default_rng(case_seed)
        case = space.sample(case_rng, overrides=overrides, max_tries=max_tries)
        cases.append((case_seed, case))
    return tuple(cases)


def _sample_value(spec: Any, current: dict[str, Any], rng: np.random.Generator) -> Any:
    if isinstance(spec, IntRange):
        return int(rng.integers(spec.low, spec.high + 1))
    if isinstance(spec, FloatRange):
        return float(rng.uniform(spec.low, spec.high))
    if isinstance(spec, Choice):
        return spec.options[int(rng.integers(0, len(spec.options)))]
    if callable(spec):
        return spec(current, rng)
    return spec
