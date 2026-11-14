# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed comparisons and invariants for RL progress reports."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INVALID = "invalid"


class Label(StrEnum):
    IMPROVES = "improves"
    TRADE_OFF = "trade-off"
    NO_CHANGE = "no change"
    REGRESSION = "regression"


class Change(StrEnum):
    BETTER = "better"
    WORSE = "worse"
    SAME = "same"
    UNAVAILABLE = "unavailable"
    CHANGED = "changed"


class ComparisonRole(StrEnum):
    OBJECTIVE = "objective"
    CONSTRAINT = "constraint"
    REQUIREMENT = "requirement"
    REPORTED = "reported"
    SLICE = "slice"


class Invariant(BaseModel):
    name: str
    value: bool | int | float
    limit: bool | int | float
    comparator: Literal["<=", ">=", "=="] | None = None
    holds: bool
    replicate: int = 0


class Comparison(BaseModel):
    metric: str
    role: ComparisonRole
    higher_is_better: bool | None = None
    run: float | None = None
    control: float | None = None
    delta: float | None = None
    delta_percent: float | None = None
    noise_percent: float | None = None
    tolerance: float | None = None
    sd: float | None = None
    df: int | None = None
    n_run: int
    n_control: int
    passes: bool | None = None
    change: Change | None = None
    tasks: int | None = None
    control_min: float | None = None
    control_max: float | None = None
