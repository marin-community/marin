# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared schema objects for roofline reports."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class RowKind(StrEnum):
    COMPUTE = "compute"
    COMM = "comm"
    MIXED = "mixed"


class ObservedTimeBasis(StrEnum):
    TRACK_SUMMED_XPROF_KERNEL_TIME = "track_summed_xprof_kernel_time_us"
    TRACK_SUMMED_PROFILE_HOT_OP_TIME = "track_summed_profile_hot_op_time_us"
    TRACE_EMPTY_TRAIN_STEP_TIME = "trace_empty_train_step_time_us"
    MANUAL = "manual"
    NONE = "none"


@dataclass(frozen=True)
class RooflineRow:
    semantic_op: str
    kind: RowKind
    source: str
    estimated_flops: float = 0.0
    estimated_bytes: float = 0.0
    ideal_time: float = 0.0
    user_efficiency: float = 1.0
    estimated_time: float = 0.0
    profile_observed_time: float | None = None
    profile_achieved_pct: float | None = None
    critical_path_observed_time: float | None = None
    track_summed_observed_time: float | None = None
    observed_time_basis: ObservedTimeBasis = ObservedTimeBasis.NONE
    display_name: str | None = None
    observed_count: int | None = None
    observed_avg_time: float | None = None
    formula: str | None = None
    matched_name: str | None = None
    observed_comparable_to_model: bool = True


@dataclass(frozen=True)
class RooflineTotals:
    compute_roofline_time: float
    comm_roofline_time: float
    observed_track_summed_time: float | None
    observed_exposed_time: float | None
    estimated_scenario_time: float


@dataclass(frozen=True)
class ImportState:
    wandb_run: str | None = None
    wandb_run_url: str | None = None
    profile_path: str | None = None
    profile_devices: int | None = None
    profile_steps: int | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RooflineReport:
    schema_version: str
    model: dict[str, Any]
    hardware: dict[str, Any]
    rows: list[RooflineRow]
    totals: RooflineTotals
    imports: ImportState
    unattributed: list[dict[str, Any]] = field(default_factory=list)
    attribution_rules: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
