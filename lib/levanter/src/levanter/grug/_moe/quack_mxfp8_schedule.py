# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch schedules (tile, cluster, persistence) of the QuACK block-scaled GEMMs in ``quack_mxfp8``.

Kept free of CuTe imports so model code can build schedule tables on any backend. A GEMM's schedule comes from the
table active at trace time (``gemm_schedule_table``); a GEMM no entry matches keeps its kind's default, which is
what every GEMM ran before the tables existed.
"""

import contextlib
import contextvars
import dataclasses
import enum
from collections.abc import Iterator, Mapping

# Block-scaled 2-CTA MMA needs a 256-row tile.
MXFP8_TILE_MN = (256, 256)
MXFP8_CLUSTER = (2, 1, 1)


class GemmKind(enum.StrEnum):
    """The block-scaled GEMMs of the MXFP8 recipe, each with its own schedule-table entries."""

    UP_RELU2 = "up_relu2"  # varlen-m, B n-major, ReLU^2 + MXFP8 epilogue (routed up)
    UP_SWIGLU = "up_swiglu"  # varlen-m, B n-major, SwiGLU + MXFP8 epilogue (routed gated up)
    DRELU2 = "drelu2"  # varlen-m, B k-major, dReLU^2 + MXFP8 epilogue (routed down backward)
    GROUPED_N = "grouped_n"  # plain varlen-m, B n-major (routed down)
    GROUPED_K = "grouped_k"  # plain varlen-m, B k-major (routed input gradients)
    WGRAD = "wgrad"  # varlen-k over tokens (routed weight gradients)
    DENSE_FWD = "dense_fwd"  # x @ w
    DENSE_DGRAD = "dense_dgrad"  # g @ w^T
    DENSE_WGRAD = "dense_wgrad"  # x^T @ g


@dataclasses.dataclass(frozen=True)
class GemmSchedule:
    """Tile shape, cluster shape and persistence of one QuACK SM100 block-scaled GEMM launch.

    ``clc`` swaps the static persistent grid for QuACK's cluster-launch-control (dynamic) persistence, which takes
    work tiles as SMs free up. ``sm_reserve`` shrinks the static persistent grid by that many SMs (rounded to whole
    clusters) so a concurrent kernel (the ragged all-to-all) keeps them; it needs ``clc=False``."""

    tile_mn: tuple[int, int] = MXFP8_TILE_MN
    cluster_mnk: tuple[int, int, int] = MXFP8_CLUSTER
    clc: bool = False
    sm_reserve: int = 0

    def __post_init__(self):
        if self.clc and self.sm_reserve:
            raise ValueError("sm_reserve caps the static persistent grid; CLC persistence launches every tile")
        if self.sm_reserve < 0:
            raise ValueError(f"sm_reserve must be >= 0, got {self.sm_reserve}")


@dataclasses.dataclass(frozen=True)
class TunedGemm:
    """One schedule-table entry: GEMMs of ``kind`` whose weight-shaped feature dims are ``features`` and whose token
    count is at least ``min_rows`` run ``schedule``. ``features`` is ``(contraction, output)`` for the activation
    GEMMs and the weight gradient's ``(rows, cols)`` for the weight gradients."""

    kind: GemmKind
    features: tuple[int, int]
    min_rows: int
    schedule: GemmSchedule


DEFAULT_SCHEDULE = GemmSchedule()
DEFAULT_WGRAD_SCHEDULE = GemmSchedule(tile_mn=(256, 256), cluster_mnk=(2, 2, 1))
_DEFAULT_SCHEDULES: Mapping[GemmKind, GemmSchedule] = {
    kind: DEFAULT_WGRAD_SCHEDULE if kind == GemmKind.WGRAD else DEFAULT_SCHEDULE for kind in GemmKind
}
_SCHEDULE_TABLE: contextvars.ContextVar[tuple[TunedGemm, ...]] = contextvars.ContextVar(
    "quack_mxfp8_schedule_table", default=()
)


@contextlib.contextmanager
def gemm_schedule_table(table: tuple[TunedGemm, ...]) -> Iterator[None]:
    """Trace the block-scaled GEMMs inside the block with ``table`` (first matching entry wins); GEMMs no entry
    matches keep the default schedule. The table is read at trace time, so wrap the whole forward and backward."""
    token = _SCHEDULE_TABLE.set(tuple(table))
    try:
        yield
    finally:
        _SCHEDULE_TABLE.reset(token)


def gemm_schedule(kind: GemmKind, rows: int, features: tuple[int, int]) -> GemmSchedule:
    """The schedule a GEMM of ``kind`` over ``rows`` tokens and weight-shaped ``features`` runs under the active
    table (``gemm_schedule_table``)."""
    for entry in _SCHEDULE_TABLE.get():
        if entry.kind == kind and entry.features == tuple(features) and rows >= entry.min_rows:
            return entry.schedule
    return _DEFAULT_SCHEDULES[kind]


# GB200 (148 SMs), from lib/levanter/scripts/bench/bench_mxfp8_gemm_schedules.py at the d6144 recipe's per-GPU shapes.
TUNED_GB200: tuple[TunedGemm, ...] = ()
