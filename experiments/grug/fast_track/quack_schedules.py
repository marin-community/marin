# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch-schedule tables for the QuACK MXFP8 GEMMs of a ``GrugModelConfig`` (``quack_gemm_*`` fields).

``quack_gemm_table`` picks the base table (``default``: every GEMM at its kind's default schedule). ``quack_a2a_*``
then overrides the persistence of the GEMMs that run while the ragged all-to-all is in flight (the routed experts'
GEMMs, and the shared experts', which ``moe_shared_placement=overlap`` places under the dispatch): ``clc`` takes work
tiles dynamically as SMs free up, ``reserve`` shrinks the static persistent grid by ``quack_a2a_sm_reserve`` SMs so
the all-to-all keeps them.
"""

import dataclasses
from enum import StrEnum

from levanter.grug._moe.quack_mxfp8_schedule import (
    TUNED_GB200,
    GemmKind,
    TunedGemm,
    gemm_schedule,
    gemm_schedule_table,
)


class QuackGemmTable(StrEnum):
    DEFAULT = "default"
    TUNED = "tuned"
    """``TUNED_GB200``: per-shape tiles and clusters from ``bench_mxfp8_gemm_schedules.py`` on GB200."""


class QuackA2aPersistence(StrEnum):
    STATIC = "static"
    """The base table's persistence."""
    CLC = "clc"
    RESERVE = "reserve"


_BASE_TABLES = {QuackGemmTable.DEFAULT: (), QuackGemmTable.TUNED: TUNED_GB200}
_LARGEST_ROWS = 1 << 62
_ROUTED_KINDS = (GemmKind.UP_RELU2, GemmKind.UP_SWIGLU, GemmKind.DRELU2, GemmKind.GROUPED_N, GemmKind.GROUPED_K)
_DENSE_KINDS = (GemmKind.DENSE_FWD, GemmKind.DENSE_DGRAD, GemmKind.DENSE_WGRAD)


def _a2a_gemms(cfg) -> list[tuple[GemmKind, tuple[int, int]]]:
    """(kind, features) of the GEMMs that overlap the ragged all-to-all under ``cfg``."""
    latent = cfg.latent_dim or cfg.hidden_dim
    width = cfg.intermediate_dim
    gemms = [(kind, features) for kind in _ROUTED_KINDS for features in ((latent, width), (width, latent))]
    gemms += [(GemmKind.WGRAD, (latent, width)), (GemmKind.WGRAD, (width, latent))]
    shared = cfg.shared_expert_intermediate_dim * cfg.num_shared_experts
    if shared:
        gemms += [(kind, f) for kind in _DENSE_KINDS for f in ((cfg.hidden_dim, shared), (shared, cfg.hidden_dim))]
    return gemms


def quack_schedule_table(cfg) -> tuple[TunedGemm, ...]:
    """The schedule table ``cfg``'s ``quack_gemm_*`` fields select (empty: every GEMM at its default)."""
    base = _BASE_TABLES[cfg.quack_gemm_table]
    mode = cfg.quack_a2a_persistence
    if mode == QuackA2aPersistence.STATIC:
        if cfg.quack_a2a_sm_reserve:
            raise ValueError("quack_a2a_sm_reserve needs quack_a2a_persistence=reserve")
        return base
    if mode == QuackA2aPersistence.RESERVE and cfg.quack_a2a_sm_reserve <= 0:
        raise ValueError("quack_a2a_persistence=reserve needs quack_a2a_sm_reserve > 0")
    if mode == QuackA2aPersistence.CLC and cfg.quack_a2a_sm_reserve:
        raise ValueError("quack_a2a_sm_reserve applies to quack_a2a_persistence=reserve only")
    overrides = []
    with gemm_schedule_table(base):
        for kind, features in _a2a_gemms(cfg):
            # These GEMMs run over every routed row or token: look up the base schedule for the largest bucket.
            schedule = gemm_schedule(kind, rows=_LARGEST_ROWS, features=features)
            if mode == QuackA2aPersistence.CLC:
                schedule = dataclasses.replace(schedule, clc=True, sm_reserve=0)
            else:
                schedule = dataclasses.replace(schedule, clc=False, sm_reserve=cfg.quack_a2a_sm_reserve)
            overrides.append(TunedGemm(kind, features, 0, schedule))
    return (*overrides, *base)


def quack_schedule_context(cfg):
    """``gemm_schedule_table`` over ``cfg``'s table; trace the forward and backward inside it."""
    return gemm_schedule_table(quack_schedule_table(cfg))
