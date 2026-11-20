# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
# Modifications Copyright The Levanter Authors, licensed Apache-2.0.
# Derived from Atomic Ops, Copyright (c) 2026 Omirbay Akseleu, MIT licensed.
# The original MIT copyright and permission notice are retained in ../LICENSE.

"""
Central configuration and shape validation
shared across the forward/backward Pallas kernels.
"""

from __future__ import annotations

import dataclasses as dc
from enum import StrEnum

import jax
import jax.numpy as jnp

_HIGHEST = jax.lax.Precision.HIGHEST


class ScoreLayout(StrEnum):
    FEATURE_LAST = "feature_last"
    FEATURE_FIRST = "feature_first"

    @classmethod
    def for_device_kind(cls, device_kind: str) -> ScoreLayout:
        """Choose a layout supported by the TPU generation."""
        return cls.FEATURE_FIRST if device_kind == "TPU v4" else cls.FEATURE_LAST


@dc.dataclass(frozen=True)
class KernelConfig:
    bt: int = 128
    mb: int = 16
    interpret: bool = False
    score_layout: ScoreLayout = ScoreLayout.FEATURE_LAST

    @property
    def bc(self) -> int:
        return self.bt // 2

    @property
    def n_micro(self) -> int:
        return self.bc // self.mb

    def __post_init__(self):
        if self.bt <= 0 or self.bt % 2:
            raise ValueError(f"bt={self.bt} must be a positive even chunk size")
        if self.mb <= 0 or self.bc % self.mb:
            raise ValueError(f"mb={self.mb} must be positive and divide bc={self.bc}")


DEFAULT_CONFIG = KernelConfig()


def _reshape_to_chunks(t, bsz, n_chunks, H, D, bt):
    t = t.reshape(bsz, n_chunks, bt, H, D)
    return jnp.moveaxis(t, (1, 3), (2, 1))


def _reshape_from_chunks(t, bsz, n_chunks, bt, H, D):
    t2 = jnp.moveaxis(t, (1, 2, 3), (3, 1, 2))
    return t2.reshape(bsz, n_chunks * bt, H, D)


def validate_inputs(q, k, v, w, b, g, h0, config: KernelConfig):
    """Shape/dtype sanity checks shared by forward and backward entry points."""
    if q.ndim != 4:
        raise ValueError(f"q must be (batch, seq_len, heads, d_head); got shape {q.shape}")
    bsz, L, H, D = q.shape
    if L == 0:
        raise ValueError("seq_len must be positive")
    if D != 128:
        raise ValueError(f"Kernels assume d_head=128 (MXU tile); got D={D}.")
    if L % config.bt != 0:
        raise ValueError(f"seq_len={L} must be divisible by config.bt={config.bt}.")

    for name, t in (("k", k), ("v", v), ("w", w), ("b", b), ("g", g)):
        if t.shape != q.shape:
            raise ValueError(f"{name}.shape={t.shape} must match q.shape={q.shape}")

    if h0 is not None:
        expected_h0 = (bsz, H, D, D)
        if h0.shape != expected_h0:
            raise ValueError(f"h0.shape={h0.shape} must be {expected_h0}")
        if h0.dtype != jnp.float32:
            raise ValueError(f"h0 must have dtype float32; got {h0.dtype}")
