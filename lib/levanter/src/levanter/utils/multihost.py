# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Host/global array gathers safe under a multi-process JAX mesh.

Every rank must call collectives in the same order. NumPy host values cross as
host data; 64-bit elements cross as pairs of int32 words to avoid x64 narrowing.
Global JAX arrays use a tiled gather when they are not fully addressable.
"""

import jax
import numpy as np
from jax.experimental import multihost_utils


def _host_value(value) -> np.ndarray:
    if isinstance(value, jax.Array):
        raise TypeError("expected a host (python or numpy) value, got a jax.Array; use gather_global")
    return np.asarray(value)


def _as_words(value: np.ndarray) -> np.ndarray:
    """``value`` (at least 1-d) with each 64-bit element as two int32 words on the last axis; others unchanged."""
    if value.dtype.itemsize != 8:
        return value
    return np.ascontiguousarray(value).view(np.int32)


def _from_words(words, dtype: np.dtype) -> np.ndarray:
    words = np.asarray(words)
    if dtype.itemsize != 8:
        return words
    return np.ascontiguousarray(words.astype(np.int32, copy=False)).view(dtype)


def _allgather_rows(value: np.ndarray) -> np.ndarray:
    """Tiled gather along axis 0 of a host array of rank >= 2, exact for 64-bit dtypes."""
    return _from_words(multihost_utils.process_allgather(_as_words(value), tiled=True), value.dtype)


def gather_host_scalar(value) -> np.ndarray:
    """Every process's ``value``, in process order: shape [process_count]."""
    value = _host_value(value).reshape(1, 1)
    return _allgather_rows(value).reshape(-1)


def gather_host_rows(value) -> np.ndarray:
    """Every process's ``value`` (same shape on every process), stacked in process order: [process_count, ...]."""
    value = _host_value(value)
    rows = value.reshape(1, *value.shape, 1)
    return _allgather_rows(rows).reshape(-1, *value.shape)


def gather_global(x: jax.Array) -> np.ndarray:
    """The whole value of a jax array as numpy on every process."""
    if x.is_fully_addressable:
        return np.asarray(x)
    return np.asarray(multihost_utils.process_allgather(x, tiled=True))
