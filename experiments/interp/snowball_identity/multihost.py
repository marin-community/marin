# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host reads and cross-process collectives that hold on a multi-host mesh.

Under the trainer's mesh, every ``jnp`` value (even a scalar) is a global array that no single process fully
addresses. ``jax.device_get`` and an untiled ``process_allgather`` refuse those. So host values cross processes
as numpy, and global arrays are read with a tiled gather. All four helpers are collective when there is more
than one process: every process must call them in the same order.

With x64 off, JAX narrows 64-bit host values to 32 bits on their way into a collective (int64 device memory sizes
once wrapped this way). So 64-bit host values cross as pairs of int32 words and are reassembled exactly.
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


def broadcast_host_scalar(value) -> np.ndarray:
    """Process 0's ``value`` on every process."""
    value = _host_value(value)
    words = multihost_utils.broadcast_one_to_all(_as_words(value.reshape(*value.shape, 1)))
    return _from_words(words, value.dtype).reshape(value.shape)


def gather_global(x: jax.Array) -> np.ndarray:
    """The whole value of a jax array as numpy on every process."""
    if x.is_fully_addressable:
        return np.asarray(x)
    return np.asarray(multihost_utils.process_allgather(x, tiled=True))
