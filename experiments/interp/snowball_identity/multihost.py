# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host reads and cross-process collectives that hold on a multi-host mesh.

Under the trainer's mesh, every ``jnp`` value (even a scalar) is a global array that no single process fully
addresses. ``jax.device_get`` and an untiled ``process_allgather`` refuse those. So host values cross processes
as numpy, and global arrays are read with a tiled gather. All four helpers are collective when there is more
than one process: every process must call them in the same order.
"""

import jax
import numpy as np
from jax.experimental import multihost_utils


def _host_value(value) -> np.ndarray:
    if isinstance(value, jax.Array):
        raise TypeError("expected a host (python or numpy) value, got a jax.Array; use gather_global")
    return np.asarray(value)


def gather_host_scalar(value) -> np.ndarray:
    """Every process's ``value``, in process order: shape [process_count]."""
    return np.asarray(multihost_utils.process_allgather(_host_value(value).reshape(1), tiled=True))


def gather_host_rows(value) -> np.ndarray:
    """Every process's ``value`` (same shape on every process), stacked in process order: [process_count, ...]."""
    value = _host_value(value)
    return np.asarray(multihost_utils.process_allgather(value[None], tiled=True))


def broadcast_host_scalar(value) -> np.ndarray:
    """Process 0's ``value`` on every process."""
    return np.asarray(multihost_utils.broadcast_one_to_all(_host_value(value)))


def gather_global(x: jax.Array) -> np.ndarray:
    """The whole value of a jax array as numpy on every process."""
    if x.is_fully_addressable:
        return np.asarray(x)
    return np.asarray(multihost_utils.process_allgather(x, tiled=True))
