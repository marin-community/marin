# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import logging
from datetime import timedelta
from enum import Enum
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _dtype_name(o) -> str | None:
    """Canonical name for a numpy/jax dtype or scalar dtype *type*, else ``None``.

    Recognizes ``np.dtype`` instances and the scalar *type* objects that show up
    as config values (``np.float32``, ``jax.numpy.bfloat16`` and other
    ``ml_dtypes`` extensions) — all of which ``np.dtype`` normalizes without a
    jax import. Returns ``None`` for scalar values, arrays, and unrelated classes
    so they follow the normal encoding path instead of being mislabeled.
    """
    if isinstance(o, np.dtype):
        return str(o)
    if not isinstance(o, type):
        return None
    try:
        dt = np.dtype(o)
    except TypeError:
        return None
    # np.dtype() coerces *any* class to a dtype (arbitrary classes -> object,
    # kind 'O'); keep only genuine scalar dtypes. bf16/fp8 extensions report
    # kind 'V', so accept that too rather than filtering to numeric kinds.
    return str(dt) if dt.kind in "biufcV" else None


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, timedelta):
            return {"days": o.days, "seconds": o.seconds, "microseconds": o.microseconds}
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, Enum):
            return o.value
        if (dtype_name := _dtype_name(o)) is not None:
            return dtype_name
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        try:
            return super().default(o)
        except TypeError:
            logger.warning(f"Could not serialize object of type {type(o)}: {o}")
            return str(o)
