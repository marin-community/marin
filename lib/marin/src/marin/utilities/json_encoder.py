# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import logging
from datetime import timedelta
from enum import Enum
from pathlib import Path

# Todo(Percy, dlwh): Can we remove this jax dependency?
from jax.numpy import bfloat16, float32

logger = logging.getLogger(__name__)


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, timedelta):
            return {"days": o.days, "seconds": o.seconds, "microseconds": o.microseconds}
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, Enum):
            return o.value
        if o in (float32, bfloat16):
            return str(o)
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        try:
            return super().default(o)
        except TypeError:
            logger.warning(f"Could not serialize object of type {type(o)}: {o}")
            return str(o)
