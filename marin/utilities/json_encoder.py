import json
import logging
import jax
from jax.numpy import float32, bfloat16
from datetime import timedelta
from pathlib import Path

logger = logging.getLogger("ray")


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, timedelta):
            return {"days": obj.days, "seconds": obj.seconds, "microseconds": obj.microseconds}
        if isinstance(obj, Path):
            return str(obj)
        if obj in (float32, bfloat16):
            return str(obj)
        try:
            return super().default(obj)
        except TypeError:
            logger.warning(f"Could not serialize object of type {type(obj)}: {obj}")
            return str(obj)
