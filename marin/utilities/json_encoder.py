import json
import logging
from datetime import timedelta

logger = logging.getLogger("ray")


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, timedelta):
            return {"days": obj.days, "seconds": obj.seconds, "microseconds": obj.microseconds}
        try:
            return super().default(obj)
        except TypeError:
            logger.warning(f"Could not serialize object of type {type(obj)}: {obj}")
            return str(obj)
