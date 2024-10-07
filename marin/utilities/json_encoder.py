import json
from datetime import timedelta


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, timedelta):
            return {"days": obj.days, "seconds": obj.seconds, "microseconds": obj.microseconds}
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)
