import json

from levanter.utils.activation import ActivationFunctionEnum


class ConfigJSONEncoder(json.JSONEncoder):
    """Supports all the custom types we put into configs."""

    def default(self, o):
        # We can probably get rid of this if we require python 3.11
        # and change ActivationFunctionEnum to a StrEnum
        if isinstance(o, ActivationFunctionEnum):
            return o.name
        return super().default(o)
