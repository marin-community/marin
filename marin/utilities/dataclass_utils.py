import dataclasses

from draccus.utils import DataclassInstance


def shallow_asdict(obj: DataclassInstance) -> dict:
    """
    Similar to dataclasses.asdict, but doesn't recurse into nested dataclasses.
    """
    return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
