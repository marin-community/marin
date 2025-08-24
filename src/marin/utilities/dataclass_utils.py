import dataclasses

from draccus.utils import DataclassInstance


def shallow_asdict(obj: DataclassInstance) -> dict:
    """
    Similar to dataclasses.asdict, but doesn't recurse into nested dataclasses.
    """
    return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}


def asdict_without_nones(obj: DataclassInstance) -> dict:
    """Convert dataclass to dictionary, omitting None values."""
    if not dataclasses.is_dataclass(obj):
        raise ValueError(f"Expected dataclass, got '{obj}'")
    return dataclasses.asdict(obj, dict_factory=lambda x: {k: v for (k, v) in x if v is not None})
