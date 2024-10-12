from dataclasses import asdict


def dataclass_to_dict(dataclass):
    """Convert dataclass to dictionary, omitting None values."""
    if not is_dataclass(obj):
        raise ValueError(f"Expected dataclass, got '{obj}'")
    return asdict(obj, dict_factory=lambda x: {k: v for (k, v) in x if v is not None})
