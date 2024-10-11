from dataclasses import asdict

def dataclass_to_dict(dataclass):
    """Convert dataclass to dictionary, omitting None values."""
    return {k: v for k, v in asdict(dataclass).items() if v is not None}

