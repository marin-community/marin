from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
import json
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Convert nested configs, dataclasses, and containers to JSON-compatible values."""
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [to_jsonable(item) for item in value]
    return str(value)


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a value using the canonical representation used for content hashes."""
    return json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
