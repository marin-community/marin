# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import contextlib
import datetime
import decimal
import enum
import functools
import json
import os
import pathlib
import random
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from typing import TypeVar

import numpy as np

F = TypeVar("F", bound=Callable)


def per_instance_lru_cache(maxsize: int | None = 128) -> Callable[[F], F]:
    """Memoizes a method, giving each instance its own LRU cache.

    `functools.lru_cache` applied to a method stores `self` in a single cache owned by the
    function object, so instances are pinned in memory for the lifetime of the process and
    `maxsize` becomes a budget shared by every live instance (which then evict each other's
    entries). This decorator instead installs a separate cache in each instance's `__dict__`
    the first time the method is called, so the cache dies with the instance.

    The cache participates in a reference cycle with its instance, so collection happens on a
    `gc` pass rather than when the last reference drops. Arguments must be hashable.
    """

    def decorator(method: F) -> F:
        cache_attr = f"_per_instance_cache_{method.__name__}"

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            cache = self.__dict__.get(cache_attr)
            if cache is None:
                cache = functools.lru_cache(maxsize=maxsize)(functools.partial(method, self))
                self.__dict__[cache_attr] = cache
            return cache(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def logical_cpu_core_count() -> int:
    """Returns the number of logical CPU cores available to the process."""
    num_cpus = os.getenv("SLURM_CPUS_ON_NODE", None)
    if num_cpus is not None:
        return int(num_cpus)

    try:
        return os.cpu_count() or 1
    except NotImplementedError:
        return 1


# https://stackoverflow.com/a/58336722/1736826 CC-BY-SA 4.0
def dataclass_with_default_init(_cls=None, *args, **kwargs):
    def wrap(cls):
        # Save the current __init__ and remove it so dataclass will
        # create the default __init__.
        user_init = getattr(cls, "__init__")
        delattr(cls, "__init__")

        # let dataclass process our class.
        result = dataclass(cls, *args, **kwargs)

        # Restore the user's __init__ save the default init to __default_init__.
        setattr(result, "__default_init__", result.__init__)
        setattr(result, "__init__", user_init)

        # Just in case that dataclass will return a new instance,
        # (currently, does not happen), restore cls's __init__.
        if result is not cls:
            setattr(cls, "__init__", user_init)

        return result

    # Support both dataclass_with_default_init() and dataclass_with_default_init
    if _cls is None:
        return wrap
    else:
        return wrap(_cls)


@contextlib.contextmanager
def set_global_rng_seeds(seed):
    current_np_seed = np.random.get_state()
    np.random.seed(seed)

    current_random_seed = random.getstate()
    random.seed(seed)

    try:
        import torch  # noqa: PLC0415  # optional dep: torch

        current_torch_seed = torch.random.get_rng_state()
        torch.manual_seed(seed)
    except ImportError:
        torch = None
        current_torch_seed = None
        pass

    try:
        yield
    finally:
        np.random.set_state(current_np_seed)
        random.setstate(current_random_seed)
        if current_torch_seed is not None:
            torch.random.set_rng_state(current_torch_seed)


class FailSafeJSONEncoder(json.JSONEncoder):
    """
    A 'never-throw' JSON encoder:
    - Handles many common non-JSON types.
    - Degrades unknowns to a safe string payload.
    - Avoids blowing up on weird __repr__ or circulars.

    NOTES [Kevin: 10/15/25]:
    Marin also has a CustomJsonEncoder:
    `https://github.com/marin-community/marin/blob/4dec0f6fdb33d72846a1a1a5279d0c6da6fc118d/src/marin/utilities/json_encoder.py#L26`.
    Hopefully after the monorepo conversion is complete, we can just have a shared CustomJsonEncoder.
    """

    def __init__(self, *args, bytes_strategy="base64", **kwargs):
        # bytes_strategy: "base64" | "repr" | "hex"
        super().__init__(*args, **kwargs)
        self.bytes_strategy = bytes_strategy

    def default(self, o):
        # Known clean conversions
        if isinstance(o, (datetime.datetime, datetime.date, datetime.time)):
            # ISO 8601; preserves tzinfo if present
            return o.isoformat()

        if isinstance(o, decimal.Decimal):
            # Prefer float; fallback to string if NaN/Inf
            f = float(o)
            if f == float("inf") or f == float("-inf") or f != f:  # NaN check
                return str(o)
            return f

        if isinstance(o, uuid.UUID):
            return str(o)

        if isinstance(o, (set, frozenset)):
            return list(o)

        if isinstance(o, pathlib.Path):
            return str(o)

        if isinstance(o, complex):
            # JSON has no complex; encode as 2-tuple
            return {"__type__": "complex", "real": o.real, "imag": o.imag}

        if isinstance(o, bytes):
            if self.bytes_strategy == "base64":
                return {"__type__": "bytes", "base64": base64.b64encode(o).decode("ascii")}
            if self.bytes_strategy == "hex":
                return {"__type__": "bytes", "hex": o.hex()}
            return repr(o)

        if isinstance(o, bytearray):
            return self.default(bytes(o))

        if isinstance(o, enum.Enum):
            # Serialize as its value when simple; else name
            val = o.value
            # Make sure the value itself is JSON-serializable
            json.dumps(val)  # quick probe
            return val

        if is_dataclass(o):
            # Convert dataclasses to dicts (lets the base encoder recurse)
            return asdict(o)

        # Functions / callables -> a safe label
        if callable(o):
            name = getattr(o, "__name__", None)
            return f"<function {name}>" if name else "<callable>"

        # Everything else: use repr()
        return repr(o)
