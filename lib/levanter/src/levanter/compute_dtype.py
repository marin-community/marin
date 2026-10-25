"""Explicit per-field exceptions to the model's compute precision policy."""

from collections.abc import Mapping
from typing import TypeVar

import haliax as hax
import jax
import jax.numpy as jnp
import jmp

Model = TypeVar("Model")


def cast_model_for_compute(
    model: Model, mp: jmp.Policy, field_dtype_overrides: Mapping[str, str] | None = None
) -> Model:
    """Cast a compute view directly from masters, with exact attribute-name exceptions.

    Overrides match array-valued attributes, including NamedArrays, at any depth.
    They do not affect storage, trainability, or operator accumulation precision.
    Missing fields and non-floating overrides fail rather than silently doing nothing.
    """
    if not field_dtype_overrides:
        return mp.cast_to_compute(model)
    dtypes = {name: jnp.dtype(dtype) for name, dtype in field_dtype_overrides.items()}
    for name, dtype in dtypes.items():
        if not jnp.issubdtype(dtype, jnp.floating):
            raise ValueError(f"Compute override for {name} must be floating point")
    matched = set()

    def cast_leaf(path, leaf):
        field = path[-1].name if path and isinstance(path[-1], jax.tree_util.GetAttrKey) else None
        if field not in dtypes:
            return mp.cast_to_compute(leaf)
        if not isinstance(leaf, (jax.Array, hax.NamedArray)) or not jnp.issubdtype(leaf.dtype, jnp.floating):
            raise ValueError(f"Compute override field {field} is not a floating array")
        matched.add(field)
        return leaf.astype(dtypes[field])

    result = jax.tree_util.tree_map_with_path(cast_leaf, model, is_leaf=lambda leaf: isinstance(leaf, hax.NamedArray))
    if missing := set(dtypes) - matched:
        raise ValueError(f"Compute override fields not found: {sorted(missing)}")
    return result
