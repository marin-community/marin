# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit per-field exceptions to the model's compute precision policy."""

import dataclasses
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


@dataclasses.dataclass(frozen=True)
class OverridingPolicy(jmp.Policy):
    """A mixed-precision policy whose ``cast_to_compute`` applies field dtype overrides.

    Code that only receives a policy, such as the evaluation loops, then builds the same compute view as training.
    """

    field_dtype_overrides: tuple[tuple[str, str], ...] = ()

    def cast_to_compute(self, x):
        base = jmp.Policy(self.param_dtype, self.compute_dtype, self.output_dtype)
        return cast_model_for_compute(x, base, dict(self.field_dtype_overrides))


def compute_policy(mp: jmp.Policy, field_dtype_overrides: Mapping[str, str] | None) -> jmp.Policy:
    """``mp`` itself without overrides; otherwise a policy that casts with them."""
    if not field_dtype_overrides:
        return mp
    return OverridingPolicy(
        mp.param_dtype, mp.compute_dtype, mp.output_dtype, tuple(sorted(field_dtype_overrides.items()))
    )
