# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Give named model fields their own floating dtype under a mixed-precision policy.

Some small arrays, such as MoE router biases, lose meaning when rounded to bfloat16. An override maps an attribute
name, matched exactly at any depth, to a floating ``jnp.dtype``; in YAML it is written ``{router_bias: float32}``.
The policy casts that field to its own dtype wherever it casts: in the trainer state, in the compute view, on HF
load and in HF exports.
"""

import dataclasses
from collections.abc import Mapping
from typing import TypeVar

import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np

T = TypeVar("T")

FieldDtypes = tuple[tuple[str, jnp.dtype], ...]
"""Overrides as ``(field name, dtype)`` pairs sorted by name, hashable so a policy can be a static jit argument."""


def sorted_field_dtypes(overrides: Mapping[str, jnp.dtype] | None) -> FieldDtypes:
    """``overrides`` as sorted, hashable ``(field name, dtype)`` pairs."""
    return tuple(sorted((overrides or {}).items()))


def _is_floating_array(leaf) -> bool:
    return isinstance(leaf, (np.ndarray, jax.Array, hax.NamedArray)) and jnp.issubdtype(leaf.dtype, jnp.floating)


def _field_name(path) -> str | None:
    last = path[-1] if path else None
    return last.name if isinstance(last, jax.tree_util.GetAttrKey) else None


def cast_with_field_dtypes(tree: T, dtype: jnp.dtype | None, field_dtypes: FieldDtypes) -> T:
    """Cast floating arrays to ``dtype``, except overridden fields, which get their own dtype.

    With ``dtype=None`` only the overridden fields are cast.
    """
    targets = dict(field_dtypes)

    def cast_leaf(path, leaf):
        if not _is_floating_array(leaf):
            return leaf
        target = targets.get(_field_name(path), dtype)
        return leaf if target is None else leaf.astype(target)

    return jax.tree_util.tree_map_with_path(cast_leaf, tree, is_leaf=lambda leaf: isinstance(leaf, hax.NamedArray))


def check_field_dtype_overrides(model, overrides: Mapping[str, jnp.dtype]) -> None:
    """Raise if an override is not a floating dtype or names no floating array in ``model``.

    A typo in a field name cannot silently do nothing. Unknown dtype names already fail when the config is decoded.
    """
    for name, dtype in overrides.items():
        if not jnp.issubdtype(dtype, jnp.floating):
            raise ValueError(f"field dtype override {name}: {dtype} is not a floating dtype")
    fields = frozenset(overrides)
    if not fields:
        return
    matched = set()

    def visit(path, leaf):
        name = _field_name(path)
        if name in fields:
            if not _is_floating_array(leaf):
                raise ValueError(f"field dtype override {name} is not a floating-point array")
            matched.add(name)
        return leaf

    jax.tree_util.tree_map_with_path(visit, model, is_leaf=lambda leaf: isinstance(leaf, hax.NamedArray))
    if missing := fields - matched:
        raise ValueError(f"field dtype overrides not found in model: {sorted(missing)}")


@dataclasses.dataclass(frozen=True)
class FieldDtypePolicy(jmp.Policy):
    """A ``jmp.Policy`` whose parameter and compute casts give overridden fields their own dtype."""

    field_dtypes: FieldDtypes = ()

    def cast_to_param(self, x):
        return cast_with_field_dtypes(x, self.param_dtype, self.field_dtypes)

    def cast_to_compute(self, x):
        return cast_with_field_dtypes(x, self.compute_dtype, self.field_dtypes)


def with_field_dtype_overrides(mp: jmp.Policy, overrides: Mapping[str, jnp.dtype]) -> jmp.Policy:
    """``mp`` itself without overrides; otherwise ``mp`` with each overridden field cast to its own dtype."""
    if not overrides:
        return mp
    return FieldDtypePolicy(mp.param_dtype, mp.compute_dtype, mp.output_dtype, sorted_field_dtypes(overrides))


def hf_load_dtype(mp: jmp.Policy) -> jnp.dtype | None:
    """The dtype for an HF load under ``mp``.

    Without overrides, the whole model loads in compute dtype. With overrides, it loads at the stored dtypes and
    ``mp.cast_to_param`` casts it afterwards; a compute-dtype load would round the overridden fields first.
    """
    return None if isinstance(mp, FieldDtypePolicy) else mp.compute_dtype
