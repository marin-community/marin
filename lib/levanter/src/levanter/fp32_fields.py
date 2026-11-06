# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep named model fields in float32 under a mixed-precision policy.

Some small arrays, such as MoE router biases, lose meaning when rounded to bfloat16. A field listed here is
matched by its exact attribute name at any depth and stays float32 wherever the policy casts: on load, in the
trainer state, in the compute view and in HF exports.
"""

import dataclasses
from collections.abc import Iterable
from typing import TypeVar

import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np

T = TypeVar("T")


def _is_floating_array(leaf) -> bool:
    return isinstance(leaf, (np.ndarray, jax.Array, hax.NamedArray)) and jnp.issubdtype(leaf.dtype, jnp.floating)


def _field_name(path) -> str | None:
    last = path[-1] if path else None
    return last.name if isinstance(last, jax.tree_util.GetAttrKey) else None


def cast_floating(tree: T, dtype: jnp.dtype | None, fp32_fields: Iterable[str]) -> T:
    """Cast floating arrays to ``dtype``, except fields named in ``fp32_fields``, which become float32.

    With ``dtype=None`` only the named fields are cast.
    """
    fields = frozenset(fp32_fields)

    def cast_leaf(path, leaf):
        if not _is_floating_array(leaf):
            return leaf
        target = jnp.float32 if _field_name(path) in fields else dtype
        return leaf if target is None else leaf.astype(target)

    return jax.tree_util.tree_map_with_path(cast_leaf, tree, is_leaf=lambda leaf: isinstance(leaf, hax.NamedArray))


def check_fp32_fields(model, fp32_fields: Iterable[str]) -> None:
    """Raise if a listed field names no floating array in ``model``, so a typo cannot silently do nothing."""
    fields = frozenset(fp32_fields)
    if not fields:
        return
    matched = set()

    def visit(path, leaf):
        name = _field_name(path)
        if name in fields:
            if not _is_floating_array(leaf):
                raise ValueError(f"fp32 field {name} is not a floating-point array")
            matched.add(name)
        return leaf

    jax.tree_util.tree_map_with_path(visit, model, is_leaf=lambda leaf: isinstance(leaf, hax.NamedArray))
    if missing := fields - matched:
        raise ValueError(f"fp32 fields not found in model: {sorted(missing)}")


@dataclasses.dataclass(frozen=True)
class Fp32FieldsPolicy(jmp.Policy):
    """A ``jmp.Policy`` whose parameter and compute casts keep the named fields in float32."""

    fp32_fields: tuple[str, ...] = ()

    def cast_to_param(self, x):
        return cast_floating(x, self.param_dtype, self.fp32_fields)

    def cast_to_compute(self, x):
        return cast_floating(x, self.compute_dtype, self.fp32_fields)


def with_fp32_fields(mp: jmp.Policy, fp32_fields: Iterable[str]) -> jmp.Policy:
    """``mp`` itself when no fields are listed; otherwise ``mp`` with the named fields kept in float32."""
    fields = tuple(fp32_fields)
    if not fields:
        return mp
    return Fp32FieldsPolicy(mp.param_dtype, mp.compute_dtype, mp.output_dtype, fields)
