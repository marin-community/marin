# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import functools as ft
import typing
import warnings
from typing import Any, Callable, Sequence

import equinox as eqx
import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jax import random as jrandom
from jax.ad_checkpoint import checkpoint_name
from jax.typing import DTypeLike
from jaxtyping import PRNGKeyArray

import haliax
from haliax.types import PrecisionLike

try:
    # jax v0.5.1 or newer
    from jax._src.numpy import (
        einsum as jax_einsum,  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
    )
except ImportError:
    # jax v0.5.0 or older
    from jax._src.numpy import lax_numpy as jax_einsum  # pylint: disable=g-import-not-at-top


F = typing.TypeVar("F", bound=Callable[..., Any])
T = typing.TypeVar("T")


class Static(eqx.Module):
    value: Any = eqx.field(static=True)


def shaped_rng_split(key, split_shape: int | Sequence[int] = 2) -> PRNGKeyArray:
    if isinstance(split_shape, int):
        num_splits = split_shape
        split_shape = (num_splits,) + key.shape
    else:
        num_splits = np.prod(split_shape)
        split_shape = tuple(split_shape) + key.shape

    if num_splits == 1:
        return jnp.reshape(key, split_shape)

    unshaped = maybe_rng_split(key, num_splits)
    return jnp.reshape(unshaped, split_shape)


def maybe_rng_split(key: PRNGKeyArray | None, num: int = 2):
    """Splits a random key into multiple random keys. If the key is None, then it replicates the None. Also handles
    num == 1 case"""
    if key is None:
        return [None] * num
    elif num == 1:
        return jnp.reshape(key, (1,) + key.shape)
    else:
        return jrandom.split(key, num)


@ft.wraps(eqx.filter_eval_shape)
def filter_eval_shape(*args, **kwargs):
    import warnings

    warnings.warn("filter_eval_shape is deprecated, use eqx.filter_eval_shape instead", DeprecationWarning)
    return eqx.filter_eval_shape(*args, **kwargs)


def filter_checkpoint(fun: Callable, *, prevent_cse: bool = True, policy: Callable[..., bool] | None = None):
    """As `jax.checkpoint`, but allows any Python object as inputs and outputs"""

    warnings.warn("filter_checkpoint is deprecated, use eqx.filter_checkpoint instead", DeprecationWarning)

    return eqx.filter_checkpoint(fun, prevent_cse=prevent_cse, policy=policy)


def is_jax_array_like(x):
    return hasattr(x, "shape") and hasattr(x, "dtype")  # and not isinstance(x, haliax.NamedArray)


# adapted from jax but exposed so i can use it
def broadcast_prefix(prefix_tree: Any, full_tree: Any, is_leaf: Callable[[Any], bool] | None = None):
    """Broadcast a prefix tree to match the structure of a full tree."""
    result = []
    num_leaves = lambda t: jax.tree_util.tree_structure(t).num_leaves  # noqa: E731
    add_leaves = lambda x, subtree: result.extend([x] * num_leaves(subtree))  # noqa: E731
    jax.tree_util.tree_map(add_leaves, prefix_tree, full_tree, is_leaf=is_leaf)
    full_structure = jax.tree_util.tree_structure(full_tree)

    return jax.tree_util.tree_unflatten(full_structure, result)


@ft.wraps(eqx.combine)
def combine(*args, **kwargs):
    import warnings

    warnings.warn("combine is deprecated, use eqx.combine instead", DeprecationWarning)
    return eqx.combine(*args, **kwargs)


def _UNSPECIFIED():
    raise ValueError("unspecified")


@typing.overload
def named_call(f: F, name: str | None = None) -> F: ...


@typing.overload
def named_call(*, name: str | None = None) -> Callable[[F], F]: ...


def named_call(f=_UNSPECIFIED, name: str | None = None):
    if f is _UNSPECIFIED:
        return lambda f: named_call(f, name)  # type: ignore
    else:
        if name is None:
            name = f.__name__
            if name == "__call__":
                if hasattr(f, "__self__"):
                    name = f.__self__.__class__.__name__  # type: ignore
                else:
                    name = f.__qualname__.rsplit(".", maxsplit=1)[0]  # type: ignore
            else:
                name = f.__qualname__

        return jax.named_scope(name)(f)


def is_in_jit():
    return isinstance(jnp.zeros((), dtype=jnp.float32), jax.core.Tracer)


def is_pallas_dslice(x: object) -> bool:
    try:
        from jax.experimental.pallas import dslice as pdslice
    except ImportError:
        return False

    _PALLAS_DSLICE_TYPE = type(pdslice(0, 1))
    return isinstance(x, _PALLAS_DSLICE_TYPE)


def is_scalarish(x):
    if isinstance(x, haliax.NamedArray):
        return x.ndim == 0
    else:
        return jnp.isscalar(x) or (getattr(x, "shape", None) == ())


def ensure_scalar(x, *, name: str = "value"):
    """Return ``x`` if it is not a :class:`NamedArray`, otherwise ensure it is a scalar.

    This is useful for APIs that can accept either Python scalars or scalar
    ``NamedArray`` objects (for example ``roll`` or ``updated_slice``).  If ``x``
    is a ``NamedArray`` with rank greater than 0 a :class:`TypeError` is raised.
    """

    if isinstance(x, haliax.NamedArray):
        if x.ndim != 0:
            raise TypeError(f"{name} must be a scalar NamedArray")
        return x.array
    return x


def is_on_mac_metal():
    return jax.devices()[0].platform.lower() == "metal"


def _jittable_dg_einsum(
    subscripts,
    /,
    *operands,
    out: None = None,
    optimize: str = "optimal",
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    _dot_general: Callable[..., Array] = jax.lax.dot_general,
) -> Array:
    """
    So we want to pass around a jittable dot_general module, but JAX's builtin version doesn't support this.
    So we copy over the implementation of jax.numpy.einsum and modify thing so that it is jittable (via
    eqx.filter_jit)

    More or less copied from AQT
    """
    operands = (subscripts, *operands)
    if out is not None:
        raise NotImplementedError("The 'out' argument to jnp.einsum is not supported.")
    spec = operands[0] if isinstance(operands[0], str) else None
    optimize = "optimal" if optimize is True else optimize

    import opt_einsum

    # Allow handling of shape polymorphism
    non_constant_dim_types = {
        type(d) for op in operands if not isinstance(op, str) for d in np.shape(op) if not jax.core.is_constant_dim(d)
    }
    if not non_constant_dim_types:
        contract_path = opt_einsum.contract_path
    else:
        ty = next(iter(non_constant_dim_types))
        contract_path = jax_einsum._poly_einsum_handlers.get(ty, jax_einsum._default_poly_einsum_handler)
    # using einsum_call=True here is an internal api for opt_einsum... sorry
    operands, contractions = contract_path(*operands, einsum_call=True, use_blas=True, optimize=optimize)

    contractions = tuple((a, frozenset(b), c) for a, b, c, *_ in contractions)

    einsum = eqx.filter_jit(jax_einsum._einsum, inline=True)
    if spec is not None:
        einsum = jax.named_call(einsum, name=spec)
    return einsum(operands, contractions, precision, preferred_element_type, _dot_general)  # type: ignore[operator]


def tree_checkpoint_name(x: T, name: str) -> T:
    """
    Checkpoint a tree of arrays with a given name. This is useful for gradient checkpointing.
    This is equivalent to calling [jax.ad_checkpoint.checkpoint_name][]
    except that it works for any PyTree, not just arrays.

    See Also:
        * [jax.ad_checkpoint.checkpoint_name][]
        * [haliax.nn.ScanCheckpointPolicy][]
    """

    def _checkpoint_leaf(x):
        if is_jax_array_like(x):
            return checkpoint_name(x, name)
        else:
            return x

    return jax.tree.map(_checkpoint_leaf, x)


def multilevel_scan(f, carry, xs, outer_size, length, reverse=False, unroll=1):
    """

    Similar to jax.lax.scan, but "nested". You take your scanned axis and break it up into outer_size chunks, then
    scan each chunk with a scan.

    You use this if you want to save memory by, e.g., implementing the sqrt(N) memory trick for checkpointing.

    This is typically ~20% slower than the O(n) memory thing, but it's often worthwhile.

    Credit to Roy and Matt.
    """

    inner_size = length // outer_size

    if inner_size * outer_size != length:
        raise ValueError(f"Length {length} must be divisible by outer_size {outer_size}")

    def _reshape(x):
        if is_jax_array_like(x) and x.shape != ():
            return x.reshape([outer_size, inner_size, *x.shape[1:]])
        else:
            return x

    xs_shaped = jax.tree.map(_reshape, xs)

    carry, scanned = jax.lax.scan(
        jax.remat(ft.partial(jax.lax.scan, f, reverse=reverse, unroll=unroll)),
        carry,
        xs_shaped,
        reverse=reverse,
        unroll=True,
    )

    def _deshape(x):
        if is_jax_array_like(x) and x.shape != ():
            return x.reshape([length, *x.shape[2:]])
        else:
            return x

    return carry, jax.tree.map(_deshape, scanned)


def to_jax_shape(shape):
    from haliax.core import Axis, ensure_tuple

    shape = ensure_tuple(shape)
    return tuple(axis.size if isinstance(axis, Axis) else axis for axis in shape)
