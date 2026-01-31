# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Template for adding a Pallas kernel safely (Tokamax-style dispatch).

This file is intentionally not “the best kernel”. It is a scaffold showing the expected structure:
1) a vanilla JAX reference implementation (the oracle)
2) one or more accelerated implementations (e.g. Pallas on TPU)
3) a stable API entrypoint that selects among implementations, following the same pattern as Tokamax:
   an explicit `implementation=` option plus a best-available default fallback order.

Batching convention:
- Internal implementations should be written for the *batched* case.
- The public API should accept either batched inputs or a single unbatched example and will add/remove a trivial
  batch dimension as needed.

See `docs/recipes/add_pallas_kernel.md` for the workflow.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias, cast

import jax
import jax.numpy as jnp


def _ensure_batched(x: jax.Array) -> tuple[jax.Array, bool]:
    """Ensure `x` has a leading batch dimension.

    Template convention:
    - if `x.ndim == 1`, treat it as a single example and add a batch axis.
    - otherwise treat it as already batched.

    Returns:
        (x_batched, added_batch_dim)
    """

    if x.ndim == 1:
        return x[None, :], True
    return x, False


def reference_impl_batched(x_batched: jax.Array) -> jax.Array:
    """Reference implementation (oracle).

    Replace this with a readable, correct baseline for your kernel. This should operate on batched inputs.
    """

    return jnp.tanh(x_batched)


def xla_impl_batched(x_batched: jax.Array) -> jax.Array:
    """Default implementation (XLA / plain JAX).

    For most kernels, this should be the same as the reference implementation.
    """

    return reference_impl_batched(x_batched)


def pallas_tpu_impl_batched(x_batched: jax.Array) -> jax.Array:
    """TPU/Pallas implementation placeholder.

    Replace this body with a real `jax.experimental.pallas` kernel. Keep the signature stable and match the reference.
    """

    return reference_impl_batched(x_batched)


Implementation: TypeAlias = Literal["pallas_tpu", "xla"]

IMPLEMENTATIONS: dict[str, Callable[..., jax.Array]] = {"xla": xla_impl_batched}
_DEFAULT_IMPLEMENTATION: tuple[Implementation, ...] = ("xla",)

try:
    # In real kernels, import the TPU backend here. Keep this optional so CPU-only users can still import the module.
    # Example:
    # from .my_kernel_pallas_tpu import pallas_tpu_impl as _pallas_impl
    IMPLEMENTATIONS["pallas_tpu"] = pallas_tpu_impl_batched
    _DEFAULT_IMPLEMENTATION = ("pallas_tpu",) + _DEFAULT_IMPLEMENTATION
except ImportError:
    # TPU/Pallas backend is not available; keep the XLA-only implementations.
    pass


def template_op(
    x: jax.Array,
    *,
    implementation: Implementation | Sequence[Implementation | Callable[..., jax.Array]] | None = None,
) -> jax.Array:
    """Public API entrypoint (dispatches to best available implementation).

    This mirrors Tokamax's pattern:
    - `implementation="xla"` or `"pallas_tpu"` forces a specific backend.
    - `implementation=None` tries a default order (best available first).
    - Advanced: `implementation=[..., callable]` allows injecting an implementation for experimentation.

    Batching convention:
    - `x` may be a single example (shape `(H,)`) or batched (shape `(B, H)` or more dims).
    - Implementations are expected to handle the *batched* form.
    """

    x_batched, added_batch_dim = _ensure_batched(x)

    if implementation is None:
        impls: Sequence[Implementation | Callable[..., jax.Array]] = _DEFAULT_IMPLEMENTATION
    elif isinstance(implementation, Sequence) and not isinstance(implementation, (str, bytes)):
        impls = cast(Sequence[Implementation | Callable[..., jax.Array]], implementation)
    else:
        impls = (cast(Implementation, implementation),)

    errors: list[Exception] = []
    for impl in impls:
        if callable(impl):
            try:
                y = impl(x_batched)
                return y[0] if added_batch_dim else y
            except NotImplementedError as e:
                errors.append(e)
                continue
        fn = IMPLEMENTATIONS.get(impl)
        if fn is None:
            raise ValueError(f"Unsupported implementation: {impl}")
        try:
            y = fn(x_batched)
            return y[0] if added_batch_dim else y
        except NotImplementedError as e:
            errors.append(e)

    raise ExceptionGroup("all implementations failed", errors)


__all__ = [
    "Implementation",
    "IMPLEMENTATIONS",
    "pallas_tpu_impl_batched",
    "reference_impl_batched",
    "template_op",
    "xla_impl_batched",
]
