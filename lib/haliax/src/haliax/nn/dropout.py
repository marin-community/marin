# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

import haliax
from haliax.axis import AxisSpec
from haliax.core import NamedArray
from haliax.util import ensure_tuple


def dropout(x, pdrop, broadcast_axes=None, *, inference, key=None):
    """Applies dropout.

    **Arguments:**

    - `x`: An any-dimensional JAX array to dropout.
    - `pdrop`: The fraction of entries to set to zero.
    - `broadcast_axes`: The dimensions to broadcast the dropout mask over. If set, these axes will share the same mask
    - `key`: A `jax.random.PRNGKey` used to provide randomness for calculating
        which elements to dropout. (Keyword only argument.)
    - `inference`: As per [`equinox.nn.Dropout.__init__`][]. If `True` or
        `False` then it will take priority over `self.inference`. If `None`
        then the value from `self.inference` will be used.
    """
    if inference:
        return x
    elif isinstance(pdrop, (int, float)) and pdrop == 0:
        return x
    elif isinstance(pdrop, (int, float)) and pdrop == 1:
        return haliax.zeros_like(x)
    elif key is None:
        raise RuntimeError("Dropout requires a key when running in non-deterministic mode.")
    else:
        with jax.named_scope(name="dropout"):
            if broadcast_axes is None:
                if isinstance(x, NamedArray):
                    shape_to_generate = x.axes
                else:
                    shape_to_generate = x.shape
            else:
                axes = ensure_tuple(broadcast_axes)
                shape_to_generate = tuple(ax for ax in x.axes if ax not in axes)

            q = 1 - pdrop
            mask: NamedArray = haliax.random.bernoulli(key, shape_to_generate, q)
            q = x.dtype.type(q)

            out = haliax.where(mask, x / q, 0)
            assert out.dtype == x.dtype
            return out


class Dropout(eqx.Module):
    """Applies dropout.

    Attributes:
        pdrop: The fraction of entries to set to zero.
        broadcast_axes: The dimensions to broadcast the dropout mask over. If set, these axes will share the same mask
    """

    # key difference from equinox: these are static fields
    pdrop: float = eqx.field(static=True)
    broadcast_axes: AxisSpec | None = eqx.field(static=True)
    inference: bool = False  # note: not static

    def __init__(
        self,
        pdrop: float = 0.5,
        broadcast_axes: AxisSpec | None = None,
        inference: bool = False,
    ):
        self.pdrop = pdrop
        self.broadcast_axes = broadcast_axes
        self.inference = inference

    @property
    def is_active(self):
        """Returns `True` if dropout is active (and therefore needs a key), `False` otherwise."""
        return not self.inference and self.pdrop > 0

    def __call__(
        self,
        x: NamedArray,
        *,
        inference: bool | None = None,
        key: PRNGKeyArray | None = None,
    ) -> NamedArray:
        """**Arguments:**

        - `x`: An any-dimensional JAX array to dropout.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for calculating
            which elements to dropout. (Keyword only argument.)
        - `inference`: As per [`equinox.nn.Dropout.__init__`][]. If `True` or
            `False` then it will take priority over `self.inference`. If `None`
            then the value from `self.inference` will be used.
        """
        if inference is None:
            inference = self.inference

        return dropout(
            x,
            self.pdrop,
            broadcast_axes=self.broadcast_axes,
            inference=inference,
            key=key,
        )
