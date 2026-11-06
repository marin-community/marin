# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Array dropout with an explicit logical mask layout."""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray


@dataclass(frozen=True)
class DropoutLayout:
    """Unflattened input shape and axis order used to sample a dropout mask.

    A smaller leading dimension shares the logical mask across replicated inputs.
    """

    shape: tuple[int, ...]
    permutation: tuple[int, ...]

    def restore(self, mask: jax.Array, input_shape: tuple[int, ...]) -> jax.Array:
        inverse = tuple(self.permutation.index(i) for i in range(len(self.permutation)))
        return mask.transpose(inverse).reshape(self.shape[0], *input_shape[1:])


class ArrayDropout(eqx.Module):
    """Dropout whose mask can precede feature flattening or batch replication."""

    pdrop: float = eqx.field(static=True)
    inference: bool = False

    def __call__(self, x, *, key: PRNGKeyArray | None = None, layout: DropoutLayout | None = None):
        if self.inference or self.pdrop == 0:
            return x
        if key is None:
            raise RuntimeError("Dropout requires a key when running in non-deterministic mode.")
        if self.pdrop == 1:
            return jnp.zeros_like(x)
        keep = 1 - self.pdrop
        shape = x.shape if layout is None else tuple(layout.shape[i] for i in layout.permutation)
        mask = jax.random.bernoulli(key, keep, shape)
        if layout is not None:
            mask = layout.restore(mask, x.shape)
        return jnp.where(mask, x / x.dtype.type(keep), 0)
