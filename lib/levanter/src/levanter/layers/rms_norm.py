# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Array-native RMS normalization."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec, reshard
from jaxtyping import Array, Float


class RmsNorm(eqx.Module):
    """Normalize in FP32 and apply affine parameters in the input dtype."""

    weight: Float[Array, "Dim"] | None
    bias: Float[Array, "Dim"] | None
    eps: float = eqx.field(static=True)
    compute_spec: PartitionSpec = eqx.field(static=True)

    @staticmethod
    def init(
        dim: int,
        *,
        eps: float,
        use_weight: bool,
        use_bias: bool,
        storage_spec: PartitionSpec,
        compute_spec: PartitionSpec,
    ) -> "RmsNorm":
        weight = reshard(jnp.ones(dim), storage_spec) if use_weight else None
        bias = reshard(jnp.zeros(dim), storage_spec) if use_bias else None
        return RmsNorm(weight, bias, eps, compute_spec)

    def __call__(self, x: Float[Array, "... Dim"]) -> Float[Array, "... Dim"]:
        normalized = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(normalized), axis=-1, keepdims=True)
        output = (normalized * jax.lax.rsqrt(variance + self.eps)).astype(x.dtype)
        if self.weight is not None:
            output = output * reshard(self.weight, self.compute_spec).astype(x.dtype)
        if self.bias is not None:
            output = output + reshard(self.bias, self.compute_spec).astype(x.dtype)
        return output
