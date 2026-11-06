# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Array-native linear projections with separate storage and compute layouts."""

from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec, reshard
from jaxtyping import Array, Float, PRNGKeyArray

from levanter.layers.dropout import DropoutLayout

INITIALIZER_TRUNCATION_BOUND = 3.0


class LinearProjection(Protocol):
    def __call__(
        self,
        x: Float[Array, "... In"],
        *,
        out_sharding: PartitionSpec,
        key: PRNGKeyArray | None = None,
        dropout_layout: DropoutLayout | None = None,
    ) -> Float[Array, "... Out"]: ...


class RawDotGeneralOp(Protocol):
    """Array contraction with an explicit result layout."""

    def __call__(
        self,
        lhs: Array,
        rhs: Array,
        dimension_numbers: tuple[tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[int, ...], tuple[int, ...]]],
        *,
        out_sharding: PartitionSpec,
    ) -> Array: ...


class Linear(eqx.Module):
    """Output-first weights, optionally gathered from an FSDP storage layout."""

    weight: Float[Array, "Out In"]
    bias: Float[Array, "Out"] | None
    compute_spec: PartitionSpec = eqx.field(static=True)
    dot_general: RawDotGeneralOp | None = None

    @staticmethod
    def init(
        in_features: int,
        out_features: int,
        *,
        key: PRNGKeyArray,
        use_bias: bool,
        initializer_std: float,
        storage_spec: PartitionSpec,
        compute_spec: PartitionSpec,
    ) -> "Linear":
        weight = initializer_std * jax.random.truncated_normal(
            key,
            -INITIALIZER_TRUNCATION_BOUND,
            INITIALIZER_TRUNCATION_BOUND,
            (out_features, in_features),
            out_sharding=storage_spec,
        )
        bias = reshard(jnp.zeros(out_features), PartitionSpec(storage_spec[0])) if use_bias else None
        return Linear(weight, bias, compute_spec)

    def __call__(
        self,
        x: Float[Array, "... In"],
        *,
        out_sharding: PartitionSpec,
        key: PRNGKeyArray | None = None,
        dropout_layout: DropoutLayout | None = None,
    ) -> Float[Array, "... Out"]:
        del key, dropout_layout
        weight = reshard(self.weight, self.compute_spec)
        if self.dot_general is None:
            output = jnp.einsum("...i,oi->...o", x, weight, out_sharding=out_sharding)
        else:
            output = self.dot_general(x, weight, (((x.ndim - 1,), (1,)), ((), ())), out_sharding=out_sharding)
        if self.bias is not None:
            output = output + reshard(self.bias, PartitionSpec(self.compute_spec[0]))
        return output
