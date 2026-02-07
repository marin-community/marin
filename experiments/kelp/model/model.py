# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared transformer building blocks for Kelp models.

Provides parameter dataclasses and primitive operations (RMS norm, SwiGLU MLP,
weight initialization) that are shared between the AR edit-prediction model
and any future model variants.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import random
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, PRNGKeyArray


@register_dataclass
@dataclass(frozen=True)
class TreeDiffusionAttentionParams:
    """Parameters for a single attention layer."""

    w_q: jax.Array
    w_k: jax.Array
    w_v: jax.Array
    w_o: jax.Array


@register_dataclass
@dataclass(frozen=True)
class TreeDiffusionBlockParams:
    """Parameters for a transformer block."""

    attn: TreeDiffusionAttentionParams
    rms_attn: jax.Array
    rms_mlp: jax.Array
    mlp_gate: jax.Array
    mlp_up: jax.Array
    mlp_down: jax.Array


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Float[Array, "..."]:
    """Initialize weights with truncated normal."""
    return std * random.truncated_normal(key, -3, 3, shape)


def rms_norm(x: Float[Array, "... D"], weight: Float[Array, "D"], eps: float) -> Float[Array, "... D"]:
    """RMS normalization.

    Kept local rather than delegating to Grug's version because Grug's
    calls unshard(weight) which requires a JAX mesh context.
    """
    dtype = x.dtype
    x = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed = x * jax.lax.rsqrt(variance + eps)
    out = normed * weight
    return out.astype(dtype)


def mlp(block: TreeDiffusionBlockParams, x: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
    """SwiGLU MLP."""
    gate = jnp.einsum("bsh,hm->bsm", x, block.mlp_gate)
    up = jnp.einsum("bsh,hm->bsm", x, block.mlp_up)
    activated = jax.nn.silu(gate) * up
    return jnp.einsum("bsm,mh->bsh", activated, block.mlp_down)
