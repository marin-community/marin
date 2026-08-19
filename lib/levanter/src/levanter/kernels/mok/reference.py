# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""A pure-JAX restatement of what :func:`levanter.kernels.mok.mok_bf16` computes.

This exists so the fused megakernel has something to be checked against without a GPU, a wheel,
or a symmetric-memory workspace. It is deliberately dense and single-device: it makes no attempt
to reproduce dispatch, capacity, or the expert-parallel all-to-all, only the arithmetic those
mechanics are supposed to preserve.

Every operand is a *canonical Marin leaf*, in exactly the orientation
:func:`levanter.kernels.mok.ffi.mok_bf16` accepts, so a caller can pass the same tuple to both.
Notably ``w_latent_down`` is ``(hidden, latent)`` and ``w_latent_up`` is ``(latent, hidden)`` --
the kernel wants each transposed, and in opposite directions, which is why the transposes live
behind the FFI boundary rather than here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _swiglu(x: jax.Array, gate: jax.Array, up: jax.Array, down: jax.Array) -> jax.Array:
    return (jax.nn.silu(x @ gate) * (x @ up)) @ down


def rmsnorm_reference(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    """The pre-dispatch latent RMSNorm, in float32, matching the hero model's ``RMSNorm``."""

    x32 = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    return x32 * jax.lax.rsqrt(variance + eps) * weight.astype(jnp.float32)


def routed_experts_reference(
    routed_input: jax.Array,
    selected_experts: jax.Array,
    router_weights: jax.Array,
    routed_gate: jax.Array,
    routed_up: jax.Array,
    routed_down: jax.Array,
) -> jax.Array:
    """Dense, dropless top-k expert combine at the routed width.

    ``routed_gate``/``routed_up`` are ``(experts, routed, intermediate)`` and ``routed_down`` is
    ``(experts, intermediate, routed)`` -- the canonical Marin layout.
    """

    routed_input = routed_input.astype(jnp.float32)
    gate = routed_gate.astype(jnp.float32)
    up = routed_up.astype(jnp.float32)
    down = routed_down.astype(jnp.float32)

    def _one_expert(expert: jax.Array) -> jax.Array:
        return _swiglu(routed_input, gate[expert], up[expert], down[expert])

    # (experts, tokens, routed); dense on purpose, so the reference has no capacity behaviour of
    # its own to disagree with.
    per_expert = jax.vmap(_one_expert)(jnp.arange(gate.shape[0]))
    contributions = jnp.take_along_axis(
        per_expert,
        selected_experts.astype(jnp.int32).T[:, :, None],
        axis=0,
    )
    return jnp.sum(contributions * router_weights.astype(jnp.float32).T[:, :, None], axis=0)


def mok_bf16_reference(
    x: jax.Array,
    selected_experts: jax.Array,
    router_weights: jax.Array,
    shared0_gate: jax.Array,
    shared0_up: jax.Array,
    shared0_down: jax.Array,
    shared1_gate: jax.Array,
    shared1_up: jax.Array,
    shared1_down: jax.Array,
    routed_gate: jax.Array,
    routed_up: jax.Array,
    routed_down: jax.Array,
    latent_down: jax.Array | None = None,
    latent_norm_weight: jax.Array | None = None,
    latent_up: jax.Array | None = None,
    *,
    latent_norm_eps: float = 1e-5,
) -> jax.Array:
    """Compute the fused call's output densely, in float32.

    With the three latent operands present the result is

        ``shared0(x) + shared1(x) + routed(rmsnorm(x @ W_down)) @ W_up``

    where the shared experts read the full ``hidden`` width and everything between the two
    projections runs at ``latent``. Omitting them (the ``latent_size == 0`` control arm) drops
    both projections and the norm, and the routed path then runs at ``hidden``.
    """

    x32 = x.astype(jnp.float32)
    shared = _swiglu(
        x32,
        shared0_gate.astype(jnp.float32),
        shared0_up.astype(jnp.float32),
        shared0_down.astype(jnp.float32),
    ) + _swiglu(
        x32,
        shared1_gate.astype(jnp.float32),
        shared1_up.astype(jnp.float32),
        shared1_down.astype(jnp.float32),
    )

    latent_operands = (latent_down, latent_norm_weight, latent_up)
    latent_enabled = all(operand is not None for operand in latent_operands)
    if not latent_enabled and any(operand is not None for operand in latent_operands):
        raise ValueError("latent weights must be passed together or omitted together")

    if latent_enabled:
        assert latent_down is not None and latent_norm_weight is not None and latent_up is not None
        routed_input = rmsnorm_reference(
            x32 @ latent_down.astype(jnp.float32),
            latent_norm_weight,
            latent_norm_eps,
        )
    else:
        routed_input = x32

    routed = routed_experts_reference(
        routed_input,
        selected_experts,
        router_weights,
        routed_gate,
        routed_up,
        routed_down,
    )
    if latent_enabled:
        assert latent_up is not None
        routed = routed @ latent_up.astype(jnp.float32)
    return shared + routed
