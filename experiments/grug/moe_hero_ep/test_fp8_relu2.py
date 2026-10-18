# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, set_mesh
from levanter.utils.activation import ActivationFunctionEnum

from experiments.grug.moe_hero_ep import model

_HIDDEN = 256
_WIDTH = 384
_TOKENS = 128


def _mesh():
    return Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1, 1),
        ("replica_dcn", "data", "context", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 5,
    )


def _config(fp8_recipe: model.Fp8Recipe) -> model.GrugModelConfig:
    return model.GrugModelConfig(
        vocab_size=128,
        hidden_dim=_HIDDEN,
        intermediate_dim=_WIDTH,
        shared_expert_intermediate_dim=_WIDTH,
        num_shared_experts=2,
        num_heads=2,
        num_kv_heads=2,
        head_dim=128,
        fp8_recipe=fp8_recipe,
    )


def _shared_experts(*, ungated: bool) -> tuple[model.DenseMLP, ...]:
    keys = jax.random.split(jax.random.key(0), 2)
    experts = tuple(model.DenseMLP.init(_HIDDEN, _WIDTH, _HIDDEN**-0.5, key=key) for key in keys)
    if ungated:
        experts = tuple(dataclasses.replace(expert, w_gate=None) for expert in experts)
    return experts


def _activations() -> jax.Array:
    return jax.random.normal(jax.random.key(1), (2, _TOKENS // 2, _HIDDEN)).astype(jnp.bfloat16)


def test_ungated_shared_expert_is_relu2_of_the_up_projection():
    with set_mesh(_mesh()):
        (expert, _) = _shared_experts(ungated=True)
        x = _activations()
        actual = expert(x, activation=ActivationFunctionEnum.relu2)
    # fast_track's ungated ReLU^2 expert: relu(x @ W_up)^2 @ W_down.
    pre = np.asarray(x, np.float32) @ np.asarray(expert.w_up, np.float32)
    expected = np.square(np.maximum(pre, 0)) @ np.asarray(expert.w_down, np.float32)
    np.testing.assert_allclose(np.asarray(actual, np.float32), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("ungated", [False, True], ids=["swiglu", "relu2"])
def test_fused_shared_experts_match_the_sum_of_experts(ungated):
    activation = ActivationFunctionEnum.relu2 if ungated else ActivationFunctionEnum.silu
    with set_mesh(_mesh()):
        experts = _shared_experts(ungated=ungated)
        x = _activations()
        fused = model._shared_experts_fused(_config(model.Fp8Recipe.NONE), experts, x, activation)
        summed = sum(expert(x, activation=activation) for expert in experts)
    np.testing.assert_allclose(np.asarray(fused, np.float32), np.asarray(summed, np.float32), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("ungated", [False, True], ids=["swiglu", "relu2"])
def test_mxfp8_dense_shared_experts_stay_within_mxfp8_error_of_bf16(ungated):
    if jax.default_backend() != "gpu" or float(jax.devices()[0].compute_capability) < 10.0:
        pytest.skip("MXFP8 projections require an SM100 GPU")
    activation = ActivationFunctionEnum.relu2 if ungated else ActivationFunctionEnum.silu
    experts = _shared_experts(ungated=ungated)
    x = _activations()
    dy = jax.random.normal(jax.random.key(2), x.shape).astype(jnp.bfloat16)

    def run(recipe):
        cfg = _config(recipe)

        def loss(x, experts):
            out = model._shared_experts_fused(cfg, experts, x, activation)
            return jnp.sum(out.astype(jnp.float32) * dy), out

        (_, out), (dx, dexperts) = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)(x, experts)
        return out, dx, dexperts[0].w_up, dexperts[0].w_down

    with set_mesh(_mesh()):
        actual = jax.jit(lambda: run(model.Fp8Recipe.MXFP8_DENSE))()
        expected = jax.jit(lambda: run(model.Fp8Recipe.NONE))()
    for name, got, want in zip(("y", "dx", "dw_up", "dw_down"), actual, expected, strict=True):
        got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
        relative = float(np.linalg.norm(got - want) / np.linalg.norm(want))
        assert 0 < relative < 0.1, f"{name}: relative L2 error {relative:.3e}"
