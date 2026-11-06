# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0


import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest

from levanter.fp32_fields import check_fp32_fields, with_fp32_fields
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.mesh import MeshConfig


class Router(eqx.Module):
    router_bias: jax.Array | hax.NamedArray
    weight: jax.Array


def router(named=False):
    bias = jnp.asarray([0.1234567, -0.7654321], dtype=jnp.float32)
    if named:
        bias = hax.named(bias, hax.Axis("expert", 2))
    return Router(bias, jnp.asarray(0.1234567, dtype=jnp.float32))


@pytest.mark.parametrize("named", [False, True])
def test_compute_view_preserves_fp32_field_bits(named):
    model = tuple(router(named) for _ in range(26))
    policy = with_fp32_fields(jmp.get_policy("p=f32,c=bf16,o=f32"), ("router_bias",))
    compute = eqx.filter_jit(policy.cast_to_compute)(model)
    for master, view in zip(model, compute, strict=True):
        original = master.router_bias.array if named else master.router_bias
        preserved = view.router_bias.array if named else view.router_bias
        np.testing.assert_array_equal(original, preserved)
        assert preserved.dtype == jnp.float32
        assert view.weight.dtype == jnp.bfloat16
        assert np.any(np.asarray(original) != np.asarray(original.astype(jnp.bfloat16).astype(jnp.float32)))


def test_trainer_loss_preserves_router_bias_and_gradient():
    policy = jmp.get_policy("p=f32,c=bf16,o=f32")
    model = router()

    def loss(view):
        assert view.router_bias.dtype == jnp.float32
        assert view.weight.dtype == jnp.bfloat16
        return jnp.sum(view.router_bias) + view.weight.astype(jnp.float32), {}

    config = TrainerConfig(
        id="compute-view-test",
        mp=policy,
        fp32_fields=("router_bias",),
        tracker=(),
        require_accelerator=False,
        mesh=MeshConfig(),
    )
    trainer = Trainer(config, optax.sgd(0.01), loss, add_default_hooks=False)
    actual, _ = trainer.loss_fn(model)
    expected = jnp.sum(model.router_bias) + model.weight.astype(jnp.bfloat16).astype(jnp.float32)
    np.testing.assert_array_equal(actual, expected)
    gradient = eqx.filter_grad(lambda model: trainer.loss_fn(model)[0])(model)
    np.testing.assert_array_equal(gradient.router_bias, jnp.ones_like(model.router_bias))
    np.testing.assert_array_equal(model.router_bias, router().router_bias)


def test_check_fp32_fields_rejects_unknown_and_non_floating_fields():
    model = Router(jnp.zeros(2, dtype=jnp.float32), jnp.zeros((), dtype=jnp.int32))
    check_fp32_fields(model, ("router_bias",))
    with pytest.raises(ValueError, match="not found"):
        check_fp32_fields(model, ("router_bais",))
    with pytest.raises(ValueError, match="not a floating-point array"):
        check_fp32_fields(model, ("weight",))
