import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest

from levanter.compute_dtype import cast_model_for_compute
from levanter.trainer import Trainer, TrainerConfig, WrappedLossFunction
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
def test_compute_override_preserves_original_bits_and_default_policy(named):
    model = tuple(router(named) for _ in range(26))
    policy = jmp.get_policy("p=f32,c=bf16,o=f32")
    default = cast_model_for_compute(model, policy)
    expected = policy.cast_to_compute(model)
    for actual, reference in zip(jax.tree.leaves(default), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(actual, reference)
        assert actual.dtype == reference.dtype
    compute = eqx.filter_jit(lambda model: cast_model_for_compute(model, policy, {"router_bias": "float32"}))(model)
    for master, view in zip(model, compute, strict=True):
        original = master.router_bias.array if named else master.router_bias
        preserved = view.router_bias.array if named else view.router_bias
        np.testing.assert_array_equal(original, preserved)
        assert preserved.dtype == jnp.float32
        assert view.weight.dtype == jnp.bfloat16
        assert np.any(np.asarray(original) != np.asarray(original.astype(jnp.bfloat16).astype(jnp.float32)))


def test_wrapped_loss_and_trainer_compute_view_share_override():
    policy = jmp.get_policy("p=f32,c=bf16,o=f32")
    model = router()

    def loss(view):
        assert view.router_bias.dtype == jnp.float32
        assert view.weight.dtype == jnp.bfloat16
        return jnp.sum(view.router_bias) + view.weight.astype(jnp.float32), {}

    config = TrainerConfig(
        id="compute-view-test",
        mp=policy,
        compute_dtype_overrides={"router_bias": "float32"},
        tracker=(),
        require_accelerator=False,
        mesh=MeshConfig(),
    )
    trainer = Trainer(config, optax.sgd(0.01), loss, add_default_hooks=False)
    wrapped = WrappedLossFunction(loss, policy, {}, config.compute_dtype_overrides)
    actual, _ = wrapped(model)
    expected, _ = loss(trainer.compute_model(model))
    np.testing.assert_array_equal(actual, expected)
    trainer_loss, _ = trainer.loss_fn(model)
    np.testing.assert_array_equal(trainer_loss, expected)
    gradient = eqx.filter_grad(lambda model: wrapped(model)[0])(model)
    np.testing.assert_array_equal(gradient.router_bias, jnp.ones_like(model.router_bias))
    np.testing.assert_array_equal(model.router_bias, router().router_bias)


@pytest.mark.parametrize("overrides", [{"missing": "float32"}, {"router_bias": "int32"}])
def test_bad_compute_override_fails(overrides):
    with pytest.raises(ValueError):
        cast_model_for_compute(router(), jmp.get_policy("p=f32,c=bf16,o=f32"), overrides)
