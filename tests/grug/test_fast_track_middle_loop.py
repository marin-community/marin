# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``middle_loop``: every middle pass restarts from the prelude (with a zero controller, the cross design's
output does not depend on the pass count; the window design's pool reads the previous pass), the router state keeps
one row per layer, and the controller's zero-initialised W_f / W_g train on Adam."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from experiments.belay.model_probe import SEQ, tiny_config, tiny_mesh
from experiments.grug.fast_track.model import Transformer, middle_loop_schedule
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig
from experiments.grug.fast_track.train import _apply_qb_betas


def _forward(middle_loop: str, w_g_scale: float = 0.0) -> np.ndarray:
    config = dataclasses.replace(tiny_config(), middle_loop=middle_loop)
    with jax.set_mesh(tiny_mesh()):
        model = Transformer.init(config, key=jax.random.key(0))
        if w_g_scale:
            w_g = w_g_scale * jax.random.normal(jax.random.key(4), model.middle_loop_w_g.shape)
            model = eqx.tree_at(lambda m: m.middle_loop_w_g, model, w_g)
        tokens = jax.random.randint(jax.random.key(1), (2, SEQ), 0, config.vocab_size)
        return np.asarray(eqx.filter_jit(lambda m: m(tokens)[0])(model), np.float32)


def test_passes_restart_from_the_prelude_and_read_the_previous_pass_only_through_attn_res():
    cross_two, cross_three = _forward("cross_x2"), _forward("cross_x3")
    # Zero W_f / W_g: the cross mix is gated off, and each pass repeats the first one exactly.
    np.testing.assert_allclose(cross_three, cross_two, atol=1e-6)
    gated_two, gated_three = _forward("cross_x2", 0.5), _forward("cross_x3", 0.5)
    assert np.abs(gated_two - cross_two).max() > 1e-3
    assert np.abs(gated_three - gated_two).max() > 1e-3
    # The window design's middle pool holds the previous pass's blocks from the start.
    window_two, window_three = _forward("window_x2"), _forward("window_x3")
    assert np.abs(window_three - window_two).max() > 1e-3
    assert np.abs(window_two - cross_two).max() > 1e-3


def test_router_state_keeps_one_row_per_layer_for_the_trainer():
    config = dataclasses.replace(tiny_config(), middle_loop="window_x3")
    assert middle_loop_schedule(config) == (0, 1, 2, 1, 2, 1, 2, 3)
    with jax.set_mesh(tiny_mesh()):
        model = Transformer.init(config, key=jax.random.key(0))
        tokens = jax.random.randint(jax.random.key(1), (2, SEQ), 0, config.vocab_size)
        weight = jnp.ones(tokens.shape, jnp.float32)

        def train_loss(m):
            return m.next_token_loss(tokens, weight, return_router_metrics=True, train_terms=True)

        loss, metrics = eqx.filter_jit(train_loss)(model)
        betas = metrics["qb_beta_per_layer"]
        biased = _apply_qb_betas(model, betas)

    assert np.isfinite(float(loss))
    assert betas.shape == (config.num_layers, config.num_experts)
    assert np.all(np.isfinite(np.asarray(betas)))
    assert biased.stacked_blocks.stacked.mlp.router_bias.shape == model.stacked_blocks.stacked.mlp.router_bias.shape


def test_zero_initialised_controller_trains_on_adam():
    config = dataclasses.replace(tiny_config(), middle_loop="cross_x2")
    with jax.set_mesh(tiny_mesh()):
        model = Transformer.init(config, key=jax.random.key(0))
        tokens = jax.random.randint(jax.random.key(1), (2, SEQ), 0, config.vocab_size)
        weight = jnp.ones(tokens.shape, jnp.float32)
        grads = eqx.filter_jit(eqx.filter_grad(lambda m: m.next_token_loss(tokens, weight)))(model)

    params = eqx.filter(model, eqx.is_array)
    optimizer_config = GrugMoeMuonHConfig(learning_rate=0.02, warmup=0.0, lr_schedule="constant")
    mask = optimizer_config.create_mask(params)
    assert (mask.middle_loop_w_f, mask.middle_loop_w_g) == ("adam", "adam")
    assert mask.middle_loop_attn_res_query_cross == "attn_res_query"
    assert float(jnp.linalg.norm(grads.middle_loop_w_g)) > 0
    assert float(jnp.linalg.norm(grads.middle_loop_w_f)) > 0

    controller = {"middle_loop_w_f": params.middle_loop_w_f, "middle_loop_w_g": params.middle_loop_w_g}
    controller_grads = {"middle_loop_w_f": grads.middle_loop_w_f, "middle_loop_w_g": grads.middle_loop_w_g}
    optimizer = optimizer_config.build(num_train_steps=100)
    updates, _ = optimizer.update(controller_grads, optimizer.init(controller), controller)
    assert float(jnp.linalg.norm(updates["middle_loop_w_f"])) > 0
    assert float(jnp.linalg.norm(updates["middle_loop_w_g"])) > 0
