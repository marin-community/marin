# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from jax._src import config as jax_config
from jax.sharding import PartitionSpec as P
from jax.sharding import use_abstract_mesh

from experiments.grug.moe.optimizer import (
    GrugMoeAdamHConfig,
    GrugMoeKLSoapHConfig,
    GrugMoeMuonHConfig,
    GrugMoeNorMuonHConfig,
    scale_with_grug_klsoaph,
    scale_with_grug_muonh,
    scale_with_grug_normuonh,
)


class _reset_abstract_mesh:
    def __enter__(self):
        self._prev = jax_config.abstract_mesh_context_manager.swap_local(jax_config.config_ext.unset)
        return self

    def __exit__(self, exc_type, exc, tb):
        jax_config.abstract_mesh_context_manager.set_local(self._prev)
        return False


def test_grug_moe_muonh_keeps_adamh_baseline_adam_group_on_adam():
    params = {
        "token_embed": jnp.ones((128, 32), dtype=jnp.float32),
        "output_proj": jnp.ones((32, 128), dtype=jnp.float32),
        "lm_head": jnp.ones((32, 128), dtype=jnp.float32),
        "blocks": (
            {
                "attn": {
                    "w_q": jnp.ones((32, 32), dtype=jnp.float32),
                    "attn_gate": jnp.ones((32, 2), dtype=jnp.float32),
                },
                "mlp": {
                    "router": jnp.ones((32, 4), dtype=jnp.float32),
                    "router_bias": jnp.ones((4,), dtype=jnp.float32),
                    "w_gate_up": jnp.ones((4, 32, 64), dtype=jnp.float32),
                    "w_down": jnp.ones((4, 64, 32), dtype=jnp.float32),
                },
                "shared": {
                    "w_up": jnp.ones((32, 64), dtype=jnp.float32),
                },
                "rms_attn": {
                    "weight": jnp.ones((32,), dtype=jnp.float32),
                },
            },
        ),
    }

    mask = GrugMoeMuonHConfig().create_mask(params)
    baseline_mask = GrugMoeAdamHConfig().create_mask(params)

    assert baseline_mask["token_embed"] == "adam"
    assert mask["token_embed"] == "adam"
    assert mask["output_proj"] == "adamh"
    assert mask["lm_head"] == "adamh"
    block_mask = mask["blocks"][0]
    baseline_block_mask = baseline_mask["blocks"][0]
    assert block_mask["attn"]["w_q"] == "muonh"
    assert baseline_block_mask["attn"]["attn_gate"] == "adam"
    assert block_mask["attn"]["attn_gate"] == "adam"
    assert baseline_block_mask["mlp"]["router"] == "adam"
    assert block_mask["mlp"]["router"] == "adam"
    assert block_mask["mlp"]["w_gate_up"] == "muonh"
    assert block_mask["mlp"]["w_down"] == "muonh"
    assert block_mask["shared"]["w_up"] == "muonh"
    assert block_mask["mlp"]["router_bias"] == "adam"
    assert block_mask["rms_attn"]["weight"] == "adam"


def test_grug_moe_normuonh_keeps_adamh_baseline_adam_group_on_adam():
    params = {
        "token_embed": jnp.ones((128, 32), dtype=jnp.float32),
        "output_proj": jnp.ones((32, 128), dtype=jnp.float32),
        "blocks": (
            {
                "attn": {
                    "attn_gate": jnp.ones((32, 2), dtype=jnp.float32),
                },
                "mlp": {
                    "router": jnp.ones((32, 4), dtype=jnp.float32),
                    "w_gate_up": jnp.ones((4, 32, 64), dtype=jnp.float32),
                    "w_down": jnp.ones((4, 64, 32), dtype=jnp.float32),
                    "router_bias": jnp.ones((4,), dtype=jnp.float32),
                },
            },
        ),
    }

    mask = GrugMoeNorMuonHConfig().create_mask(params)
    baseline_mask = GrugMoeAdamHConfig().create_mask(params)

    assert baseline_mask["token_embed"] == "adam"
    assert mask["token_embed"] == "adam"
    assert mask["output_proj"] == "adamh"
    block_mask = mask["blocks"][0]
    baseline_block_mask = baseline_mask["blocks"][0]
    assert baseline_block_mask["attn"]["attn_gate"] == "adam"
    assert block_mask["attn"]["attn_gate"] == "adam"
    assert baseline_block_mask["mlp"]["router"] == "adam"
    assert block_mask["mlp"]["router"] == "adam"
    assert block_mask["mlp"]["w_gate_up"] == "normuonh"
    assert block_mask["mlp"]["w_down"] == "normuonh"
    assert block_mask["mlp"]["router_bias"] == "adam"


def test_grug_normuonh_tracks_output_axis_for_fan_in_fan_out_arrays():
    params = {
        "dense": jnp.ones((2, 3), dtype=jnp.float32),
        "experts": jnp.ones((5, 2, 3), dtype=jnp.float32),
    }
    updates = jax.tree.map(lambda x: jnp.arange(x.size, dtype=x.dtype).reshape(x.shape) + 1, params)
    transform = scale_with_grug_normuonh(
        momentum=0.0,
        nesterov=False,
        beta2=0.0,
        steps=0,
        learning_rate=0.01,
    )

    _, state = transform.update(updates, transform.init(params), params)

    assert state.row_nu["dense"].shape == (3,)
    assert state.row_nu["experts"].shape == (5, 3)


@pytest.mark.parametrize(
    "transform",
    [
        scale_with_grug_muonh(momentum=0.0, nesterov=False, steps=1, learning_rate=0.01),
        scale_with_grug_normuonh(momentum=0.0, nesterov=False, beta2=0.0, steps=1, learning_rate=0.01),
        scale_with_grug_klsoaph(precond_freq=1, learning_rate=0.01),
    ],
    ids=["muonh", "normuonh", "klsoaph"],
)
def test_grug_hyperball_update_handles_expert_parameter_sharding(transform):
    mesh = jax.sharding.AbstractMesh(
        axis_sizes=(2, 2, 1),
        axis_names=("data", "expert", "model"),
        axis_types=(
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
        ),
    )

    def update_expert_weight():
        param = jax.sharding.reshard(jnp.ones((4, 8, 16), dtype=jnp.float32), P("expert", "data", "model"))
        update = jax.sharding.reshard(jnp.ones_like(param), P("expert", "data", "model"))
        params = {"w_gate_up": param}
        updates = {"w_gate_up": update}
        out, _ = transform.update(updates, transform.init(params), params)
        return out["w_gate_up"]

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        out_shape = jax.eval_shape(update_expert_weight)

    assert out_shape.shape == (4, 8, 16)


def test_muonh_matrix_sweep_gate1_builds_readme_gate_steps():
    from experiments.grug.moe import muonh_matrix_sweep

    steps = muonh_matrix_sweep._build_steps("1")

    assert [step.name for step in steps] == [
        "grug/muonh_matrix_sweep/muonh-matrix-d512-2.19e17",
        "grug/muonh_matrix_sweep/muonh-matrix-d768-1.70e18",
    ]
    assert all(isinstance(step.config.optimizer.value, GrugMoeMuonHConfig) for step in steps)


def test_muonh_matrix_sweep_suffix_builds_distinct_relaunch_steps():
    from experiments.grug.moe import muonh_matrix_sweep

    steps = muonh_matrix_sweep._build_steps("1", run_suffix="baseline-adam-mask")

    assert [step.name for step in steps] == [
        "grug/muonh_matrix_sweep/muonh-matrix-baseline-adam-mask-d512-2.19e17",
        "grug/muonh_matrix_sweep/muonh-matrix-baseline-adam-mask-d768-1.70e18",
    ]


def test_normuonh_matrix_sweep_gate1_builds_readme_gate_steps():
    from experiments.grug.moe import normuonh_matrix_sweep

    steps = normuonh_matrix_sweep._build_steps("1")

    assert [step.name for step in steps] == [
        "grug/normuonh_matrix_sweep/normuonh-matrix-d512-2.19e17",
        "grug/normuonh_matrix_sweep/normuonh-matrix-d768-1.70e18",
    ]
    assert all(isinstance(step.config.optimizer.value, GrugMoeNorMuonHConfig) for step in steps)


def test_normuonh_matrix_sweep_suffix_builds_distinct_relaunch_steps():
    from experiments.grug.moe import normuonh_matrix_sweep

    steps = normuonh_matrix_sweep._build_steps("1", run_suffix="baseline-adam-mask")

    assert [step.name for step in steps] == [
        "grug/normuonh_matrix_sweep/normuonh-matrix-baseline-adam-mask-d512-2.19e17",
        "grug/normuonh_matrix_sweep/normuonh-matrix-baseline-adam-mask-d768-1.70e18",
    ]


def test_klsoaph_sweep_gate1_builds_two_steps():
    from experiments.grug.moe import klsoaph_sweep

    steps = klsoaph_sweep._build_steps("1")

    assert [step.name for step in steps] == [
        "grug/klsoaph_sweep/klsoaph-d512-2.19e17",
        "grug/klsoaph_sweep/klsoaph-d768-1.70e18",
    ]
    assert all(isinstance(step.config.optimizer.value, GrugMoeKLSoapHConfig) for step in steps)


def test_klsoaph_sweep_gate2_builds_two_steps():
    from experiments.grug.moe import klsoaph_sweep

    steps = klsoaph_sweep._build_steps("2")

    assert [step.name for step in steps] == [
        "grug/klsoaph_sweep/klsoaph-d1024-9.00e18",
        "grug/klsoaph_sweep/klsoaph-d1280-2.83e19",
    ]


def test_klsoaph_sweep_propagates_precond_freq_and_no_warmup():
    from experiments.grug.moe import klsoaph_sweep

    step = klsoaph_sweep._build_steps("1")[0]
    optimizer = step.config.optimizer.value

    assert isinstance(optimizer, GrugMoeKLSoapHConfig)
    assert optimizer.precond_freq == 5
    assert optimizer.warmup == 0.0


def test_grug_moe_muonh_optimizer_update_runs_on_single_device():
    params = {
        "token_embed": jnp.ones((8, 4), dtype=jnp.float32),
        "output_proj": jnp.ones((4, 8), dtype=jnp.float32),
        "block": {
            "router": jnp.ones((4, 2), dtype=jnp.float32),
            "w_gate_up": jnp.ones((2, 4, 6), dtype=jnp.float32),
            "norm": jnp.ones((4,), dtype=jnp.float32),
        },
    }
    updates = jax.tree.map(lambda x: jnp.full_like(x, 0.1), params)
    optimizer = GrugMoeMuonHConfig(learning_rate=0.01, adam_lr=0.001, warmup=0).build(10)

    out, _ = optimizer.update(updates, optimizer.init(params), params)

    assert jax.tree.map(lambda x: x.shape, out) == jax.tree.map(lambda x: x.shape, params)


def test_grug_moe_normuonh_optimizer_update_runs_on_single_device():
    params = {
        "token_embed": jnp.ones((8, 4), dtype=jnp.float32),
        "output_proj": jnp.ones((4, 8), dtype=jnp.float32),
        "block": {
            "router": jnp.ones((4, 2), dtype=jnp.float32),
            "w_gate_up": jnp.ones((2, 4, 6), dtype=jnp.float32),
            "norm": jnp.ones((4,), dtype=jnp.float32),
        },
    }
    updates = jax.tree.map(lambda x: jnp.full_like(x, 0.1), params)
    optimizer = GrugMoeNorMuonHConfig(learning_rate=0.01, adam_lr=0.001, warmup=0).build(10)

    out, _ = optimizer.update(updates, optimizer.init(params), params)

    assert jax.tree.map(lambda x: x.shape, out) == jax.tree.map(lambda x: x.shape, params)


def test_grug_moe_klsoaph_routes_attn_gate_to_matrix_group():
    """KL Soap H variant override: attn_gate goes to klsoaph, not adam."""
    params = {
        "token_embed": jnp.ones((128, 32), dtype=jnp.float32),
        "output_proj": jnp.ones((32, 128), dtype=jnp.float32),
        "lm_head": jnp.ones((32, 128), dtype=jnp.float32),
        "blocks": (
            {
                "attn": {
                    "w_q": jnp.ones((32, 32), dtype=jnp.float32),
                    "attn_gate": jnp.ones((32, 2), dtype=jnp.float32),
                },
                "mlp": {
                    "router": jnp.ones((32, 4), dtype=jnp.float32),
                    "router_bias": jnp.ones((4,), dtype=jnp.float32),
                    "w_gate_up": jnp.ones((4, 32, 64), dtype=jnp.float32),
                    "w_down": jnp.ones((4, 64, 32), dtype=jnp.float32),
                },
                "shared": {
                    "w_up": jnp.ones((32, 64), dtype=jnp.float32),
                },
                "rms_attn": {
                    "weight": jnp.ones((32,), dtype=jnp.float32),
                },
            },
        ),
    }

    mask = GrugMoeKLSoapHConfig().create_mask(params)
    baseline_mask = GrugMoeAdamHConfig().create_mask(params)
    muonh_mask = GrugMoeMuonHConfig().create_mask(params)

    assert mask["token_embed"] == "adam"
    assert mask["output_proj"] == "adamh"
    assert mask["lm_head"] == "adamh"
    block_mask = mask["blocks"][0]
    # The variant override: attn_gate routes to klsoaph (vs. adam for AdamH/MuonH).
    assert baseline_mask["blocks"][0]["attn"]["attn_gate"] == "adam"
    assert muonh_mask["blocks"][0]["attn"]["attn_gate"] == "adam"
    assert block_mask["attn"]["attn_gate"] == "klsoaph"
    # Other matrix params land in klsoaph.
    assert block_mask["attn"]["w_q"] == "klsoaph"
    assert block_mask["mlp"]["w_gate_up"] == "klsoaph"
    assert block_mask["mlp"]["w_down"] == "klsoaph"
    assert block_mask["shared"]["w_up"] == "klsoaph"
    # Embedding-table / 1-D leaves stay on adam.
    assert block_mask["mlp"]["router"] == "adam"
    assert block_mask["mlp"]["router_bias"] == "adam"
    assert block_mask["rms_attn"]["weight"] == "adam"


def test_grug_klsoaph_state_shapes_match_soap_geometry():
    """Per-leaf SOAP state: Q_L rows x rows, Q_R cols x cols, exp_avg rows x cols.

    Higher-rank (3-D) leaves carry an extra leading axis on every state
    tensor so the transform can vmap the 2-D update over experts.
    """
    params = {
        "dense": jnp.ones((4, 6), dtype=jnp.float32),
        "experts": jnp.ones((3, 4, 6), dtype=jnp.float32),
    }
    transform = scale_with_grug_klsoaph(precond_freq=1, learning_rate=0.01)

    state = transform.init(params)
    soap_state = state.inner_state if hasattr(state, "inner_state") else state

    assert soap_state.exp_avg["dense"].shape == (4, 6)
    assert soap_state.exp_avg_sq["dense"].shape == (4, 6)
    assert soap_state.gg_l["dense"].shape == (4, 4)
    assert soap_state.gg_r["dense"].shape == (6, 6)
    assert soap_state.q_l["dense"].shape == (4, 4)
    assert soap_state.q_r["dense"].shape == (6, 6)
    assert soap_state.esi_l["dense"].shape == (4,)
    assert soap_state.esi_r["dense"].shape == (6,)
    assert soap_state.exp_avg["experts"].shape == (3, 4, 6)
    assert soap_state.exp_avg_sq["experts"].shape == (3, 4, 6)
    assert soap_state.gg_l["experts"].shape == (3, 4, 4)
    assert soap_state.gg_r["experts"].shape == (3, 6, 6)
    assert soap_state.q_l["experts"].shape == (3, 4, 4)
    assert soap_state.q_r["experts"].shape == (3, 6, 6)
    assert soap_state.esi_l["experts"].shape == (3, 4)
    assert soap_state.esi_r["experts"].shape == (3, 6)
    # esi initialized at init_factor**-0.5 (default init_factor=0.1 → sqrt(10)).
    import math

    assert jnp.allclose(soap_state.esi_l["dense"], math.sqrt(10.0))
    assert jnp.allclose(soap_state.esi_r["dense"], math.sqrt(10.0))


def test_grug_moe_klsoaph_optimizer_update_runs_on_single_device():
    params = {
        "token_embed": jnp.ones((8, 4), dtype=jnp.float32),
        "output_proj": jnp.ones((4, 8), dtype=jnp.float32),
        "block": {
            "attn": {"attn_gate": jnp.ones((4, 2), dtype=jnp.float32)},
            "router": jnp.ones((4, 2), dtype=jnp.float32),
            "w_gate_up": jnp.ones((2, 4, 6), dtype=jnp.float32),
            "norm": jnp.ones((4,), dtype=jnp.float32),
        },
    }
    updates = jax.tree.map(lambda x: jnp.full_like(x, 0.1), params)
    optimizer = GrugMoeKLSoapHConfig(learning_rate=0.01, adam_lr=0.001, warmup=0).build(10)

    out, _ = optimizer.update(updates, optimizer.init(params), params)

    assert jax.tree.map(lambda x: x.shape, out) == jax.tree.map(lambda x: x.shape, params)
