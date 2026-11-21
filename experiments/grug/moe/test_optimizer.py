# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec
from levanter.testing.devices import use_test_devices

from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHConfig, scale_with_grug_muonh
from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig as HeroMuonHConfig
from experiments.grug.moe_hero_ep.train import clip_by_total_norm

_NUM_EXPERTS, _INTERMEDIATE, _HIDDEN, _VOCAB = 2, 8, 16, 32
_UINT_DTYPE = {"bfloat16": jnp.uint16, "float32": jnp.uint32}
# The scale-invariant step re-projects each matrix onto its previous Frobenius norm,
# so stored-norm drift stays at BF16 storage-rounding scale: at most 6e-4 over 15
# steps on this tree with float32 norms. A parameter-space norm computed in BF16
# instead biases the re-projection at every step and drifts 1.2e-3 to 6.0e-3 over
# the same steps.
_NORM_DRIFT_TOLERANCE = 1e-3


def _moe_params(dtype):
    """A minimal tree with one leaf per optimizer group of both MoE configs."""

    def normal(shape, scale, key):
        return (jax.random.normal(key, shape) * scale).astype(dtype)

    k = jax.random.split(jax.random.PRNGKey(0), 12)
    return {
        "token_embed": normal((_VOCAB, _HIDDEN), 0.02, k[0]),
        "output_proj": normal((_VOCAB, _HIDDEN), 0.05, k[1]),
        "blocks": {
            "0": {
                "attn": {
                    "w_q": normal((_HIDDEN, _HIDDEN), 0.05, k[2]),
                    "attn_gate": normal((_HIDDEN,), 0.1, k[3]),
                },
                "mlp": {
                    "router": normal((_HIDDEN, _NUM_EXPERTS), 0.05, k[4]),
                    "router_bias": normal((_NUM_EXPERTS,), 0.1, k[5]),
                    "expert_mlp": {
                        "w_gate": normal((_NUM_EXPERTS, _INTERMEDIATE, _HIDDEN), 0.05, k[6]),
                        "w_down": normal((_NUM_EXPERTS, _HIDDEN, _INTERMEDIATE), 0.05, k[7]),
                    },
                    "shared": {"w_gate": normal((_INTERMEDIATE, _HIDDEN), 0.05, k[8])},
                },
                "gated_norm": {"weight": normal((_HIDDEN,), 0.5, k[9])},
            }
        },
    }


def _norm_test_params(dtype):
    """Larger matrices than _moe_params so norm drift reflects accumulated rounding."""

    def normal(shape, scale, key):
        return (jax.random.normal(key, shape) * scale).astype(dtype)

    k = jax.random.split(jax.random.PRNGKey(0), 6)
    return {
        "token_embed": normal((128, 64), 0.02, k[0]),
        "output_proj": normal((128, 64), 0.05, k[1]),
        "blocks": {
            "0": {
                "attn": {"w_q": normal((64, 64), 0.05, k[2])},
                "mlp": {
                    "router": normal((64, 4), 0.05, k[3]),
                    "expert_mlp": {"w_gate": normal((4, 64, 64), 0.05, k[4])},
                    "shared": {"w_gate": normal((64, 64), 0.05, k[5])},
                },
            }
        },
    }


def _run_steps(config, params, num_steps):
    mesh = Mesh(jax.devices(), axis_names=("i",))
    sharding = NamedSharding(mesh, PartitionSpec())
    params = jax.tree.map(lambda leaf: jax.device_put(leaf, sharding), params)
    initial = params

    optimizer = config.build(num_train_steps=num_steps)
    opt_state = optimizer.init(params)

    @jax.jit
    def step_fn(params, opt_state, grads):
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state

    key = jax.random.PRNGKey(1)
    for _ in range(num_steps):
        key, subkey = jax.random.split(key)
        grads = jax.tree.map(lambda p, step_key=subkey: jax.random.normal(step_key, p.shape).astype(p.dtype), params)
        params, opt_state = step_fn(params, opt_state, grads)
    return initial, params


def _bits_differ(reference, current):
    uint = _UINT_DTYPE[reference.dtype.name]
    # The runtime stores or exports parameters in the policy dtype; a zero-rate
    # update is exactly zero, so this cast never hides a change.
    reference_bits = jax.lax.bitcast_convert_type(reference, uint)
    current_bits = jax.lax.bitcast_convert_type(current.astype(reference.dtype), uint)
    return reference_bits != current_bits


def _norm_drift(reference, current):
    """Largest relative change of per-slice Frobenius norms between two leaves."""

    def per_slice_norms(leaf):
        x = leaf.astype(jnp.float32)
        if x.ndim == 2:
            return jnp.linalg.norm(x)[None]
        return jnp.linalg.norm(x, axis=tuple(range(1, x.ndim)))

    return float(jnp.max(jnp.abs(per_slice_norms(current) / per_slice_norms(reference) - 1.0)))


@pytest.mark.parametrize(
    ("config_factory", "dtype"),
    [
        (GrugMoeMuonHConfig, jnp.bfloat16),
        (GrugMoeMuonHConfig, jnp.float32),
        (GrugMoeAdamHConfig, jnp.bfloat16),
        (GrugMoeAdamHConfig, jnp.float32),
    ],
)
def test_grug_moe_optimizer_zero_lr_leaves_params_bit_identical(config_factory, dtype):
    config = config_factory(learning_rate=0.0, adam_lr=0.0, lr_schedule="constant")

    initial, final = _run_steps(config, _moe_params(dtype), num_steps=3)

    initial_leaves = jax.tree_util.tree_flatten_with_path(initial)[0]
    final_leaves = jax.tree_util.tree_flatten_with_path(final)[0]
    changed = [
        f"{path} ({reference.dtype.name})"
        for (path, reference), (_, current) in zip(initial_leaves, final_leaves, strict=True)
        if bool(_bits_differ(reference, current).any())
    ]
    assert changed == [], f"parameters changed at learning rate 0: {changed}"


@pytest.mark.parametrize("config_factory", [GrugMoeMuonHConfig, GrugMoeAdamHConfig])
def test_grug_moe_optimizer_positive_lr_updates_params(config_factory):
    config = config_factory(learning_rate=0.01, adam_lr=0.01, lr_schedule="constant")

    initial, final = _run_steps(config, _moe_params(jnp.float32), num_steps=2)

    initial_leaves = jax.tree_util.tree_flatten_with_path(initial)[0]
    final_leaves = jax.tree_util.tree_flatten_with_path(final)[0]
    # Matrix leaves cover every scale-invariant re-projection; one-dimensional
    # MuonH leaves only change sign, so they are excluded.
    unchanged = [
        f"{path}"
        for (path, reference), (_, current) in zip(initial_leaves, final_leaves, strict=True)
        if reference.ndim >= 2 and not _bits_differ(reference, current).any()
    ]
    assert unchanged == [], f"matrix leaves failed to update at a positive learning rate: {unchanged}"


@pytest.mark.parametrize("config_factory", [GrugMoeMuonHConfig, GrugMoeAdamHConfig])
def test_grug_moe_optimizer_positive_lr_preserves_bfloat16_matrix_norms(config_factory):
    config = config_factory(learning_rate=0.01, adam_lr=0.01, lr_schedule="constant")

    initial, final = _run_steps(config, _norm_test_params(jnp.bfloat16), num_steps=15)

    initial_leaves = jax.tree_util.tree_flatten_with_path(initial)[0]
    final_leaves = jax.tree_util.tree_flatten_with_path(final)[0]
    group_leaves = jax.tree_util.tree_flatten_with_path(config.create_mask(initial))[0]
    # Only the scale-invariant groups re-project onto the previous Frobenius norm.
    # Adam-group matrices may change norm freely.
    scale_invariant_groups = {"muonh", "adamh", "adamh_expert"}
    drifts = [
        (str(path), _norm_drift(reference, current))
        for (path, reference), (_, current), (_, group) in zip(initial_leaves, final_leaves, group_leaves, strict=True)
        if group in scale_invariant_groups
    ]
    drifted = [f"{name} {drift:.3e}" for name, drift in drifts if drift > _NORM_DRIFT_TOLERANCE]
    assert drifted == [], f"BF16 matrix norms drifted at a positive learning rate: {drifted}"


def test_grug_moe_adamh_mask_routes_expert_mlp_weights_to_expert_group():
    params = {
        "blocks": {
            "0": {
                "mlp": {
                    "router": jnp.ones((8, 4), dtype=jnp.float32),
                    "expert_mlp": {
                        "w_gate": jnp.ones((4, 8, 16), dtype=jnp.float32),
                        "w_up": jnp.ones((4, 8, 16), dtype=jnp.float32),
                        "w_down": jnp.ones((4, 16, 8), dtype=jnp.float32),
                    },
                },
                "shared": {
                    "w_gate": jnp.ones((8, 16), dtype=jnp.float32),
                },
            },
        },
        "token_embed": jnp.ones((128, 8), dtype=jnp.float32),
    }

    mask = GrugMoeAdamHConfig().create_mask(params)

    block_mask = mask["blocks"]["0"]
    assert block_mask["mlp"]["router"] == "adam"
    assert block_mask["mlp"]["expert_mlp"]["w_gate"] == "adamh_expert"
    assert block_mask["mlp"]["expert_mlp"]["w_up"] == "adamh_expert"
    assert block_mask["mlp"]["expert_mlp"]["w_down"] == "adamh_expert"
    assert block_mask["shared"]["w_gate"] == "adamh_expert"
    assert mask["token_embed"] == "adam"


@pytest.mark.parametrize("stacked", [False, True])
def test_muonh_groups_match_hero_for_stacked_and_unstacked_models(stacked):
    layer = (2,) if stacked else ()
    params = {
        "blocks": {
            "rms_attn": {"weight": jnp.ones((*layer, 4))},
            "rms_mlp": {"weight": jnp.ones((*layer, 4))},
            "mlp": {
                "expert_mlp": {name: jnp.ones((*layer, 2, 4, 3)) for name in ("w_gate", "w_up", "w_down")},
                "router": jnp.ones((*layer, 4, 2)),
            },
            "attn": {"w_q": jnp.ones((*layer, 4, 4)), "attn_gate": jnp.ones((*layer, 4))},
            "attn_gated_norm": {"w_up": jnp.ones((*layer, 4, 2))},
        },
        "token_embed": jnp.ones((8, 4)),
        "output_proj": jnp.ones((4, 8)),
    }
    actual = GrugMoeMuonHConfig().create_mask(params)
    assert actual == HeroMuonHConfig().create_mask(params)
    assert actual["blocks"]["rms_attn"]["weight"] == "adam"
    assert actual["blocks"]["rms_mlp"]["weight"] == "adam"
    assert set(actual["blocks"]["mlp"]["expert_mlp"].values()) == {"muonh"}
    assert actual["blocks"]["attn_gated_norm"]["w_up"] == "muonh"
    assert actual["blocks"]["mlp"]["router"] == "adam"
    assert actual["blocks"]["attn"]["w_q"] == "muonh"
    assert actual["blocks"]["attn"]["attn_gate"] == "adam"
    assert actual["token_embed"] == "adam"
    assert actual["output_proj"] == "adamh"


@pytest.mark.parametrize("clip", [None, 0.1])
def test_muonh_update_matches_hero_with_optional_global_clip(clip):
    params = {
        "matrix": jnp.arange(1, 13, dtype=jnp.float32).reshape(3, 4),
        "stacked_matrix": jnp.arange(1, 25, dtype=jnp.float32).reshape(2, 3, 4),
        "mlp": {"expert_mlp": {"w_gate": jnp.arange(1, 49, dtype=jnp.float32).reshape(2, 2, 3, 4)}},
        "rms_attn": {"weight": jnp.ones((2, 4))},
    }
    settings = dict(learning_rate=0.02, warmup=0, max_grad_norm=None, use_syrk=False, expert_normalization="all_experts")
    actual = GrugMoeMuonHConfig(**settings, global_grad_norm=clip).build(100)
    reference = HeroMuonHConfig(**settings).build(100)
    actual_state = actual.init(params)
    expected_state = reference.init(params)
    actual_params = params
    expected_params = params
    for step in range(2):
        scale = step + 1
        grads = jax.tree.map(lambda x, s=scale: jnp.sin(x * s) * 0.2, actual_params)
        reference_grads = clip_by_total_norm(grads, clip).grads if clip is not None else grads
        actual_updates, actual_state = actual.update(grads, actual_state, actual_params)
        expected_updates, expected_state = reference.update(reference_grads, expected_state, expected_params)
        for actual_leaf, expected_leaf in zip(
            jax.tree.leaves(actual_updates), jax.tree.leaves(expected_updates), strict=True
        ):
            np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=0, atol=3e-6)
        if clip is None:
            np.testing.assert_array_equal(
                actual_updates["mlp"]["expert_mlp"]["w_gate"], expected_updates["mlp"]["expert_mlp"]["w_gate"]
            )
        actual_params = jax.tree.map(lambda p, u: p + u, actual_params, actual_updates)
        expected_params = jax.tree.map(lambda p, u: p + u, expected_params, expected_updates)


@pytest.mark.parametrize("stacked", [False, True])
def test_muonh_state_matches_hero_layout(stacked):
    shape = (2, 2, 3, 4) if stacked else (2, 3, 4)
    params = {"mlp": {"expert_mlp": {"w_gate": jnp.ones(shape)}}, "rms_attn": {"weight": jnp.ones(4)}}
    actual = GrugMoeMuonHConfig(use_syrk=False).build(100).init(params)
    expected = HeroMuonHConfig(use_syrk=False).build(100).init(params)
    actual_leaves = jax.tree_util.tree_flatten_with_path(actual)[0]
    expected_leaves = jax.tree_util.tree_flatten_with_path(expected)[0]
    assert [(p, x.shape, x.dtype) for p, x in actual_leaves] == [(p, x.shape, x.dtype) for p, x in expected_leaves]


@pytest.mark.multidevice
@pytest.mark.parametrize("clip", [None, 0.5])
def test_muonh_sharded_expert_update_matches_hero(clip):
    with use_test_devices(2) as devices:
        mesh = Mesh(np.asarray(devices), ("data",), axis_types=(AxisType.Explicit,))
        with jax.set_mesh(mesh):
            expert_sharding = NamedSharding(mesh, PartitionSpec(None, "data", None, None))
            matrix_sharding = NamedSharding(mesh, PartitionSpec("data", None))
            params = {
                "matrix": jax.device_put(jnp.arange(1, 17, dtype=jnp.float32).reshape(4, 4), matrix_sharding),
                "mlp": {
                    "expert_mlp": {
                        "w_gate": jax.device_put(
                            jnp.arange(1, 49, dtype=jnp.float32).reshape(2, 2, 3, 4), expert_sharding
                        )
                    }
                },
            }
            grads = jax.tree.map(lambda x: jax.sharding.reshard(jnp.sin(x), x.sharding), params)
            settings = dict(learning_rate=0.02, warmup=0, use_syrk=False, expert_normalization="all_experts")
            actual = GrugMoeMuonHConfig(**settings, global_grad_norm=clip).build(100)
            reference = HeroMuonHConfig(**settings).build(100)
            reference_grads = clip_by_total_norm(grads, clip).grads if clip is not None else grads
            actual_updates, _ = jax.jit(actual.update)(grads, actual.init(params), params)
            expected_updates, _ = jax.jit(reference.update)(reference_grads, reference.init(params), params)
            for actual_leaf, expected_leaf in zip(
                jax.tree.leaves(actual_updates), jax.tree.leaves(expected_updates), strict=True
            ):
                np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=0, atol=3e-6)


@pytest.mark.multidevice
@pytest.mark.parametrize(
    ("mesh_shape", "layers"),
    [
        ((1, 2, 2, 1, 1), 4),  # expert=1, context=2, data=2 (divisible layers)
        ((1, 2, 2, 1, 1), 3),  # expert=1, context=2, data=2 (padded layers)
        ((1, 2, 1, 2, 1), 3),  # expert=2, context=1, data=2 (hero layout, padded layers)
    ],
)
def test_muonh_4d_expert_layer_split_matches_unsharded_reference(mesh_shape, layers):
    rng = np.random.default_rng(0)
    w_gate_np = rng.normal(0.0, 0.05, size=(layers, 4, 8, 6)).astype(np.float32)
    w_down_np = rng.normal(0.0, 0.05, size=(layers, 4, 6, 8)).astype(np.float32)
    g_gate_np = rng.normal(0.0, 0.05, size=(layers, 4, 8, 6)).astype(np.float32)
    g_down_np = rng.normal(0.0, 0.05, size=(layers, 4, 6, 8)).astype(np.float32)

    cfg = GrugMoeMuonHConfig(
        learning_rate=0.02,
        adam_lr=0.02,
        warmup=0,
        use_syrk=False,
        expert_normalization="all_experts",
        global_grad_norm=1.0,
    )
    opt = cfg.build(100)

    ref_params = {
        "blocks": {"mlp": {"expert_mlp": {"w_gate": jnp.asarray(w_gate_np), "w_down": jnp.asarray(w_down_np)}}}
    }
    ref_grads = {"blocks": {"mlp": {"expert_mlp": {"w_gate": jnp.asarray(g_gate_np), "w_down": jnp.asarray(g_down_np)}}}}
    ref_updates, _ = opt.update(ref_grads, opt.init(ref_params), ref_params)
    ref_updates_np = jax.tree.map(np.asarray, ref_updates)

    with use_test_devices(4) as devices:
        mesh = Mesh(
            np.asarray(devices).reshape(mesh_shape),
            ("replica_dcn", "data", "context", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 5,
        )
        gate_sharding = NamedSharding(mesh, PartitionSpec(None, ("expert", "context"), "data", "model"))
        down_sharding = NamedSharding(mesh, PartitionSpec(None, ("expert", "context"), "model", "data"))
        with jax.set_mesh(mesh):
            params = {
                "blocks": {
                    "mlp": {
                        "expert_mlp": {
                            "w_gate": jax.device_put(jnp.asarray(w_gate_np), gate_sharding),
                            "w_down": jax.device_put(jnp.asarray(w_down_np), down_sharding),
                        }
                    }
                }
            }
            grads = {
                "blocks": {
                    "mlp": {
                        "expert_mlp": {
                            "w_gate": jax.device_put(jnp.asarray(g_gate_np), gate_sharding),
                            "w_down": jax.device_put(jnp.asarray(g_down_np), down_sharding),
                        }
                    }
                }
            }
            actual_updates, _ = jax.jit(opt.update)(grads, opt.init(params), params)
            for actual_leaf, param_leaf, expected_leaf in zip(
                jax.tree.leaves(actual_updates),
                jax.tree.leaves(params),
                jax.tree.leaves(ref_updates_np),
                strict=True,
            ):
                assert actual_leaf.sharding == param_leaf.sharding
                np.testing.assert_allclose(np.asarray(actual_leaf), expected_leaf, rtol=1e-2, atol=1e-4)


def test_scale_with_grug_muonh_rejects_4d_updates():
    # _grug_scale_with_muon skips Newton-Schulz on 4-D leaves; direct callers must fail fast.
    params = {"mlp": {"expert_mlp": {"w_gate": jnp.ones((2, 2, 3, 4), dtype=jnp.float32)}}}
    grads = jax.tree.map(jnp.ones_like, params)
    transform = scale_with_grug_muonh()
    with pytest.raises(ValueError, match="4-D"):
        transform.update(grads, transform.init(params), params)


@pytest.mark.parametrize("expert_normalization", ["per_expert", "all_experts"])
def test_muonh_stacked_4d_bfloat16_zero_lr_exact_and_positive_lr_preserves_norm(expert_normalization):
    w_gate = (jax.random.normal(jax.random.PRNGKey(0), (2, 4, 64, 64)) * 0.05).astype(jnp.bfloat16)
    params = {"blocks": {"mlp": {"expert_mlp": {"w_gate": w_gate}}}}

    zero_cfg = GrugMoeMuonHConfig(
        learning_rate=0.0,
        adam_lr=0.0,
        lr_schedule="constant",
        use_syrk=False,
        expert_normalization=expert_normalization,
    )
    initial, final = _run_steps(zero_cfg, params, num_steps=3)
    assert not bool(
        _bits_differ(
            initial["blocks"]["mlp"]["expert_mlp"]["w_gate"],
            final["blocks"]["mlp"]["expert_mlp"]["w_gate"],
        ).any()
    )

    pos_cfg = GrugMoeMuonHConfig(
        learning_rate=0.01,
        adam_lr=0.01,
        lr_schedule="constant",
        use_syrk=False,
        expert_normalization=expert_normalization,
    )
    initial_pos, final_pos = _run_steps(pos_cfg, params, num_steps=15)
    axes = (1, 2, 3) if expert_normalization == "all_experts" else (2, 3)
    ref_norm = jnp.sqrt(
        jnp.sum(jnp.square(initial_pos["blocks"]["mlp"]["expert_mlp"]["w_gate"].astype(jnp.float32)), axis=axes)
    )
    cur_norm = jnp.sqrt(
        jnp.sum(jnp.square(final_pos["blocks"]["mlp"]["expert_mlp"]["w_gate"].astype(jnp.float32)), axis=axes)
    )
    drift = float(jnp.max(jnp.abs(cur_norm / ref_norm - 1.0)))
    assert drift <= _NORM_DRIFT_TOLERANCE, f"4-D BF16 expert norm drifted ({expert_normalization}): {drift:.3e}"


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)])
def test_muonh_unstacked_update_matches_existing_backend(shape):
    size = int(np.prod(shape))
    param = jnp.arange(1, size + 1, dtype=jnp.float32).reshape(shape)
    grad = jnp.linspace(-0.5, 0.7, size).reshape(shape)
    params = {"mlp": {"expert_mlp": {"w_gate": param}}}
    grads = {"mlp": {"expert_mlp": {"w_gate": grad}}}
    config = GrugMoeMuonHConfig(learning_rate=0.02, warmup=0, max_grad_norm=None, use_syrk=False)
    optimizer = config.build(100)
    update, _ = optimizer.update(grads, optimizer.init(params), params)
    backend = scale_with_grug_muonh(learning_rate=config.lr_scheduler(100)(0))
    expected, _ = backend.update(grads, backend.init(params), params)
    np.testing.assert_array_equal(update["mlp"]["expert_mlp"]["w_gate"], expected["mlp"]["expert_mlp"]["w_gate"])
