# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
from jax.sharding import AxisType, Mesh, reshard
from levanter.data.text.datasets import LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.testing.devices import use_test_devices

from experiments.grug.moe_hero_ep import model
from experiments.grug.moe_hero_ep.expert_placement import (
    ExpertPlacement,
    balanced_expert_order,
    projected_drop_fraction,
    source_order_loads,
)
from experiments.grug.moe_hero_ep.launch_hero_sft import (
    SFT_COMPONENT_PREFIX,
    SFT_POOLED_COMPONENT,
    SftHandoff,
    SftMixturePosition,
    SftStore,
    one_epoch_weight_stages,
    plan_sft_mixture,
    sft_components,
    sft_handoff,
    sft_steps,
)
from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe_hero_ep.train import (
    RouterBiasMode,
    SpecialTokenInit,
    _make_train_step,
    _place_restored_experts,
    initial_state,
    permute_experts,
    reseed_special_token_columns,
    scale_special_token_updates,
)

_FP32 = jmp.get_policy("params=float32,compute=float32,output=float32")


def _store(packed_sequences: int) -> SftStore:
    return SftStore(cache_path="s3://bucket/store", tokenizer="tok", max_length=8, packed_sequences=packed_sequences)


def _tiny_config():
    return model.GrugModelConfig(
        vocab_size=64,
        hidden_dim=32,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        local_kv_heads=2,
        global_kv_heads=2,
        head_dim=8,
        max_seq_len=8,
        sliding_window=4,
        global_every=2,
        capacity_factor=4.0,
        initializer_std=0.5 / math.sqrt(32),
        qk_mult=1.3,
        attention_implementation="reference",
        moe_implementation="fixed_all_to_all",
    )


def _one_device_mesh():
    return Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1, 1),
        ("replica_dcn", "data", "expert", "context", "model"),
        axis_types=(AxisType.Explicit,) * 5,
    )


def _optimizer():
    return GrugMoeMuonHConfig(learning_rate=1e-2, adam_lr=1e-2, warmup=0, use_syrk=False).build(100)


def test_one_epoch_stages_read_every_sft_store_exactly_once():
    stores = {"a": _store(600_000), "b": _store(200_000), "c": _store(50)}
    plan = plan_sft_mixture(stores, batch_size=16, replay_fraction=0.2)
    assert plan.num_blocks > 1
    _, sft_counts = sft_components(stores, plan)
    stages = one_epoch_weight_stages(sft_counts, {"replay/x": 3.0, "replay/y": 1.0}, plan)

    assert len(stages) == plan.num_blocks
    assert [start for start, _ in stages] == [block * plan.block_size for block in range(plan.num_blocks)]
    draws: dict[str, int] = {}
    for _, weights in stages:
        counts = {name: round(weight * plan.block_size) for name, weight in weights.items()}
        assert sum(counts.values()) == plan.block_size
        assert counts["replay/x"] >= 1 and counts["replay/y"] >= 1
        for name, count in counts.items():
            draws[name] = draws.get(name, 0) + count
    assert {name: draws[name] for name in sft_counts} == sft_counts
    assert draws["replay/x"] + draws["replay/y"] == plan.run_sequences - plan.sft_sequences


def test_a_handoff_reads_the_rest_of_the_epoch_once_at_the_new_batch():
    plan = plan_sft_mixture({"a": _store(900_000), "b": _store(68_596)}, batch_size=176, replay_fraction=0.2035)
    handoff = SftHandoff("s3://bucket/step-394552", 394_552, plan_dp_racks=11, sequence=(394_552 - 393_491) * 176)

    steps = sft_steps(plan, start_step=393_491, batch_size=144, handoff=handoff)

    read = steps.first_sequence + (steps.num_train_steps - steps.first_step) * 144
    assert plan.run_sequences - 144 < read <= plan.run_sequences
    assert abs(steps.decay_steps * 144 - plan.decay_steps * 176) <= 72


def _write_sft_checkpoint(path, step: int, position: SftMixturePosition | None = None) -> str:
    path.mkdir(parents=True)
    recorded = {} if position is None else position.checkpoint_metadata()
    metadata = {"step": step, "timestamp": "2026-11-11T00:00:00", "is_temporary": False, **recorded}
    (path / "metadata.json").write_text(json.dumps(metadata))
    return str(path)


def test_a_second_handoff_resumes_where_the_first_one_left_off(tmp_path):
    # An 11-rack run written before positions were recorded, handed to 9 racks, then to 8.
    first = sft_handoff(_write_sft_checkpoint(tmp_path / "step-394552", 394_552), start_step=393_491, parent_dp_racks=11)
    nine_racks = SftMixturePosition(plan_dp_racks=11, step=first.step, sequence=first.sequence, batch_size=144)
    second_checkpoint = _write_sft_checkpoint(tmp_path / "nine" / "step-395000", 395_000, nine_racks)

    second = sft_handoff(second_checkpoint, start_step=393_491, parent_dp_racks=None)

    assert first.sequence == (394_552 - 393_491) * 176
    assert second == SftHandoff(second_checkpoint, 395_000, plan_dp_racks=11, sequence=first.sequence + 448 * 144)


def test_stores_smaller_than_the_block_count_share_one_component():
    stores = {"big": _store(1_000_000), "tiny1": _store(9), "tiny2": _store(15)}
    plan = plan_sft_mixture(stores, batch_size=16, replay_fraction=0.2)
    assert 15 < plan.num_blocks <= 24

    components, counts = sft_components(stores, plan)

    assert set(components) == {SFT_COMPONENT_PREFIX + "big", SFT_POOLED_COMPONENT}
    assert counts[SFT_POOLED_COMPONENT] == 24


def test_sft_stores_offer_no_validation_split():
    # Evaluation asks every component for a validation split; the SFT stores hold only train.
    stores = {"a": _store(600_000), "tiny": _store(14)}
    plan = plan_sft_mixture(stores, batch_size=16, replay_fraction=0.2)
    components, _ = sft_components(stores, plan)
    config = LmDataConfig(tokenizer="gpt2", cache_dir=None, components=components)

    assert config.build_caches("validation") == {}


def test_special_token_updates_scale_only_the_listed_rows():
    with jax.set_mesh(_one_device_mesh()):
        updates = jax.tree.map(jnp.ones_like, model.Transformer.init(_tiny_config(), key=jax.random.key(0)))
        scaled = scale_special_token_updates(updates, (3, 9), 4.0)

    embed_scale = np.asarray(scaled.token_embed)[:, 0]
    head_scale = np.asarray(scaled.output_proj)[0, :]
    expected = np.ones(64)
    expected[[3, 9]] = 4.0
    np.testing.assert_array_equal(embed_scale, expected)
    np.testing.assert_array_equal(head_scale, expected)
    np.testing.assert_array_equal(np.asarray(scaled.stacked_blocks.stacked.mlp.router_bias), 1.0)


def test_reseed_copies_anchor_columns_and_zeros_only_their_moments():
    special = SpecialTokenInit(token_ids=(5, 7), anchor_ids=(1, 2), start_step=0, lr_multiplier=2.0)
    with jax.set_mesh(_one_device_mesh()):
        state = initial_state(_tiny_config(), optimizer=_optimizer(), mp=_FP32, key=jax.random.key(0), ema_beta=None)
        # A resumed optimizer has nonzero moments everywhere.
        trained_moments = jax.tree.map(
            lambda x: jnp.ones_like(x) if jnp.issubdtype(x.dtype, jnp.floating) else x, state.opt_state
        )
        state = state.__class__(**{**vars(state), "opt_state": trained_moments})
        before = np.asarray(state.params.output_proj)
        embed_before = np.asarray(state.params.token_embed)
        reseeded = reseed_special_token_columns(state, special, offload_opt_state=False)

    head = np.asarray(reseeded.params.output_proj)
    np.testing.assert_array_equal(head[:, [5, 7]], before[:, [1, 2]])
    others = [i for i in range(64) if i not in (5, 7)]
    np.testing.assert_array_equal(head[:, others], before[:, others])
    np.testing.assert_array_equal(np.asarray(reseeded.params.token_embed), embed_before)

    head_moments = 0
    for path, leaf in jax.tree_util.tree_leaves_with_path(reseeded.opt_state):
        if not jnp.issubdtype(leaf.dtype, jnp.floating):
            continue
        values = np.asarray(leaf)
        if str(path[-1]) == ".output_proj":
            head_moments += 1
            np.testing.assert_array_equal(values[:, [5, 7]], 0.0)
            np.testing.assert_array_equal(values[:, others], 1.0)
        else:
            np.testing.assert_array_equal(values, 1.0)
    assert head_moments >= 2  # AdamH keeps a first and a second moment for the LM head


@pytest.mark.parametrize("mode", [RouterBiasMode.FROZEN, RouterBiasMode.UPDATE])
def test_frozen_router_bias_keeps_the_restored_betas(mode):
    config = _tiny_config()
    tokens = jnp.asarray(np.random.default_rng(0).integers(0, config.vocab_size, size=(2, 8), dtype=np.int32))
    batch = GrugLmExample(tokens=tokens, loss_weight=jnp.ones((2, 8), jnp.float32).at[:, -1].set(0.0))
    # Adam keeps momentum on the router bias, which a frozen bias must not follow.
    optimizer = optax.adam(1e-2)
    with jax.set_mesh(_one_device_mesh()):
        state = initial_state(config, optimizer=optimizer, mp=_FP32, key=jax.random.key(0), ema_beta=None)
        restored_betas = jax.random.normal(jax.random.key(1), state.pending_qb_betas.shape)
        state = state.__class__(**{**vars(state), "pending_qb_betas": restored_betas})
        # The step donates its state, so keep host copies of what it restores.
        restored_betas = np.asarray(restored_betas)
        expected_bias = np.asarray(
            model.apply_qb_betas(state.params, restored_betas).stacked_blocks.stacked.mlp.router_bias
        )
        step = _make_train_step(optimizer, _FP32, z_loss_weight=1e-4, ema_beta=None, router_bias_mode=mode)
        for _ in range(2):
            state, _, _ = step(state, batch)

    betas = np.asarray(state.pending_qb_betas)
    bias = np.asarray(state.params.stacked_blocks.stacked.mlp.router_bias)
    if mode == RouterBiasMode.FROZEN:
        np.testing.assert_array_equal(betas, restored_betas)
        np.testing.assert_allclose(bias, expected_bias, rtol=0, atol=0)
    else:
        assert not np.array_equal(betas, restored_betas)


def _placement(order) -> ExpertPlacement:
    return ExpertPlacement(
        source_checkpoint="s3://bucket/step-0",
        expert_axis_size=1,
        block_experts=len(order[0]),
        order=tuple(tuple(layer) for layer in order),
        loads_from="test",
    )


def test_balanced_order_spreads_hot_experts_so_capacity_stops_dropping():
    # Two hot experts share the first block in expert-id order and overflow it.
    loads = np.array([[10.0, 9.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5]])
    identity = np.arange(8)[None, :]

    order = balanced_expert_order(loads, block_experts=4)

    assert sorted(order[0]) == list(range(8))
    assert projected_drop_fraction(loads, identity, block_experts=4, capacity_factor=1.5) > 0.1
    assert projected_drop_fraction(loads, order, block_experts=4, capacity_factor=1.5) == 0.0


def test_loads_logged_by_a_placed_run_map_back_to_source_order():
    placement = _placement([[2, 0, 3, 1], [1, 3, 0, 2]])
    source_loads = np.array([[4.0, 3.0, 2.0, 1.0], [8.0, 6.0, 7.0, 5.0]])
    # A placed run logs slot i for the source expert order[layer][i].
    logged = np.take_along_axis(source_loads, np.asarray(placement.order), axis=1)

    np.testing.assert_array_equal(
        source_order_loads(logged, placement, source_checkpoint=placement.source_checkpoint), source_loads
    )
    with pytest.raises(ValueError, match="placed relative to"):
        source_order_loads(logged, placement, source_checkpoint="s3://bucket/other")


def _trained_state(config, optimizer, mesh):
    """A restored-looking state: nonzero optimizer moments and QB betas, as an SFT start checkpoint has."""
    with jax.set_mesh(mesh):
        state = initial_state(config, optimizer=optimizer, mp=_FP32, key=jax.random.key(0), ema_beta=None)
        keys = iter(jax.random.split(jax.random.key(1), len(jax.tree.leaves(state.opt_state)) + 1))

        def noise_like(x):
            # Positive, so Adam's second moment stays valid.
            noise = jax.random.uniform(next(keys), x.shape, x.dtype, minval=0.5, maxval=1.5)
            return reshard(noise, jax.typeof(x).sharding.spec)

        moments = jax.tree.map(lambda x: noise_like(x) if jnp.issubdtype(x.dtype, jnp.floating) else x, state.opt_state)
        betas = noise_like(state.pending_qb_betas)
        return state.__class__(**{**vars(state), "opt_state": moments, "pending_qb_betas": betas})


def _assert_trees_close(a, b):
    for (path, x), y in zip(jax.tree_util.tree_leaves_with_path(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_allclose(
            np.asarray(x), np.asarray(y), rtol=1e-5, atol=1e-6, err_msg=jax.tree_util.keystr(path)
        )


def _expert_sharded_mesh(devices):
    return Mesh(
        np.asarray(devices).reshape(1, 2, 2, 1, 1),
        ("replica_dcn", "data", "expert", "context", "model"),
        axis_types=(AxisType.Explicit,) * 5,
    )


@pytest.mark.multidevice
def test_expert_placement_commutes_with_a_training_step():
    # Placing then stepping equals stepping then placing only if every expert-indexed parameter and
    # optimizer moment moved together; a missed leaf pairs an expert with another expert's state.
    config = _tiny_config()
    placement = _placement([[2, 0, 3, 1], [1, 3, 0, 2]])
    tokens = jnp.asarray(np.random.default_rng(0).integers(0, config.vocab_size, size=(4, 8), dtype=np.int32))
    batch = GrugLmExample(tokens=tokens, loss_weight=jnp.ones((4, 8), jnp.float32).at[:, -1].set(0.0))
    optimizer = _optimizer()
    step = _make_train_step(optimizer, _FP32, z_loss_weight=1e-4, ema_beta=None, router_bias_mode=RouterBiasMode.FROZEN)

    with use_test_devices(4) as devices:
        mesh = _expert_sharded_mesh(devices)
        with jax.set_mesh(mesh):
            placed_first, placed_metrics, _ = step(
                permute_experts(_trained_state(config, optimizer, mesh), placement), batch
            )
            stepped, metrics, _ = step(_trained_state(config, optimizer, mesh), batch)
            stepped_then_placed = permute_experts(stepped, placement)
        np.testing.assert_allclose(float(placed_metrics["train/loss"]), float(metrics["train/loss"]), rtol=1e-6)
        _assert_trees_close(placed_first.params, stepped_then_placed.params)
        _assert_trees_close(placed_first.opt_state, stepped_then_placed.opt_state)
        np.testing.assert_array_equal(
            np.asarray(placed_first.pending_qb_betas), np.asarray(stepped_then_placed.pending_qb_betas)
        )


@pytest.mark.multidevice
def test_expert_placement_moves_expert_sharded_leaves():
    config = _tiny_config()
    placement = _placement([[3, 2, 1, 0], [1, 0, 3, 2]])
    order = np.asarray(placement.order)
    with use_test_devices(4) as devices:
        mesh = _expert_sharded_mesh(devices)
        state = _trained_state(config, _optimizer(), mesh)
        mlp = state.params.stacked_blocks.stacked.mlp
        before = {
            "w_gate": np.asarray(mlp.expert_mlp.w_gate),
            "router": np.asarray(mlp.router),
            "bias": np.asarray(mlp.router_bias),
        }
        with jax.set_mesh(mesh):
            placed = permute_experts(state, placement).params.stacked_blocks.stacked.mlp
        after = {
            "w_gate": np.asarray(placed.expert_mlp.w_gate),
            "router": np.asarray(placed.router),
            "bias": np.asarray(placed.router_bias),
        }

    layers = np.arange(order.shape[0])[:, None]
    np.testing.assert_array_equal(after["w_gate"], before["w_gate"][layers, order])
    np.testing.assert_array_equal(after["bias"], before["bias"][layers, order])
    expected_router = np.stack([router[:, o] for router, o in zip(before["router"], order, strict=True)])
    np.testing.assert_array_equal(after["router"], expected_router)


def test_expert_placement_round_trips_through_its_inverse():
    config = _tiny_config()
    placement = _placement([[2, 0, 3, 1], [1, 3, 0, 2]])
    mesh = _one_device_mesh()
    original = _trained_state(config, optax.adam(1e-2), mesh)
    # Placement donates the leaves it moves, so compare against host copies.
    expected = jax.tree.map(np.asarray, original)
    with jax.set_mesh(mesh):
        placed = permute_experts(original, placement)
        moved = np.asarray(placed.params.stacked_blocks.stacked.mlp.expert_mlp.w_gate)
        restored = permute_experts(placed, placement.inverse())

    assert not np.array_equal(moved, expected.params.stacked_blocks.stacked.mlp.expert_mlp.w_gate)
    for (path, want), have in zip(jax.tree_util.tree_leaves_with_path(expected), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(np.asarray(have), want, err_msg=jax.tree_util.keystr(path))


def test_expert_placement_moves_host_offloaded_optimizer_state():
    # The hero offloads optimizer state to pinned host memory; placement must move those leaves too.
    config = _tiny_config()
    placement = _placement([[2, 0, 3, 1], [1, 3, 0, 2]])
    order = np.asarray(placement.order)
    mesh = _one_device_mesh()
    with jax.set_mesh(mesh):
        # The trainer builds its state under jit, where the offload's memory-kind moves can trace.
        state = jax.jit(
            lambda key: initial_state(
                config, optimizer=_optimizer(), mp=_FP32, key=key, ema_beta=None, offload_opt_state=True
            )
        )(jax.random.key(0))
        # Fresh moments are zero, and permuted zeros equal zeros, so give the host leaves distinct values.
        rng = np.random.default_rng(1)

        def nonzero_on_host(leaf):
            if not (eqx.is_array(leaf) and leaf.sharding.memory_kind == "pinned_host"):
                return leaf
            return jax.device_put(rng.uniform(0.5, 1.5, leaf.shape).astype(leaf.dtype), leaf.sharding)

        state = eqx.tree_at(lambda s: s.opt_state, state, jax.tree.map(nonzero_on_host, state.opt_state))
        host_leaves = {
            jax.tree_util.keystr(path): np.asarray(leaf)
            for path, leaf in jax.tree_util.tree_leaves_with_path(state.opt_state)
            if eqx.is_array(leaf) and leaf.sharding.memory_kind == "pinned_host"
        }
        placed = permute_experts(state, placement)

    gate_moments = 0
    layers = np.arange(order.shape[0])[:, None]
    for path, leaf in jax.tree_util.tree_leaves_with_path(placed.opt_state):
        key = jax.tree_util.keystr(path)
        if key not in host_leaves:
            continue
        assert leaf.sharding.memory_kind == "pinned_host"
        if key.endswith(".expert_mlp.w_gate"):
            gate_moments += 1
            np.testing.assert_array_equal(np.asarray(leaf), host_leaves[key][layers, order])
    assert gate_moments >= 1


def _write_checkpoint_metadata(path, placement: ExpertPlacement | None) -> str:
    path.mkdir(parents=True)
    recorded = {} if placement is None else placement.checkpoint_metadata()
    (path / "metadata.json").write_text(json.dumps({"step": 0, "timestamp": "2026-11-09T00:00:00", **recorded}))
    return str(path)


def test_restore_places_only_the_source_and_trusts_only_a_matching_record(tmp_path):
    config = _tiny_config()
    mesh = _one_device_mesh()
    source = str(tmp_path / "source")
    placement = dataclasses.replace(_placement([[2, 0, 3, 1], [1, 3, 0, 2]]), source_checkpoint=source)
    other = dataclasses.replace(placement, order=((0, 1, 2, 3), (0, 1, 2, 3)))
    _write_checkpoint_metadata(tmp_path / "source", None)
    placed_save = _write_checkpoint_metadata(tmp_path / "placed" / "step-5", placement)
    unrecorded_save = _write_checkpoint_metadata(tmp_path / "unrecorded" / "step-5", None)
    other_save = _write_checkpoint_metadata(tmp_path / "other" / "step-5", other)

    def gate(state):
        return np.asarray(state.params.stacked_blocks.stacked.mlp.expert_mlp.w_gate)

    with jax.set_mesh(mesh):
        state = _trained_state(config, _optimizer(), mesh)
        before = gate(state)
        layers = np.arange(2)[:, None]
        np.testing.assert_array_equal(
            gate(_place_restored_experts(state, placement, loaded_checkpoint=source)),
            before[layers, np.asarray(placement.order)],
        )
        state = _trained_state(config, _optimizer(), mesh)
        np.testing.assert_array_equal(
            gate(_place_restored_experts(state, placement, loaded_checkpoint=placed_save)), before
        )
        np.testing.assert_array_equal(gate(_place_restored_experts(state, None, loaded_checkpoint=source)), before)
        with pytest.raises(ValueError, match="records no expert placement"):
            _place_restored_experts(state, placement, loaded_checkpoint=unrecorded_save)
        with pytest.raises(ValueError, match=f"this run uses expert placement {placement.fingerprint}"):
            _place_restored_experts(state, placement, loaded_checkpoint=other_save)
        with pytest.raises(ValueError, match="this run uses no expert placement"):
            _place_restored_experts(state, None, loaded_checkpoint=placed_save)
