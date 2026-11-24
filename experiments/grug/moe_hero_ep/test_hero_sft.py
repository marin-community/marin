# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import math
import time
from datetime import timedelta
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import levanter.training_budget as training_budget
import marin.training.levanter as training_runtime
import numpy as np
import optax
import pytest
from fray.types import ResourceConfig
from jax.sharding import AxisType, Mesh, NamedSharding, reshard
from jax.sharding import PartitionSpec as P
from levanter.checkpoint import CheckpointerConfig, load_checkpoint_metadata, save_checkpoint
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig
from levanter.optim.config import AdamConfig
from levanter.schedule import BatchSchedule
from levanter.testing.devices import use_test_devices
from levanter.tracker import NoopConfig, NoopTracker, current_tracker
from levanter.trainer import TrainerConfig
from levanter.training_budget import TrainingBudgetConfig

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
    SftStore,
    one_epoch_weight_stages,
    plan_sft_mixture,
    sft_components,
)
from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe_hero_ep.train import (
    GradientAccumulatorMemory,
    GrugRunConfig,
    GrugTrainerConfig,
    RouterBiasMode,
    SpecialTokenInit,
    TrainingDataMode,
    _loss_and_grads,
    _make_train_step,
    _place_restored_experts,
    _run_grug_local,
    build_train_loader,
    grug_trainer_mesh_config,
    initial_state,
    initialize_hero_sft_weights,
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
    assert [step for step, _ in stages] == [block * plan.block_size // 16 for block in range(plan.num_blocks)]
    draws: dict[str, int] = {}
    for _, weights in stages:
        counts = {name: round(weight * plan.block_size) for name, weight in weights.items()}
        assert sum(counts.values()) == plan.block_size
        assert counts["replay/x"] >= 1 and counts["replay/y"] >= 1
        for name, count in counts.items():
            draws[name] = draws.get(name, 0) + count
    assert {name: draws[name] for name in sft_counts} == sft_counts
    assert draws["replay/x"] + draws["replay/y"] == plan.run_sequences - plan.sft_sequences


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


@pytest.mark.multidevice
@pytest.mark.parametrize("devices", [1, 4])
@pytest.mark.timeout(120)
def test_176_sequence_update_matches_six_padded_microbatches_with_one_clip_and_update(devices):
    with use_test_devices(devices) as selected:
        mesh = Mesh(
            np.asarray(selected).reshape(1, devices, 1, 1, 1),
            ("replica_dcn", "data", "expert", "context", "model"),
            axis_types=(AxisType.Explicit,) * 5,
        )
        with jax.set_mesh(mesh):
            optimizer = optax.sgd(1e-2, momentum=0.9)
            state = initial_state(_tiny_config(), optimizer=optimizer, mp=_FP32, key=jax.random.key(12), ema_beta=None)
            params = state.params
            tokens = jnp.arange(176 * 8, dtype=jnp.int32).reshape(176, 8) % 64
            weights = (jnp.arange(176 * 8).reshape(176, 8) % 3 != 0).astype(jnp.float32)
            weights = weights.at[:, -1].set(0)
            batch = GrugLmExample(tokens=tokens, loss_weight=weights)
            padded = GrugLmExample(
                tokens=jnp.pad(tokens, ((0, 16), (0, 0)), constant_values=63),
                # Nonzero garbage in padding must never contribute to the update.
                loss_weight=jnp.pad(weights, ((0, 16), (0, 0)), constant_values=1),
            )
            batch = jax.tree.map(lambda x: jax.device_put(x, NamedSharding(mesh, P("data", None))), batch)
            padded = jax.tree.map(lambda x: jax.device_put(x, NamedSharding(mesh, P("data", None))), padded)
            (loss, _), grads = _loss_and_grads(params, batch, _FP32, 1e-4)
            clipped, _ = optax.clip_by_global_norm(0.01).update(grads, optax.EmptyState())
            updates, expected_opt_state = optimizer.update(clipped, state.opt_state, params)
            expected_params = optax.apply_updates(params, updates)
            step = _make_train_step(
                optimizer,
                _FP32,
                z_loss_weight=1e-4,
                ema_beta=None,
                max_grad_norm=0.01,
                router_bias_mode=RouterBiasMode.FROZEN,
                microbatch_size=32,
                logical_batch_size=176,
            )
            state, metrics, _ = step(state, padded)
            assert int(state.step) == 1
            assert float(metrics["train/grad_clipped"]) == 1.0
            np.testing.assert_allclose(np.asarray(metrics["train/loss"]), np.asarray(loss), rtol=2e-6, atol=1e-6)
            for expected, actual in zip(jax.tree.leaves(expected_params), jax.tree.leaves(state.params), strict=True):
                np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-4, atol=2e-6)

            for expected, actual in zip(
                jax.tree.leaves(expected_opt_state), jax.tree.leaves(state.opt_state), strict=True
            ):
                np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-4, atol=2e-6)


@pytest.mark.timeout(120)
def test_hero_budget_stops_between_updates_and_commits_permanent_checkpoint(tmp_path, monkeypatch):
    # CPU devices expose no allocator statistics; emulate that hardware I/O boundary.
    monkeypatch.setattr(
        type(jax.local_devices()[0]),
        "memory_stats",
        lambda _: {
            "peak_bytes_in_use": 0,
            "bytes_in_use": 0,
            "bytes_limit": 1,
        },
    )
    started = time.time()
    clock = iter([0.0, 0.0, 1000.0])
    monkeypatch.setattr(training_budget, "monotonic", lambda: next(clock))
    monkeypatch.setattr(training_budget, "time", lambda: started)
    context = SimpleNamespace(
        job_id="cpu-clock-test",
        client=SimpleNamespace(
            list_tasks=lambda _: [SimpleNamespace(started_at=SimpleNamespace(epoch_seconds=lambda: started))]
        ),
    )
    monkeypatch.setattr(training_runtime, "iris_ctx", lambda: context)
    checkpoint = tmp_path / "checkpoints"
    trainer = TrainerConfig(
        id="hero-clock-test",
        train_batch_size=1,
        num_train_steps=10,
        mp=_FP32,
        tracker=NoopConfig(),
        require_accelerator=False,
        mesh=grug_trainer_mesh_config(1),
        distributed=DistributedConfig(initialize_jax_distributed=False),
        log_jaxprs=False,
        log_xla_hlo=False,
        load_checkpoint=False,
        log_dir=tmp_path / "logs",
        native_telemetry_root=str(tmp_path / "telemetry"),
        checkpointer=CheckpointerConfig(
            base_path=str(checkpoint),
            temporary_base_path=str(tmp_path / "resume"),
            append_run_id_to_base_path=False,
            save_interval=timedelta(hours=2),
            keep=None,
        ),
        training_budget=TrainingBudgetConfig(timedelta(seconds=100), timedelta(seconds=20), timedelta(seconds=20)),
    )
    with current_tracker(NoopTracker()):
        _run_grug_local(
            GrugRunConfig(
                model=_tiny_config(),
                data=LmDataConfig(tokenizer="gpt2", components={}),
                resources=ResourceConfig(),
                optimizer=AdamConfig(learning_rate=1e-3, warmup=0),
                eval=None,
                trainer=GrugTrainerConfig(
                    trainer=trainer,
                    save_checkpoints=True,
                    training_data_mode=TrainingDataMode.SYNTHETIC,
                    replica_axis_size=1,
                ),
            )
        )
    receipt = json.loads((tmp_path / "telemetry" / "completion.json").read_text())
    assert receipt["stop_reason"] == "time_bound"
    assert receipt["restored_step"] == 0 and receipt["final_step"] == 1
    metadata = load_checkpoint_metadata(receipt["checkpoint"])
    assert metadata["step"] == 1 and not metadata["is_temporary"]
    assert receipt["finalized_epoch_seconds"] <= receipt["deadline_epoch_seconds"]


@pytest.mark.multidevice
def test_hero_loader_padding_does_not_consume_the_next_logical_batch():
    with use_test_devices(32) as devices:
        mesh = Mesh(
            np.asarray(devices).reshape(2, 1, 16, 1, 1),
            ("replica_dcn", "data", "expert", "context", "model"),
            axis_types=(AxisType.Explicit,) * 5,
        )
        dataset = ListAsyncDataset(
            [
                GrugLmExample(tokens=np.full(8, index, dtype=np.int32), loss_weight=np.ones(8, dtype=np.float32))
                for index in range(352)
            ]
        )
        loader = build_train_loader(
            dataset,
            batch_schedule=BatchSchedule(176),
            mesh=mesh,
            allow_nondivisible_batch_size=True,
        )
        with jax.set_mesh(mesh):
            iterator = loader.iter_from_step(0)
            for ordinal in range(2):
                batch = next(iterator)
                assert batch.tokens.shape == (192, 8)
                np.testing.assert_array_equal(
                    np.asarray(batch.tokens)[:176, 0], np.arange(ordinal * 176, (ordinal + 1) * 176)
                )
                np.testing.assert_array_equal(np.asarray(batch.loss_weight)[176:], 0)


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


@pytest.mark.multidevice
@pytest.mark.timeout(120)
@pytest.mark.parametrize("precision", [_FP32, jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")])
def test_recomputing_moe_output_preserves_accumulated_updates(precision):
    with use_test_devices(4) as devices:
        mesh = _expert_sharded_mesh(devices)
        with jax.set_mesh(mesh):
            optimizer = _optimizer()
            configs = [dataclasses.replace(_tiny_config(), remat_mode=mode) for mode in model.OFFLOAD_CARRY_REMAT_MODES]
            tokens = jnp.asarray(np.random.default_rng(2).integers(0, 64, size=(8, 8), dtype=np.int32))
            batch = GrugLmExample(tokens=tokens, loss_weight=jnp.ones((8, 8), jnp.float32).at[:, -1].set(0.0))
            step = _make_train_step(
                optimizer,
                precision,
                z_loss_weight=1e-4,
                ema_beta=None,
                max_grad_norm=1.0,
                offload_opt_state=True,
                router_bias_mode=RouterBiasMode.FROZEN,
                microbatch_size=4,
                logical_batch_size=8,
            )
            states = [_trained_state(config, optimizer, mesh) for config in configs]
            for _ in range(4):
                results = [step(state, batch) for state in states]
                states = [result[0] for result in results]
                for left, right in zip(jax.tree.leaves(states[0]), jax.tree.leaves(states[1]), strict=True):
                    np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
                np.testing.assert_array_equal(
                    np.asarray(results[0][1]["train/loss"]), np.asarray(results[1][1]["train/loss"])
                )


@pytest.mark.multidevice
@pytest.mark.timeout(120)
@pytest.mark.parametrize("precision", [_FP32, jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")])
def test_host_gradient_accumulator_preserves_padded_updates(precision):
    with use_test_devices(4) as devices:
        mesh = _expert_sharded_mesh(devices)
        with jax.set_mesh(mesh):
            optimizer = _optimizer()
            config = dataclasses.replace(_tiny_config(), remat_mode=model.OFFLOAD_CARRY_RECOMPUTE_MOE_REMAT_MODE)
            tokens = jnp.asarray(np.random.default_rng(3).integers(0, 64, size=(12, 8), dtype=np.int32))
            batch = GrugLmExample(tokens=tokens, loss_weight=jnp.ones((12, 8), jnp.float32).at[:, -1].set(0.0))
            steps = [
                _make_train_step(
                    optimizer,
                    precision,
                    z_loss_weight=1e-4,
                    ema_beta=None,
                    max_grad_norm=1.0,
                    offload_opt_state=True,
                    router_bias_mode=RouterBiasMode.FROZEN,
                    microbatch_size=4,
                    logical_batch_size=10,
                    gradient_accumulator_memory=memory,
                )
                for memory in GradientAccumulatorMemory
            ]
            states = [_trained_state(config, optimizer, mesh) for _ in steps]
            for _ in range(4):
                results = [step(state, batch) for step, state in zip(steps, states, strict=True)]
                states = [result[0] for result in results]
                for left, right in zip(jax.tree.leaves(states[0]), jax.tree.leaves(states[1]), strict=True):
                    np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
                np.testing.assert_array_equal(
                    np.asarray(results[0][1]["train/loss"]), np.asarray(results[1][1]["train/loss"])
                )


@pytest.mark.multidevice
@pytest.mark.timeout(180)
@pytest.mark.parametrize("precision", [_FP32, jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")])
def test_grouped_remat_preserves_padded_host_accumulated_updates(precision):
    with use_test_devices(4) as devices:
        mesh = _expert_sharded_mesh(devices)
        with jax.set_mesh(mesh):
            optimizer = _optimizer()
            configs = [
                dataclasses.replace(
                    _tiny_config(),
                    num_layers=6,
                    remat_mode=model.OFFLOAD_CARRY_RECOMPUTE_MOE_REMAT_MODE,
                    remat_group_size=group_size,
                )
                for group_size in (None, 2)
            ]
            tokens = jnp.asarray(np.random.default_rng(4).integers(0, 64, size=(12, 8), dtype=np.int32))
            batch = GrugLmExample(tokens=tokens, loss_weight=jnp.ones((12, 8), jnp.float32).at[:, -1].set(0.0))
            step = _make_train_step(
                optimizer,
                precision,
                z_loss_weight=1e-4,
                ema_beta=None,
                max_grad_norm=1.0,
                offload_opt_state=True,
                router_bias_mode=RouterBiasMode.FROZEN,
                microbatch_size=4,
                logical_batch_size=10,
                gradient_accumulator_memory=GradientAccumulatorMemory.PINNED_HOST,
            )
            states = [_trained_state(config, optimizer, mesh) for config in configs]
            for _ in range(4):
                results = [step(state, batch) for state in states]
                states = [result[0] for result in results]
                for left, right in zip(jax.tree.leaves(states[0]), jax.tree.leaves(states[1]), strict=True):
                    np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
                np.testing.assert_array_equal(
                    np.asarray(results[0][1]["train/loss"]), np.asarray(results[1][1]["train/loss"])
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


def test_initialize_hero_sft_weights_keeps_fresh_optimizer(tmp_path):
    config = _tiny_config()
    mesh = _one_device_mesh()
    checkpoint = str(tmp_path / "step-393491")
    placement = dataclasses.replace(_placement([[2, 0, 3, 1], [1, 3, 0, 2]]), source_checkpoint=checkpoint)
    pretrained = _trained_state(config, _optimizer(), mesh)
    bank_config = GrugMoeMuonHConfig(muonh_param_ndims=(2, 3), use_syrk=False)
    with jax.set_mesh(mesh):
        save_checkpoint(pretrained, step=393491, checkpoint_path=checkpoint, is_temporary=False)
        fresh = initial_state(config, optimizer=bank_config.build(20), mp=_FP32, key=jax.random.key(8), ema_beta=None)
        initialized = initialize_hero_sft_weights(fresh, checkpoint, mesh=mesh, placement=placement)

    order = np.asarray(placement.order)
    layers = np.arange(config.num_layers)[:, None]

    def bank(params):
        return params.stacked_blocks.stacked.mlp.expert_mlp.w_gate

    np.testing.assert_array_equal(bank(initialized.params), np.asarray(bank(pretrained.params))[layers, order])
    np.testing.assert_array_equal(initialized.pending_qb_betas, np.asarray(pretrained.pending_qb_betas)[layers, order])
    np.testing.assert_array_equal(initialized.params.output_proj, pretrained.params.output_proj)
    assert int(initialized.step) == 0
    _assert_trees_close(initialized.opt_state, fresh.opt_state)
    assert bank(bank_config.create_mask(initialized.params)) == "adam"
