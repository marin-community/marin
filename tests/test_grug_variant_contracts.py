# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for grug variants under experiments/grug/*.

These checks are intentionally variant-discovered: if a subdirectory contains
`model.py` and/or `train.py`, it is expected to satisfy the corresponding
lowering and training contracts.
"""

import asyncio
import dataclasses
import importlib
import json
import logging
import pickle
import uuid
from io import StringIO
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import optax
import pytest
from fray.cluster import ResourceConfig
from jax._src import config as jax_config
from jax.sharding import PartitionSpec as P
from jax.sharding import use_abstract_mesh
from levanter.checkpoint import CheckpointerConfig
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text import DatasetComponent, DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig
from levanter.grug.attention import AttentionMask as GrugAttentionMask
from levanter.grug.attention import with_fa4_cute_metadata
from levanter.grug.sharding import _compact_grug_mesh_shape
from levanter.schedule import BatchSchedule
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import TrainerConfig


def _discover_grug_variants_with_file(filename: str) -> list[str]:
    grug_dir = Path(__file__).resolve().parents[1] / "experiments" / "grug"
    variants: list[str] = []
    found_any = False
    for child in sorted(grug_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("__"):
            continue
        if (child / filename).is_file():
            found_any = True
            if _variant_has_noverify(child):
                continue
            variants.append(child.name)
    if not variants and not found_any:
        raise AssertionError(f"No grug variants with {filename} found under {grug_dir}")
    return variants


def _variant_module_name(variant: str, module: str) -> str:
    return f"experiments.grug.{variant}.{module}"


def test_compact_grug_mesh_shape_allows_expert_axis_to_span_processes():
    assert _compact_grug_mesh_shape(
        process_count=32,
        local_device_count=4,
        expert_axis_size=16,
        replica_axis_size=4,
        model_axis_size=1,
    ) == (4, 2, 16, 1)


def test_compact_grug_mesh_shape_keeps_expert_axis_at_size_one():
    """Standardized contract: compact_grug_mesh always carries the expert axis.

    The data-loader and model code reference "expert" unconditionally; we keep the axis at
    size 1 instead of dropping it so size-1 cases (e.g. the GPU canary) don't fall through
    a separate "axis absent" code path. See #6252 for the bug this contract prevents.
    """
    assert _compact_grug_mesh_shape(
        process_count=1,
        local_device_count=4,
        expert_axis_size=1,
        replica_axis_size=1,
        model_axis_size=1,
    ) == (1, 4, 1, 1)


def test_grug_coreweave_axes_keep_expert_and_model_groups_local():
    launch_module = importlib.import_module("experiments.grug.moe.launch")

    launch_module.validate_local_expert_model_axes(
        expert_axis=2,
        model_axis=4,
        local_device_count=8,
        env_prefix="SCALE",
    )


def test_grug_coreweave_axes_reject_cross_node_expert_model_product():
    launch_module = importlib.import_module("experiments.grug.moe.launch")

    with pytest.raises(ValueError, match="SCALE_EXPERT_AXIS \\* SCALE_MODEL_AXIS"):
        launch_module.validate_local_expert_model_axes(
            expert_axis=8,
            model_axis=4,
            local_device_count=8,
            env_prefix="SCALE",
        )


def test_grug_coreweave_ring_ep_rejects_simultaneous_expert_and_model_axes():
    launch_module = importlib.import_module("experiments.grug.moe.launch")

    with pytest.raises(ValueError, match="SCALE_MOE_IMPLEMENTATION=ring"):
        launch_module.validate_ring_expert_model_axes(
            expert_axis=2,
            model_axis=4,
            moe_implementation="ring",
            env_prefix="SCALE",
        )


def test_grug_coreweave_non_ring_backend_can_try_simultaneous_expert_and_model_axes():
    launch_module = importlib.import_module("experiments.grug.moe.launch")

    launch_module.validate_ring_expert_model_axes(
        expert_axis=2,
        model_axis=4,
        moe_implementation="ragged_all_to_all",
        env_prefix="SCALE",
    )


def test_grug_moe_synthetic_dataset_vectorizes_token_generation():
    launch_module = importlib.import_module("experiments.grug.moe.launch")
    dataset = launch_module.SyntheticGrugDataset(seq_len=8, vocab_size=128, num_examples=16)

    examples = asyncio.run(dataset.get_batch([0, 3, 5]))

    assert len(examples) == 3
    for index, example in zip([0, 3, 5], examples, strict=True):
        expected_tokens = (jnp.arange(8, dtype=jnp.int32) + index * 9973) % 128
        assert jnp.array_equal(example.tokens, expected_tokens)
        assert jnp.array_equal(example.loss_weight, GrugLmExample.causal_loss_mask(8).astype(jnp.float32))
        assert example.attn_mask.segment_ids is None


def test_grug_moe_synthetic_dataset_preserves_eos_segments():
    launch_module = importlib.import_module("experiments.grug.moe.launch")
    dataset = launch_module.SyntheticGrugDataset(seq_len=8, vocab_size=16, num_examples=16, eos_id=15, eos_interval=4)

    [example] = asyncio.run(dataset.get_batch([1]))
    expected_tokens = (jnp.arange(8, dtype=jnp.int32) + 9973) % 16
    expected_tokens = expected_tokens.at[3::4].set(15)
    expected = GrugLmExample.causal(expected_tokens, eos_id=15)

    assert jnp.array_equal(example.tokens, expected.tokens)
    assert jnp.array_equal(example.loss_weight, expected.loss_weight)
    assert example.attn_mask.segment_ids is not None
    assert expected.attn_mask.segment_ids is not None
    assert jnp.array_equal(example.attn_mask.segment_ids[0], expected.attn_mask.segment_ids[0])


def test_grug_moe_synthetic_dataset_is_dataclass_replaceable():
    launch_module = importlib.import_module("experiments.grug.moe.launch")
    dataset = launch_module.SyntheticGrugDataset(seq_len=8, vocab_size=128, num_examples=16)

    replaced = dataclasses.replace(dataset, num_examples=32)

    field_names = {field.name for field in dataclasses.fields(replaced)}
    assert "_positions" not in field_names
    assert "_loss_weight" not in field_names
    assert "_attn_mask" not in field_names
    assert asyncio.run(replaced.async_len()) == 32


def test_grug_moe_synthetic_dataset_serializes_without_jax_arrays():
    launch_module = importlib.import_module("experiments.grug.moe.launch")
    dataset = launch_module.SyntheticGrugDataset(seq_len=8, vocab_size=128, num_examples=16)

    assert not any(isinstance(value, jax.Array) for value in dataset.__dict__.values())

    restored = pickle.loads(pickle.dumps(dataset))

    assert not any(isinstance(value, jax.Array) for value in restored.__dict__.values())
    [example] = asyncio.run(restored.get_batch([0]))
    assert isinstance(example.tokens, jax.Array)
    assert isinstance(example.loss_weight, jax.Array)


def _variant_has_noverify(variant_dir: Path) -> bool:
    train_file = variant_dir / "train.py"
    if not train_file.is_file():
        return False
    return "# GRUG NOVERIFY" in train_file.read_text(encoding="utf-8")


class _reset_abstract_mesh:
    def __enter__(self):
        self._prev = jax_config.abstract_mesh_context_manager.swap_local(jax_config.config_ext.unset)
        return self

    def __exit__(self, exc_type, exc, tb):
        jax_config.abstract_mesh_context_manager.set_local(self._prev)
        return False


def _jaxpr_has_primitive(jaxpr_like, primitive_name: str) -> bool:
    jaxpr = getattr(jaxpr_like, "jaxpr", jaxpr_like)
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == primitive_name:
            return True
        for param in eqn.params.values():
            if _jaxpr_param_has_primitive(param, primitive_name):
                return True
    return False


def _jaxpr_param_has_primitive(param, primitive_name: str) -> bool:
    if hasattr(param, "jaxpr") or hasattr(param, "eqns"):
        return _jaxpr_has_primitive(param, primitive_name)
    if isinstance(param, (tuple, list)):
        return any(_jaxpr_param_has_primitive(item, primitive_name) for item in param)
    return False


def _discover_grug_variants_with_model_and_train() -> list[str]:
    model_variants = set(_discover_grug_variants_with_file("model.py"))
    train_variants = set(_discover_grug_variants_with_file("train.py"))
    variants = sorted(model_variants & train_variants)
    if not variants and model_variants and train_variants:
        return []
    if not variants:
        raise AssertionError("No grug variants with both model.py and train.py found")
    return variants


def _small_model_config(model_config_cls, *, vocab_size: int, seq_len: int):
    base_kwargs = {
        "vocab_size": vocab_size,
        "hidden_dim": 32,
        "intermediate_dim": 64,
        "num_layers": 2,
        "num_heads": 2,
        "num_kv_heads": 2,
        "max_seq_len": seq_len,
        "num_experts": 4,
        "num_experts_per_token": 2,
        "shared_expert_intermediate_dim": 64,
    }
    field_names = {field.name for field in dataclasses.fields(model_config_cls)}
    kwargs = {k: v for k, v in base_kwargs.items() if k in field_names}
    return model_config_cls(**kwargs)


def test_grug_moe_layer_masks_preserve_thd_segment_metadata():
    model_module = importlib.import_module("experiments.grug.moe.model")
    mask = GrugAttentionMask.causal().with_segment_ids(
        jnp.array([[0, 0, 1, 1, -1, -1]], dtype=jnp.int32),
        max_segments=3,
    )

    short_mask, long_mask = model_module._layer_attention_masks(mask, sliding_window=12)

    assert short_mask.thd_segment_metadata is mask.thd_segment_metadata
    assert long_mask.thd_segment_metadata is mask.thd_segment_metadata
    assert short_mask.segment_ids is mask.segment_ids
    assert long_mask.segment_ids is mask.segment_ids


def test_coreweave_thd_canary_uses_fixed_shape_training_segments(monkeypatch):
    monkeypatch.setenv("CANARY_ACCELERATOR", "gpu")
    monkeypatch.setenv("CANARY_ATTENTION_IMPLEMENTATION", "gpu_fa4_thd")
    monkeypatch.setenv("CANARY_TRACKER", "json_logger")
    monkeypatch.setenv("RUN_ID", "test-thd")

    canary_ferry = importlib.import_module("experiments.ferries.canary_ferry")
    canary_ferry = importlib.reload(canary_ferry)
    data = canary_ferry.canary_moe_step.config.data

    components = list(data.components.values())
    assert components
    assert all(isinstance(component, DatasetComponent) for component in components)
    assert {component.pack for component in components} == {1}


@pytest.mark.parametrize(
    "variant",
    _discover_grug_variants_with_model_and_train(),
)
def test_grug_variant_one_step_contract_lowers_with_default_ctor(variant: str):
    train_module = importlib.import_module(_variant_module_name(variant, "train"))
    model_module = importlib.import_module(_variant_module_name(variant, "model"))
    model_config_cls = model_module.GrugModelConfig
    make_train_step = train_module._make_train_step
    initial_state = train_module.initial_state
    mesh_fn = getattr(model_module, "debug_mesh_and_token_pspec", None)
    if mesh_fn is None:
        raise AssertionError(f"{_variant_module_name(variant, 'model')} must define debug_mesh_and_token_pspec")

    cfg = model_config_cls(vocab_size=1024)
    optimizer = optax.adam(1e-2)
    mp = jmp.get_policy("f32")
    train_step = make_train_step(optimizer, mp, z_loss_weight=0.0, ema_beta=None)
    mesh, token_pspec = mesh_fn(num_devices=4)
    batch = GrugLmExample(
        tokens=jnp.zeros((8, 4), dtype=jnp.int32),
        loss_weight=jnp.ones((8, 4), dtype=jnp.float32),
        attn_mask=GrugAttentionMask.causal(),
    )

    def one_step():
        sharded_batch = dataclasses.replace(
            batch,
            tokens=jax.sharding.reshard(batch.tokens, token_pspec),
            loss_weight=jax.sharding.reshard(batch.loss_weight, token_pspec),
        )
        state = initial_state(cfg, optimizer=optimizer, mp=mp, key=jax.random.PRNGKey(0), ema_beta=None)
        return train_step(state, sharded_batch, compute_watch=False)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        out_state_shape, out_metrics_shape, out_watch_shape = eqx.filter_eval_shape(one_step)

    assert out_state_shape.step.shape == ()
    assert "train/loss" in out_metrics_shape
    assert out_metrics_shape["train/loss"].shape == ()
    assert out_watch_shape is None


def test_grug_moe_variant_threads_moe_implementation_to_kernel():
    train_module = importlib.import_module("experiments.grug.moe.train")
    model_module = importlib.import_module("experiments.grug.moe.model")
    make_train_step = train_module._make_train_step
    initial_state = train_module.initial_state
    mesh_fn = getattr(model_module, "debug_mesh_and_token_pspec", None)
    if mesh_fn is None:
        raise AssertionError("experiments.grug.moe.model must define debug_mesh_and_token_pspec")

    cfg = _small_model_config(model_module.GrugModelConfig, vocab_size=1024, seq_len=4)
    cfg = dataclasses.replace(cfg, moe_implementation="ragged_all_to_all")
    optimizer = optax.adam(1e-2)
    mp = jmp.get_policy("f32")
    train_step = make_train_step(optimizer, mp, z_loss_weight=0.0, ema_beta=None)
    mesh, token_pspec = mesh_fn(num_devices=4)
    batch = GrugLmExample(
        tokens=jnp.zeros((8, 4), dtype=jnp.int32),
        loss_weight=jnp.ones((8, 4), dtype=jnp.float32),
        attn_mask=GrugAttentionMask.causal(),
    )

    def one_step():
        sharded_batch = dataclasses.replace(
            batch,
            tokens=jax.sharding.reshard(batch.tokens, token_pspec),
            loss_weight=jax.sharding.reshard(batch.loss_weight, token_pspec),
        )
        state = initial_state(cfg, optimizer=optimizer, mp=mp, key=jax.random.PRNGKey(0), ema_beta=None)
        return train_step(state, sharded_batch, compute_watch=False)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        closed_jaxpr, _, _ = eqx.filter_make_jaxpr(one_step)()

    assert "ragged_all_to_all" in str(closed_jaxpr)


def test_grug_moe_may_recipe_attention_flags_lower():
    train_module = importlib.import_module("experiments.grug.moe.train")
    model_module = importlib.import_module("experiments.grug.moe.model")
    make_train_step = train_module._make_train_step
    initial_state = train_module.initial_state
    mesh_fn = getattr(model_module, "debug_mesh_and_token_pspec", None)
    if mesh_fn is None:
        raise AssertionError("experiments.grug.moe.model must define debug_mesh_and_token_pspec")

    cfg = _small_model_config(model_module.GrugModelConfig, vocab_size=1024, seq_len=4)
    cfg = dataclasses.replace(
        cfg,
        routing_renorm_sum=2.5,
        use_half_rope=True,
        use_pko=True,
        cross_entropy_implementation="xla",
        router_z_loss_coef=0.0,
    )
    optimizer = optax.adam(1e-2)
    mp = jmp.get_policy("f32")
    train_step = make_train_step(optimizer, mp, z_loss_weight=0.0, ema_beta=None)
    mesh, token_pspec = mesh_fn(num_devices=4)
    segment_ids = jnp.array([[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5], [6, 6, 7, 7]], dtype=jnp.int32)
    batch = GrugLmExample(
        tokens=jnp.zeros((4, 4), dtype=jnp.int32),
        loss_weight=jnp.ones((4, 4), dtype=jnp.float32),
        attn_mask=GrugAttentionMask.causal().with_segment_ids(segment_ids),
    )

    def one_step():
        sharded_batch = dataclasses.replace(
            batch,
            tokens=jax.sharding.reshard(batch.tokens, token_pspec),
            loss_weight=jax.sharding.reshard(batch.loss_weight, token_pspec),
        )
        state = initial_state(cfg, optimizer=optimizer, mp=mp, key=jax.random.PRNGKey(0), ema_beta=None)
        return train_step(state, sharded_batch, compute_watch=False)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        out_state_shape, out_metrics_shape, _ = eqx.filter_eval_shape(one_step)

    assert out_state_shape.step.shape == ()
    assert "train/loss" in out_metrics_shape


@pytest.mark.parametrize(
    ("remat_mode", "expects_remat"),
    [
        ("none", False),
        ("recompute_all", True),
        ("save_moe", True),
    ],
)
def test_grug_moe_remat_mode_controls_checkpoint_boundary(remat_mode: str, expects_remat: bool):
    train_module = importlib.import_module("experiments.grug.moe.train")
    model_module = importlib.import_module("experiments.grug.moe.model")
    make_train_step = train_module._make_train_step
    initial_state = train_module.initial_state
    mesh_fn = getattr(model_module, "debug_mesh_and_token_pspec", None)
    if mesh_fn is None:
        raise AssertionError("experiments.grug.moe.model must define debug_mesh_and_token_pspec")

    cfg = _small_model_config(model_module.GrugModelConfig, vocab_size=1024, seq_len=4)
    cfg = dataclasses.replace(
        cfg,
        num_layers=1,
        remat_mode=remat_mode,
        cross_entropy_implementation="xla",
        router_z_loss_coef=0.0,
    )
    optimizer = optax.adam(1e-2)
    mp = jmp.get_policy("f32")
    train_step = make_train_step(optimizer, mp, z_loss_weight=0.0, ema_beta=None)
    mesh, token_pspec = mesh_fn(num_devices=4)
    batch = GrugLmExample(
        tokens=jnp.zeros((8, 4), dtype=jnp.int32),
        loss_weight=jnp.ones((8, 4), dtype=jnp.float32),
        attn_mask=GrugAttentionMask.causal(),
    )

    def one_step():
        sharded_batch = dataclasses.replace(
            batch,
            tokens=jax.sharding.reshard(batch.tokens, token_pspec),
            loss_weight=jax.sharding.reshard(batch.loss_weight, token_pspec),
        )
        state = initial_state(cfg, optimizer=optimizer, mp=mp, key=jax.random.PRNGKey(0), ema_beta=None)
        return train_step(state, sharded_batch, compute_watch=False)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        closed_jaxpr, _, _ = eqx.filter_make_jaxpr(one_step)()

    assert _jaxpr_has_primitive(closed_jaxpr, "remat2") is expects_remat


def test_grug_moe_may_launcher_diagnostic_overrides(monkeypatch):
    monkeypatch.setenv("MAY_NUM_LAYERS", "3")
    monkeypatch.setenv("MAY_USE_PKO", "false")
    monkeypatch.setenv("MAY_PKO_ON_LAST_LAYER", "false")

    launch_module = importlib.import_module("experiments.grug.moe.launch_cw_may_d2560")
    model = launch_module.build_may_model()

    assert model.num_layers == 3
    assert model.use_pko is False
    assert model.pko_on_last_layer is False


def test_grug_moe_pko_attention_accepts_precomputed_segment_starts(monkeypatch):
    model_module = importlib.import_module("experiments.grug.moe.model")
    mesh, _ = model_module.debug_mesh_and_token_pspec(num_devices=4)
    cfg = _small_model_config(model_module.GrugModelConfig, vocab_size=1024, seq_len=4)
    cfg = dataclasses.replace(cfg, use_pko=True)
    attn = model_module.CausalSelfAttention(
        w_q=jnp.ones((32, 32), dtype=jnp.bfloat16),
        w_k=jnp.ones((32, 32), dtype=jnp.bfloat16),
        w_v=jnp.ones((32, 32), dtype=jnp.bfloat16),
        w_o=jnp.ones((32, 32), dtype=jnp.bfloat16),
        attn_gate=jnp.zeros((32, 2), dtype=jnp.float32),
        cfg=cfg,
    )
    segment_ids = jnp.array([[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5], [6, 6, 7, 7]], dtype=jnp.int32)
    mask = GrugAttentionMask.causal().with_segment_ids(segment_ids)
    pko_doc_starts = model_module._segment_start_mask(mask, batch_size=4, seq_len=4)

    def fail_segment_start_mask(*args, **kwargs):
        del args, kwargs
        raise AssertionError("PKO should reuse precomputed document starts")

    def fake_attention(q, k, v, mask, *, implementation=None):
        del k, v, mask, implementation
        return jnp.zeros_like(q)

    monkeypatch.setattr(model_module, "_segment_start_mask", fail_segment_start_mask)
    monkeypatch.setattr(model_module, "attention", fake_attention)
    x = jnp.ones((4, 4, 32), dtype=jnp.bfloat16)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        out_shape = eqx.filter_eval_shape(lambda y: attn(y, mask, use_pko=True, pko_doc_starts=pko_doc_starts), x)

    assert out_shape.shape == x.shape


def test_grug_moe_segment_start_mask_keeps_token_sharding():
    model_module = importlib.import_module("experiments.grug.moe.model")
    mesh, _ = model_module.debug_mesh_and_token_pspec(num_devices=4)
    segment_ids = jnp.array([[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5], [6, 6, 7, 7]], dtype=jnp.int32)

    def starts_for(ids):
        mask = GrugAttentionMask.causal().with_segment_ids(ids)
        return model_module._segment_start_mask(mask, batch_size=4, seq_len=4)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        closed_jaxpr = jax.make_jaxpr(starts_for)(segment_ids)

    assert closed_jaxpr.jaxpr.outvars[0].aval.sharding.spec == P(("replica_dcn", "data", "expert"), None)


def test_grug_moe_fa4_metadata_keeps_token_sharding():
    model_module = importlib.import_module("experiments.grug.moe.model")
    mesh, _ = model_module.debug_mesh_and_token_pspec(num_devices=4)
    segment_ids = jnp.array([[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5], [6, 6, 7, 7]], dtype=jnp.int32)

    def lower_bounds_for(ids):
        mask = GrugAttentionMask.causal(sliding_window=2).with_segment_ids(ids)
        mask = with_fa4_cute_metadata(mask, batch_size=4, seq_len=4)
        assert mask.fa4_cute_metadata is not None
        return mask.fa4_cute_metadata.lower_bounds

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        closed_jaxpr = jax.make_jaxpr(lower_bounds_for)(segment_ids)

    assert closed_jaxpr.jaxpr.outvars[0].aval.sharding.spec == P(("replica_dcn", "data", "expert"), None)


def test_grug_moe_shared_dense_intermediates_keep_token_sharding():
    model_module = importlib.import_module("experiments.grug.moe.model")
    mesh, _ = model_module.debug_mesh_and_token_pspec(num_devices=4)
    dense = model_module.DenseMLP(
        w_gate=jnp.ones((32, 64), dtype=jnp.bfloat16),
        w_up=jnp.ones((32, 64), dtype=jnp.bfloat16),
        w_down=jnp.ones((64, 32), dtype=jnp.bfloat16),
    )
    x = jnp.ones((8, 4, 32), dtype=jnp.bfloat16)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        closed_jaxpr = jax.make_jaxpr(lambda y: dense(y))(x)

    dot_specs = [
        eqn.outvars[0].aval.sharding.spec for eqn in closed_jaxpr.jaxpr.eqns if eqn.primitive.name == "dot_general"
    ]

    assert dot_specs == [P(("replica_dcn", "data", "expert"), None)] * 3


def test_grug_moe_gated_norm_intermediates_keep_batch_sharding():
    model_module = importlib.import_module("experiments.grug.moe.model")
    mesh, _ = model_module.debug_mesh_and_token_pspec(num_devices=4)
    gated_norm = model_module.GatedNorm(
        w_down=jnp.ones((32, 8), dtype=jnp.bfloat16),
        w_up=jnp.ones((8, 32), dtype=jnp.bfloat16),
    )
    x = jnp.ones((8, 4, 32), dtype=jnp.bfloat16)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        closed_jaxpr = jax.make_jaxpr(lambda y: gated_norm(y))(x)

    dot_specs = [
        eqn.outvars[0].aval.sharding.spec for eqn in closed_jaxpr.jaxpr.eqns if eqn.primitive.name == "dot_general"
    ]

    assert dot_specs == [P(("replica_dcn", "data", "expert"), None, None)] * 2


def test_grug_moe_data_loaders_build_against_single_expert_mesh():
    """Regression: build_train_loader / build_tagged_evaluator must work when the
    compact mesh's expert axis has size 1 (canary configuration).

    See https://github.com/marin-community/marin/issues/6252 — canary configurations
    always have expert_axis_size == 1. Under the standardized
    ``(replica_dcn, data, expert, model)`` contract the "expert" axis is kept at length 1
    instead of being dropped, so the data-loader pspec can name it unconditionally.
    """
    train_module = importlib.import_module("experiments.grug.moe.train")
    compact_grug_mesh = importlib.import_module("levanter.grug.sharding").compact_grug_mesh

    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    assert mesh.shape.get("expert") == 1, "fixture must reproduce the canary single-expert layout"

    dataset = ListAsyncDataset(
        [
            GrugLmExample(
                tokens=jnp.zeros((4,), dtype=jnp.int32),
                loss_weight=jnp.ones((4,), dtype=jnp.float32),
                attn_mask=GrugAttentionMask.causal(),
            )
        ]
    )
    batch_schedule = BatchSchedule(max(1, len(jax.devices())))

    # This used to raise: "Resource axis: expert ... is not found in mesh: (..., model)".
    loader = train_module.build_train_loader(dataset, batch_schedule=batch_schedule, mesh=mesh)
    assert loader is not None


def test_grug_moe_model_init_against_single_expert_mesh():
    """Regression: MoEMLP.init must build when the compact mesh's expert axis has size 1.

    See https://github.com/marin-community/marin/issues/6252 — canary configurations
    have expert_axis_size == 1. Under the standardized
    ``(replica_dcn, data, expert, model)`` contract the "expert" axis is kept at length 1,
    so MoEMLP.init reads ``mesh.shape["expert"] == 1`` rather than hitting an
    "axis absent" branch.
    """
    train_module = importlib.import_module("experiments.grug.moe.train")
    model_module = importlib.import_module("experiments.grug.moe.model")
    compact_grug_mesh = importlib.import_module("levanter.grug.sharding").compact_grug_mesh

    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    assert mesh.shape.get("expert") == 1, "fixture must reproduce the canary single-expert layout"

    cfg = _small_model_config(model_module.GrugModelConfig, vocab_size=1024, seq_len=4)
    optimizer = optax.adam(1e-2)
    mp = jmp.get_policy("f32")

    def build():
        return train_module.initial_state(cfg, optimizer=optimizer, mp=mp, key=jax.random.PRNGKey(0), ema_beta=None)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh.abstract_mesh):
        state_shape = eqx.filter_eval_shape(build)

    assert state_shape.params is not None


def _floating_leaf_dtypes(tree) -> set:
    return {
        leaf.dtype
        for leaf in jax.tree_util.tree_leaves(tree)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact)
    }


def test_grug_moe_compute_live_params_keep_fp32_master():
    train_module = importlib.import_module("experiments.grug.moe.train")
    model_module = importlib.import_module("experiments.grug.moe.model")
    mesh_fn = getattr(model_module, "debug_mesh_and_token_pspec", None)
    if mesh_fn is None:
        raise AssertionError("experiments.grug.moe.model must define debug_mesh_and_token_pspec")

    cfg = _small_model_config(model_module.GrugModelConfig, vocab_size=1024, seq_len=4)
    optimizer = optax.adam(1e-2)
    mp = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    mesh, _ = mesh_fn(num_devices=4)

    def build():
        return train_module.initial_state(
            cfg,
            optimizer=optimizer,
            mp=mp,
            key=jax.random.PRNGKey(0),
            ema_beta=None,
            live_param_mode="compute_with_master",
        )

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        state_shape = eqx.filter_eval_shape(build)

    assert state_shape.master_params is not None
    assert _floating_leaf_dtypes(state_shape.params) == {jnp.dtype(jnp.bfloat16)}
    assert _floating_leaf_dtypes(state_shape.master_params) == {jnp.dtype(jnp.float32)}


def test_grug_moe_compute_live_params_one_step_lowers():
    train_module = importlib.import_module("experiments.grug.moe.train")
    model_module = importlib.import_module("experiments.grug.moe.model")
    mesh_fn = getattr(model_module, "debug_mesh_and_token_pspec", None)
    if mesh_fn is None:
        raise AssertionError("experiments.grug.moe.model must define debug_mesh_and_token_pspec")

    cfg = _small_model_config(model_module.GrugModelConfig, vocab_size=1024, seq_len=4)
    optimizer = optax.adam(1e-2)
    mp = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    train_step = train_module._make_train_step(optimizer, mp, z_loss_weight=0.0, ema_beta=None)
    mesh, token_pspec = mesh_fn(num_devices=4)
    batch = GrugLmExample(
        tokens=jnp.zeros((8, 4), dtype=jnp.int32),
        loss_weight=jnp.ones((8, 4), dtype=jnp.float32),
        attn_mask=GrugAttentionMask.causal(),
    )

    def one_step():
        sharded_batch = dataclasses.replace(
            batch,
            tokens=jax.sharding.reshard(batch.tokens, token_pspec),
            loss_weight=jax.sharding.reshard(batch.loss_weight, token_pspec),
        )
        state = train_module.initial_state(
            cfg,
            optimizer=optimizer,
            mp=mp,
            key=jax.random.PRNGKey(0),
            ema_beta=None,
            live_param_mode="compute_with_master",
        )
        return train_step(state, sharded_batch, compute_watch=False)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        out_state_shape, out_metrics_shape, _ = eqx.filter_eval_shape(one_step)

    assert out_state_shape.master_params is not None
    assert _floating_leaf_dtypes(out_state_shape.params) == {jnp.dtype(jnp.bfloat16)}
    assert _floating_leaf_dtypes(out_state_shape.master_params) == {jnp.dtype(jnp.float32)}
    assert "train/loss" in out_metrics_shape


def test_grug_moe_log_every_gates_explicit_loop_metrics(tmp_path: Path):
    train_module = importlib.import_module("experiments.grug.moe.train")
    model_module = importlib.import_module("experiments.grug.moe.model")
    launch_module = importlib.import_module("experiments.grug.moe.launch")

    vocab_size = 128
    seq_len = 8
    examples = []
    for i in range(8):
        tokens = (jnp.arange(seq_len, dtype=jnp.int32) + i) % vocab_size
        examples.append(GrugLmExample.causal(tokens))

    train_dataset = ListAsyncDataset(examples)
    data_config = LmDataConfig(
        components={"direct": DirectDatasetComponent(datasets={"train": train_dataset})},
        vocab_size=vocab_size,
        tokenizer="passthrough",
    )

    logger_name = f"test_grug_json_tracker_moe_{uuid.uuid4().hex}"
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    try:
        variant_tmp = tmp_path / "moe"
        variant_tmp.mkdir(parents=True, exist_ok=True)
        trainer_config = TrainerConfig(
            id="test-grug-moe-log-every",
            num_train_steps=3,
            train_batch_size=max(1, len(jax.devices())),
            tracker=JsonLoggerConfig(logger_name=logger_name),
            require_accelerator=False,
            use_explicit_mesh_axes=True,
            distributed=DistributedConfig(initialize_jax_distributed=False),
            log_dir=variant_tmp / "logs",
            checkpointer=launch_module.DisabledCheckpointerConfig(base_path=str(variant_tmp / "checkpoints")),
            load_checkpoint=False,
            log_jaxprs=False,
            log_xla_hlo=False,
        )

        model_config = _small_model_config(model_module.GrugModelConfig, vocab_size=vocab_size, seq_len=seq_len)
        model_config = dataclasses.replace(
            model_config,
            cross_entropy_implementation="xla",
            router_z_loss_coef=0.0,
        )
        run_cfg = train_module.GrugRunConfig(
            model=model_config,
            data=data_config,
            resources=ResourceConfig.with_cpu(),
            trainer=train_module.GrugTrainerConfig(
                trainer=trainer_config,
                log_every=2,
                z_loss_weight=0.0,
                ema_beta=None,
                expert_axis_size=1,
                replica_axis_size=1,
            ),
            eval=None,
        )
        train_module.run_grug(run_cfg)
    finally:
        logger.removeHandler(handler)

    records = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    log_records = [record for record in records if record.get("event") == "log"]

    def steps_with_metric(metric_name: str) -> list[int]:
        return [record["step"] for record in log_records if metric_name in record.get("metrics", {})]

    assert steps_with_metric("throughput/loading_time") == [0, 2]
    assert steps_with_metric("train/cross_entropy_loss") == [0, 2]
    assert steps_with_metric("train/router/router_z_loss") == [0, 2]


@pytest.mark.parametrize(
    "variant",
    _discover_grug_variants_with_model_and_train(),
)
def test_grug_variant_initial_state_only_stores_ema_when_enabled(variant: str):
    train_module = importlib.import_module(_variant_module_name(variant, "train"))
    model_module = importlib.import_module(_variant_module_name(variant, "model"))
    model_config_cls = model_module.GrugModelConfig
    initial_state = train_module.initial_state
    mesh_fn = getattr(model_module, "debug_mesh_and_token_pspec", None)
    if mesh_fn is None:
        raise AssertionError(f"{_variant_module_name(variant, 'model')} must define debug_mesh_and_token_pspec")

    cfg = model_config_cls(vocab_size=1024)
    optimizer = optax.adam(1e-2)
    mp = jmp.get_policy("f32")
    mesh, _ = mesh_fn(num_devices=4)

    def init_state_shape(*, ema_beta: float | None):
        def build():
            return initial_state(cfg, optimizer=optimizer, mp=mp, key=jax.random.PRNGKey(0), ema_beta=ema_beta)

        with _reset_abstract_mesh(), use_abstract_mesh(mesh):
            return eqx.filter_eval_shape(build)

    no_ema_state_shape = init_state_shape(ema_beta=None)
    assert no_ema_state_shape.ema_params is None

    with_ema_state_shape = init_state_shape(ema_beta=0.999)
    assert with_ema_state_shape.ema_params is not None


def test_grug_base_run_emits_expected_metrics_with_json_tracker(tmp_path: Path):
    train_module = importlib.import_module("experiments.grug.base.train")
    model_module = importlib.import_module("experiments.grug.base.model")

    vocab_size = 128
    seq_len = 32
    examples = []
    for i in range(8):
        tokens = (jnp.arange(seq_len, dtype=jnp.int32) + i) % vocab_size
        examples.append(GrugLmExample.causal(tokens))
    eval_examples = [GrugLmExample.causal((jnp.arange(seq_len, dtype=jnp.int32) + 100) % vocab_size)]

    train_dataset = ListAsyncDataset(examples)
    eval_dataset = ListAsyncDataset(eval_examples)
    data_config = LmDataConfig(
        components={"direct": DirectDatasetComponent(datasets={"train": train_dataset, "validation": eval_dataset})},
        vocab_size=vocab_size,
        tokenizer="passthrough",
    )

    logger_name = f"test_grug_json_tracker_base_{uuid.uuid4().hex}"
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    try:
        variant_tmp = tmp_path / "base"
        variant_tmp.mkdir(parents=True, exist_ok=True)
        trainer_config = TrainerConfig(
            id="test-grug-base-metrics",
            num_train_steps=1,
            train_batch_size=max(1, len(jax.devices())),
            tracker=JsonLoggerConfig(logger_name=logger_name),
            require_accelerator=False,
            use_explicit_mesh_axes=True,
            distributed=DistributedConfig(initialize_jax_distributed=False),
            log_dir=variant_tmp / "logs",
            checkpointer=CheckpointerConfig(base_path=str(variant_tmp / "checkpoints")),
        )

        run_cfg = train_module.GrugRunConfig(
            model=_small_model_config(model_module.GrugModelConfig, vocab_size=vocab_size, seq_len=seq_len),
            data=data_config,
            resources=ResourceConfig.with_cpu(),
            trainer=train_module.GrugTrainerConfig(
                trainer=trainer_config,
                log_every=1,
                backward_flow=train_module.BackwardFlowConfig(interval=0),
            ),
            eval=train_module.GrugEvalConfig(
                eval_batch_size=1,
                steps_per_eval=1,
                max_eval_batches=1,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            ),
        )
        train_module.run_grug(run_cfg)
    finally:
        logger.removeHandler(handler)

    records = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    finish_records = [record for record in records if record.get("event") == "finish"]
    assert len(finish_records) == 1
    summary = finish_records[0]["summary"]

    required_keys = [
        "train/loss",
        "global_step",
        "throughput/duration",
        "throughput/hook_time",
        "throughput/loading_time",
        "throughput/total_tokens",
        "throughput/examples_per_second",
        "throughput/tokens_per_second",
        "throughput/flops_per_example_analytic",
        "eval/loss",
        "eval/loading_time",
        "eval/total_time",
    ]
    for key in required_keys:
        assert key in summary
