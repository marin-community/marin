# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for grug variants under experiments/grug/*.

These checks are intentionally variant-discovered: if a subdirectory contains
`model.py` and/or `train.py`, it is expected to satisfy the corresponding
lowering and training contracts.
"""

import dataclasses
import importlib
import json
import logging
import uuid
from io import StringIO
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
from fray.cluster import ResourceConfig
from jax._src import config as jax_config
from jax.sharding import use_abstract_mesh
from levanter.checkpoint import CheckpointerConfig
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DatasetComponent, DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig
from levanter.grug.attention import AttentionMask as GrugAttentionMask
from levanter.grug.attention._fa4_thd import _jax_fa4_thd_attention, _thd_kernel_config
from levanter.grug.sharding import _compact_grug_mesh_shape, compact_grug_mesh
from levanter.schedule import BatchSchedule
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import TrainerConfig
from marin.execution.artifact import ArtifactRecord, write_record
from marin.execution.lazy import materialized_config
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.ferries import canary_ferry
from experiments.grug.moe import launch_nested_experts
from experiments.llama import llama3_tokenizer

_TOKENIZED_CACHE = f"{TokenizedCache.__module__}.{TokenizedCache.__qualname__}"


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

    assert short_mask.sliding_window == 12
    assert long_mask.sliding_window is None
    assert short_mask.thd_segment_metadata is mask.thd_segment_metadata
    assert long_mask.thd_segment_metadata is mask.thd_segment_metadata
    assert short_mask.segment_ids is mask.segment_ids
    assert long_mask.segment_ids is mask.segment_ids


def test_grug_moe_xsa_forward_lowers_with_gpu_fa4_thd_gqa_sharding():
    if jax.default_backend() != "gpu":
        pytest.skip("gpu_fa4_thd requires the JAX GPU backend")

    model_module = importlib.import_module("experiments.grug.moe.model")
    mesh, _ = model_module.debug_mesh_and_token_pspec(num_devices=8)
    cfg = model_module.GrugModelConfig(
        vocab_size=128,
        hidden_dim=512,
        intermediate_dim=64,
        shared_expert_intermediate_dim=64,
        num_layers=1,
        num_heads=4,
        num_kv_heads=1,
        max_seq_len=8,
        sliding_window=8,
        num_experts=4,
        num_experts_per_token=2,
        attention_implementation="gpu_fa4_thd",
    )

    def forward():
        attn = model_module.CausalSelfAttention.init(cfg, key=jax.random.PRNGKey(0))
        x = jax.sharding.reshard(
            jnp.zeros((8, 16, cfg.hidden_dim), dtype=jnp.bfloat16),
            jax.sharding.PartitionSpec(("replica_dcn", "data", "expert"), None, None),
        )
        segment_ids = jnp.zeros((8, 16), dtype=jnp.int32)
        mask = GrugAttentionMask.causal().with_segment_ids(segment_ids, max_segments=1)
        return attn(x, mask)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        out_shape = eqx.filter_eval_shape(forward)

    assert out_shape.shape == (8, 16, cfg.hidden_dim)


def test_grug_fa4_thd_forward_and_backward_compile_on_gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("gpu_fa4_thd requires the JAX GPU backend")

    tokens = 128
    q = jnp.zeros((tokens, 4, 128), dtype=jnp.bfloat16)
    k = jnp.zeros((tokens, 1, 128), dtype=jnp.bfloat16)
    v = jnp.zeros((tokens, 1, 128), dtype=jnp.bfloat16)
    cu_seqlens = jnp.array([0, tokens], dtype=jnp.int32)
    kernel_config = _thd_kernel_config(128)

    def loss(q, k, v):
        out = _jax_fa4_thd_attention(q, k, v, cu_seqlens, 128**-0.5, kernel_config, None)
        return jnp.sum(out.astype(jnp.float32))

    value, gradients = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))(q, k, v)
    assert jnp.isfinite(value)
    assert all(jnp.all(jnp.isfinite(gradient)) for gradient in gradients)


def _seed_cache_records(step, prefix: str) -> None:
    """Write the minimal record a built ``TokenizedCache`` dep would leave, so the run-time
    ``mixture`` can read each dataset's tokenizer/format offline (mirrors a real run, where the
    datasets materialize first as build dependencies)."""
    for dep in step.deps:
        if dep.artifact_type is TokenizedCache:
            write_record(
                ArtifactRecord(
                    name=dep.name,
                    version=dep.version,
                    output_path=dep.path(prefix),
                    result_type=_TOKENIZED_CACHE,
                    config={"tokenizer": llama3_tokenizer, "format": {"text_key": "text"}},
                )
            )


def test_coreweave_thd_canary_uses_fixed_shape_training_segments(monkeypatch, tmp_path):
    monkeypatch.setenv("CANARY_ACCELERATOR", "gpu")
    monkeypatch.setenv("CANARY_ATTENTION_IMPLEMENTATION", "gpu_fa4_thd")
    monkeypatch.setenv("CANARY_TRACKER", "json_logger")
    monkeypatch.setenv("RUN_ID", "test-thd")
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))

    # build() reads the env at call time, so set it above before resolving the config.
    step = canary_ferry.build()
    _seed_cache_records(step, str(tmp_path))
    data = materialized_config(step, str(tmp_path)).data

    components = list(data.components.values())
    assert components
    assert all(isinstance(component, DatasetComponent) for component in components)
    assert {component.pack for component in components} == {1}


def test_nested_moe_launcher_uses_fixed_shape_thd_segments(monkeypatch, tmp_path):
    monkeypatch.setenv("NESTED_ARM", "nested25")
    monkeypatch.setenv("NESTED_PHASE", "smoke")
    monkeypatch.setenv("NESTED_ATTENTION", "gpu_fa4_thd")
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))

    step = launch_nested_experts.build(version="dev")
    _seed_cache_records(step, str(tmp_path))
    data = materialized_config(step, str(tmp_path)).data

    components = list(data.components.values())
    assert components
    assert all(isinstance(component, DatasetComponent) for component in components)
    assert {component.pack for component in components} == {1}


def test_nested_moe_launcher_reference_attention_uses_causal_examples(monkeypatch, tmp_path):
    monkeypatch.setenv("NESTED_ARM", "nested25")
    monkeypatch.setenv("NESTED_PHASE", "smoke")
    monkeypatch.setenv("NESTED_ATTENTION", "reference")
    monkeypatch.setenv("NESTED_HIDDEN_DIM", "768")
    monkeypatch.setenv("NESTED_SEQUENCE_LENGTH", "2048")
    monkeypatch.setenv("NESTED_BATCH", "1024")
    monkeypatch.setenv("NESTED_CAPACITY_FACTOR", "1.25")
    monkeypatch.setenv("NESTED_RUN_SUFFIX", "finite")
    monkeypatch.setenv("NESTED_MP", "params=float32,compute=float32,output=float32")
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))

    assert launch_nested_experts._attention_implementation() == "reference"
    step = launch_nested_experts.build(version="dev")
    _seed_cache_records(step, str(tmp_path))
    config = materialized_config(step, str(tmp_path))
    data = config.data

    components = list(data.components.values())
    assert components
    assert all(isinstance(component, DatasetComponent) for component in components)
    assert {component.pack for component in components} == {None}
    assert config.model.hidden_dim == 768
    assert config.model.max_seq_len == 2048
    assert config.batch_size == 1024
    assert config.model.capacity_factor == 1.25
    assert config.optimizer.warmup == 5
    assert config.run_id.endswith("-finite")
    assert config.mp == "params=float32,compute=float32,output=float32"


def test_nested_moe_launcher_evaluates_untreated_control_subset(monkeypatch, tmp_path):
    monkeypatch.setenv("NESTED_ARM", "large")
    monkeypatch.setenv("NESTED_PHASE", "smoke")
    monkeypatch.setenv("NESTED_EVAL_EXPERTS", "128")
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))

    step = launch_nested_experts.build(version="dev")
    _seed_cache_records(step, str(tmp_path))
    config = materialized_config(step, str(tmp_path))

    assert config.model.num_experts == 256
    assert config.model.nested_expert_count == 128
    assert config.model.nested_batch_fraction == 0.0


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
        tokens=jnp.zeros((32, 4), dtype=jnp.int32),
        loss_weight=jnp.ones((32, 4), dtype=jnp.float32),
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


def _nested_grug_model_config(model_module):
    return model_module.GrugModelConfig(
        vocab_size=48,
        hidden_dim=16,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_experts=8,
        num_experts_per_token=2,
        nested_expert_count=4,
        nested_batch_fraction=0.5,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=8,
        sliding_window=8,
        moe_implementation="ring",
    )


def test_grug_moe_nested_experts_are_interleaved_across_expert_ranks():
    model_module = importlib.import_module("experiments.grug.moe.model")

    eligible = np.asarray(model_module.nested_expert_eligibility(8, 4))

    assert eligible.tolist() == [True, False, True, False, True, False, True, False]
    assert np.all(eligible.reshape(4, 2).sum(axis=1) == 1)


def test_grug_moe_extracted_nested_model_matches_restricted_forward():
    model_module = importlib.import_module("experiments.grug.moe.model")
    config = _nested_grug_model_config(model_module)
    tokens = jnp.arange(16, dtype=jnp.int32).reshape(2, 8) % config.vocab_size
    nested_rows = jnp.ones((tokens.shape[0],), dtype=jnp.bool_)

    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)):
        model = model_module.Transformer.init(config, key=jax.random.PRNGKey(0))
        restricted_logits = model.logits(tokens, nested_rows=nested_rows)
        extracted_model = model_module.extract_nested_expert_model(model)
        extracted_logits = extracted_model.logits(tokens)

    np.testing.assert_allclose(restricted_logits, extracted_logits, atol=1e-5, rtol=1e-5)
    assert extracted_model.config.num_experts == config.nested_expert_count
    assert extracted_model.config.nested_expert_count is None


def test_grug_moe_full_rows_preserve_router_and_reach_both_expert_banks():
    model_module = importlib.import_module("experiments.grug.moe.model")
    config = _nested_grug_model_config(model_module)
    assert config.nested_expert_count is not None
    x = jax.random.normal(jax.random.PRNGKey(1), (8, 8, config.hidden_dim))
    full_rows = jnp.zeros((x.shape[0],), dtype=jnp.bool_)
    nested_experts = np.asarray(model_module.nested_expert_eligibility(config.num_experts, config.nested_expert_count))

    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)):
        mlp = model_module.MoEMLP.init(config, key=jax.random.PRNGKey(2))
        original_out, original_stats = mlp(x)
        full_out, full_stats = mlp(x, nested_rows=full_rows)
        grads = jax.grad(lambda candidate: jnp.sum(candidate(x, nested_rows=full_rows)[0]))(mlp)

    np.testing.assert_array_equal(original_out, full_out)
    np.testing.assert_array_equal(original_stats["routing_counts"], full_stats["routing_counts"])
    assert np.any(np.asarray(grads.expert_mlp.w_gate)[nested_experts] != 0)
    assert np.any(np.asarray(grads.expert_mlp.w_gate)[~nested_experts] != 0)
    assert np.any(np.asarray(grads.router)[:, nested_experts] != 0)
    assert np.any(np.asarray(grads.router)[:, ~nested_experts] != 0)


def test_grug_moe_nested_forward_has_no_outer_expert_gradients():
    model_module = importlib.import_module("experiments.grug.moe.model")
    config = _nested_grug_model_config(model_module)
    assert config.nested_expert_count is not None
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 4, config.hidden_dim))
    nested_rows = jnp.ones((x.shape[0],), dtype=jnp.bool_)
    nested_experts = np.asarray(model_module.nested_expert_eligibility(config.num_experts, config.nested_expert_count))

    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)):
        mlp = model_module.MoEMLP.init(config, key=jax.random.PRNGKey(2))
        grads = jax.grad(lambda candidate: jnp.sum(candidate(x, nested_rows=nested_rows)[0]))(mlp)

    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree.leaves(grads))
    assert np.any(np.asarray(grads.expert_mlp.w_gate)[nested_experts] != 0)
    assert np.all(np.asarray(grads.expert_mlp.w_gate)[~nested_experts] == 0)
    assert np.any(np.asarray(grads.router)[:, nested_experts] != 0)
    assert np.all(np.asarray(grads.router)[:, ~nested_experts] == 0)


def test_grug_moe_nested_row_schedule_is_fixed_and_step_balanced():
    model_module = importlib.import_module("experiments.grug.moe.model")
    train_module = importlib.import_module("experiments.grug.moe.train")
    config = _nested_grug_model_config(model_module)

    step_zero = train_module._training_nested_rows(config, batch_size=8, step=jnp.array(0))
    step_one = train_module._training_nested_rows(config, batch_size=8, step=jnp.array(1))

    assert step_zero is not None
    assert step_one is not None
    assert step_zero.tolist() == [True, False, True, False, True, False, True, False]
    assert step_one.tolist() == [False, True, False, True, False, True, False, True]


def test_grug_moe_nested_train_step_lowers():
    model_module = importlib.import_module("experiments.grug.moe.model")
    train_module = importlib.import_module("experiments.grug.moe.train")
    config = _nested_grug_model_config(model_module)
    optimizer = optax.adam(1e-2)
    mp = jmp.get_policy("f32")
    train_step = train_module._make_train_step(optimizer, mp, z_loss_weight=0.0, ema_beta=None)
    mesh, token_pspec = model_module.debug_mesh_and_token_pspec(num_devices=4)
    batch = GrugLmExample(
        tokens=jnp.zeros((32, 8), dtype=jnp.int32),
        loss_weight=jnp.ones((32, 8), dtype=jnp.float32),
        attn_mask=GrugAttentionMask.causal(),
    )

    def one_step():
        sharded_batch = dataclasses.replace(
            batch,
            tokens=jax.sharding.reshard(batch.tokens, token_pspec),
            loss_weight=jax.sharding.reshard(batch.loss_weight, token_pspec),
        )
        state = train_module.initial_state(
            config,
            optimizer=optimizer,
            mp=mp,
            key=jax.random.PRNGKey(0),
            ema_beta=None,
        )
        return train_step(state, sharded_batch, compute_watch=False)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        next_state, metrics, watch_stats = eqx.filter_eval_shape(one_step)

    assert next_state.step.shape == ()
    assert metrics["train/nested/sequence_fraction"].shape == ()
    assert metrics["train/router/nested_assignment_cv"].shape == ()
    assert metrics["train/router/outer_assignment_cv"].shape == ()
    assert watch_stats is None


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
