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
import optax
import pytest
from fray.cluster import ResourceConfig
from haliax.partitioning import set_mesh
from jax._src import config as jax_config
from jax.sharding import use_abstract_mesh
from levanter.checkpoint import CheckpointerConfig
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text import DatasetComponent, DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig
from levanter.grug.attention import AttentionMask as GrugAttentionMask
from levanter.grug.sharding import _compact_grug_mesh_shape, compact_grug_mesh
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


def test_reentrant_recurrent_config_lowers_and_keeps_router_stats_per_unique_block():
    """The re-entrant variant must lower a full train step with recurrence enabled,
    and (the load-bearing invariant) keep ``qb_beta_per_layer`` 1:1 with the unique
    blocks even though the weight-tied core is applied multiple times per forward.

    A mismatch here would break ``_apply_qb_betas`` (which maps betas positionally
    onto ``model.blocks``) and silently corrupt the QB load-balancing bias update.
    """
    train_module = importlib.import_module("experiments.grug.reentrant.train")
    model_module = importlib.import_module("experiments.grug.reentrant.model")

    # 1 prelude + 1 weight-tied core looped 4x + 1 coda => 3 unique blocks, effective depth 6.
    cfg = model_module.GrugModelConfig(
        vocab_size=1024,
        num_layers=3,
        num_prelude_layers=1,
        num_coda_layers=1,
        recurrence_steps=4,
    )
    assert cfg.num_core_layers == 1
    assert cfg.effective_depth == 6

    optimizer = optax.adam(1e-2)
    mp = jmp.get_policy("f32")
    train_step = train_module._make_train_step(optimizer, mp, z_loss_weight=0.0, ema_beta=None)
    mesh, token_pspec = model_module.debug_mesh_and_token_pspec(num_devices=4)
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
        state = train_module.initial_state(cfg, optimizer=optimizer, mp=mp, key=jax.random.PRNGKey(0), ema_beta=None)
        return train_step(state, sharded_batch, compute_watch=False)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        out_state_shape, out_metrics_shape, _ = eqx.filter_eval_shape(one_step)

    assert out_metrics_shape["train/loss"].shape == ()
    # One QB beta row per unique block (3), NOT per block application (6).
    assert out_metrics_shape["qb_beta_per_layer"].shape[0] == cfg.num_layers
    # pending_qb_betas in the next state must match the unique-block router biases.
    assert out_state_shape.pending_qb_betas.shape[0] == cfg.num_layers


def test_reentrant_iteration_film_is_identity_at_init():
    """E2's per-iteration FiLM must be identity at init: a model built with
    ``iteration_film=True`` and one with ``iteration_film=False`` from the SAME
    PRNG key must produce numerically identical forward outputs.

    FiLM is initialized to zeros and consumes no PRNG key, so every other param is
    bit-identical and ``x * (1 + 0) + 0 == x``. This is the contract that lets E2 be
    a strict superset of E1 (it can only diverge once the FiLM tables learn).
    """
    model_module = importlib.import_module("experiments.grug.reentrant.model")

    # Same small re-entrant config as the recurrence test: 1 prelude + 1 core looped
    # 4x + 1 coda => 3 unique blocks, effective depth 6.
    base_kwargs = dict(
        vocab_size=128,
        hidden_dim=32,
        intermediate_dim=64,
        num_layers=3,
        num_prelude_layers=1,
        num_coda_layers=1,
        recurrence_steps=4,
        num_heads=2,
        num_kv_heads=2,
        num_experts=4,
        num_experts_per_token=2,
        shared_expert_intermediate_dim=64,
        max_seq_len=8,
    )
    cfg_e1 = model_module.GrugModelConfig(iteration_film=False, **base_kwargs)
    cfg_e2 = model_module.GrugModelConfig(iteration_film=True, **base_kwargs)

    # Concrete mesh so we can compare actual forward values (not just shapes). The
    # default expert_axis_size=1 mesh works on a single CPU device; FiLM identity
    # holds independent of mesh shape.
    mesh = compact_grug_mesh()
    key = jax.random.PRNGKey(0)
    tokens = jnp.arange(2 * 8, dtype=jnp.int32).reshape(2, 8) % base_kwargs["vocab_size"]
    token_pspec = jax.sharding.PartitionSpec(("replica_dcn", "data", "expert"), None)

    def forward(cfg):
        sharded_tokens = jax.sharding.reshard(tokens, token_pspec)
        model = model_module.Transformer.init(cfg, key=key)
        return model, jax.jit(lambda m, t: m.logits(t))(model, sharded_tokens)

    with _reset_abstract_mesh(), set_mesh(mesh):
        model_e1, logits_e1 = forward(cfg_e1)
        model_e2, logits_e2 = forward(cfg_e2)

    # E1 carries no FiLM tables; E2 carries zero-initialized ones of the expected shape.
    assert model_e1.core_film_scale is None
    assert model_e1.core_film_shift is None
    assert model_e2.core_film_scale is not None
    assert model_e2.core_film_shift is not None
    expected_film_shape = (cfg_e2.recurrence_steps, cfg_e2.num_core_layers, cfg_e2.hidden_dim)
    assert model_e2.core_film_scale.shape == expected_film_shape
    assert model_e2.core_film_shift.shape == expected_film_shape

    # Identity FiLM => bit-identical forward outputs from the same key.
    assert jnp.array_equal(logits_e1, logits_e2)
