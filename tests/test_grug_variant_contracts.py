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
from jax._src import config as jax_config
from jax.sharding import use_abstract_mesh
from marin.execution.executor import VersionedValue

from levanter.checkpoint import CheckpointerConfig
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig, RayConfig
from levanter.grug.attention import AttentionMask as GrugAttentionMask
from levanter.optim import AdamConfig
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
            ray=RayConfig(auto_start_cluster=False),
            log_dir=variant_tmp / "logs",
            checkpointer=CheckpointerConfig(base_path=str(variant_tmp / "checkpoints")),
        )

        run_cfg = train_module.GrugRunConfig(
            model=_small_model_config(model_module.GrugModelConfig, vocab_size=vocab_size, seq_len=seq_len),
            data=data_config,
            resources=ResourceConfig.with_cpu(),
            trainer=train_module.GrugTrainerConfig(trainer=trainer_config, log_every=1),
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


def test_grug_iteration_02_v5p8_uses_expert_mesh_axis_one() -> None:
    launch_module = importlib.import_module("experiments.grug.moe_scaling_iteration_02.launch")

    assert launch_module._mesh_expert_axis(ResourceConfig.with_tpu("v5p-8")) == 1
    assert launch_module._mesh_expert_axis(ResourceConfig.with_tpu("v5p-16")) == 1
    assert launch_module._mesh_expert_axis(ResourceConfig.with_tpu("v5litepod-16")) == 4


def test_grug_iteration_02_gated_norm_one_step_contract_lowers() -> None:
    train_module = importlib.import_module("experiments.grug.moe_scaling_iteration_02.train")
    model_module = importlib.import_module("experiments.grug.moe_scaling_iteration_02.model")

    cfg = dataclasses.replace(
        _small_model_config(model_module.GrugModelConfig, vocab_size=1024, seq_len=32),
        gated_norm_rank=8,
        num_dense_layers=1,
    )
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
        state = train_module.initial_state(
            cfg,
            optimizer=optimizer,
            mp=mp,
            key=jax.random.PRNGKey(0),
        )
        return train_step(state, sharded_batch, compute_watch=False)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        out_state_shape, out_metrics_shape, out_watch_shape = eqx.filter_eval_shape(one_step)

    assert out_state_shape.step.shape == ()
    assert "train/loss" in out_metrics_shape
    assert out_metrics_shape["train/loss"].shape == ()
    assert out_watch_shape is None


def test_grug_iteration_02_launch_preserves_trainer_parallelism_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launch_module = importlib.import_module("experiments.grug.moe_scaling_iteration_02.launch")

    captured: dict[str, object] = {}

    def fake_run_grug(run_config) -> None:
        captured["run_config"] = run_config

    monkeypatch.setattr(launch_module, "run_grug", fake_run_grug)

    trainer_override = TrainerConfig(
        per_device_parallelism=2,
        per_device_eval_parallelism=8,
        distributed=DistributedConfig(initialize_jax_distributed=False),
        ray=RayConfig(auto_start_cluster=False),
        require_accelerator=False,
    )
    config = launch_module.GrugMoeLaunchConfig(
        model=launch_module._build_model_config(512),
        data=launch_module.NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=str(tmp_path),
        run_id="test-lowparallelism-override",
        resources=ResourceConfig.with_tpu("v5p-8"),
        steps=123,
        batch_size=64,
        seed=7,
        mp="params=float32,compute=bfloat16,output=bfloat16",
        tracker=JsonLoggerConfig(logger_name="test_grug_launch_parallelism"),
        optimizer=AdamConfig(learning_rate=1e-3),
        grug_trainer=launch_module.GrugTrainerConfig(trainer=trainer_override, log_every=1),
    )

    launch_module.run_grug_moe_trial(config)

    run_config = captured["run_config"]
    assert run_config.trainer.trainer.per_device_parallelism == 2
    assert run_config.trainer.trainer.per_device_eval_parallelism == 8
    assert run_config.trainer.trainer.train_batch_size == 64
    assert run_config.trainer.trainer.num_train_steps == 123


def test_grug_iteration_02_v4_256_scaleup_launcher_bakes_expected_schedule() -> None:
    launch_module = importlib.import_module(
        "experiments.grug.moe_scaling_iteration_02.launch_isoflop_moe_adamh_gatednorm_v4_256_1e21_d2304_scaleup"
    )

    step = launch_module.scaleup_step
    config = step.config

    assert config.run_id == "isoflop-moe-adamh-gatednorm-v4-256-scaleup-r1-1e21-d2304"
    assert isinstance(config.resources, VersionedValue)
    assert config.resources.value.device.variant == "v4-256"
    assert isinstance(config.batch_size, VersionedValue)
    assert config.batch_size.value == 2048
    assert isinstance(config.steps, VersionedValue)
    assert config.steps.value == 9286
    assert isinstance(config.grug_trainer, VersionedValue)
    assert config.grug_trainer.value.trainer.per_device_parallelism == 2
    assert isinstance(config.eval, VersionedValue)
    assert config.eval.value.eval_batch_size == 64
    assert isinstance(config.model, VersionedValue)
    assert config.model.value.hidden_dim == 2304
    assert config.model.value.gated_norm_rank == 16


def test_grug_iteration_02_v5p256_h2h_launcher_bakes_expected_schedule() -> None:
    launch_module = importlib.import_module(
        "experiments.grug.moe_scaling_iteration_02.launch_isoflop_moe_adamh_gatednorm_v5p256_1e21_d2304_h2h"
    )

    step = launch_module.scaleup_step
    config = step.config

    assert config.run_id == "isoflop-moe-adamh-gatednorm-v5p256-h2h-r1-1e21-d2304"
    assert isinstance(config.resources, VersionedValue)
    assert config.resources.value.device.variant == "v5p-256"
    assert config.resources.value.regions == ("us-central1",)
    assert isinstance(config.batch_size, VersionedValue)
    assert config.batch_size.value == 512
    assert isinstance(config.steps, VersionedValue)
    assert config.steps.value == 35802
    assert isinstance(config.grug_trainer, VersionedValue)
    assert config.grug_trainer.value.trainer.per_device_parallelism == 2
    assert config.grug_trainer.value.trainer.per_device_eval_parallelism == 1
    assert isinstance(config.eval, VersionedValue)
    assert config.eval.value.eval_batch_size == 128
    assert isinstance(config.model, VersionedValue)
    assert config.model.value.hidden_dim == 2304
    assert config.model.value.num_layers == 24
    assert config.model.value.gated_norm_rank == 16


def test_grug_iteration_02_v5p64_h2h_launcher_bakes_expected_schedule() -> None:
    launch_module = importlib.import_module(
        "experiments.grug.moe_scaling_iteration_02.launch_isoflop_moe_adamh_gatednorm_v5p64_1e21_d2304_h2h"
    )

    step = launch_module.scaleup_step
    config = step.config

    assert config.run_id == "isoflop-moe-adamh-gatednorm-v5p64-h2h-r1-1e21-d2304"
    assert isinstance(config.resources, VersionedValue)
    assert config.resources.value.device.variant == "v5p-64"
    assert config.resources.value.regions == ("us-central1",)
    assert isinstance(config.batch_size, VersionedValue)
    assert config.batch_size.value == 512
    assert isinstance(config.steps, VersionedValue)
    assert config.steps.value == 35802
    assert isinstance(config.grug_trainer, VersionedValue)
    assert config.grug_trainer.value.trainer.per_device_parallelism == 2
    assert config.grug_trainer.value.trainer.per_device_eval_parallelism == 1
    assert isinstance(config.eval, VersionedValue)
    assert config.eval.value.eval_batch_size == 128
    assert isinstance(config.model, VersionedValue)
    assert config.model.value.hidden_dim == 2304
    assert config.model.value.num_layers == 24
    assert config.model.value.gated_norm_rank == 16


def test_grug_iteration_02_v5p64_1e20_d1536_launcher_schedule_is_valid() -> None:
    launch_module = importlib.import_module(
        "experiments.grug.moe_scaling_iteration_02.launch_isoflop_moe_adamh_gatednorm_v5p64_1e20_d1536"
    )

    step = launch_module.scaleup_step
    config = step.config

    assert config.run_id == "isoflop-moe-adamh-gatednorm-v5p64-r1-1e20-d1536"
    assert isinstance(config.resources, VersionedValue)
    assert config.resources.value.device.variant == "v5p-64"
    assert config.resources.value.regions == ("us-central1",)
    assert isinstance(config.batch_size, VersionedValue)
    assert config.batch_size.value == 256
    assert isinstance(config.steps, VersionedValue)
    assert config.steps.value == 17822
    assert isinstance(config.grug_trainer, VersionedValue)
    assert config.grug_trainer.value.trainer.per_device_parallelism == 2
    assert config.grug_trainer.value.trainer.per_device_eval_parallelism == 1
    assert isinstance(config.eval, VersionedValue)
    assert config.eval.value.eval_batch_size == 128
    assert isinstance(config.model, VersionedValue)
    assert config.model.value.hidden_dim == 1536
    assert config.model.value.num_layers == 16
    assert config.model.value.gated_norm_rank == 16
