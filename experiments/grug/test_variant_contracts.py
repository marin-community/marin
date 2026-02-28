# Copyright 2025 The Marin Authors
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
from jax._src import config as jax_config
from jax.sharding import NamedSharding, PartitionSpec as P, use_abstract_mesh

from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig, RayConfig
from levanter.grug.attention import AttentionMask as GrugAttentionMask
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import TrainerConfig


def _discover_grug_variants_with_file(filename: str) -> list[str]:
    grug_dir = Path(__file__).resolve().parent
    variants: list[str] = []
    for child in sorted(grug_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("__"):
            continue
        if (child / filename).is_file():
            variants.append(child.name)
    if not variants:
        raise AssertionError(f"No grug variants with {filename} found under {grug_dir}")
    return variants


def _variant_module_name(variant: str, module: str) -> str:
    return f"experiments.grug.{variant}.{module}"


class _reset_abstract_mesh:
    def __enter__(self):
        self._prev = jax_config.abstract_mesh_context_manager.swap_local(jax_config.config_ext.unset)
        return self

    def __exit__(self, exc_type, exc, tb):
        jax_config.abstract_mesh_context_manager.set_local(self._prev)
        return False


def _infer_loss_fn_name(params) -> str:
    if hasattr(params, "compute_next_token_loss"):
        return "compute_next_token_loss"
    if hasattr(params, "next_token_loss"):
        return "next_token_loss"
    raise AssertionError("Transformer variant must define either compute_next_token_loss or next_token_loss")


@pytest.mark.parametrize(
    "variant",
    _discover_grug_variants_with_file("model.py"),
)
def test_grug_variant_loss_lowers_on_abstract_mesh(variant: str):
    module_name = _variant_module_name(variant, "model")
    module = importlib.import_module(module_name)
    config_cls = module.GrugModelConfig
    transformer_cls = module.Transformer

    seq = 256 if jax.default_backend() == "tpu" else 16
    cfg = config_cls(vocab_size=256, max_seq_len=seq)
    mesh_fn = getattr(module, "debug_mesh_and_token_pspec", None)
    if mesh_fn is None:
        raise AssertionError(f"{module_name} must define debug_mesh_and_token_pspec(num_devices)")
    mesh, token_pspec = mesh_fn(num_devices=4)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        key = jax.ShapeDtypeStruct(shape=(2,), dtype=jnp.uint32, sharding=NamedSharding(mesh, P()))

        def init_model(k):
            return transformer_cls.init(cfg, key=k)

        params = jax.eval_shape(init_model, key)
        loss_fn_name = _infer_loss_fn_name(params)

        def loss_fn(p):
            token_ids = jnp.zeros((8, seq), dtype=jnp.int32)
            token_ids = jax.sharding.reshard(token_ids, token_pspec)
            loss_weight = jnp.ones((8, seq), dtype=jnp.float32)
            loss_weight = jax.sharding.reshard(loss_weight, token_pspec)
            if loss_fn_name == "compute_next_token_loss":
                return p.compute_next_token_loss(
                    token_ids,
                    loss_weight,
                    mask=GrugAttentionMask.causal(),
                    reduction="mean",
                )
            return p.next_token_loss(
                token_ids,
                loss_weight,
                mask=GrugAttentionMask.causal(),
                reduction="mean",
            )

        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = jax.jit(loss_fn).trace(params).lower(lowering_platforms=(platform,))

    assert lowered is not None


class DummyModel(eqx.Module):
    w: jax.Array

    def compute_next_token_loss(
        self,
        token_ids: jax.Array,
        loss_weight: jax.Array,
        *,
        mask=None,
        reduction: str = "mean",
        logsumexp_weight: float | None = None,
    ) -> jax.Array:
        del token_ids, loss_weight, mask, reduction, logsumexp_weight
        return jnp.mean(jnp.square(self.w))

    def next_token_loss(
        self,
        token_ids: jax.Array,
        loss_weight: jax.Array,
        *,
        mask=None,
        reduction: str = "mean",
        logsumexp_weight: float | None = None,
    ) -> jax.Array:
        return self.compute_next_token_loss(
            token_ids,
            loss_weight,
            mask=mask,
            reduction=reduction,
            logsumexp_weight=logsumexp_weight,
        )


def _build_state(train_state_cls, params: DummyModel, optimizer: optax.GradientTransformation):
    return train_state_cls(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        opt_state=optimizer.init(params),
        ema_params=params,
    )


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
    _discover_grug_variants_with_file("train.py"),
)
def test_grug_variant_train_step_with_watch_matches_base_step(variant: str):
    module = importlib.import_module(_variant_module_name(variant, "train"))
    make_train_step = module._make_train_step
    train_state_cls = module.GrugTrainState

    optimizer = optax.adam(1e-2)
    mp = jmp.get_policy("f32")

    state_for_base = _build_state(train_state_cls, DummyModel(jnp.array([1.0, -2.0], dtype=jnp.float32)), optimizer)
    state_for_watch = _build_state(train_state_cls, DummyModel(jnp.array([1.0, -2.0], dtype=jnp.float32)), optimizer)
    batch = GrugLmExample(
        tokens=jnp.zeros((1, 4), dtype=jnp.int32),
        loss_weight=jnp.ones((1, 4), dtype=jnp.float32),
        attn_mask=GrugAttentionMask.causal(),
    )

    base_step = make_train_step(optimizer, mp, z_loss_weight=0.0, ema_beta=None)
    watch_step = make_train_step(
        optimizer,
        mp,
        z_loss_weight=0.0,
        ema_beta=None,
        watch_config=WatchConfig(
            watch_targets=["grads", "params", "updates"],
            include_norms=True,
            include_per_parameter_norms=True,
            include_histograms=False,
            split_scan_layers=True,
            interval=1,
        ),
    )

    next_base, metrics_base, base_watch_stats = base_step(state_for_base, batch, compute_watch=False)
    next_watch, metrics_watch, watch_stats = watch_step(state_for_watch, batch, compute_watch=True)

    assert int(next_base.step) == 1
    assert int(next_watch.step) == 1
    assert jnp.allclose(next_base.params.w, next_watch.params.w)
    assert jnp.allclose(next_base.ema_params.w, next_watch.ema_params.w)
    assert jnp.allclose(metrics_base["train/loss"], metrics_watch["train/loss"])
    assert base_watch_stats is None
    assert watch_stats
    assert any(key.startswith("grad/") for key in watch_stats)
    assert any(key.startswith("params/") for key in watch_stats)
    assert any(key.startswith("updates/") for key in watch_stats)


def test_grug_base_run_emits_expected_metrics_with_json_tracker(tmp_path: Path):
    train_module = importlib.import_module("experiments.grug.base.train")
    model_module = importlib.import_module("experiments.grug.base.model")

    vocab_size = 128
    seq_len = 32
    examples = []
    for i in range(8):
        tokens = (jnp.arange(seq_len, dtype=jnp.int32) + i) % vocab_size
        examples.append(GrugLmExample.causal(tokens))

    dataset = ListAsyncDataset(examples)
    data_config = LmDataConfig(
        components={"direct": DirectDatasetComponent(datasets={"train": dataset})},
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
            trainer=train_module.GrugTrainerConfig(trainer=trainer_config, log_every=1),
            eval=None,
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
    ]
    for key in required_keys:
        assert key in summary
