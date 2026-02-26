# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import tempfile
import uuid
from io import StringIO
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import optax

from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig, RayConfig
from levanter.grug.attention import AttentionMask as GrugAttentionMask
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import TrainerConfig

from experiments.grug.base.model import GrugModelConfig, Transformer
from experiments.grug.base.train import (
    GrugRunConfig,
    GrugTrainerConfig,
    GrugTrainState,
    _compute_flops,
    _make_train_step,
    run_grug,
)


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


def _build_state(params: DummyModel, optimizer: optax.GradientTransformation) -> GrugTrainState:
    return GrugTrainState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        opt_state=optimizer.init(params),
        ema_params=params,
    )


def test_grug_base_train_step_with_watch_matches_base_step():
    optimizer = optax.adam(1e-2)
    mp = jmp.get_policy("f32")

    state_for_base = _build_state(DummyModel(jnp.array([1.0, -2.0], dtype=jnp.float32)), optimizer)
    state_for_watch = _build_state(DummyModel(jnp.array([1.0, -2.0], dtype=jnp.float32)), optimizer)
    batch = GrugLmExample(
        tokens=jnp.zeros((1, 4), dtype=jnp.int32),
        loss_weight=jnp.ones((1, 4), dtype=jnp.float32),
        attn_mask=GrugAttentionMask.causal(),
    )

    base_step = _make_train_step(optimizer, mp, z_loss_weight=0.0, ema_beta=None)
    watch_step = _make_train_step(
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


def test_grug_base_run_emits_expected_metrics_with_json_tracker():
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

    logger_name = f"test_grug_base_json_tracker_{uuid.uuid4().hex}"
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer_config = TrainerConfig(
            id="test-grug-base-metrics",
            num_train_steps=1,
            train_batch_size=max(1, len(jax.devices())),
            tracker=JsonLoggerConfig(logger_name=logger_name),
            require_accelerator=False,
            use_explicit_mesh_axes=True,
            distributed=DistributedConfig(initialize_jax_distributed=False),
            ray=RayConfig(auto_start_cluster=False),
            log_dir=Path(tmpdir) / "logs",
            checkpointer=CheckpointerConfig(base_path=str(Path(tmpdir) / "checkpoints")),
        )

        run_grug(
            GrugRunConfig(
                model=GrugModelConfig(
                    vocab_size=vocab_size,
                    hidden_dim=32,
                    intermediate_dim=64,
                    num_layers=2,
                    num_heads=2,
                    num_kv_heads=2,
                    max_seq_len=seq_len,
                ),
                data=data_config,
                trainer=GrugTrainerConfig(trainer=trainer_config, log_every=1),
                eval=None,
            )
        )

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


def test_compute_flops_emits_jax_metrics_with_explicit_mesh():
    cfg = GrugModelConfig(
        vocab_size=1024,
        hidden_dim=64,
        intermediate_dim=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        max_seq_len=128,
    )
    trainer = TrainerConfig(
        train_batch_size=8,
        num_train_steps=1,
        use_explicit_mesh_axes=True,
        require_accelerator=False,
    )
    mesh = trainer.device_mesh

    with trainer.use_device_mesh():
        params = Transformer.init(cfg, key=jax.random.PRNGKey(0))
        _, flops_per_example_jax, flops_summary = _compute_flops(
            model_config=cfg,
            params=params,
            mp=jmp.get_policy("f32"),
            z_loss_weight=0.0,
            mesh=mesh,
        )

    assert flops_per_example_jax is not None
    assert flops_per_example_jax > 0
    assert "throughput_jax/flops_per_example_fwd_bwd_est" in flops_summary
    assert flops_summary["throughput_jax/flops_per_example_fwd_bwd_est"] > 0
