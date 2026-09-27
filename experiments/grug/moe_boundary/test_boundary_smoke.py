# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke test: a few real training steps with the boundary operator on.

Covers the wiring the unit tests cannot: eval, checkpointing, and the training
loss actually going down with the operator active.
"""

import dataclasses
import glob
import json
import logging
import uuid
from io import StringIO

import jax
import jax.numpy as jnp
import pytest
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig
from levanter.optim.config import AdamConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import TrainerConfig

from experiments.grug.moe_boundary import train as train_module
from experiments.grug.moe_boundary.model import GrugModelConfig

_VOCAB = 128
_SEQ = 32


def _small_boundary_config() -> GrugModelConfig:
    """6-layer MoE config with the boundary operator on (paper split 2/2/2)."""
    field_names = {f.name for f in dataclasses.fields(GrugModelConfig)}
    kwargs = {
        k: v
        for k, v in {
            "vocab_size": _VOCAB,
            "hidden_dim": 32,
            "intermediate_dim": 64,
            "num_layers": 6,
            "num_heads": 2,
            "num_kv_heads": 2,
            "max_seq_len": _SEQ,
            "num_experts": 4,
            "num_experts_per_token": 2,
            "shared_expert_intermediate_dim": 64,
        }.items()
        if k in field_names
    }
    kwargs.update(boundary_operator=True, prelude_len=2, coda_len=2, injection_scale=0.707)
    return GrugModelConfig(**kwargs)


@pytest.mark.timeout(300)
def test_boundary_training_smoke_loss_decreases(tmp_path):
    """Train ~10 steps with the operator on; loss must decrease and the run must finish."""
    logger_name = f"test-grug-boundary-smoke-{uuid.uuid4().hex}"
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    seq_len = _SEQ
    vocab_size = _VOCAB
    examples = []
    for i in range(8):
        tokens = (jnp.arange(seq_len, dtype=jnp.int32) + i) % vocab_size
        examples.append(GrugLmExample.causal(tokens))
    eval_examples = [GrugLmExample.causal((jnp.arange(seq_len, dtype=jnp.int32) + 100) % vocab_size)]
    data_config = LmDataConfig(
        components={
            "direct": DirectDatasetComponent(
                datasets={"train": ListAsyncDataset(examples), "validation": ListAsyncDataset(eval_examples)}
            )
        },
        vocab_size=vocab_size,
        tokenizer="passthrough",
    )

    # 6 layers, paper split 2/2/2, operator on.
    cfg = _small_boundary_config()

    trainer_config = TrainerConfig(
        id="test-grug-boundary-smoke",
        num_train_steps=10,
        train_batch_size=max(1, len(jax.devices())),
        tracker=JsonLoggerConfig(logger_name=logger_name),
        require_accelerator=False,
        use_explicit_mesh_axes=True,
        distributed=DistributedConfig(initialize_jax_distributed=False),
        log_dir=tmp_path / "logs",
        checkpointer=CheckpointerConfig(base_path=str(tmp_path / "checkpoints")),
    )

    run_cfg = train_module.GrugRunConfig(
        model=cfg,
        data=data_config,
        resources=ResourceConfig.with_cpu(),
        trainer=train_module.GrugTrainerConfig(trainer=trainer_config, log_every=1, z_loss_weight=0.0, ema_beta=None),
        eval=train_module.GrugEvalConfig(
            eval_batch_size=1,
            steps_per_eval=5,
            max_eval_batches=1,
            eval_current=True,
            eval_ema=False,
        ),
        optimizer=AdamConfig(learning_rate=1e-3),
    )
    try:
        train_module.run_grug(run_cfg)
    finally:
        logger.removeHandler(handler)

    records = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    train_losses = [
        r["metrics"]["train/loss"] for r in records if r.get("event") == "log" and "train/loss" in r.get("metrics", {})
    ]
    assert len(train_losses) >= 2, "expected at least two logged training steps"
    assert (
        train_losses[-1] < train_losses[0]
    ), f"training loss did not decrease with the boundary operator on: {train_losses}"
    finish_records = [r for r in records if r.get("event") == "finish"]
    assert len(finish_records) == 1, "run must finish exactly once"
    assert "throughput/total_tokens" in finish_records[0]["summary"]


def _run_smoke_run(tmp_path, checkpoint_base, num_train_steps, logger_name):
    """Run the smoke config for ``num_train_steps`` steps, capturing log records."""
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    seq_len = _SEQ
    vocab_size = _VOCAB
    examples = []
    for i in range(8):
        tokens = (jnp.arange(seq_len, dtype=jnp.int32) + i) % vocab_size
        examples.append(GrugLmExample.causal(tokens))
    eval_examples = [GrugLmExample.causal((jnp.arange(seq_len, dtype=jnp.int32) + 100) % vocab_size)]
    data_config = LmDataConfig(
        components={
            "direct": DirectDatasetComponent(
                datasets={"train": ListAsyncDataset(examples), "validation": ListAsyncDataset(eval_examples)}
            )
        },
        vocab_size=vocab_size,
        tokenizer="passthrough",
    )

    cfg = _small_boundary_config()
    trainer_config = TrainerConfig(
        id="test-grug-boundary-resume",
        num_train_steps=num_train_steps,
        train_batch_size=max(1, len(jax.devices())),
        tracker=JsonLoggerConfig(logger_name=logger_name),
        require_accelerator=False,
        use_explicit_mesh_axes=True,
        distributed=DistributedConfig(initialize_jax_distributed=False),
        log_dir=tmp_path / "logs",
        checkpointer=CheckpointerConfig(base_path=str(checkpoint_base)),
    )

    run_cfg = train_module.GrugRunConfig(
        model=cfg,
        data=data_config,
        resources=ResourceConfig.with_cpu(),
        trainer=train_module.GrugTrainerConfig(trainer=trainer_config, log_every=1, z_loss_weight=0.0, ema_beta=None),
        eval=train_module.GrugEvalConfig(
            eval_batch_size=1,
            steps_per_eval=5,
            max_eval_batches=1,
            eval_current=True,
            eval_ema=False,
        ),
        optimizer=AdamConfig(learning_rate=1e-3),
    )
    try:
        train_module.run_grug(run_cfg)
    finally:
        logger.removeHandler(handler)

    records = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    return records


@pytest.mark.timeout(600)
def test_boundary_training_smoke_resume_from_checkpoint(tmp_path):
    """Train 6 steps, then resume with a longer horizon in the same checkpoint root.

    The resume path releases the initialized state to ShapeDtypeStructs before restoring
    (checkpoint restore must not stage shards on top of live init state). It must pick up the
    saved step (6), continue to the new horizon (11), and keep the loss decreasing.
    """
    checkpoint_base = tmp_path / "checkpoints"

    first = _run_smoke_run(tmp_path, checkpoint_base, num_train_steps=6, logger_name="test-grug-boundary-resume-first")
    first_steps = [r["step"] for r in first if r.get("event") == "log"]
    assert first_steps, "first run logged no steps"
    final_step = first_steps[-1]
    assert final_step >= 5, f"first run should reach at least step 5, got {final_step}"

    # Checkpoint must exist on disk before the resume run.
    saved = sorted(glob.glob(str(checkpoint_base / "**" / "step-*"), recursive=True))
    assert saved, "no step-* checkpoint written by the first run"

    second = _run_smoke_run(
        tmp_path, checkpoint_base, num_train_steps=11, logger_name="test-grug-boundary-resume-second"
    )
    second_steps = [r["step"] for r in second if r.get("event") == "log"]
    assert second_steps, "resume run logged no steps"
    # Resume must not repeat the checkpointed step: it continues past it.
    assert (
        min(second_steps) > final_step
    ), f"resume run restarted from scratch: first logged step {min(second_steps)} <= checkpointed {final_step}"
    # num_train_steps=11 trains steps 7..11, whose loss is logged with the *pre-step* state.
    assert max(second_steps) == 10, f"resume run should log through step 10, got {max(second_steps)}"

    first_losses = [
        r["metrics"]["train/loss"] for r in first if r.get("event") == "log" and "train/loss" in r.get("metrics", {})
    ]
    second_losses = [
        r["metrics"]["train/loss"] for r in second if r.get("event") == "log" and "train/loss" in r.get("metrics", {})
    ]
    assert (
        second_losses[-1] < first_losses[0]
    ), f"resumed run's final loss {second_losses[-1]} should beat the first run's initial loss {first_losses[0]}"
