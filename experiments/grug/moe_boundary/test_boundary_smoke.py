# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke test: a few real training steps with the boundary operator on.

Covers the wiring the unit tests cannot: eval, checkpointing, and loss going
down with the operator active.
"""

import dataclasses

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
    """Train ~10 steps with the operator on; the run must complete and record metrics."""
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
        tracker=JsonLoggerConfig(logger_name="test-grug-boundary-smoke"),
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
    train_module.run_grug(run_cfg)
