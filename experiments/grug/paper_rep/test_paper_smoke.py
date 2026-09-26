# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke test: a few real training steps with the paper optimizer.

Covers the wiring the unit tests cannot: the three optimizer groups
(muon / adamw_embed / adamw_head) all receive updates, the paper LR schedule
warms up and warms down, eval runs, and the training loss goes down with the
boundary operator on.
"""

import dataclasses
import json
import logging
import uuid
from io import StringIO

import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig
from levanter.grug.sharding import compact_grug_mesh
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import TrainerConfig

from experiments.grug.paper_rep import train as train_module
from experiments.grug.paper_rep.model import GrugModelConfig, split_prelude_core_coda
from experiments.grug.paper_rep.recipes import VANILLA_RECIPE, optimizer_config

_VOCAB = 128
_SEQ = 32


def _small_boundary_config() -> GrugModelConfig:
    """6-layer config with the boundary operator on (paper split 2/2/2)."""
    split = split_prelude_core_coda(6)
    return GrugModelConfig(
        vocab_size=_VOCAB,
        hidden_dim=32,
        intermediate_dim=96,
        num_layers=6,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=_SEQ,
        boundary_operator=True,
        prelude_len=split.prelude,
        coda_len=split.coda,
        injection_scale=1.0,
    )


def test_paper_lr_schedule_warms_up_then_down():
    """The recipe schedule: warmup WU steps, stable, linear warmdown over the last WDR to zero."""
    config = optimizer_config(VANILLA_RECIPE)  # WU=40, WDR=0.6
    num_train_steps = 1000
    schedule = config.lr_scheduler(num_train_steps)

    peak = float(schedule(40))
    np.testing.assert_allclose(peak, VANILLA_RECIPE.glr, rtol=1e-6)
    np.testing.assert_allclose(float(schedule(0)), 0.0, atol=1e-9)
    # Warmdown covers the last 60% of training, decaying to zero.
    np.testing.assert_allclose(float(schedule(399)), VANILLA_RECIPE.glr, rtol=1e-6)
    assert float(schedule(600)) < VANILLA_RECIPE.glr
    # The last executed step is one linear increment above zero (the schedule
    # reaches exactly zero at step 1000, which never executes).
    assert float(schedule(num_train_steps - 1)) < 1e-4

    # The embed/head schedules ride the same shape at their LR multipliers.
    embed_schedule = config.lr_scheduler(num_train_steps, override_lr=VANILLA_RECIPE.glr * VANILLA_RECIPE.elrm)
    np.testing.assert_allclose(float(embed_schedule(40)), VANILLA_RECIPE.glr * VANILLA_RECIPE.elrm, rtol=1e-6)


def test_paper_optimizer_updates_all_three_groups():
    """token_embed -> adamw_embed, output_proj -> adamw_head, matrices -> muon.

    The first ~WU steps sit in the linear warmup (LR = 0 at step 0), so the
    check advances past the Vanilla warmup (40 steps) before asserting.
    """
    config = optimizer_config(VANILLA_RECIPE)
    optimizer = config.build(num_train_steps=1000)
    cfg = _small_boundary_config()
    mp = jmp.get_policy("f32")

    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        state = train_module.initial_state(cfg, optimizer=optimizer, mp=mp, key=jax.random.PRNGKey(0), ema_beta=None)
        grads = jax.tree_util.tree_map(lambda p: jnp.ones_like(p) * 1e-3, state.params)
        updates = None
        for _ in range(50):
            updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
            state = dataclasses.replace(state, opt_state=opt_state)

    update_norms = jax.tree_util.tree_map(lambda u: float(jnp.max(jnp.abs(u))), updates)
    # tree_map preserves the Transformer structure, so leaves are floats.
    assert update_norms.token_embed > 0
    assert update_norms.output_proj > 0
    for block in update_norms.blocks:
        assert block.attn.w_q > 0
        assert block.mlp.w_down > 0


@pytest.mark.timeout(300)
def test_paper_training_smoke_loss_decreases(tmp_path):
    """Train ~10 steps with the operator on; loss must decrease and the run must finish."""
    logger_name = f"test-grug-paper-smoke-{uuid.uuid4().hex}"
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

    trainer_config = TrainerConfig(
        id="test-grug-paper-smoke",
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
        model=_small_boundary_config(),
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
        optimizer=optimizer_config(VANILLA_RECIPE),
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
    assert train_losses[-1] < train_losses[0], f"training loss did not decrease: {train_losses}"
    finish_records = [r for r in records if r.get("event") == "finish"]
    assert len(finish_records) == 1, "run must finish exactly once"
    assert "throughput/total_tokens" in finish_records[0]["summary"]
    # Eval must have run against the validation split.
    eval_losses = [
        r["metrics"]
        for r in records
        if r.get("event") == "log" and any(k.startswith("eval") for k in r.get("metrics", {}))
    ]
    assert eval_losses, "expected at least one eval log"
