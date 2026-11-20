# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint behavior of the June training loop (``_run_grug_local``), run on CPU with in-memory data.

A run stopped at a checkpoint and resumed must match an uninterrupted run bit for bit. This covers the
checkpointer, own-run auto-resume, ``iter_from_step`` data resume, the MuonH optimizer, EMA and QB router betas;
the stopped run's second process sees only what the checkpoint holds. A step whose update makes the params
non-finite must stop the run before that state is checkpointed.
"""

import dataclasses
import json
import logging
from pathlib import Path

import jax
import jmp
import numpy as np
import pytest
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig, load_checkpoint
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
from levanter.distributed import DistributedConfig
from levanter.grug.sharding import compact_grug_mesh
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import TrainerConfig

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig
from experiments.june_tpu_67b_a2b.moe.optimizer import GrugMoeMuonHConfig
from experiments.june_tpu_67b_a2b.moe.synthetic_benchmark import synthetic_examples
from experiments.june_tpu_67b_a2b.moe.train import GrugRunConfig, GrugTrainerConfig, _run_grug_local, initial_state

TOTAL_STEPS = 6
STOP_STEP = 3
BATCH_SIZE = 4
MODEL = GrugModelConfig(
    vocab_size=64,
    hidden_dim=32,
    intermediate_dim=32,
    shared_expert_intermediate_dim=32,
    num_experts=4,
    num_experts_per_token=2,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    head_dim=8,
    max_seq_len=16,
    sliding_window=4,
    disable_pko=True,
    disable_long_rope=True,
    use_array_stacked_blocks=True,
)
# The hero runs' mixed-precision policy, so bf16 compute paths are covered.
MP = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
# The LR schedule spans TOTAL_STEPS in both runs, as it does for a run preempted mid-way.
OPTIMIZER = GrugMoeMuonHConfig(
    learning_rate=0.02,
    adam_lr=0.004,
    warmup=0.2,
    min_lr_ratio=0.1,
    rmsnorm_to_adam=True,
    schedule_num_train_steps_override=TOTAL_STEPS,
)


class LossesByStep(logging.Handler):
    """Collects the per-step training loss the loop logs through the JSON tracker."""

    def __init__(self) -> None:
        super().__init__()
        self.by_step: dict[int, float] = {}

    def emit(self, record: logging.LogRecord) -> None:
        event = json.loads(record.getMessage())
        loss = event.get("metrics", {}).get("train/loss") if event.get("event") == "log" else None
        if loss is not None:
            self.by_step[event["step"]] = loss


def _run(
    checkpoint_dir: Path,
    num_train_steps: int,
    losses: LossesByStep,
    optimizer: GrugMoeMuonHConfig = OPTIMIZER,
) -> None:
    examples = synthetic_examples(TOTAL_STEPS * BATCH_SIZE, MODEL.max_seq_len, MODEL.vocab_size)
    data = LmDataConfig(
        components={"synthetic": DirectDatasetComponent(datasets={"train": ListAsyncDataset(examples)})},
        tokenizer="passthrough",
        vocab_size=MODEL.vocab_size,
        mixture_block_size=len(examples),
    )
    trainer = TrainerConfig(
        id="resume-test",
        use_explicit_mesh_axes=True,
        require_accelerator=False,
        mp=MP,
        train_batch_size=BATCH_SIZE,
        num_train_steps=num_train_steps,
        tracker=JsonLoggerConfig(),
        checkpointer=CheckpointerConfig(
            base_path=str(checkpoint_dir),
            append_run_id_to_base_path=False,
            save_interval=None,
            keep=[dict(every=1)],
        ),
        distributed=DistributedConfig(initialize_jax_distributed=False),
        log_jaxprs=False,
        log_xla_hlo=False,
    )
    config = GrugRunConfig(
        model=MODEL,
        data=data,
        resources=ResourceConfig(),
        optimizer=optimizer,
        trainer=GrugTrainerConfig(trainer=trainer, ema_beta=0.9, z_loss_weight=1e-4),
        eval=None,
    )
    tracker_log = logging.getLogger(JsonLoggerConfig.logger_name)
    tracker_log.addHandler(losses)
    try:
        _run_grug_local(config)
    finally:
        tracker_log.removeHandler(losses)


def _final_state(checkpoint_dir: Path):
    mesh = compact_grug_mesh(expert_axis_size=1)
    with jax.set_mesh(mesh):
        exemplar = initial_state(
            MODEL, optimizer=OPTIMIZER.build(TOTAL_STEPS), mp=MP, key=jax.random.PRNGKey(1), ema_beta=0.9
        )
        return load_checkpoint(exemplar, str(checkpoint_dir / f"step-{TOTAL_STEPS}"), mesh=mesh)


def test_resume_from_checkpoint_is_bitwise_identical(tmp_path):
    straight_losses = LossesByStep()
    _run(tmp_path / "straight", TOTAL_STEPS, straight_losses)

    before_stop = LossesByStep()
    _run(tmp_path / "resumed", STOP_STEP, before_stop)
    after_stop = LossesByStep()
    _run(tmp_path / "resumed", TOTAL_STEPS, after_stop)

    assert sorted(straight_losses.by_step) == list(range(TOTAL_STEPS))
    # The second process trains only the remaining steps, so it must have started from the checkpoint.
    assert sorted(after_stop.by_step) == list(range(STOP_STEP, TOTAL_STEPS))
    assert {**before_stop.by_step, **after_stop.by_step} == straight_losses.by_step

    straight = _final_state(tmp_path / "straight")
    resumed = _final_state(tmp_path / "resumed")
    assert int(straight.step) == int(resumed.step) == TOTAL_STEPS
    straight_leaves, treedef = jax.tree_util.tree_flatten_with_path(straight)
    resumed_leaves, resumed_treedef = jax.tree_util.tree_flatten_with_path(resumed)
    assert treedef == resumed_treedef
    for (path, a), (_, b) in zip(straight_leaves, resumed_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=jax.tree_util.keystr(path))


def test_nonfinite_update_stops_the_run_before_its_checkpoint(tmp_path):
    # The loss is computed before the update, so it stays finite while the infinite learning rate breaks the params.
    losses = LossesByStep()
    with pytest.raises(FloatingPointError, match="Non-finite Grug update at step 1"):
        _run(tmp_path, TOTAL_STEPS, losses, optimizer=dataclasses.replace(OPTIMIZER, learning_rate=float("inf")))
    assert not list(tmp_path.glob("step-*"))
