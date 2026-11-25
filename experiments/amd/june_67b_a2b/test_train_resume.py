# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A run of the AMD June training loop stopped at a checkpoint and resumed matches an uninterrupted run bit for bit.

Runs ``_run_grug_local`` on four CPU devices with in-memory data, on the expert-parallel mesh and, in the second
case, with two microbatches and ZeRO-1 over two replicas. It covers the checkpointer, own-run auto-resume, the
data loader's ``iter_from_step`` resume, the rebuilt compute copy (checkpoints leave it out), MuonH, EMA and the QB
router betas; the resumed process sees only what the checkpoint holds.
"""

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
from levanter.testing.devices import use_test_devices
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import TrainerConfig

from experiments.amd.june_67b_a2b.train import (
    GrugRunConfig,
    GrugTrainerConfig,
    _checkpointed,
    _run_grug_local,
    initial_state,
)
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig
from experiments.june_tpu_67b_a2b.moe.optimizer import GrugMoeMuonHConfig
from experiments.june_tpu_67b_a2b.moe.synthetic_benchmark import synthetic_examples

TOTAL_STEPS = 6
STOP_STEP = 3
BATCH_SIZE = 8
DEVICES = 4
MODEL = GrugModelConfig(
    vocab_size=64,
    hidden_dim=128,
    intermediate_dim=32,
    shared_expert_intermediate_dim=32,
    num_experts=4,
    num_experts_per_token=2,
    num_layers=2,
    num_heads=2,
    num_kv_heads=1,
    head_dim=64,
    max_seq_len=16,
    sliding_window=4,
    capacity_factor=8.0,
    pooled_transport_capacity_factor=4.0,
    attention_implementation="reference",
    disable_pko=True,
    disable_long_rope=True,
    use_array_stacked_blocks=True,
)
# Float32 parameters and a bfloat16 compute copy, as the MI355X runs train.
MP = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
OPTIMIZER = GrugMoeMuonHConfig(
    learning_rate=0.02,
    adam_lr=0.004,
    warmup=0.2,
    min_lr_ratio=0.1,
    rmsnorm_to_adam=True,
    schedule_num_train_steps_override=TOTAL_STEPS,
)
# (expert axis, replica axis, microbatches, zero1)
MESHES = {"expert_parallel": (2, 1, 1, False), "zero1_microbatched": (2, 2, 2, True)}


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


def _run(checkpoint_dir: Path, num_train_steps: int, mesh: tuple[int, int, int, bool]) -> dict[int, float]:
    expert, replica, microbatches, zero1 = mesh
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
            base_path=str(checkpoint_dir), append_run_id_to_base_path=False, save_interval=None, keep=[dict(every=1)]
        ),
        distributed=DistributedConfig(initialize_jax_distributed=False),
        log_jaxprs=False,
        log_xla_hlo=False,
    )
    config = GrugRunConfig(
        model=MODEL,
        data=data,
        resources=ResourceConfig(),
        optimizer=OPTIMIZER,
        trainer=GrugTrainerConfig(
            trainer=trainer,
            ema_beta=0.9,
            z_loss_weight=1e-4,
            expert_axis_size=expert,
            replica_axis_size=replica,
            microbatches=microbatches,
            zero1=zero1,
        ),
        eval=None,
    )
    losses = LossesByStep()
    tracker_log = logging.getLogger(JsonLoggerConfig.logger_name)
    tracker_log.addHandler(losses)
    try:
        _run_grug_local(config)
    finally:
        tracker_log.removeHandler(losses)
    return losses.by_step


def _final_state(checkpoint_dir: Path, mesh: tuple[int, int, int, bool]):
    expert, replica, _, zero1 = mesh
    grug_mesh = compact_grug_mesh(expert_axis_size=expert, replica_axis_size=replica)
    with jax.set_mesh(grug_mesh):
        exemplar = initial_state(
            MODEL, optimizer=OPTIMIZER.build(TOTAL_STEPS), mp=MP, key=jax.random.PRNGKey(1), ema_beta=0.9, zero1=zero1
        )
        return load_checkpoint(_checkpointed(exemplar), str(checkpoint_dir / f"step-{TOTAL_STEPS}"), mesh=grug_mesh)


@pytest.mark.parametrize("mesh", MESHES.values(), ids=MESHES.keys())
def test_resume_from_checkpoint_is_bitwise_identical(tmp_path, mesh):
    with use_test_devices(DEVICES):
        straight_losses = _run(tmp_path / "straight", TOTAL_STEPS, mesh)
        before_stop = _run(tmp_path / "resumed", STOP_STEP, mesh)
        after_stop = _run(tmp_path / "resumed", TOTAL_STEPS, mesh)

        assert sorted(straight_losses) == list(range(TOTAL_STEPS))
        # The second process trains only the remaining steps, so it must have started from the checkpoint.
        assert sorted(after_stop) == list(range(STOP_STEP, TOTAL_STEPS))
        assert {**before_stop, **after_stop} == straight_losses

        straight = _final_state(tmp_path / "straight", mesh)
        resumed = _final_state(tmp_path / "resumed", mesh)
    assert int(straight.step) == int(resumed.step) == TOTAL_STEPS
    straight_leaves, treedef = jax.tree_util.tree_flatten_with_path(straight)
    resumed_leaves, resumed_treedef = jax.tree_util.tree_flatten_with_path(resumed)
    assert treedef == resumed_treedef
    for (path, a), (_, b) in zip(straight_leaves, resumed_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=jax.tree_util.keystr(path))
