# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay the first steps of the 10.05 Grug SFT from its base on RNO2A, probing re-closes every step.

10.05 (``special_token_full_ab_grouped.py``) fully restored the pretraining state at step 157000, reset the
seven special-token LM-head rows from anchors, and trained 4,750 steps at a constant 5e-5 (linear decay only
over the last 475). This run repeats that start exactly: same restored state and row reset, the same LR
schedule (``schedule_num_train_steps_override``), and the same batches in the same order (exported by
``export_sft_batches.py``). Differences from 10.05:

* H100s with FA4 attention and blocked-vocab cross-entropy; each 256-sequence batch runs as sequential
  microbatches with gradient accumulation;
* the teacher-forced re-close probe (``reclose_eval.py``) runs inside the loop instead of exporting checkpoints;
  ``--checkpoint-root`` keeps one rolling checkpoint so a preemption resumes instead of restarting.

``--close-seed mean`` seeds the <|end_think|> LM-head row from the mean row instead of " answer", to test whether
that anchor drives the re-close behavior. ``--trainable close-row`` freezes every parameter except that row, to test
whether the base model's hidden states already mark where a close belongs.

``--start sft1005`` scores the final 10.05 state with the same probe and no training, as the endpoint reference.
"""

import argparse
import dataclasses
import enum
import json
import logging
import math
from datetime import timedelta

import jmp
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.compat.hf_checkpoints import load_tokenizer
from levanter.data.text.datasets import LmDataConfig
from levanter.tensorstore_serialization import TensorStoreWriteConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from rigging.filesystem.buckets import filesystem_for
from rigging.log_setup import configure_logging

from experiments.grug_sft.reclose_eval import RecloseEvalConfig
from experiments.grug_sft.sft10_continuation import GPUS_PER_NODE, SFT10_STEPS, _reno
from experiments.grug_sft.sft10_continuation import RENO as SFT10_RENO
from experiments.grug_sft.special_token_lr import ANCHORS, BATCH, CONTEXT, EXPECTED_ANCHOR_IDS, START_STEP
from experiments.grug_sft.special_token_train import GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.june_tpu_67b_a2b.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.june_tpu_67b_a2b.moe.optimizer import GrugMoeMuonHConfig

logger = logging.getLogger(__name__)

ROOT = "s3://marin-rno2a/tmp/ttl=30d/dml/sft_replay_10400"
BASE = f"{ROOT}/start/step-157000"
END_THINK_ID = 128003
SFT10_FINAL_STEP = START_STEP + SFT10_STEPS


class Start(enum.StrEnum):
    BASE = "base"
    SFT1005 = "sft1005"


class CloseSeed(enum.StrEnum):
    """Source of the reset <|end_think|> LM-head row: 10.05's " answer" anchor, or the mean of the preserved rows."""

    ANSWER = "answer"
    MEAN = "mean"


class Trainable(enum.StrEnum):
    """Parameters the replay updates: all of them as in 10.05, or only the <|end_think|> LM-head row."""

    ALL = "all"
    CLOSE_ROW = "close-row"


# One rolling checkpoint so a preemption resumes, not restarts. Splitting a replicated array's write across
# replicas is disabled: a 10-step CoreWeave run with replica-split writes kept only one host's share.
CHECKPOINT_INTERVAL = timedelta(minutes=60)


def train(
    run_id: str,
    steps: int,
    nodes: int,
    context_axis_size: int,
    start: Start,
    data_root: str,
    close_seed: CloseSeed,
    checkpoint_root: str | None,
    trainable: Trainable,
) -> None:
    initialize_from, first_step = (BASE, START_STEP) if start == Start.BASE else (SFT10_RENO.start, SFT10_FINAL_STEP)
    if start == Start.SFT1005 and steps != 0:
        raise ValueError("The 10.05 endpoint is scored without training")
    if start == Start.SFT1005 and close_seed != CloseSeed.ANSWER:
        raise ValueError("The close seed applies only when starting from the base")
    if start == Start.SFT1005 and trainable != Trainable.ALL:
        raise ValueError("The 10.05 endpoint is scored without training")
    fs, metadata_path = filesystem_for(f"{initialize_from}/metadata.json")
    with fs.open(metadata_path) as handle:
        assert json.load(handle)["step"] == first_step
    tokenizer = load_tokenizer(SFT10_RENO.tokenizer)
    token_ids = tuple(token_id for token_id, _ in ANCHORS)
    anchor_ids = tuple(tuple(tokenizer.encode(text, add_special_tokens=False)) for _, text in ANCHORS)
    assert anchor_ids == EXPECTED_ANCHOR_IDS, anchor_ids
    assert tokenizer.convert_tokens_to_ids("<|end_think|>") == END_THINK_ID

    microbatch = nodes * GPUS_PER_NODE // context_axis_size
    if BATCH % microbatch:
        raise ValueError(f"{nodes} nodes give microbatch {microbatch}, which does not divide batch {BATCH}")
    model = dataclasses.replace(
        MoeMuonHHeuristic(min_lr_ratio=0.05).build_model_config(2560, seq_len=CONTEXT),
        disable_pko=True,
        disable_long_rope=True,
        sliding_window=2048,
        use_array_stacked_blocks=True,
        qk_mult=1.75,
        max_seq_len=CONTEXT,
        attention_implementation="gpu_fa4_cute",
        ce_implementation="batched_xla",
    )
    trainer = TrainerConfig(
        id=run_id,
        seed=0,
        train_batch_size=BATCH,
        per_device_parallelism=-1,
        num_train_steps=first_step + steps,
        mp=jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(entity="marin-community", project="marin_moe_sft", name=run_id, id=run_id, resume="allow"),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": 1, "context": context_axis_size}, compute_mapping={"batch": ["data", "expert"]}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        initialize_from=initialize_from,
        load_checkpoint=None,
        # The grad/param-norm step variant needs a large extra buffer; after a restore it ran out of memory.
        watch=WatchConfig(interval=0),
        checkpointer=(
            CheckpointerConfig(
                base_path=f"{checkpoint_root}/{run_id}",
                temporary_base_path=f"{checkpoint_root}/{run_id}",
                append_run_id_to_base_path=False,
                save_interval=CHECKPOINT_INTERVAL,
                keep=None,
                keep_last_temporary_checkpoints=1,
                write=TensorStoreWriteConfig(max_write_replicas=1),
            )
            if checkpoint_root is not None
            else TrainerConfig().checkpointer
        ),
    )
    optimizer = GrugMoeMuonHConfig(
        learning_rate=5e-5,
        adam_lr=5e-5,
        beta1=0.9062,
        beta2=0.95,
        epsilon=3.8339433005718795e-15,
        weight_decay=0.0,
        max_grad_norm=None,
        min_lr_ratio=0.1,
        warmup=0,
        decay=SFT10_STEPS // 10,
        lr_schedule="linear",
        rmsnorm_to_adam=True,
        schedule_num_train_steps_override=SFT10_FINAL_STEP,
    )
    run_grug(
        GrugRunConfig(
            model=model,
            data=LmDataConfig(tokenizer=SFT10_RENO.tokenizer),
            optimizer=optimizer,
            resources=_reno(nodes),
            trainer=GrugTrainerConfig(
                trainer=trainer,
                sft_weights_only_init=False,
                data_start_step=first_step,
                max_data_epochs=None,
                special_token_lr_ids=token_ids,
                special_token_lr_multiplier=math.sqrt(32),
                reinitialize_token_ids=token_ids if start == Start.BASE else (),
                reinitialize_token_anchors=anchor_ids if start == Start.BASE else (),
                mean_seeded_token_ids=(END_THINK_ID,) if close_seed == CloseSeed.MEAN else (),
                trainable_output_rows=(END_THINK_ID,) if trainable == Trainable.CLOSE_ROW else None,
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
                replica_axis_size=1,
                context_axis_size=context_axis_size,
                save_checkpoints=checkpoint_root is not None,
                microbatches=BATCH // microbatch,
                replay_batches=f"{data_root}/batches",
                reclose_eval=RecloseEvalConfig(
                    trajectories=f"{data_root}/trajectories.jsonl",
                    close_token_id=END_THINK_ID,
                    every=1,
                    output=f"{data_root}/reclose/{run_id}",
                ),
            ),
            eval=None,
        )
    )


if __name__ == "__main__":
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--nodes", type=int, choices=(16, 32, 64), required=True)
    parser.add_argument("--context-axis-size", type=int, choices=(4, 8), required=True, help="GPUs sharing one sequence")
    parser.add_argument("--start", type=Start, choices=list(Start), default=Start.BASE)
    parser.add_argument(
        "--data-root",
        default=ROOT,
        help="Prefix holding batches/ and trajectories.jsonl; use a bucket in the training cluster's region",
    )
    parser.add_argument("--close-seed", type=CloseSeed, choices=list(CloseSeed), default=CloseSeed.ANSWER)
    parser.add_argument("--trainable", type=Trainable, choices=list(Trainable), default=Trainable.ALL)
    parser.add_argument("--checkpoint-root", help="Write a rolling resume checkpoint under this prefix; none if unset")
    args = parser.parse_args()
    train(
        args.run_id,
        args.steps,
        args.nodes,
        args.context_axis_size,
        args.start,
        args.data_root,
        args.close_seed,
        args.checkpoint_root,
        args.trainable,
    )
