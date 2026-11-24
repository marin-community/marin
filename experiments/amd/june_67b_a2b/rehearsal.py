# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scaled-down Snowball rehearsal on MI355X: every phase of the planned run, on a small June model.

The phases follow the June 67B-A2B schedule (pretrain, batch doubling at step 15k, 64K cooldown at step 39k) with
the steps scaled down:

* ``pretrain``: 8,192 tokens per sequence, ``BATCH`` sequences, steps 0 to ``PRETRAIN_STEPS``. The learning rate
  follows a ``LR_HORIZON``-step warmup-and-decay schedule, as June's pretrain follows its 10T-token horizon.
* ``double``: from the pretrain checkpoint at ``DOUBLE_STEP``, twice the batch to ``DOUBLE_END_STEP``, through
  ``launch_2x_bs`` (piecewise batch schedule, piecewise learning rate, ``initialize_from``).
* ``cooldown``: from the doubled run's last checkpoint, 65,536 tokens per sequence, YaRN ``qk_mult``, the phase-1
  cache, ``COOLDOWN_MICROBATCHES`` microbatches with ZeRO-1, the learning rate decaying to the floor.

The model is June's d768 shape (8 layers, 6 query heads per KV head, 2,048-token window; layers 3 and 7 are the
full-causal "long" layers, 2 of 8 against June's 7 of 26). Data are the pre-mixed Snowball-mix flat caches
(``snowball_phase0``, ``snowball_phase1``) and the Paloma and Uncheatable Eval caches, all on a shared filesystem.

Launch one process per node from the repository root, after ``jax.distributed`` can find the Slurm job::

    srun --ntasks-per-node=1 python -m experiments.amd.june_67b_a2b.rehearsal --phase pretrain --run-id ID \\
        --data-root DIR --validation-root DIR --output-root DIR
"""

import argparse
import dataclasses
import logging
import math
import os
from datetime import timedelta

import jax
import jmp
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.distributed import DistributedConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from rigging.log_setup import configure_logging

from experiments.amd.june_67b_a2b.launch_2x_bs import AmdLaunchConfig2xBS, build_2x_bs_run_config
from experiments.amd.june_67b_a2b.launch_env import check_launch_env
from experiments.amd.june_67b_a2b.model import MOE_IMPLEMENTATION
from experiments.amd.june_67b_a2b.train import GrugRunConfig, GrugTrainerConfig, _run_grug_local
from experiments.june_tpu_67b_a2b.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.june_tpu_67b_a2b.moe.launch_2x_bs import GrugMoeMuonHResumeConfig
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig
from experiments.june_tpu_67b_a2b.moe.optimizer import GrugMoeMuonHConfig
from experiments.june_tpu_67b_a2b.moe.train import GrugEvalConfig
from experiments.marin_tokenizer import marin_tokenizer

logger = logging.getLogger(__name__)

HIDDEN_DIM = 768
SEQ_LEN = 8192
COOLDOWN_SEQ_LEN = 65_536
SLIDING_WINDOW = 2048
ATTENTION = "gpu_pallas_triton_flash"
POOLED_TRANSPORT_CAPACITY_FACTOR = 1.1
MIXED_PRECISION = "params=float32,compute=bfloat16,output=bfloat16"
SEED = 0

GPUS_PER_NODE = 8
EXPERT_AXIS = 8

BATCH = 128
DOUBLED_BATCH = 256
# The cooldown keeps the doubled phase's tokens per step, as June's does (1,024 x 65,536 = 8,192 x 8,192).
COOLDOWN_BATCH = DOUBLED_BATCH * SEQ_LEN // COOLDOWN_SEQ_LEN
COOLDOWN_MICROBATCHES = 2

PRETRAIN_STEPS = 400
DOUBLE_STEP = 300
DOUBLE_RAMP_STEPS = 20
DOUBLE_END_STEP = 400
COOLDOWN_STEPS = 30
LR_HORIZON = 1000
"""Steps of the pretrain and doubled-batch learning-rate schedule; training stops before its end, as June's
pretrain stopped at step 39k of a 10T-token schedule."""
MIN_LR_RATIO = 0.05

CHECKPOINT_EVERY = 50
STEPS_PER_EVAL = 100
COOLDOWN_STEPS_PER_EVAL = 15
STEP_TIMEOUT = 600.0

# YaRN attention temperature for 65,536 tokens, as in June's cooldown: 1.3 x (0.1 ln(65536 / 8192) + 1).
COOLDOWN_QK_MULT = 1.3 * (0.1 * math.log(COOLDOWN_SEQ_LEN / SEQ_LEN) + 1.0)

PHASE_0_CACHE = "snowball_phase0"
PHASE_1_CACHE = "snowball_phase1"
VALIDATION_SETS = (
    "paloma/4chan",
    "paloma/c4_100_domains",
    "paloma/c4_en",
    "paloma/dolma-v1_5",
    "paloma/dolma_100_programing_languages",
    "paloma/dolma_100_subreddits",
    "paloma/falcon-refinedweb",
    "paloma/gab",
    "paloma/m2d2_s2orc_unsplit",
    "paloma/m2d2_wikipedia_unsplit",
    "paloma/manosphere_meta_sep",
    "paloma/mc4",
    "paloma/ptb",
    "paloma/redpajama",
    "paloma/twitterAAE_HELM_fixed",
    "paloma/wikitext_103",
    "uncheatable_eval/ao3_english",
    "uncheatable_eval/arxiv_computer_science",
    "uncheatable_eval/arxiv_physics",
    "uncheatable_eval/bbc_news",
    "uncheatable_eval/github_cpp",
    "uncheatable_eval/github_python",
    "uncheatable_eval/wikipedia_english",
)

WANDB_PROJECT = "marin_moe"
WANDB_GROUP = "amd-snowball-e2e-rehearsal"

_HEURISTIC = MoeMuonHHeuristic(min_lr_ratio=MIN_LR_RATIO)


def build_model(seq_len: int) -> GrugModelConfig:
    base = _HEURISTIC.build_model_config(HIDDEN_DIM, seq_len=seq_len)
    model = dataclasses.replace(
        base,
        disable_pko=True,
        disable_long_rope=True,
        sliding_window=SLIDING_WINDOW,
        use_array_stacked_blocks=True,
        attention_implementation=ATTENTION,
        moe_implementation=MOE_IMPLEMENTATION,
        pooled_transport_capacity_factor=POOLED_TRANSPORT_CAPACITY_FACTOR,
    )
    if seq_len == COOLDOWN_SEQ_LEN:
        model = dataclasses.replace(model, qk_mult=COOLDOWN_QK_MULT)
    return model


def _horizon_tokens(batch: int) -> float:
    return float(LR_HORIZON * batch * SEQ_LEN)


def pretrain_optimizer() -> GrugMoeMuonHConfig:
    base = _HEURISTIC.build_muonh_config(BATCH, _horizon_tokens(BATCH), HIDDEN_DIM, seq_len=SEQ_LEN)
    return dataclasses.replace(base, schedule_num_train_steps_override=LR_HORIZON)


def _doubled_base() -> GrugMoeMuonHConfig:
    # Same token horizon at twice the batch: the heuristic's learning rate grows by sqrt(2), as June's did.
    return _HEURISTIC.build_muonh_config(DOUBLED_BATCH, _horizon_tokens(BATCH), HIDDEN_DIM, seq_len=SEQ_LEN)


def _doubled_lr_at(step: int) -> float:
    base = _doubled_base()
    return float(base.lr_scheduler(LR_HORIZON)(step))


def double_optimizer() -> GrugMoeMuonHResumeConfig:
    """Join the doubled-batch schedule: hold the pretrain LR at the resume, ramp to where the doubled schedule is."""
    lr_at_resume = float(pretrain_optimizer().lr_scheduler(LR_HORIZON)(DOUBLE_STEP))
    return GrugMoeMuonHResumeConfig(
        **dataclasses.asdict(_doubled_base()),
        resume_step=DOUBLE_STEP,
        ramp_end_step=DOUBLE_STEP + DOUBLE_RAMP_STEPS,
        end_step=LR_HORIZON,
        lr_at_resume=lr_at_resume,
        lr_at_ramp_end=_doubled_lr_at(DOUBLE_STEP + DOUBLE_RAMP_STEPS),
    )


def cooldown_optimizer() -> GrugMoeMuonHResumeConfig:
    """Decay linearly from the doubled schedule's LR at the cooldown start to its floor, as June's cooldown does."""
    base = _HEURISTIC.build_muonh_config(COOLDOWN_BATCH, _horizon_tokens(BATCH), HIDDEN_DIM, seq_len=COOLDOWN_SEQ_LEN)
    doubled = _doubled_base()
    lr_at_resume = _doubled_lr_at(DOUBLE_END_STEP)
    # Keep the doubled run's peak, so the floor (min_lr_ratio x peak) and the adam/muonh ratio are its own.
    base = dataclasses.replace(base, learning_rate=doubled.learning_rate, adam_lr=doubled.adam_lr)
    return GrugMoeMuonHResumeConfig(
        **dataclasses.asdict(base),
        resume_step=DOUBLE_END_STEP,
        ramp_end_step=DOUBLE_END_STEP + 1,
        end_step=DOUBLE_END_STEP + COOLDOWN_STEPS,
        lr_at_resume=lr_at_resume,
        lr_at_ramp_end=lr_at_resume,
    )


def _flat_component(cache_dir: str, tag: str) -> DatasetComponent:
    return DatasetComponent(source=None, cache_dir=cache_dir, format=TextLmDatasetFormat(), tags=[tag], flat_cache=True)


def _validation_component(cache_dir: str) -> DatasetComponent:
    return DatasetComponent(source=None, cache_dir=cache_dir, format=TextLmDatasetFormat(), tags=[])


def build_data(*, train_cache: str, data_root: str, validation_root: str) -> LmDataConfig:
    validation = {name: _validation_component(os.path.join(validation_root, name)) for name in VALIDATION_SETS}
    return LmDataConfig(
        tokenizer=marin_tokenizer,
        cache_dir=None,
        components={train_cache: _flat_component(os.path.join(data_root, train_cache), train_cache), **validation},
        train_weights=[(0, {train_cache: 1.0, **{name: 0.0 for name in validation}})],
        # The caches are stored grouped by mixture cell; a full permutation mixes them.
        shuffle=True,
        auto_build_caches=False,
    )


def _tracker(run_id: str, phase: str) -> tuple[TrackerConfig, ...]:
    """W&B, and every step's metrics as JSON lines in process 0's log for exact comparisons between runs."""
    wandb = WandbConfig(
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        name=run_id,
        tags=["moe", "mi355x", "vultr", "e2e_rehearsal", f"d{HIDDEN_DIM}", phase],
    )
    return (wandb, JsonLoggerConfig())


def _checkpointer(checkpoint_dir: str) -> CheckpointerConfig:
    return CheckpointerConfig(
        base_path=os.path.join(checkpoint_dir, "checkpoints"),
        temporary_base_path=os.path.join(checkpoint_dir, "checkpoints-temp"),
        append_run_id_to_base_path=False,
        # Only the step policy saves; a time-based temporary save would make two runs' checkpoints differ.
        save_interval=timedelta(days=1),
        keep=[{"every": CHECKPOINT_EVERY}],
    )


def pretrain_config(args: argparse.Namespace) -> GrugRunConfig:
    checkpoint_dir = os.path.join(args.output_root, args.run_id)
    trainer = TrainerConfig(
        id=args.run_id,
        seed=SEED,
        train_batch_size=BATCH,
        num_train_steps=args.steps or PRETRAIN_STEPS,
        mp=jmp.get_policy(MIXED_PRECISION),
        tracker=_tracker(args.run_id, "pretrain"),
        log_dir=args.log_dir,
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        distributed=DistributedConfig(initialize_jax_distributed=False),
        checkpointer=_checkpointer(checkpoint_dir),
    )
    return GrugRunConfig(
        model=build_model(SEQ_LEN),
        data=build_data(train_cache=PHASE_0_CACHE, data_root=args.data_root, validation_root=args.validation_root),
        resources=ResourceConfig(),
        optimizer=pretrain_optimizer(),
        trainer=GrugTrainerConfig(
            trainer=trainer,
            expert_axis_size=EXPERT_AXIS,
            replica_axis_size=args.nodes,
            diagnose_numerics=True,
            step_timeout=STEP_TIMEOUT,
        ),
        eval=GrugEvalConfig(
            eval_batch_size=args.eval_batch_size,
            steps_per_eval=STEPS_PER_EVAL,
            max_eval_batches=args.max_eval_batches,
            eval_current=True,
            eval_ema=False,
        ),
    )


def double_config(args: argparse.Namespace) -> GrugRunConfig:
    if args.initialize_from is None:
        raise ValueError("--phase double needs --initialize-from (the pretrain checkpoint at DOUBLE_STEP)")
    return build_2x_bs_run_config(
        AmdLaunchConfig2xBS(
            model=build_model(SEQ_LEN),
            data=build_data(train_cache=PHASE_0_CACHE, data_root=args.data_root, validation_root=args.validation_root),
            run_id=args.run_id,
            checkpoint_dir=os.path.join(args.output_root, args.run_id),
            steps=args.steps or DOUBLE_END_STEP,
            batch_size=DOUBLED_BATCH,
            seed=SEED,
            mp=MIXED_PRECISION,
            tracker=_tracker(args.run_id, "double"),
            optimizer=double_optimizer(),
            eval=GrugEvalConfig(
                eval_batch_size=args.eval_batch_size,
                steps_per_eval=STEPS_PER_EVAL,
                max_eval_batches=args.max_eval_batches,
                eval_current=True,
                eval_ema=False,
            ),
            expert_parallel=EXPERT_AXIS,
            replica_parallel=args.nodes,
            diagnose_numerics=True,
            step_timeout=STEP_TIMEOUT,
            checkpoint_keep=[{"every": CHECKPOINT_EVERY}],
            save_interval=timedelta(days=1),
            initialize_from_path=args.initialize_from,
            source_batch_size=BATCH,
            resume_step=DOUBLE_STEP,
            log_dir=args.log_dir,
            initialize_jax_distributed=False,
        )
    )


def cooldown_config(args: argparse.Namespace) -> GrugRunConfig:
    if args.initialize_from is None:
        raise ValueError("--phase cooldown needs --initialize-from (the doubled run's last checkpoint)")
    return build_2x_bs_run_config(
        AmdLaunchConfig2xBS(
            model=build_model(COOLDOWN_SEQ_LEN),
            data=build_data(train_cache=PHASE_1_CACHE, data_root=args.data_root, validation_root=args.validation_root),
            run_id=args.run_id,
            checkpoint_dir=os.path.join(args.output_root, args.run_id),
            steps=args.steps or DOUBLE_END_STEP + COOLDOWN_STEPS,
            batch_size=COOLDOWN_BATCH,
            seed=SEED,
            mp=MIXED_PRECISION,
            tracker=_tracker(args.run_id, "cooldown"),
            optimizer=cooldown_optimizer(),
            eval=GrugEvalConfig(
                eval_batch_size=args.cooldown_eval_batch_size,
                steps_per_eval=COOLDOWN_STEPS_PER_EVAL,
                max_eval_batches=args.max_eval_batches,
                eval_current=True,
                eval_ema=False,
            ),
            expert_parallel=EXPERT_AXIS,
            replica_parallel=args.nodes,
            microbatches=COOLDOWN_MICROBATCHES,
            zero1=args.zero1,
            diagnose_numerics=True,
            step_timeout=STEP_TIMEOUT,
            checkpoint_keep=[{"every": CHECKPOINT_EVERY}],
            save_interval=timedelta(days=1),
            initialize_from_path=args.initialize_from,
            log_dir=args.log_dir,
            initialize_jax_distributed=False,
        )
    )


PHASES = {"pretrain": pretrain_config, "double": double_config, "cooldown": cooldown_config}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=list(PHASES), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-root", required=True, help="Directory holding the two flat Snowball-mix caches.")
    parser.add_argument("--validation-root", required=True, help="Directory holding the VALIDATION_SETS caches.")
    parser.add_argument("--output-root", required=True, help="Checkpoints go to <output-root>/<run id>.")
    parser.add_argument("--initialize-from", help="Source checkpoint for the double and cooldown phases.")
    parser.add_argument("--nodes", type=int, default=2, help="One replica per node.")
    parser.add_argument("--steps", type=int, help="Override the phase's last step.")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--cooldown-eval-batch-size", type=int, default=16)
    parser.add_argument("--max-eval-batches", type=int, default=4)
    parser.add_argument("--no-zero1", dest="zero1", action="store_false", help="Cooldown without ZeRO-1.")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--local-devices", type=int, default=GPUS_PER_NODE)
    parser.add_argument(
        "--no-distributed", dest="distributed", action="store_false", help="One process, no jax.distributed."
    )
    args = parser.parse_args()

    configure_logging(logging.INFO)
    if args.distributed:
        # Before any JAX call that starts the backend; without explicit ids JAX gives each process one GPU.
        DistributedConfig(local_device_ids=list(range(args.local_devices))).initialize()
    check_launch_env(jax.process_count())
    config = PHASES[args.phase](args)
    logger.info("phase %s run %s optimizer %s", args.phase, args.run_id, config.optimizer)
    _run_grug_local(config)
    for device in jax.local_devices():
        stats = device.memory_stats() or {}
        logger.info(
            "device %s peak memory %.1f GiB of %.1f GiB",
            device.id,
            stats.get("peak_bytes_in_use", 0) / 2**30,
            stats.get("bytes_limit", 0) / 2**30,
        )


if __name__ == "__main__":
    main()
