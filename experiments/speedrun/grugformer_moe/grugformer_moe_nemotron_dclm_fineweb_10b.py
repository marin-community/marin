# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# nodryrun
"""Grugformer+MoE launcher with Nemotron+DCLM+FineWeb (10B) mixture and eval-harness.

Intended usage: match the runtime knobs from `experiments/speedrun/olmoe_1b7b_nemotron_40b.py`, but
train the experiment-only Grugformer MoE model.

Defaults (requested):
- Dataset: nemotron_dclm_fineweb_10b
- Shuffle: feistel
- TPU: v5p-32
- Global batch size: 64
- Seq len: 4096
- Model shape: OLMoE 1B/7B geometry (D=2048, I=1024, L=16, heads=16, kv_heads=8, E=64, K=8)
- Eval-harness: core_plus_leaderboard, both during and post training

Important: `--dataset-tokenizer` does not retokenize data; it only controls which tokenizer is loaded
for vocab size / special ids / eval decoding. It MUST match the tokenizer used to pretokenize the dataset.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
from datetime import timedelta

from experiments.defaults import default_train
from experiments.evals.task_configs import CORE_TASKS, CORE_TASKS_PLUS_LEADERBOARD, CORE_TASKS_PLUS_MMLU
from experiments.speedrun.grugformer_moe.grugformer_moe import GrugformerMoeConfig
from experiments.speedrun.olmoe_1b7b_nemotron_40b import COMPOSITE_TOKEN_TARGET, DATASET_OPTIONS, DEFAULT_TOKEN_TARGET
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from levanter.data.text import LMMixtureDatasetConfig
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main, output_path_of

from experiments.speedrun.olmoe_1b7b_nemotron_40b import (
    LevanterEvalHarnessStepConfig,
    run_levanter_checkpoint_eval_harness,
)

LEARNING_RATE = 4e-4
WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95
EPSILON = 1e-8
MAX_GRAD_NORM = 1.0
WARMUP_STEPS = 2000
LR_SCHEDULE = "cosine"
MIN_LR_RATIO = 0.125
Z_LOSS_WEIGHT = 1e-4
STEPS_PER_EVAL = 5000
STEPS_PER_EXPORT = 20_000
DEFAULT_PROFILER_START_STEP = 5
DEFAULT_PROFILER_NUM_STEPS = 10

_EVAL_SUITES: dict[str, tuple] = {
    "none": (),
    "core": CORE_TASKS,
    "core_plus_mmlu": CORE_TASKS_PLUS_MMLU,
    "core_plus_leaderboard": CORE_TASKS_PLUS_LEADERBOARD,
}

_DEFAULT_TPU_TYPE = "v5p-32"
_DEFAULT_SEQ_LEN = 4096
_DEFAULT_GLOBAL_BATCH_SIZE = 64
_DEFAULT_EVAL_SUITE = "core_plus_leaderboard"
_DEFAULT_EVAL_SUITE_MODE = "both"

_SMOKE_TPU_TYPE = "v5p-16"
# Keep smoke tests small enough to avoid fused CE vmem regressions on small slices.
_SMOKE_SEQ_LEN = 1024
_SMOKE_GLOBAL_BATCH_SIZE = 32
_SMOKE_NUM_TRAIN_STEPS = 5
_SMOKE_EVAL_SUITE = "none"
_SMOKE_EVAL_SUITE_MODE = "post_train"


def _steps_for_token_target(token_target: int, global_batch_size: int, seq_len: int) -> int:
    return math.ceil(token_target / (global_batch_size * seq_len))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a fast TPU sanity check (defaults: v5p-16, bs=32, seq=1024, steps=5, eval-suite=none). "
            "If you also pass an explicit flag (e.g., --seq-len), that value is kept."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_OPTIONS.keys(),
        default="nemotron_dclm_fineweb_10b",
        help="Which tokenized dataset preset to train on.",
    )
    parser.add_argument("--tpu-type", default=_DEFAULT_TPU_TYPE)
    parser.add_argument("--seq-len", type=int, default=_DEFAULT_SEQ_LEN)
    parser.add_argument("--global-batch-size", type=int, default=_DEFAULT_GLOBAL_BATCH_SIZE)
    parser.add_argument(
        "--per-device-parallelism",
        type=int,
        default=-1,
        help=(
            "How many examples to process in parallel on each device. -1 (default) chooses a value based on "
            "global batch size and device count. Set explicitly (e.g. 8) to reproduce high-MFU legacy-axis runs; "
            "train_batch_size must be divisible by per_device_parallelism * data_axis_size."
        ),
    )
    parser.add_argument(
        "--token-target",
        type=int,
        default=None,
        help=(
            "Total token budget used to compute default --num-train-steps when that flag is omitted. "
            f"Defaults to {DEFAULT_TOKEN_TARGET} for single-corpus runs and {COMPOSITE_TOKEN_TARGET} for the composite "
            "mixture."
        ),
    )
    parser.add_argument(
        "--num-train-steps",
        type=int,
        default=None,
        help="Number of training steps to run (default: computed from --token-target, --global-batch-size, --seq-len).",
    )
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--run-suffix", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--use-default-validation", action="store_true")
    parser.add_argument(
        "--eval-suite",
        choices=tuple(_EVAL_SUITES.keys()),
        default=_DEFAULT_EVAL_SUITE,
        help="Eval-harness suite to run (during training, post-training, or both).",
    )
    parser.add_argument(
        "--eval-suite-mode",
        choices=("post_train", "during_train", "both"),
        default=_DEFAULT_EVAL_SUITE_MODE,
        help="When to run eval-harness: post_train, during_train, or both.",
    )
    parser.add_argument(
        "--steps-per-task-eval",
        type=int,
        default=STEPS_PER_EVAL,
        help="How often to run eval-harness tasks during training when eval-suite-mode includes during_train.",
    )
    parser.add_argument(
        "--permutation-type",
        choices=("feistel", "linear"),
        default="feistel",
        help="Shuffle permutation type for mixture datasets.",
    )
    parser.add_argument(
        "--dataset-tokenizer",
        type=str,
        default="stanford-crfm/marin-tokenizer",
        help=(
            "Tokenizer spec for loading vocab size/special ids (does not retokenize the dataset). "
            "Must match the tokenizer used when pretokenizing."
        ),
    )
    parser.add_argument("--single-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-save-minutes", type=int, default=60)
    parser.add_argument(
        "--cross-entropy-block-size",
        type=int,
        default=512,
        help=(
            "Vocab block size for the fused next-token loss. Smaller blocks reduce peak memory at the cost of "
            "more blocks. Use a multiple of 128 when using the Pallas kernel."
        ),
    )
    parser.add_argument(
        "--cross-entropy-implementation",
        choices=("auto", "xla", "pallas_tpu", "reference"),
        default="auto",
        help=(
            "Cross-entropy backend. 'auto' tries Pallas on TPU v5+ and falls back to XLA when unsupported (e.g. TPU v4)."
        ),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable JAX profiling and upload `jax_profile` artifacts for Perfetto analysis.",
    )
    parser.add_argument(
        "--profile-start-step",
        type=int,
        default=DEFAULT_PROFILER_START_STEP,
        help="Step to start profiling.",
    )
    parser.add_argument(
        "--profile-num-steps",
        type=int,
        default=DEFAULT_PROFILER_NUM_STEPS,
        help="Number of steps to capture after profiling starts.",
    )
    parser.add_argument(
        "--profile-perfetto-link",
        action="store_true",
        help="Generate a Perfetto link when the profiler trace is finalized.",
    )
    parser.add_argument(
        "--log-jaxprs",
        action="store_true",
        help="Log the training step jaxpr to W&B artifacts (slow; off by default).",
    )
    parser.add_argument(
        "--log-xla-hlo",
        action="store_true",
        help="Log the training step StableHLO text to W&B artifacts (very slow; off by default).",
    )
    # Default to non-explicit mesh axes for higher MFU on v5p (matches Levanter's MoE runs).
    parser.set_defaults(explicit_mesh_axes=False)
    parser.add_argument(
        "--explicit-mesh-axes",
        dest="explicit_mesh_axes",
        action="store_true",
        help="Use explicit mesh axes in TrainerConfig.",
    )
    parser.add_argument(
        "--no-explicit-mesh-axes",
        dest="explicit_mesh_axes",
        action="store_false",
        help="Disable explicit mesh axes in TrainerConfig (default).",
    )

    # Default to the "legacy" DP sharding used by high-MFU Levanter MoE runs:
    # token/token_repeat/batch -> (replica, data) and params sharded over embed -> data.
    #
    # The newer default MeshConfig maps batch over (replica_dcn, replica, data). If Grugformer hardcodes
    # (replica, data) in shard_map specs, XLA will insert thousands of tiny reshard collectives.
    parser.set_defaults(legacy_axis_resources=True)
    parser.set_defaults(use_gmm=True)
    parser.add_argument(
        "--use-gmm",
        dest="use_gmm",
        action="store_true",
        help="Use Megablox GMM for expert matmuls (default).",
    )
    parser.add_argument(
        "--no-use-gmm",
        dest="use_gmm",
        action="store_false",
        help="Use ragged_dot_general for expert matmuls (debug/ablation).",
    )

    parser.add_argument(
        "--legacy-axis-resources",
        dest="legacy_axis_resources",
        action="store_true",
        help=(
            "Use a December-style axis mapping equivalent to axis_resources with "
            "token/token_repeat/batch -> (replica, data) and embed -> data."
        ),
    )

    return parser.parse_args()


def _patch_trainer_sharding_ablations(
    train_step: ExecutorStep,
    *,
    tpu_type: str,
    explicit_mesh_axes: bool,
    legacy_axis_resources: bool,
    profiler_perfetto_link: bool,
    log_jaxprs: bool,
    log_xla_hlo: bool,
) -> ExecutorStep:
    config = train_step.config
    inner = config.train_config
    trainer = inner.trainer
    mesh = trainer.mesh

    if legacy_axis_resources:
        mesh = dataclasses.replace(
            mesh,
            compute_mapping={
                "batch": ("replica", "data"),
                "token": ("replica", "data"),
                "token_repeat": ("replica", "data"),
            },
            param_mapping={"embed": "data"},
        )

    trainer = dataclasses.replace(
        trainer,
        mesh=mesh,
        use_explicit_mesh_axes=explicit_mesh_axes,
        profiler_perfetto_link=profiler_perfetto_link,
        log_jaxprs=log_jaxprs,
        log_xla_hlo=log_xla_hlo,
    )
    inner = dataclasses.replace(inner, trainer=trainer)
    config = dataclasses.replace(config, train_config=inner)
    return dataclasses.replace(train_step, config=config)


def main() -> None:
    args = _parse_args()
    tpu_type = args.tpu_type
    seq_len = args.seq_len
    global_batch_size = args.global_batch_size
    num_train_steps_override = args.num_train_steps
    eval_suite = args.eval_suite
    eval_suite_mode = args.eval_suite_mode

    if args.smoke:
        if tpu_type == _DEFAULT_TPU_TYPE:
            tpu_type = _SMOKE_TPU_TYPE
        if seq_len == _DEFAULT_SEQ_LEN:
            seq_len = _SMOKE_SEQ_LEN
        if global_batch_size == _DEFAULT_GLOBAL_BATCH_SIZE:
            global_batch_size = _SMOKE_GLOBAL_BATCH_SIZE
        if eval_suite == _DEFAULT_EVAL_SUITE:
            eval_suite = _SMOKE_EVAL_SUITE
        if eval_suite_mode == _DEFAULT_EVAL_SUITE_MODE:
            eval_suite_mode = _SMOKE_EVAL_SUITE_MODE
        if num_train_steps_override is None:
            num_train_steps_override = _SMOKE_NUM_TRAIN_STEPS

    if args.cross_entropy_implementation in ("auto", "pallas_tpu") and args.cross_entropy_block_size % 128 != 0:
        raise ValueError(
            "--cross-entropy-block-size must be a multiple of 128 when using the Pallas kernel "
            f"(got {args.cross_entropy_block_size})."
        )

    cross_entropy_implementation = (
        None if args.cross_entropy_implementation == "auto" else args.cross_entropy_implementation
    )

    token_target = args.token_target
    if token_target is None:
        token_target = COMPOSITE_TOKEN_TARGET if args.dataset == "nemotron_dclm_fineweb_10b" else DEFAULT_TOKEN_TARGET

    num_train_steps = (
        num_train_steps_override
        if num_train_steps_override is not None
        else _steps_for_token_target(token_target, global_batch_size, seq_len)
    )

    warmup_steps = max(0, int(args.warmup_steps))
    if LR_SCHEDULE == "cosine" and warmup_steps >= num_train_steps:
        warmup_steps = max(0, num_train_steps - 1)

    if args.smoke:
        # Keep smoke tests fast: the full OLMoE-1B7B geometry has very long compile times and can
        # be sensitive to TPU slice health. This smaller shape still exercises the MoE routing
        # + ragged expert MLP path.
        model_cfg = GrugformerMoeConfig(
            max_seq_len=seq_len,
            hidden_dim=512,
            intermediate_dim=1024,
            num_layers=4,
            num_heads=8,
            num_kv_heads=8,
            head_dim=None,
            n_routed_experts=8,
            num_experts_per_tok=2,
            lbl_coef=None,
            rzl_coef=None,
            router_fp32=True,
            router_topk_then_softmax=True,
            use_gmm=bool(args.use_gmm),
            cross_entropy_block_size=int(args.cross_entropy_block_size),
            cross_entropy_implementation=cross_entropy_implementation,
        )
    else:
        model_cfg = GrugformerMoeConfig(
            max_seq_len=seq_len,
            hidden_dim=2048,
            intermediate_dim=1024,
            num_layers=16,
            num_heads=16,
            num_kv_heads=8,
            head_dim=None,
            n_routed_experts=64,
            num_experts_per_tok=8,
            use_gmm=bool(args.use_gmm),
            # Avoid XLA allocation-size overflow on large (tokens x vocab) logits tiles.
            # 32k blocks are fine for smaller local token counts, but can exceed XLA's 32-bit
            # allocation checks at large global batch/seq settings.
            cross_entropy_block_size=int(args.cross_entropy_block_size),
            cross_entropy_implementation=cross_entropy_implementation,
        )

    tokenized = DATASET_OPTIONS[args.dataset]
    if not isinstance(tokenized, LMMixtureDatasetConfig):
        raise ValueError(
            f"--dataset {args.dataset} is not a mixture dataset; cannot set permutation_type={args.permutation_type}"
        )
    tokenized = dataclasses.replace(
        tokenized,
        permutation_type=args.permutation_type,
        tokenizer=args.dataset_tokenizer,
    )

    evals = _EVAL_SUITES[eval_suite]
    eval_harness_tasks = ()
    if eval_suite_mode in ("during_train", "both"):
        eval_harness_tasks = evals

    train = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu(tpu_type=tpu_type),
        train_seq_len=seq_len,
        train_batch_size=global_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        beta1=BETA1,
        beta2=BETA2,
        epsilon=EPSILON,
        max_grad_norm=MAX_GRAD_NORM,
        warmup=warmup_steps,
        lr_schedule=LR_SCHEDULE,
        min_lr_ratio=MIN_LR_RATIO,
        z_loss_weight=Z_LOSS_WEIGHT,
        steps_per_eval=STEPS_PER_EVAL,
        steps_per_export=STEPS_PER_EXPORT,
        steps_per_task_eval=args.steps_per_task_eval,
        steps_per_hf_export=-1,
        per_device_parallelism=int(args.per_device_parallelism),
        explicit_mesh_axes=bool(args.explicit_mesh_axes),
        profiler=bool(args.profile),
        profiler_start_step=int(args.profile_start_step),
        profiler_num_steps=int(args.profile_num_steps),
    )

    default_suffix = f"grugformer_moe_olmoe1b7b_{tpu_type}_bs{global_batch_size}_{args.dataset}_seq{seq_len}"
    run_suffix = args.run_suffix or default_suffix
    wandb_group = args.wandb_group if args.wandb_group is not None else os.environ.get("WANDB_GROUP")

    train_step = default_train(
        name=f"speedrun/{run_suffix}",
        tokenized=tokenized,
        model_config=model_cfg,
        train_config=train,
        tags=[
            "speedrun",
            "grugformer_moe",
            "olmoe_1b7b",
            tpu_type,
            f"b{global_batch_size}",
            f"s{seq_len}",
            f"perm={args.permutation_type}",
            f"pdp={int(args.per_device_parallelism)}",
            f"explicit_mesh_axes={int(bool(args.explicit_mesh_axes))}",
            f"legacy_axis_resources={int(bool(args.legacy_axis_resources))}",
        ],
        eval_harness_tasks=eval_harness_tasks,
        wandb_name=run_suffix,
        wandb_group=wandb_group,
        use_default_validation=args.use_default_validation,
        checkpointer_save_interval=timedelta(minutes=int(args.checkpoint_save_minutes)),
        checkpointer_keep=[] if args.single_checkpoint else None,
    )
    train_step = _patch_trainer_sharding_ablations(
        train_step,
        tpu_type=tpu_type,
        explicit_mesh_axes=bool(args.explicit_mesh_axes),
        legacy_axis_resources=bool(args.legacy_axis_resources),
        profiler_perfetto_link=bool(args.profile_perfetto_link),
        log_jaxprs=bool(args.log_jaxprs),
        log_xla_hlo=bool(args.log_xla_hlo),
    )

    steps: list[ExecutorStep] = [train_step]
    if eval_suite_mode in ("post_train", "both") and eval_suite != "none":
        steps.append(
            ExecutorStep(
                name=f"evaluation/levanter_eval_harness/{run_suffix}/{eval_suite}",
                fn=run_levanter_checkpoint_eval_harness,
                config=LevanterEvalHarnessStepConfig(
                    model_name=f"{run_suffix}_{eval_suite}",
                    model_config=model_cfg,
                    tokenizer=args.dataset_tokenizer,
                    checkpoint_root=train_step / "checkpoints",
                    evals=evals,
                    max_eval_instances=None,
                    output_path=output_path_of(train_step, f"eval_harness/{eval_suite}"),
                    wandb_project=os.environ.get("WANDB_PROJECT") or "marin",
                    wandb_group=wandb_group,
                ),
                resources=ResourceConfig.with_tpu(tpu_type),
                pip_dependency_groups=["tpu", "eval"],
            )
        )

    # `executor_main` is draccus-wrapped to provide a standardized CLI. This experiment uses argparse
    # for its own runtime flags, so we call the undecorated function to avoid draccus attempting to
    # parse the training arguments.
    executor_main.__wrapped__(
        ExecutorMainConfig(prefix=os.environ.get("MARIN_PREFIX")),
        steps=steps,
        description="Grugformer MoE (OLMoE-1B7B shape) on Nemotron+DCLM+FineWeb (feistel + eval harness).",
    )


if __name__ == "__main__":
    main()
