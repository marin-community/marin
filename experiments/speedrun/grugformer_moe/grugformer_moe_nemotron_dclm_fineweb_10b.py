# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

_EVAL_SUITES: dict[str, tuple] = {
    "none": (),
    "core": CORE_TASKS,
    "core_plus_mmlu": CORE_TASKS_PLUS_MMLU,
    "core_plus_leaderboard": CORE_TASKS_PLUS_LEADERBOARD,
}


def _steps_for_token_target(token_target: int, global_batch_size: int, seq_len: int) -> int:
    return math.ceil(token_target / (global_batch_size * seq_len))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=DATASET_OPTIONS.keys(),
        default="nemotron_dclm_fineweb_10b",
        help="Which tokenized dataset preset to train on.",
    )
    parser.add_argument("--tpu-type", default="v5p-32")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--global-batch-size", type=int, default=64)
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
        default="core_plus_leaderboard",
        help="Eval-harness suite to run (during training, post-training, or both).",
    )
    parser.add_argument(
        "--eval-suite-mode",
        choices=("post_train", "during_train", "both"),
        default="both",
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    token_target = args.token_target
    if token_target is None:
        token_target = COMPOSITE_TOKEN_TARGET if args.dataset == "nemotron_dclm_fineweb_10b" else DEFAULT_TOKEN_TARGET

    num_train_steps = (
        args.num_train_steps
        if args.num_train_steps is not None
        else _steps_for_token_target(token_target, args.global_batch_size, args.seq_len)
    )

    warmup_steps = max(0, int(args.warmup_steps))
    if LR_SCHEDULE == "cosine" and warmup_steps >= num_train_steps:
        warmup_steps = max(0, num_train_steps - 1)

    model_cfg = GrugformerMoeConfig(
        max_seq_len=args.seq_len,
        hidden_dim=2048,
        intermediate_dim=1024,
        num_layers=16,
        num_heads=16,
        num_kv_heads=8,
        head_dim=None,
        n_routed_experts=64,
        num_experts_per_tok=8,
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

    evals = _EVAL_SUITES[args.eval_suite]
    eval_harness_tasks = ()
    if args.eval_suite_mode in ("during_train", "both"):
        eval_harness_tasks = evals

    train = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu(tpu_type=args.tpu_type),
        train_seq_len=args.seq_len,
        train_batch_size=args.global_batch_size,
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
        explicit_mesh_axes=True,
    )

    default_suffix = f"grugformer_moe_olmoe1b7b_{args.tpu_type}_bs{args.global_batch_size}_{args.dataset}_seq{args.seq_len}"
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
            args.tpu_type,
            f"b{args.global_batch_size}",
            f"s{args.seq_len}",
            f"perm={args.permutation_type}",
        ],
        eval_harness_tasks=eval_harness_tasks,
        wandb_name=run_suffix,
        wandb_group=wandb_group,
        use_default_validation=args.use_default_validation,
        checkpointer_save_interval=timedelta(minutes=int(args.checkpoint_save_minutes)),
        checkpointer_keep=[] if args.single_checkpoint else None,
    )

    steps: list[ExecutorStep] = [train_step]
    if args.eval_suite_mode in ("post_train", "both") and args.eval_suite != "none":
        steps.append(
            ExecutorStep(
                name=f"evaluation/levanter_eval_harness/{run_suffix}/{args.eval_suite}",
                fn=run_levanter_checkpoint_eval_harness,
                config=LevanterEvalHarnessStepConfig(
                    model_name=f"{run_suffix}_{args.eval_suite}",
                    model_config=model_cfg,
                    tokenizer=args.dataset_tokenizer,
                    checkpoint_root=train_step / "checkpoints",
                    evals=evals,
                    max_eval_instances=None,
                    output_path=output_path_of(train_step, f"eval_harness/{args.eval_suite}"),
                    wandb_group=wandb_group,
                ),
                resources=ResourceConfig.with_tpu(args.tpu_type),
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
