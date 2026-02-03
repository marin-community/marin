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

"""
OLMoE-M LR sweep on the Nemotron+DCLM+FineWeb mixture (fixed token budget).

This mirrors `experiments/speedrun/olmoe_s_dclm10b_moe_vs_dense_lr_sweep.py`, but:
- uses the composite Nemotron+DCLM+FineWeb mixture (tokenized) from `olmoe_1b7b_nemotron_40b.py`
- runs OLMoE-M geometry (16 experts, top-2 routing)
- compares 3 MoE variants across 4 learning-rate multipliers:
  1) `olmoe_m` (vanilla)
  2) `olmoe_m_bilinear` (bilinear expert MLPs; SwiGLU -> (W1 x) * (W3 x))
  3) `olmoe_m_stab5` (five stability measures, including fp32 router compute):
     - QK-norm
     - topk-then-softmax routing
     - auxiliary-free load balancing (ALF-LB)
     - dense routing for first 2 blocks
     - fp32 router compute

W&B:
- set `WANDB_PROJECT=olmoe_m` (or pass `--wandb-project olmoe_m`) to keep these runs in a separate project.
"""

# nodryrun

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
from datetime import timedelta

from experiments.defaults import default_train
from experiments.simple_train_config import SimpleTrainConfig
from experiments.speedrun.custom_mixtral import MixtralConfig
from experiments.speedrun.olmoe_1b7b_nemotron_40b import DATASET_OPTIONS
from fray.cluster import ResourceConfig
from levanter.data.text import LMMixtureDatasetConfig
from levanter.utils.activation import ActivationFunctionEnum
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main

logger = logging.getLogger("ray")

OLMOE_1B7B_REFERENCE_CHECKPOINT = "allenai/OLMoE-1B-7B-0125"


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _format_lr_tag(lr: float) -> str:
    s = f"{lr:.2e}"
    mantissa, exp = s.split("e", 1)
    mantissa = mantissa.replace(".", "p")
    exp_i = int(exp)
    exp_tag = f"em{abs(exp_i):02d}" if exp_i < 0 else f"e{exp_i:02d}"
    return f"{mantissa}{exp_tag}"


def _build_olmoe_m_config(seq_len: int) -> MixtralConfig:
    # Keep expert granularity fixed: topk/n_experts = 2/16 = 1/8.
    return MixtralConfig(
        seq_len=seq_len,
        hidden_dim=1024,
        intermediate_dim=512,
        num_layers=12,
        num_heads=8,
        num_kv_heads=4,
        n_routed_experts=16,
        num_experts_per_tok=2,
        layer_norm_epsilon=1e-5,
        gradient_checkpointing=True,
        scan_layers=True,
        use_gmm=True,
        # Keep the CE vocab block size modest: very large (e.g. 32k) blocks can trigger XLA allocation-size
        # overflows for (tokens x vocab_block) intermediates at long seq / large batch settings.
        cross_entropy_block_size=4096,
        cross_entropy_implementation="xla",
        flash_attention_block_size=None,
        reference_checkpoint=OLMOE_1B7B_REFERENCE_CHECKPOINT,
        tokenizer=OLMOE_1B7B_REFERENCE_CHECKPOINT,
    )


def _make_tags(
    *,
    variant: str,
    lr: float,
    seq_len: int,
    global_batch_size: int,
    token_target: int,
    permutation_type: str,
    use_qk_norm: bool,
    router_topk_then_softmax: bool,
    alf_lb_loss_scale: float,
    dense_first_n_layers: int,
    router_fp32: bool,
    extra_tags: list[str],
) -> list[str]:
    return [
        "exp=olmoe_m_lr_sweep",
        "data=nemotron_dclm_fineweb",
        f"token_target={token_target}",
        f"perm={permutation_type}",
        f"seq={seq_len}",
        f"bs={global_batch_size}",
        "opt=adamw_b0.9_0.95",
        f"lr={lr:.2e}",
        f"variant={variant}",
        f"stab_qk_norm={int(use_qk_norm)}",
        f"stab_topk_then_softmax={int(router_topk_then_softmax)}",
        f"stab_alf_lb={int(alf_lb_loss_scale > 0)}",
        f"stab_dense_first2={int(dense_first_n_layers >= 2)}",
        f"stab_router_fp32={int(router_fp32)}",
        *extra_tags,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OLMoE-M LR sweep on Nemotron+DCLM+FineWeb (token budget; 3 variants x 4 LR multipliers)."
    )
    parser.add_argument("--tpu-type", default="v5p-16")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--global-batch-size", type=int, default=512)
    parser.add_argument("--token-target", type=int, default=40_000_000_000)

    parser.add_argument("--base-lr", type=float, default=1e-4)
    parser.add_argument(
        "--lr-multipliers",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0, 2.0],
        help="Learning-rate multipliers applied to --base-lr.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument(
        "--steps-per-eval",
        type=int,
        default=5000,
        help="How often to run Levanter-native validation losses during training.",
    )
    parser.add_argument(
        "--disable-default-validation",
        action="store_true",
        help="Disable default Levanter validation losses (Paloma + uncheatable).",
    )

    parser.add_argument(
        "--permutation-type",
        choices=("feistel", "linear"),
        default="feistel",
        help="Shuffle permutation type for the mixture dataset.",
    )
    parser.add_argument(
        "--dataset-tokenizer",
        type=str,
        default="stanford-crfm/marin-tokenizer",
        help=(
            "Optional tokenizer name/path used for vocab size / special ids. "
            "Must match the tokenizer used to pretokenize the dataset."
        ),
    )

    parser.add_argument(
        "--single-checkpoint",
        action="store_true",
        help=(
            "Only keep one (temporary) checkpoint at a time to reduce disk pressure. "
            "This disables permanent step-based checkpoints."
        ),
    )
    parser.add_argument("--checkpoint-save-minutes", type=int, default=60)

    parser.add_argument("--wandb-project", type=str, default="olmoe_m")
    parser.add_argument("--wandb-name-suffix", type=str, default=None)
    parser.add_argument("--run-suffix", type=str, default=None)
    parser.add_argument("--extra-tag", action="append", default=[], help="Additional W&B tag (repeatable).")

    parser.add_argument("--stab-alf-lb-loss-scale", type=float, default=0.01)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("olmoe_m", "olmoe_m_bilinear", "olmoe_m_stab5"),
        default=["olmoe_m", "olmoe_m_bilinear", "olmoe_m_stab5"],
        help="Which variants to run (default: all).",
    )

    # Executor controls (so this script can be run under ray_run without draccus CLI conflicts).
    parser.add_argument("--prefix", default=os.getenv("MARIN_PREFIX"))
    parser.add_argument("--executor-info-base-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help=(
            "Maximum number of training steps to run concurrently within this sweep driver. "
            "Set to 1 to use a single TPU slice per submission (sequential LR runs)."
        ),
    )
    parser.set_defaults(force_run_failed=True)
    parser.add_argument(
        "--no-force-run-failed",
        dest="force_run_failed",
        action="store_false",
        help="If set, do not retry steps that failed previously (executor will stop on FAILED status).",
    )
    parser.add_argument("--run-only", nargs="*", default=None)
    args = parser.parse_args()

    use_default_validation = not args.disable_default_validation

    tokens_per_step = args.global_batch_size * args.seq_len
    num_train_steps = max(1, _ceil_div(args.token_target, tokens_per_step))
    logger.info(
        "Token budget=%d, tokens/step=%d => num_train_steps=%d", args.token_target, tokens_per_step, num_train_steps
    )

    base_train_config = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu(tpu_type=args.tpu_type),
        train_batch_size=args.global_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=args.base_lr,
        train_seq_len=args.seq_len,
        weight_decay=args.weight_decay,
        beta1=0.9,
        beta2=0.95,
        steps_per_eval=args.steps_per_eval,
        steps_per_export=100_000_000,
        steps_per_hf_export=-1,
    )

    tokenized = DATASET_OPTIONS["nemotron_dclm_fineweb_10b"]
    if not isinstance(tokenized, LMMixtureDatasetConfig):
        raise ValueError("Expected nemotron_dclm_fineweb_10b to be a mixture dataset config")
    tokenized = dataclasses.replace(
        tokenized,
        permutation_type=args.permutation_type,
        tokenizer=args.dataset_tokenizer,
    )

    olmoe_m = _build_olmoe_m_config(args.seq_len)
    olmoe_m_bilinear = dataclasses.replace(olmoe_m, activation_function=ActivationFunctionEnum.linear)
    olmoe_m_stab5 = dataclasses.replace(
        olmoe_m,
        use_qk_norm=True,
        router_topk_then_softmax=True,
        router_fp32=True,
        alf_lb_loss_scale=args.stab_alf_lb_loss_scale,
        dense_first_n_layers=2,
    )

    variants: list[tuple[str, MixtralConfig]] = [
        ("olmoe_m", olmoe_m),
        ("olmoe_m_bilinear", olmoe_m_bilinear),
        ("olmoe_m_stab5", olmoe_m_stab5),
    ]
    selected_variants = {v.strip() for v in args.variants}
    variants = [v for v in variants if v[0] in selected_variants]

    steps: list[ExecutorStep] = []
    for mult in args.lr_multipliers:
        lr = float(args.base_lr * mult)
        lr_tag = _format_lr_tag(lr)
        train_cfg = dataclasses.replace(base_train_config, learning_rate=lr)

        for variant, model_cfg in variants:
            base_name = f"olmoe_m_40b/{variant}/lr_{lr_tag}/s{args.seq_len}_b{args.global_batch_size}"

            suffix = f"_{args.wandb_name_suffix}" if args.wandb_name_suffix else ""
            wandb_name = f"olmoe_m_{variant}_s{args.seq_len}_b{args.global_batch_size}_lr{lr_tag}{suffix}"

            use_qk_norm = bool(getattr(model_cfg, "use_qk_norm", False))
            router_topk_then_softmax = bool(getattr(model_cfg, "router_topk_then_softmax", False))
            alf_lb_loss_scale = float(getattr(model_cfg, "alf_lb_loss_scale", 0.0) or 0.0)
            dense_first_n_layers = int(getattr(model_cfg, "dense_first_n_layers", 0) or 0)
            router_fp32 = bool(getattr(model_cfg, "router_fp32", False))

            extra_tags = list(args.extra_tag)
            if args.run_suffix:
                extra_tags.append(f"run_suffix={args.run_suffix}")

            tags = _make_tags(
                variant=variant,
                lr=lr,
                seq_len=args.seq_len,
                global_batch_size=args.global_batch_size,
                token_target=args.token_target,
                permutation_type=args.permutation_type,
                use_qk_norm=use_qk_norm,
                router_topk_then_softmax=router_topk_then_softmax,
                alf_lb_loss_scale=alf_lb_loss_scale,
                dense_first_n_layers=dense_first_n_layers,
                router_fp32=router_fp32,
                extra_tags=extra_tags,
            )

            run_suffix = f"_{args.run_suffix}" if args.run_suffix else ""
            steps.append(
                default_train(
                    name=f"{base_name}{run_suffix}",
                    tokenized=tokenized,
                    model_config=model_cfg,
                    train_config=train_cfg,
                    tags=tags,
                    use_default_validation=use_default_validation,
                    eval_harness_tasks=(),
                    wandb_name=wandb_name,
                    wandb_project=args.wandb_project,
                    checkpointer_save_interval=timedelta(minutes=int(args.checkpoint_save_minutes)),
                    checkpointer_keep=[] if args.single_checkpoint else None,
                )
            )

    executor_cfg = ExecutorMainConfig(
        prefix=args.prefix,
        executor_info_base_path=args.executor_info_base_path,
        dry_run=args.dry_run,
        force_run_failed=args.force_run_failed,
        run_only=args.run_only,
        max_concurrent=args.max_concurrent,
    )
    executor_main.__wrapped__(executor_cfg, steps=steps, description="OLMoE-M LR sweep (Nemotron+DCLM+FineWeb; 40B)")


if __name__ == "__main__":
    main()
