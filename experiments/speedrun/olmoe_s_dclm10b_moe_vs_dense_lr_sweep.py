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
OLMoE vs OLMoE+stability vs dense LR sweep (DCLM baseline-only).

This experiment is intended to be a more stable, low-variance comparison than large multi-size sweeps:
- Dataset: DCLM baseline-only (tokenized) with a fixed token budget (default: 10B tokens).
- Variants:
  1) MoE baseline (OLMoE-ish)
  2) MoE + 5 stability measures:
     - QK-norm
     - topk-then-softmax routing
     - auxiliary-free load balancing (ALF-LB)
     - dense routing for first 2 blocks
     - fp32 router compute
  3) Dense baseline (hackable transformer 130m preset)
- Optimizer: AdamW (via Levanter AdamConfig defaults, betas 0.9/0.95)
- Logging: per-layer and global load-violation maxima (see `experiments/speedrun/custom_mixtral.py`)
"""

# nodryrun

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
from datetime import timedelta

from fray.cluster import ResourceConfig

from experiments.defaults import default_train
from experiments.pretraining_datasets.dclm import dclm_baseline_only_mixture_config_llama3
from experiments.simple_train_config import SimpleTrainConfig
from experiments.speedrun.custom_mixtral import MixtralConfig
from experiments.speedrun.hackable_transformer_starter.hackable_transformer_attn_sink import HackableTransformerConfig
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main

logger = logging.getLogger("ray")

OLMOE_1B7B_REFERENCE_CHECKPOINT = "allenai/OLMoE-1B-7B-0125"


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _format_lr_tag(lr: float) -> str:
    # Keep this short and filesystem-safe (used in run/output identifiers).
    s = f"{lr:.2e}"
    mantissa, exp = s.split("e", 1)
    mantissa = mantissa.replace(".", "p")
    exp_i = int(exp)
    if exp_i < 0:
        exp_tag = f"em{abs(exp_i):02d}"
    else:
        exp_tag = f"e{exp_i:02d}"
    return f"{mantissa}{exp_tag}"


def _build_moe_approx_125m_config(seq_len: int) -> MixtralConfig:
    # Approximate ~125M "active" params/token by reducing hidden/intermediate sizes while keeping
    # the expert granularity ratio fixed (topk/n_experts = 1/8).
    return MixtralConfig(
        seq_len=seq_len,
        # Megablox/Pallas GMM kernels require certain block-shape divisibility constraints on TPU. In practice, using a
        # hidden dim divisible by 128 avoids compilation failures.
        hidden_dim=384,
        intermediate_dim=768,
        num_layers=8,
        num_heads=6,
        num_kv_heads=3,
        n_routed_experts=8,
        num_experts_per_tok=1,
        layer_norm_epsilon=1e-5,
        gradient_checkpointing=True,
        scan_layers=True,
        use_gmm=True,
        cross_entropy_block_size=32000,
        flash_attention_block_size=None,
        reference_checkpoint=OLMOE_1B7B_REFERENCE_CHECKPOINT,
        tokenizer=OLMOE_1B7B_REFERENCE_CHECKPOINT,
    )


def _build_dense_130m_config(seq_len: int) -> HackableTransformerConfig:
    # Reuse the hackable-transformer 130m preset geometry (dense baseline).
    return HackableTransformerConfig(
        max_seq_len=seq_len,
        hidden_dim=512,
        intermediate_dim=1792,
        num_layers=6,
        num_heads=8,
        num_kv_heads=8,
        tie_word_embeddings=False,
    )


def _model_family(variant: str) -> str:
    """Returns a simplified model-family label for W&B naming/tags.

    For operator convenience:
    - all MoE variants (OLMoE-based) are labeled "olmoe"
    - all dense baselines are labeled "llama_dense"
    """

    if variant.startswith("moe"):
        return "olmoe"
    return "llama_dense"


def _make_tags(
    *,
    model_family: str,
    variant: str,
    size_tag: str,
    lr: float,
    seq_len: int,
    global_batch_size: int,
    token_target: int,
    use_qk_norm: bool,
    router_topk_then_softmax: bool,
    alf_lb_loss_scale: float,
    dense_first_n_layers: int,
    router_fp32: bool,
    extra_tags: list[str],
) -> list[str]:
    return [
        "exp=olmoe_vs_dense_dclm10b_lr_sweep",
        "data=dclm_baseline_only",
        f"model={model_family}",
        f"token_target={token_target}",
        f"seq={seq_len}",
        f"bs={global_batch_size}",
        "opt=adamw_b0.9_0.95",
        f"lr={lr:.2e}",
        f"variant={variant}",
        f"size={size_tag}",
        f"stab_qk_norm={int(use_qk_norm)}",
        f"stab_topk_then_softmax={int(router_topk_then_softmax)}",
        f"stab_alf_lb={int(alf_lb_loss_scale > 0)}",
        f"stab_dense_first2={int(dense_first_n_layers >= 2)}",
        f"stab_router_fp32={int(router_fp32)}",
        *extra_tags,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OLMoE vs OLMoE+stability vs dense sweep on DCLM baseline-only (token budget)."
    )
    parser.add_argument("--tpu-type", default="v5p-16")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--token-target", type=int, default=10_000_000_000)

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
        default=1000,
        help="How often to run Levanter-native validation losses during training.",
    )
    parser.add_argument(
        "--disable-default-validation",
        action="store_true",
        help="Disable default Levanter validation losses (Paloma + uncheatable).",
    )
    parser.add_argument(
        "--single-checkpoint",
        action="store_true",
        help=(
            "Only keep one (temporary) checkpoint at a time to reduce disk pressure. "
            "This disables permanent step-based checkpoints."
        ),
    )
    parser.add_argument(
        "--checkpoint-save-minutes",
        type=int,
        default=60,
        help="How often to save temporary checkpoints when --single-checkpoint is set.",
    )
    parser.add_argument(
        "--wandb-name-suffix",
        type=str,
        default=None,
        help="Optional suffix appended to the W&B display name for all runs (e.g. `t$(date +%Y%m%d_%H%M%S)`).",
    )
    parser.add_argument(
        "--run-suffix",
        type=str,
        default=None,
        help=(
            "Optional suffix appended to the *output/run identifier* for all runs, to create a fresh sweep without "
            "overwriting prior outputs (e.g. `$(git rev-parse --short HEAD)` or `t$(date +%Y%m%d_%H%M%S)`)."
        ),
    )
    parser.add_argument("--extra-tag", action="append", default=[], help="Additional W&B tag (repeatable).")

    # MoE stability knobs (only applied to the 'moe_stab' variant)
    parser.add_argument("--stab-alf-lb-loss-scale", type=float, default=0.01)

    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("moe", "moe_stab5", "dense"),
        default=["moe", "moe_stab5", "dense"],
        help="Which model family variants to run (default: all three).",
    )

    # Executor controls (so this script can be run under ray_run without draccus CLI conflicts).
    parser.add_argument("--prefix", default=os.getenv("MARIN_PREFIX"))
    parser.add_argument("--executor-info-base-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-run-failed", action="store_true")
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
        steps_per_export=100_000_000,  # irrelevant when --single-checkpoint, but keep large to avoid step-based exports
        steps_per_hf_export=-1,  # disable HF exports (disk-heavy and not needed for this comparison)
    )

    moe_base = _build_moe_approx_125m_config(args.seq_len)
    moe_stab = dataclasses.replace(
        moe_base,
        use_qk_norm=True,
        router_topk_then_softmax=True,
        router_fp32=True,
        alf_lb_loss_scale=args.stab_alf_lb_loss_scale,
        dense_first_n_layers=2,
    )
    dense = _build_dense_130m_config(args.seq_len)

    variants: list[tuple[str, object, str]] = [
        ("moe", moe_base, "olmoe_s125m"),
        ("moe_stab5", moe_stab, "olmoe_s125m"),
        ("dense", dense, "llama_dense_130m"),
    ]
    selected_variants = {v.strip() for v in args.variants}
    variants = [v for v in variants if v[0] in selected_variants]

    steps: list[ExecutorStep] = []
    for mult in args.lr_multipliers:
        lr = float(args.base_lr * mult)
        lr_tag = _format_lr_tag(lr)
        train_cfg = dataclasses.replace(base_train_config, learning_rate=lr)

        for variant, model_cfg, size_tag in variants:
            base_name = f"olmoe_vs_dense_dclm10b/{variant}/lr_{lr_tag}/s{args.seq_len}_b{args.global_batch_size}"

            suffix = f"_{args.wandb_name_suffix}" if args.wandb_name_suffix else ""
            model_family = _model_family(variant)
            # Keep W&B names short and easy to scan; avoid redundant "dense_..._dense" prefixes.
            if variant == "dense":
                wandb_name = f"od_dclm10b_{model_family}_s{args.seq_len}_b{args.global_batch_size}_lr{lr_tag}{suffix}"
            else:
                wandb_name = (
                    f"od_dclm10b_{model_family}_{variant}_s{args.seq_len}_b{args.global_batch_size}_lr{lr_tag}{suffix}"
                )

            use_qk_norm = bool(getattr(model_cfg, "use_qk_norm", False))
            router_topk_then_softmax = bool(getattr(model_cfg, "router_topk_then_softmax", False))
            alf_lb_loss_scale = float(getattr(model_cfg, "alf_lb_loss_scale", 0.0) or 0.0)
            dense_first_n_layers = int(getattr(model_cfg, "dense_first_n_layers", 0) or 0)
            router_fp32 = bool(getattr(model_cfg, "router_fp32", False))

            tags = _make_tags(
                model_family=model_family,
                variant=variant,
                size_tag=size_tag,
                lr=lr,
                seq_len=args.seq_len,
                global_batch_size=args.global_batch_size,
                token_target=args.token_target,
                use_qk_norm=use_qk_norm,
                router_topk_then_softmax=router_topk_then_softmax,
                alf_lb_loss_scale=alf_lb_loss_scale,
                dense_first_n_layers=dense_first_n_layers,
                router_fp32=router_fp32,
                extra_tags=([*list(args.extra_tag), f"run_suffix={args.run_suffix}"] if args.run_suffix else list(args.extra_tag)),
            )

            run_suffix = f"_{args.run_suffix}" if args.run_suffix else ""
            steps.append(
                default_train(
                    name=f"{base_name}{run_suffix}",
                    tokenized=dclm_baseline_only_mixture_config_llama3,
                    model_config=model_cfg,  # type: ignore[arg-type]
                    train_config=train_cfg,
                    tags=tags,
                    use_default_validation=use_default_validation,
                    eval_harness_tasks=(),
                    wandb_name=wandb_name,
                    checkpointer_save_interval=timedelta(minutes=args.checkpoint_save_minutes),
                    checkpointer_keep=[] if args.single_checkpoint else None,
                )
            )

    executor_cfg = ExecutorMainConfig(
        prefix=args.prefix,
        executor_info_base_path=args.executor_info_base_path,
        dry_run=args.dry_run,
        force_run_failed=args.force_run_failed,
        run_only=args.run_only,
    )
    executor_main.__wrapped__(executor_cfg, steps=steps, description="OLMoE vs dense LR sweep (DCLM baseline-only)")


if __name__ == "__main__":
    main()
