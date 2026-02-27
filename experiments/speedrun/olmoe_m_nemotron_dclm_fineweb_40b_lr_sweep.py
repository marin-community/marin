# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
OLMoE-M AdamC sweep on configurable Nemotron-family datasets (fixed token budget).

This mirrors `experiments/speedrun/olmoe_s_dclm10b_moe_vs_dense_lr_sweep.py`, but:
- uses tokenized datasets from `olmoe_1b7b_nemotron_40b.py`:
  - `nemotron_cc` (full Nemotron-CC)
  - `nemotron_dclm_fineweb_10b` (composite Nemotron+DCLM+FineWeb mixture)
- runs OLMoE-M geometry (16 experts, top-2 routing)
- compares 5 MoE variants across 4 learning rates:
  1) `olmoe_m` (vanilla)
  2) `olmoe_m_bilinear` (bilinear expert MLPs; SwiGLU -> (W1 x) * (W3 x))
  3) `olmoe_m_stab2` (two stability measures):
     - auxiliary-free load balancing (ALF-LB)
     - fp32 router compute
  4) `olmoe_m_stab3` (three stability measures):
     - QK-norm
     - topk-then-softmax routing
     - fp32 router compute
  5) `olmoe_m_stab5` (five stability measures, including fp32 router compute):
     - QK-norm
     - topk-then-softmax routing
     - auxiliary-free load balancing (ALF-LB)
     - dense routing for first 2 blocks
     - fp32 router compute

W&B:
- choose a project with `--wandb-project` (for example `olmoe_m` or `olmoe_m_nemotron`).
- default learning-rate sweep is `[8e-4, 1e-3, 2e-3, 4e-3]`.
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
from levanter.optim import AdamConfig
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


def _identity_activation(x):
    return x


def _build_olmoe_m_config(
    seq_len: int,
    *,
    cross_entropy_block_size: int,
    cross_entropy_implementation: str | None,
) -> MixtralConfig:
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
        # Default keeps the existing high-MFU fused CE behavior; callers can override for stability/debugging.
        cross_entropy_block_size=cross_entropy_block_size,
        cross_entropy_implementation=cross_entropy_implementation,
        flash_attention_block_size=None,
        reference_checkpoint=OLMOE_1B7B_REFERENCE_CHECKPOINT,
        tokenizer=OLMOE_1B7B_REFERENCE_CHECKPOINT,
    )


def _dataset_tag(dataset_name: str) -> str:
    if dataset_name == "nemotron_dclm_fineweb_10b":
        return "nemotron_dclm_fineweb"
    return dataset_name


def _experiment_tag(dataset_name: str) -> str:
    if dataset_name == "nemotron_cc":
        return "exp=olmoe_m_nemotron_lr_sweep"
    return "exp=olmoe_m_lr_sweep"


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
    dataset_name: str,
    extra_tags: list[str],
) -> list[str]:
    return [
        _experiment_tag(dataset_name),
        f"data={_dataset_tag(dataset_name)}",
        f"token_target={token_target}",
        f"perm={permutation_type}",
        f"seq={seq_len}",
        f"bs={global_batch_size}",
        "opt=adamc_b0.9_0.95",
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
    variant_choices = ("olmoe_m", "olmoe_m_bilinear", "olmoe_m_stab2", "olmoe_m_stab3", "olmoe_m_stab5")
    parser = argparse.ArgumentParser(
        description=(
            "OLMoE-M LR sweep on configurable Nemotron-family datasets "
            "(token budget; configurable variants x 4 learning rates)."
        )
    )
    parser.add_argument("--tpu-type", default="v5p-16")
    parser.add_argument(
        "--tpu-head-resource",
        action="append",
        default=[],
        help=(
            "Additional Ray custom resource required on the TPU slice head when acquiring a slice. "
            "Use KEY or KEY=VALUE. Repeatable. Useful for pinning runs to specific labeled TPU workers."
        ),
    )
    parser.add_argument(
        "--variant-head-resource",
        action="append",
        default=[],
        metavar="VARIANT=RESOURCE",
        help=(
            "Pin a specific variant to a TPU worker by requiring RESOURCE on the TPU slice head when acquiring a "
            "slice. Repeatable. Example: --variant-head-resource olmoe_m=olmoe_base"
        ),
    )
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--token-target", type=int, default=40_000_000_000)
    parser.add_argument(
        "--cross-entropy-block-size",
        type=int,
        default=2048,
        help="Vocab block size for fused CE. Smaller values reduce TPU/XLA allocation pressure.",
    )
    parser.add_argument(
        "--cross-entropy-implementation",
        type=str,
        choices=("auto", "pallas_tpu", "xla", "reference"),
        default="pallas_tpu",
        help="Cross-entropy backend. Use xla to avoid pallas scoped-vmem pressure.",
    )

    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[8e-4, 1e-3, 2e-3, 4e-3],
        help="Explicit learning-rate sweep values.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=float, default=2000)
    parser.add_argument("--min-lr-ratio", type=float, default=0.125)
    parser.add_argument(
        "--steps-per-eval",
        type=int,
        default=5000,
        help="How often to run Levanter-native validation losses during training.",
    )
    parser.add_argument(
        "--disable-default-validation",
        dest="disable_default_validation",
        action="store_true",
        help="Disable default Levanter validation losses (Paloma + uncheatable).",
    )
    parser.add_argument(
        "--enable-default-validation",
        dest="disable_default_validation",
        action="store_false",
        help="Enable default Levanter validation losses (Paloma + uncheatable).",
    )

    parser.add_argument(
        "--permutation-type",
        choices=("feistel", "linear"),
        default="feistel",
        help="Shuffle permutation type for the selected dataset.",
    )
    parser.add_argument(
        "--dataset",
        choices=("nemotron_cc", "nemotron_dclm_fineweb_10b"),
        default="nemotron_cc",
        help="Tokenized dataset preset to use for training.",
    )
    parser.add_argument(
        "--dataset-tokenizer",
        type=str,
        # Nemotron-CC tokenized caches are built with the Llama 3 tokenizer.
        default="meta-llama/Meta-Llama-3.1-8B",
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

    # This sweep is Nemotron-family specific; default to the Nemotron project to
    # avoid accidental logging into the generic OLMoE project.
    parser.add_argument("--wandb-project", type=str, default="olmoe_m_nemotron")
    parser.add_argument("--wandb-name-suffix", type=str, default=None)
    parser.add_argument("--run-suffix", type=str, default=None)
    parser.add_argument("--extra-tag", action="append", default=[], help="Additional W&B tag (repeatable).")

    parser.add_argument("--stab-alf-lb-loss-scale", type=float, default=0.01)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=variant_choices,
        default=["olmoe_m", "olmoe_m_bilinear", "olmoe_m_stab3", "olmoe_m_stab5"],
        help=("Which variants to run (default: " "olmoe_m olmoe_m_bilinear olmoe_m_stab3 olmoe_m_stab5)."),
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
    parser.set_defaults(force_run_failed=True, disable_default_validation=True)
    parser.add_argument(
        "--no-force-run-failed",
        dest="force_run_failed",
        action="store_false",
        help="If set, do not retry steps that failed previously (executor will stop on FAILED status).",
    )
    parser.add_argument("--run-only", nargs="*", default=None)
    args = parser.parse_args()
    if args.cross_entropy_block_size <= 0:
        raise ValueError("--cross-entropy-block-size must be > 0")
    if not args.learning_rates:
        raise ValueError("--learning-rates must include at least one value.")

    use_default_validation = not args.disable_default_validation

    tpu_head_resources: dict[str, float] = {}
    for item in args.tpu_head_resource:
        item = (item or "").strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"Invalid --tpu-head-resource '{item}': key must be non-empty.")
            tpu_head_resources[key] = float(value)
        else:
            tpu_head_resources[item] = 1.0

    variant_head_resource: dict[str, str] = {}
    for item in args.variant_head_resource:
        item = (item or "").strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid --variant-head-resource '{item}': expected VARIANT=RESOURCE "
                "(for example: --variant-head-resource olmoe_m=olmoe_base)."
            )
        variant, resource = item.split("=", 1)
        variant = variant.strip()
        resource = resource.strip()
        if not variant or not resource:
            raise ValueError(f"Invalid --variant-head-resource '{item}': variant and resource must be non-empty.")
        if variant not in variant_choices:
            raise ValueError(
                f"Invalid --variant-head-resource '{item}': unknown variant '{variant}'. "
                f"Expected one of: {', '.join(variant_choices)}."
            )
        variant_head_resource[variant] = resource

    tokens_per_step = args.global_batch_size * args.seq_len
    num_train_steps = max(1, _ceil_div(args.token_target, tokens_per_step))
    logger.info(
        "Token budget=%d, tokens/step=%d => num_train_steps=%d", args.token_target, tokens_per_step, num_train_steps
    )

    base_optimizer = AdamConfig(
        learning_rate=float(args.learning_rates[0]),
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        max_grad_norm=args.max_grad_norm,
        warmup=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        lr_schedule="cosine",
        adamc_weight_decay=True,
    )
    base_train_config = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu(tpu_type=args.tpu_type, tpu_head_resources=tpu_head_resources),
        train_batch_size=args.global_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=float(args.learning_rates[0]),
        train_seq_len=args.seq_len,
        optimizer_config=base_optimizer,
        steps_per_eval=args.steps_per_eval,
        steps_per_export=100_000_000,
        steps_per_hf_export=-1,
    )

    tokenized = DATASET_OPTIONS[args.dataset]
    if not isinstance(tokenized, LMMixtureDatasetConfig):
        raise ValueError(f"Expected {args.dataset} to resolve to a mixture dataset config.")
    tokenized = dataclasses.replace(
        tokenized,
        permutation_type=args.permutation_type,
        tokenizer=args.dataset_tokenizer,
        auto_build_caches=False,
    )

    cross_entropy_implementation = (
        None if args.cross_entropy_implementation == "auto" else args.cross_entropy_implementation
    )
    olmoe_m = _build_olmoe_m_config(
        args.seq_len,
        cross_entropy_block_size=args.cross_entropy_block_size,
        cross_entropy_implementation=cross_entropy_implementation,
    )
    olmoe_m_bilinear = dataclasses.replace(olmoe_m, activation_function=_identity_activation)
    olmoe_m_stab2 = dataclasses.replace(
        olmoe_m,
        router_fp32=True,
        alf_lb_loss_scale=args.stab_alf_lb_loss_scale,
    )
    olmoe_m_stab3 = dataclasses.replace(
        olmoe_m,
        use_qk_norm=True,
        router_topk_then_softmax=True,
        router_fp32=True,
    )
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
        ("olmoe_m_stab2", olmoe_m_stab2),
        ("olmoe_m_stab3", olmoe_m_stab3),
        ("olmoe_m_stab5", olmoe_m_stab5),
    ]
    selected_variants = {v.strip() for v in args.variants}
    variants = [v for v in variants if v[0] in selected_variants]

    steps: list[ExecutorStep] = []
    for lr in args.learning_rates:
        lr = float(lr)
        lr_tag = _format_lr_tag(lr)
        optimizer_cfg = dataclasses.replace(base_optimizer, learning_rate=lr)
        train_cfg = dataclasses.replace(base_train_config, learning_rate=lr, optimizer_config=optimizer_cfg)

        for variant, model_cfg in variants:
            variant_resources = dict(tpu_head_resources)
            variant_resource_key = variant_head_resource.get(variant)
            if variant_resource_key:
                variant_resources[variant_resource_key] = 1.0

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
                dataset_name=args.dataset,
                extra_tags=extra_tags,
            )

            run_suffix = f"_{args.run_suffix}" if args.run_suffix else ""
            train_cfg_variant = dataclasses.replace(
                train_cfg,
                resources=dataclasses.replace(train_cfg.resources, tpu_head_resources=variant_resources),
            )
            steps.append(
                default_train(
                    name=f"{base_name}{run_suffix}",
                    tokenized=tokenized,
                    model_config=model_cfg,
                    train_config=train_cfg_variant,
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
    executor_main.__wrapped__(
        executor_cfg,
        steps=steps,
        description="OLMoE-M AdamC LR sweep (Nemotron-family datasets; 40B tokens)",
    )


if __name__ == "__main__":
    main()
