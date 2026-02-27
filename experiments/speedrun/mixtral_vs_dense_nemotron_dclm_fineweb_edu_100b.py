# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# nodryrun
"""Launch a Mixtral-8x7B vs Llama-13B comparison run (100B tokens).

This script trains both models from scratch on the composite Nemotron + DCLM + FineWeb-Edu mixture,
with:
- feistel shuffling (default),
- Levanter default validation enabled (Paloma + uncheatable),
- eval-harness "core" suite (during + post train by default),
- AdamW-style optimizer (Adam + weight decay), lr=1e-4 (default),
- microbatching via `trainer.per_device_parallelism` (default 1) to reduce peak memory.

Intended usage is via:
`python -m marin.run.ray_run ... -- python -m experiments.speedrun.mixtral_vs_dense_nemotron_dclm_fineweb_edu_100b ...`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from datetime import timedelta

import fsspec
import jmp
from experiments.defaults import default_train
from experiments.evals.task_configs import CORE_TASKS, CORE_TASKS_PLUS_LEADERBOARD, CORE_TASKS_PLUS_MMLU
from experiments.llama import llama_13b
from experiments.simple_train_config import SimpleTrainConfig
from experiments.speedrun.custom_mixtral import MixtralConfig
from experiments.speedrun.olmoe_1b7b_nemotron_40b import DATASET_OPTIONS
from fray.cluster import ResourceConfig
from levanter.checkpoint import discover_latest_checkpoint
from levanter.data.text import LMMixtureDatasetConfig
from levanter.distributed import RayConfig
from levanter.eval_harness import EvalHarnessMainConfig, LmEvalHarnessConfig, run_eval_harness_main
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main, output_path_of

from experiments.evals.task_configs import convert_to_levanter_task_config

DEFAULT_TPU_TYPE = "v5p-32"
DEFAULT_SEQ_LEN = 4096
DEFAULT_GLOBAL_BATCH_SIZE = 192
DEFAULT_TOKEN_TARGET = 100_000_000_000  # 100B tokens

DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 0.1
DEFAULT_BETA1 = 0.9
DEFAULT_BETA2 = 0.95
DEFAULT_EPSILON = 1e-8
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_WARMUP_STEPS = 2000
LR_SCHEDULE = "cosine"
MIN_LR_RATIO = 0.125
Z_LOSS_WEIGHT = 1e-4
DEFAULT_MIXTRAL_LBL_COEF = 0.01
DEFAULT_MIXTRAL_RZL_COEF = 0.001
STEPS_PER_EVAL = 5000
STEPS_PER_EXPORT = 20_000

DEFAULT_EVAL_SUITE = "core"
DEFAULT_EVAL_SUITE_MODE = "both"
DEFAULT_STEPS_PER_TASK_EVAL = 5000

MODEL_LLAMA_13B = "llama_13b"
MODEL_MIXTRAL_8X7B = "mixtral_8x7b"

_FORWARDED_ENV_PREFIXES = ("JAX_", "LIBTPU_", "XLA_", "WANDB_", "HF_")
_FORWARDED_ENV_KEYS = (
    "PIP_IGNORE_INSTALLED",
    "PIP_NO_CACHE_DIR",
    "RAY_TMPDIR",
    "TMPDIR",
)
_MAXTEXT_V5P_LIBTPU_INIT_ARGS = (
    "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true "
    "--xla_tpu_megacore_fusion_allow_ags=false "
    "--xla_enable_async_collective_permute=true "
    "--xla_tpu_enable_ag_backward_pipelining=true "
    "--xla_tpu_enable_data_parallel_all_reduce_opt=true "
    "--xla_tpu_data_parallel_opt_different_sized_ops=true "
    "--xla_tpu_enable_async_collective_fusion=true "
    "--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
    "--xla_tpu_overlap_compute_collective_tc=true "
    "--xla_enable_async_all_gather=true"
)

_EVAL_SUITES: dict[str, tuple] = {
    "none": (),
    "core": CORE_TASKS,
    "core_plus_mmlu": CORE_TASKS_PLUS_MMLU,
    "core_plus_leaderboard": CORE_TASKS_PLUS_LEADERBOARD,
}


@dataclasses.dataclass(frozen=True)
class LevanterEvalHarnessStepConfig:
    """Config for running Levanter's eval-harness on a Levanter (non-HF) checkpoint."""

    model_name: str
    model_config: object
    tokenizer: str
    checkpoint_root: str
    evals: tuple
    max_eval_instances: int | None
    output_path: str
    wandb_project: str
    apply_chat_template: bool = False
    wandb_group: str | None = None


def run_levanter_checkpoint_eval_harness(config: LevanterEvalHarnessStepConfig) -> None:
    checkpoint_path = discover_latest_checkpoint(config.checkpoint_root)
    if checkpoint_path is None:
        raise ValueError(f"No checkpoints found under {config.checkpoint_root}")

    trainer_config = TrainerConfig(
        tracker=WandbConfig(
            entity=os.environ.get("WANDB_ENTITY"),
            project=config.wandb_project,
            tags=["eval_harness"],
            name=config.model_name,
            group=config.wandb_group,
            mode=os.environ.get("WANDB_MODE"),
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        per_device_eval_parallelism=1,
        ray=RayConfig(auto_start_cluster=False),
    )

    eval_config = EvalHarnessMainConfig(
        eval_harness=LmEvalHarnessConfig(
            task_spec=convert_to_levanter_task_config(config.evals),
            max_examples=config.max_eval_instances,
            log_samples=False,
            confirm_run_unsafe_code=True,
        ),
        tokenizer=config.tokenizer,
        checkpoint_path=checkpoint_path,
        checkpoint_is_hf=False,
        apply_chat_template=config.apply_chat_template,
        trainer=trainer_config,
        model=config.model_config,  # type: ignore[arg-type]
    )

    results = run_eval_harness_main(eval_config)

    fs = fsspec.filesystem("gcs") if config.output_path.startswith("gs://") else fsspec.filesystem("file")
    output_path = config.output_path.rstrip("/") + "/results.json"
    with fs.open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _steps_for_token_target(token_target: int, global_batch_size: int, seq_len: int) -> int:
    return _ceil_div(token_target, global_batch_size * seq_len)


def _build_mixtral_8x7b_config(
    *,
    seq_len: int,
    max_seq_len: int,
    use_gmm: bool,
    use_qk_norm: bool,
    router_topk_then_softmax: bool,
    router_fp32: bool,
    flash_attention_block_size: int | None,
    cross_entropy_block_size: int | None,
    cross_entropy_b_block_size: int | None,
    cross_entropy_h_block_size: int | None,
    cross_entropy_implementation: str | None,
    lbl_coef: float | None,
    rzl_coef: float | None,
) -> MixtralConfig:
    return MixtralConfig(
        seq_len=seq_len,
        max_seq_len=max_seq_len,
        hidden_dim=4096,
        intermediate_dim=14336,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        n_routed_experts=8,
        num_experts_per_tok=2,
        layer_norm_epsilon=1e-5,
        gradient_checkpointing=True,
        scan_layers=True,
        use_gmm=use_gmm,
        use_qk_norm=use_qk_norm,
        router_topk_then_softmax=router_topk_then_softmax,
        router_fp32=router_fp32,
        flash_attention_block_size=flash_attention_block_size,
        lbl_coef=lbl_coef,
        rzl_coef=rzl_coef,
        cross_entropy_block_size=cross_entropy_block_size,
        cross_entropy_b_block_size=cross_entropy_b_block_size,
        cross_entropy_h_block_size=cross_entropy_h_block_size,
        cross_entropy_implementation=cross_entropy_implementation,
    )


def _patch_per_device_parallelism(
    train_step: ExecutorStep,
    *,
    per_device_parallelism: int | None,
) -> ExecutorStep:
    if per_device_parallelism is None:
        return train_step
    if per_device_parallelism <= 0:
        raise ValueError("--per-device-parallelism must be >= 1")

    cfg = train_step.config
    inner = cfg.train_config
    trainer = dataclasses.replace(inner.trainer, per_device_parallelism=per_device_parallelism)
    inner = dataclasses.replace(inner, trainer=trainer)
    cfg = dataclasses.replace(cfg, train_config=inner)
    return dataclasses.replace(train_step, config=cfg)


def _collect_forwarded_runtime_env() -> dict[str, str]:
    forwarded: dict[str, str] = {}
    for key, value in os.environ.items():
        if key in _FORWARDED_ENV_KEYS or any(key.startswith(prefix) for prefix in _FORWARDED_ENV_PREFIXES):
            if value:
                forwarded[key] = value
    return forwarded


def _patch_train_step_env_vars(train_step: ExecutorStep, *, env_vars: dict[str, str]) -> ExecutorStep:
    if not env_vars:
        return train_step

    config = train_step.config
    patched_env = dict(getattr(config, "env_vars", None) or {})
    changed = False
    for key, value in env_vars.items():
        if patched_env.get(key) != value:
            patched_env[key] = value
            changed = True

    if not changed:
        return train_step

    return dataclasses.replace(train_step, config=dataclasses.replace(config, env_vars=patched_env))


def _default_libtpu_init_args_for_tpu(tpu_type: str) -> str | None:
    if tpu_type.startswith("v5p-"):
        return _MAXTEXT_V5P_LIBTPU_INIT_ARGS
    return None


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("1", "true", "t", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def _patch_trainer_profiler_perfetto_link(
    train_step: ExecutorStep,
    *,
    profiler_perfetto_link: bool,
) -> ExecutorStep:
    if not profiler_perfetto_link:
        return train_step
    config = train_step.config
    inner = config.train_config
    trainer = dataclasses.replace(inner.trainer, profiler_perfetto_link=True)
    inner = dataclasses.replace(inner, trainer=trainer)
    config = dataclasses.replace(config, train_config=inner)
    return dataclasses.replace(train_step, config=config)


def _patch_trainer_sharding_ablations(
    train_step: ExecutorStep,
    *,
    explicit_mesh_axes: bool,
    legacy_axis_resources: bool,
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

    trainer = dataclasses.replace(trainer, mesh=mesh, use_explicit_mesh_axes=explicit_mesh_axes)
    inner = dataclasses.replace(inner, trainer=trainer)
    config = dataclasses.replace(config, train_config=inner)
    return dataclasses.replace(train_step, config=config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument(
        "--train-seq-len",
        type=int,
        default=None,
        help=(
            "Training sequence length. Defaults to --seq-len. This is useful when using sliding-window attention "
            "where the model attention window (--seq-len) can be smaller than the packed training sequences."
        ),
    )
    parser.add_argument("--global-batch-size", type=int, default=DEFAULT_GLOBAL_BATCH_SIZE)
    parser.add_argument("--token-target", type=int, default=DEFAULT_TOKEN_TARGET)
    parser.add_argument(
        "--models",
        choices=(MODEL_LLAMA_13B, MODEL_MIXTRAL_8X7B, "both"),
        default="both",
        help="Which model(s) to run. Use this to submit separate Ray jobs per model for stability.",
    )
    parser.add_argument(
        "--num-train-steps",
        type=int,
        default=None,
        help="If omitted, computed from --token-target / (--global-batch-size * --seq-len).",
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--beta1", type=float, default=DEFAULT_BETA1)
    parser.add_argument("--beta2", type=float, default=DEFAULT_BETA2)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "adamc"),
        default="adamw",
        help="Optimizer mode. `adamc` enables AdamC-corrected weight decay.",
    )
    parser.add_argument(
        "--per-device-parallelism",
        type=int,
        default=1,
        help=(
            "Microbatch size per device. Use 1 to reduce peak memory; Levanter will use gradient accumulation to "
            "reach the requested global batch size."
        ),
    )
    parser.set_defaults(explicit_mesh_axes=False)
    parser.add_argument(
        "--explicit-mesh-axes",
        dest="explicit_mesh_axes",
        action="store_true",
        help="Enable explicit mesh axes in TrainerConfig.",
    )
    parser.add_argument(
        "--no-explicit-mesh-axes",
        dest="explicit_mesh_axes",
        action="store_false",
        help="Disable explicit mesh axes in TrainerConfig (default).",
    )
    parser.set_defaults(legacy_axis_resources=True)
    parser.add_argument(
        "--legacy-axis-resources",
        dest="legacy_axis_resources",
        action="store_true",
        help=(
            "Use a December-style axis mapping equivalent to axis_resources with "
            "token/token_repeat/batch -> (replica, data) and embed -> data."
        ),
    )
    parser.add_argument(
        "--no-legacy-axis-resources",
        dest="legacy_axis_resources",
        action="store_false",
        help="Use the current mesh compute mapping path (default uses legacy mapping).",
    )

    parser.add_argument(
        "--permutation-type",
        choices=("feistel", "linear"),
        default="feistel",
        help="Shuffle permutation type for the mixture dataset.",
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_OPTIONS.keys()),
        default="nemotron_dclm_fineweb_10b",
        help="Dataset preset from DATASET_OPTIONS (e.g. nemotron_cc).",
    )
    parser.add_argument(
        "--dataset-tokenizer",
        type=str,
        default="stanford-crfm/marin-tokenizer",
        help=(
            "Tokenizer spec for vocab size / special ids / eval decoding (does not retokenize). Must match the "
            "tokenizer used when building the tokenized dataset."
        ),
    )

    parser.add_argument("--wandb-project", type=str, default="mixtral_vs_dense")
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--run-suffix", type=str, default=None)
    parser.add_argument("--extra-tag", action="append", default=[], help="Additional W&B tag (repeatable).")

    parser.add_argument(
        "--disable-default-validation",
        action="store_true",
        help="Disable default Levanter validation losses (Paloma + uncheatable).",
    )
    parser.add_argument(
        "--steps-per-eval",
        type=int,
        default=STEPS_PER_EVAL,
        help="How often (in steps) to run default Levanter validation losses when enabled.",
    )
    parser.add_argument(
        "--eval-suite",
        choices=tuple(_EVAL_SUITES.keys()),
        default=DEFAULT_EVAL_SUITE,
        help="Eval-harness suite to run (during training, post-training, or both).",
    )
    parser.add_argument(
        "--eval-suite-mode",
        choices=("post_train", "during_train", "both"),
        default=DEFAULT_EVAL_SUITE_MODE,
        help="When to run eval-harness: post_train, during_train, or both.",
    )
    parser.add_argument(
        "--steps-per-task-eval",
        type=int,
        default=DEFAULT_STEPS_PER_TASK_EVAL,
        help="How often to run eval-harness tasks during training when eval-suite-mode includes during_train.",
    )

    parser.add_argument(
        "--trainer.profiler",
        dest="trainer_profiler",
        type=_parse_bool,
        default=False,
        help="Enable the JAX profiler (writes traces under ./logs/<run_id>/profiler and uploads to W&B).",
    )
    parser.add_argument(
        "--trainer.profiler_start_step",
        dest="trainer_profiler_start_step",
        type=int,
        default=5,
        help="Step to start profiling (Levanter TrainerConfig.profiler_start_step).",
    )
    parser.add_argument(
        "--trainer.profiler_num_steps",
        dest="trainer_profiler_num_steps",
        type=int,
        default=100,
        help="Number of steps to capture once profiling starts (Levanter TrainerConfig.profiler_num_steps).",
    )
    parser.add_argument(
        "--trainer.profiler_perfetto_link",
        dest="trainer_profiler_perfetto_link",
        type=_parse_bool,
        default=False,
        help="Generate a Perfetto link when stopping the profiler (see lib/levanter/docs/Performance-Guide.md).",
    )

    parser.set_defaults(mixtral_use_gmm=True)
    parser.add_argument(
        "--mixtral-use-gmm",
        dest="mixtral_use_gmm",
        action="store_true",
        help="Use Megablox/GMM MoE kernels for Mixtral (default).",
    )
    parser.add_argument(
        "--mixtral-no-gmm",
        dest="mixtral_use_gmm",
        action="store_false",
        help="Disable Megablox/GMM MoE kernels for Mixtral.",
    )
    parser.add_argument(
        "--mixtral-flash-attention-block-size",
        type=int,
        default=None,
        help="Flash-attention block size for Mixtral. Set to <=0 to use auto block sizing.",
    )
    parser.add_argument(
        "--mixtral-cross-entropy-block-size",
        type=int,
        default=1024,
        help=(
            "Vocab block size for Mixtral fused next-token loss (default: 1024). " "Set <=0 to disable fused block loss."
        ),
    )
    parser.add_argument(
        "--mixtral-cross-entropy-b-block-size",
        type=int,
        default=None,
        help="Batch tile size for Mixtral fused CE Pallas kernel (multiple of 128; TPU v5p typically needs >=1024).",
    )
    parser.add_argument(
        "--mixtral-cross-entropy-h-block-size",
        type=int,
        default=None,
        help="Hidden tile size for Mixtral fused CE Pallas kernel (multiple of 128).",
    )
    parser.add_argument(
        "--mixtral-cross-entropy-implementation",
        choices=("auto", "legacy", "xla", "pallas_tpu", "reference"),
        default="legacy",
        help=(
            "Backend for Mixtral next-token loss. `legacy` uses the December-era blockwise CE (custom_vjp); "
            "`auto` tries Pallas first."
        ),
    )
    parser.add_argument(
        "--mixtral-use-qk-norm",
        action="store_true",
        help="Enable Mixtral QK normalization.",
    )
    parser.add_argument(
        "--mixtral-router-topk-then-softmax",
        action="store_true",
        help="Enable top-k-then-softmax routing in Mixtral.",
    )
    parser.add_argument(
        "--mixtral-router-fp32",
        action="store_true",
        help="Compute Mixtral router/gating math in fp32.",
    )
    parser.add_argument(
        "--mixtral-lbl-coef",
        type=float,
        default=DEFAULT_MIXTRAL_LBL_COEF,
        help="Mixtral router auxiliary load-balancing loss coefficient.",
    )
    parser.add_argument(
        "--mixtral-rzl-coef",
        type=float,
        default=DEFAULT_MIXTRAL_RZL_COEF,
        help="Mixtral router z-loss coefficient.",
    )
    parser.set_defaults(use_maxtext_libtpu_flags=False)
    parser.add_argument(
        "--use-maxtext-libtpu-flags",
        dest="use_maxtext_libtpu_flags",
        action="store_true",
        help="Use MaxText-style LIBTPU_INIT_ARGS on v5p when not explicitly set.",
    )
    parser.add_argument(
        "--no-maxtext-libtpu-flags",
        dest="use_maxtext_libtpu_flags",
        action="store_false",
        help="Do not auto-set MaxText-style LIBTPU_INIT_ARGS.",
    )
    parser.add_argument(
        "--libtpu-init-args",
        type=str,
        default=None,
        help="Explicit LIBTPU_INIT_ARGS override for train steps.",
    )

    parser.add_argument("--single-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-save-minutes", type=int, default=60)
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Maximum number of experiment steps to run concurrently. Use 1 for sequential smoke runs.",
    )
    args = parser.parse_args()
    forwarded_env = _collect_forwarded_runtime_env()
    if args.libtpu_init_args:
        forwarded_env["LIBTPU_INIT_ARGS"] = args.libtpu_init_args.strip()
    elif args.use_maxtext_libtpu_flags and "LIBTPU_INIT_ARGS" not in forwarded_env:
        default_libtpu_init_args = _default_libtpu_init_args_for_tpu(args.tpu_type)
        if default_libtpu_init_args is not None:
            forwarded_env["LIBTPU_INIT_ARGS"] = default_libtpu_init_args

    use_default_validation = not args.disable_default_validation

    train_seq_len = int(args.train_seq_len) if args.train_seq_len is not None else int(args.seq_len)

    num_train_steps = (
        int(args.num_train_steps)
        if args.num_train_steps is not None
        else _steps_for_token_target(args.token_target, args.global_batch_size, train_seq_len)
    )

    warmup_steps = max(0, int(args.warmup_steps))
    if LR_SCHEDULE == "cosine" and warmup_steps >= num_train_steps:
        warmup_steps = max(0, num_train_steps - 1)

    tokenized = DATASET_OPTIONS[args.dataset]
    if not isinstance(tokenized, LMMixtureDatasetConfig):
        raise ValueError(f"Expected {args.dataset} to be a mixture dataset config")
    tokenized = dataclasses.replace(
        tokenized,
        permutation_type=args.permutation_type,
        tokenizer=args.dataset_tokenizer,
    )

    evals = _EVAL_SUITES[args.eval_suite]
    eval_harness_tasks = ()
    if args.eval_suite_mode in ("during_train", "both"):
        eval_harness_tasks = evals

    optimizer_cfg = AdamConfig(
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        beta1=float(args.beta1),
        beta2=float(args.beta2),
        epsilon=float(args.epsilon),
        max_grad_norm=float(args.max_grad_norm),
        warmup=float(warmup_steps),
        lr_schedule=LR_SCHEDULE,
        min_lr_ratio=float(MIN_LR_RATIO),
        adamc_weight_decay=bool(args.optimizer == "adamc"),
    )

    base_train_config = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu(tpu_type=args.tpu_type),
        train_seq_len=train_seq_len,
        train_batch_size=args.global_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=float(args.learning_rate),
        optimizer_config=optimizer_cfg,
        z_loss_weight=Z_LOSS_WEIGHT,
        steps_per_eval=int(args.steps_per_eval),
        steps_per_export=STEPS_PER_EXPORT,
        steps_per_task_eval=int(args.steps_per_task_eval),
        steps_per_hf_export=-1,
        explicit_mesh_axes=bool(args.explicit_mesh_axes),
        profiler=bool(args.trainer_profiler),
        profiler_start_step=int(args.trainer_profiler_start_step),
        profiler_num_steps=int(args.trainer_profiler_num_steps),
    )

    run_suffix = args.run_suffix
    if not run_suffix:
        raise ValueError(
            "--run-suffix is required to ensure a fresh output path (avoids accidentally resuming prior runs)."
        )

    wandb_group = args.wandb_group if args.wandb_group is not None else os.environ.get("WANDB_GROUP")

    def _make_tags(*, model_name: str) -> list[str]:
        return [
            "exp=mixtral_vs_dense",
            f"data={args.dataset}",
            f"model={model_name}",
            f"token_target={args.token_target}",
            f"perm={args.permutation_type}",
            f"seq={args.seq_len}",
            f"bs={args.global_batch_size}",
            f"pdp={args.per_device_parallelism}",
            f"explicit_mesh_axes={int(bool(args.explicit_mesh_axes))}",
            f"legacy_axis_resources={int(bool(args.legacy_axis_resources))}",
            f"eval_suite={args.eval_suite}",
            f"eval_mode={args.eval_suite_mode}",
            f"mixtral_use_gmm={int(args.mixtral_use_gmm)}",
            f"mixtral_ce_impl={args.mixtral_cross_entropy_implementation}",
            f"mixtral_ce_block={args.mixtral_cross_entropy_block_size}",
            f"mixtral_ce_b_block={args.mixtral_cross_entropy_b_block_size}",
            f"mixtral_ce_h_block={args.mixtral_cross_entropy_h_block_size}",
            f"mixtral_qk_norm={int(args.mixtral_use_qk_norm)}",
            f"mixtral_router_topk_then_softmax={int(args.mixtral_router_topk_then_softmax)}",
            f"mixtral_router_fp32={int(args.mixtral_router_fp32)}",
            f"mixtral_lbl_coef={args.mixtral_lbl_coef:.3g}",
            f"mixtral_rzl_coef={args.mixtral_rzl_coef:.3g}",
            (
                f"opt=adamc_b{args.beta1:.2f}_{args.beta2:.2f}"
                if args.optimizer == "adamc"
                else f"opt=adamw_b{args.beta1:.2f}_{args.beta2:.2f}"
            ),
            f"lr={args.learning_rate:.2e}",
            *list(args.extra_tag),
        ]

    llama_cfg = dataclasses.replace(llama_13b, max_seq_len=args.seq_len)
    mixtral_ce_impl = (
        None if args.mixtral_cross_entropy_implementation == "auto" else args.mixtral_cross_entropy_implementation
    )
    mixtral_ce_block_size = (
        int(args.mixtral_cross_entropy_block_size)
        if args.mixtral_cross_entropy_block_size is not None and int(args.mixtral_cross_entropy_block_size) > 0
        else None
    )
    if mixtral_ce_block_size is None and (
        args.mixtral_cross_entropy_b_block_size is not None or args.mixtral_cross_entropy_h_block_size is not None
    ):
        raise ValueError(
            "--mixtral-cross-entropy-b-block-size/--mixtral-cross-entropy-h-block-size require "
            "--mixtral-cross-entropy-block-size > 0."
        )
    mixtral_cfg = _build_mixtral_8x7b_config(
        seq_len=args.seq_len,
        max_seq_len=train_seq_len,
        use_gmm=bool(args.mixtral_use_gmm),
        use_qk_norm=bool(args.mixtral_use_qk_norm),
        router_topk_then_softmax=bool(args.mixtral_router_topk_then_softmax),
        router_fp32=bool(args.mixtral_router_fp32),
        flash_attention_block_size=(
            int(args.mixtral_flash_attention_block_size)
            if args.mixtral_flash_attention_block_size is not None and int(args.mixtral_flash_attention_block_size) > 0
            else None
        ),
        cross_entropy_block_size=mixtral_ce_block_size,
        cross_entropy_b_block_size=(
            int(args.mixtral_cross_entropy_b_block_size)
            if args.mixtral_cross_entropy_b_block_size is not None and int(args.mixtral_cross_entropy_b_block_size) > 0
            else None
        ),
        cross_entropy_h_block_size=(
            int(args.mixtral_cross_entropy_h_block_size)
            if args.mixtral_cross_entropy_h_block_size is not None and int(args.mixtral_cross_entropy_h_block_size) > 0
            else None
        ),
        cross_entropy_implementation=mixtral_ce_impl,
        lbl_coef=float(args.mixtral_lbl_coef) if args.mixtral_lbl_coef > 0 else None,
        rzl_coef=float(args.mixtral_rzl_coef) if args.mixtral_rzl_coef > 0 else None,
    )

    llama_name = f"{MODEL_LLAMA_13B}_{run_suffix}"
    mixtral_name = f"{MODEL_MIXTRAL_8X7B}_{run_suffix}"

    selected = []
    if args.models in ("both", MODEL_LLAMA_13B):
        llama_train_step = default_train(
            name=f"mixtral_vs_dense/{MODEL_LLAMA_13B}/{run_suffix}",
            tokenized=tokenized,
            model_config=llama_cfg,
            train_config=base_train_config,
            tags=_make_tags(model_name=MODEL_LLAMA_13B),
            eval_harness_tasks=eval_harness_tasks,
            wandb_name=llama_name,
            wandb_group=wandb_group,
            wandb_project=args.wandb_project,
            use_default_validation=use_default_validation,
            checkpointer_save_interval=timedelta(minutes=int(args.checkpoint_save_minutes)),
            checkpointer_keep=[] if args.single_checkpoint else None,
        )
        llama_train_step = _patch_per_device_parallelism(
            llama_train_step,
            per_device_parallelism=args.per_device_parallelism,
        )
        llama_train_step = _patch_train_step_env_vars(llama_train_step, env_vars=forwarded_env)
        llama_train_step = _patch_trainer_profiler_perfetto_link(
            llama_train_step,
            profiler_perfetto_link=bool(args.trainer_profiler_perfetto_link),
        )
        llama_train_step = _patch_trainer_sharding_ablations(
            llama_train_step,
            explicit_mesh_axes=bool(args.explicit_mesh_axes),
            legacy_axis_resources=bool(args.legacy_axis_resources),
        )
        selected.append((MODEL_LLAMA_13B, llama_cfg, llama_train_step))

    if args.models in ("both", MODEL_MIXTRAL_8X7B):
        mixtral_train_step = default_train(
            name=f"mixtral_vs_dense/{MODEL_MIXTRAL_8X7B}/{run_suffix}",
            tokenized=tokenized,
            model_config=mixtral_cfg,
            train_config=base_train_config,
            tags=_make_tags(model_name=MODEL_MIXTRAL_8X7B),
            eval_harness_tasks=eval_harness_tasks,
            wandb_name=mixtral_name,
            wandb_group=wandb_group,
            wandb_project=args.wandb_project,
            use_default_validation=use_default_validation,
            checkpointer_save_interval=timedelta(minutes=int(args.checkpoint_save_minutes)),
            checkpointer_keep=[] if args.single_checkpoint else None,
        )
        mixtral_train_step = _patch_per_device_parallelism(
            mixtral_train_step,
            per_device_parallelism=args.per_device_parallelism,
        )
        mixtral_train_step = _patch_train_step_env_vars(mixtral_train_step, env_vars=forwarded_env)
        mixtral_train_step = _patch_trainer_profiler_perfetto_link(
            mixtral_train_step,
            profiler_perfetto_link=bool(args.trainer_profiler_perfetto_link),
        )
        mixtral_train_step = _patch_trainer_sharding_ablations(
            mixtral_train_step,
            explicit_mesh_axes=bool(args.explicit_mesh_axes),
            legacy_axis_resources=bool(args.legacy_axis_resources),
        )
        selected.append((MODEL_MIXTRAL_8X7B, mixtral_cfg, mixtral_train_step))

    steps: list[ExecutorStep] = [train_step for _, _, train_step in selected]
    if args.eval_suite_mode in ("post_train", "both") and args.eval_suite != "none":
        for model_name, model_cfg, train_step in selected:
            steps.append(
                ExecutorStep(
                    name=f"evaluation/levanter_eval_harness/{model_name}/{run_suffix}/{args.eval_suite}",
                    fn=run_levanter_checkpoint_eval_harness,
                    config=LevanterEvalHarnessStepConfig(
                        model_name=f"{model_name}_{run_suffix}_{args.eval_suite}",
                        model_config=model_cfg,
                        tokenizer=args.dataset_tokenizer,
                        checkpoint_root=train_step / "checkpoints",
                        evals=evals,
                        max_eval_instances=None,
                        output_path=output_path_of(train_step, f"eval_harness/{args.eval_suite}"),
                        wandb_project=args.wandb_project,
                        wandb_group=wandb_group,
                    ),
                    resources=ResourceConfig.with_tpu(args.tpu_type),
                    pip_dependency_groups=["tpu", "eval"],
                )
            )

    executor_main.__wrapped__(
        ExecutorMainConfig(prefix=os.environ.get("MARIN_PREFIX"), max_concurrent=args.max_concurrent),
        steps=steps,
        description=(
            "Mixtral 8x7B vs Llama 13B "
            f"(dataset={args.dataset}, perm={args.permutation_type}, eval_suite={args.eval_suite}, "
            f"eval_mode={args.eval_suite_mode}, token_target={args.token_target})."
        ),
    )


if __name__ == "__main__":
    main()
