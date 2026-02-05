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
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main, output_path_of

from experiments.evals.task_configs import convert_to_levanter_task_config

DEFAULT_TPU_TYPE = "v5p-32"
DEFAULT_SEQ_LEN = 4096
DEFAULT_GLOBAL_BATCH_SIZE = 192
DEFAULT_TOKEN_TARGET = 100_000_000_000  # 100B tokens

LEARNING_RATE = 1e-4
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

DEFAULT_EVAL_SUITE = "core"
DEFAULT_EVAL_SUITE_MODE = "both"
DEFAULT_STEPS_PER_TASK_EVAL = 5000

MODEL_LLAMA_13B = "llama_13b"
MODEL_MIXTRAL_8X7B = "mixtral_8x7b"

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
    use_gmm: bool,
    cross_entropy_block_size: int | None,
    cross_entropy_implementation: str | None,
) -> MixtralConfig:
    return MixtralConfig(
        seq_len=seq_len,
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
        lbl_coef=None,
        rzl_coef=None,
        cross_entropy_block_size=cross_entropy_block_size,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--global-batch-size", type=int, default=DEFAULT_GLOBAL_BATCH_SIZE)
    parser.add_argument("--token-target", type=int, default=DEFAULT_TOKEN_TARGET)
    parser.add_argument(
        "--num-train-steps",
        type=int,
        default=None,
        help="If omitted, computed from --token-target / (--global-batch-size * --seq-len).",
    )
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument(
        "--per-device-parallelism",
        type=int,
        default=1,
        help=(
            "Microbatch size per device. Use 1 to reduce peak memory; Levanter will use gradient accumulation to "
            "reach the requested global batch size."
        ),
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
        "--mixtral-use-gmm",
        action="store_true",
        help="Use Megablox/GMM MoE kernels for Mixtral (may have better MFU, but can be shape-sensitive).",
    )
    parser.add_argument(
        "--mixtral-cross-entropy-block-size",
        type=int,
        default=4096,
        help="Vocab block size for Mixtral fused next-token loss.",
    )
    parser.add_argument(
        "--mixtral-cross-entropy-implementation",
        choices=("auto", "xla", "pallas_tpu", "reference"),
        default="xla",
        help="Backend for Mixtral fused next-token loss (use 'auto' to try Pallas first).",
    )

    parser.add_argument("--single-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-save-minutes", type=int, default=60)
    args = parser.parse_args()

    use_default_validation = not args.disable_default_validation

    num_train_steps = (
        int(args.num_train_steps)
        if args.num_train_steps is not None
        else _steps_for_token_target(args.token_target, args.global_batch_size, args.seq_len)
    )

    warmup_steps = max(0, int(args.warmup_steps))
    if LR_SCHEDULE == "cosine" and warmup_steps >= num_train_steps:
        warmup_steps = max(0, num_train_steps - 1)

    tokenized = DATASET_OPTIONS["nemotron_dclm_fineweb_10b"]
    if not isinstance(tokenized, LMMixtureDatasetConfig):
        raise ValueError("Expected nemotron_dclm_fineweb_10b to be a mixture dataset config")
    tokenized = dataclasses.replace(
        tokenized,
        permutation_type=args.permutation_type,
        tokenizer=args.dataset_tokenizer,
    )

    evals = _EVAL_SUITES[args.eval_suite]
    eval_harness_tasks = ()
    if args.eval_suite_mode in ("during_train", "both"):
        eval_harness_tasks = evals

    base_train_config = SimpleTrainConfig(
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
        steps_per_task_eval=int(args.steps_per_task_eval),
        steps_per_hf_export=-1,
        explicit_mesh_axes=True,
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
            "data=nemotron_dclm_fineweb_edu",
            f"model={model_name}",
            f"token_target={args.token_target}",
            f"perm={args.permutation_type}",
            f"seq={args.seq_len}",
            f"bs={args.global_batch_size}",
            f"pdp={args.per_device_parallelism}",
            f"eval_suite={args.eval_suite}",
            f"eval_mode={args.eval_suite_mode}",
            f"mixtral_use_gmm={int(args.mixtral_use_gmm)}",
            f"mixtral_ce_impl={args.mixtral_cross_entropy_implementation}",
            f"mixtral_ce_block={args.mixtral_cross_entropy_block_size}",
            "opt=adamw_b0.9_0.95",
            f"lr={LEARNING_RATE:.2e}",
            *list(args.extra_tag),
        ]

    llama_cfg = dataclasses.replace(llama_13b, max_seq_len=args.seq_len)
    mixtral_ce_impl = (
        None if args.mixtral_cross_entropy_implementation == "auto" else args.mixtral_cross_entropy_implementation
    )
    mixtral_cfg = _build_mixtral_8x7b_config(
        seq_len=args.seq_len,
        use_gmm=bool(args.mixtral_use_gmm),
        cross_entropy_block_size=int(args.mixtral_cross_entropy_block_size),
        cross_entropy_implementation=mixtral_ce_impl,
    )

    llama_name = f"{MODEL_LLAMA_13B}_{run_suffix}"
    mixtral_name = f"{MODEL_MIXTRAL_8X7B}_{run_suffix}"

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

    llama_train_step = _patch_per_device_parallelism(
        llama_train_step,
        per_device_parallelism=args.per_device_parallelism,
    )
    mixtral_train_step = _patch_per_device_parallelism(
        mixtral_train_step,
        per_device_parallelism=args.per_device_parallelism,
    )

    steps: list[ExecutorStep] = [llama_train_step, mixtral_train_step]
    if args.eval_suite_mode in ("post_train", "both") and args.eval_suite != "none":
        for model_name, model_cfg, train_step in (
            (MODEL_LLAMA_13B, llama_cfg, llama_train_step),
            (MODEL_MIXTRAL_8X7B, mixtral_cfg, mixtral_train_step),
        ):
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
        ExecutorMainConfig(prefix=os.environ.get("MARIN_PREFIX")),
        steps=steps,
        description="Mixtral 8x7B vs Llama 13B (Nemotron+DCLM+FineWeb-Edu, feistel, core eval suite, 100B tokens).",
    )


if __name__ == "__main__":
    main()
