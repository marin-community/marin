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
"""Nemotron speedrun launcher for several model presets.

Notable presets:
- `olmoe_1b7b`: OLMoE-style 1B/7B MoE geometry (64 experts, 8 routed/token).
- `olmoe_1b7b_bilinear`: same geometry, but with *bilinear* expert MLPs
  (SwiGLU -> (W1 x) * (W3 x)) via `activation_function=linear`.
- `llama_1_4b_bilinear`: same idea for a dense Llama 1.4B.

Evaluation options:
- `--eval-suite ...` selects which eval-harness task suite to run.
- `--eval-suite-mode ...` chooses whether to run it during training, after training, or both.
"""

import argparse
import dataclasses
import json
import logging
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

import fsspec
import jmp
from experiments.defaults import default_train
from experiments.evals.evals import default_eval
from experiments.evals.task_configs import (
    CORE_TASKS,
    CORE_TASKS_PLUS_LEADERBOARD,
    CORE_TASKS_PLUS_MMLU,
    convert_to_levanter_task_config,
)
from experiments.pretraining_datasets import NEMOTRON_WEIGHTS, tokenize_nemotron
from experiments.pretraining_datasets.dclm import (
    DCLM_MIXTURE_WEIGHTS,
    dclm_baseline_only_mixture_config_llama3,
    dclm_components_llama3,
    dclm_mixture_config_llama3,
)
from experiments.llama import llama3_tokenizer, llama_1_4b
from experiments.speedrun.prebuilt_caches import fineweb_edu_subcache_10B
from experiments.speedrun.custom_mixtral import MixtralConfig
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from levanter.checkpoint import discover_latest_checkpoint
from levanter.eval_harness import EvalHarnessMainConfig, LmEvalHarnessConfig, run_eval_harness_main
from levanter.distributed import RayConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.activation import ActivationFunctionEnum
from levanter.infra.cli_helpers import load_config
from marin.execution.executor import ExecutorStep, InputName, executor_main, output_path_of
from marin.processing.tokenize.download_pretokenized import download_pretokenized_cache
from marin.processing.tokenize import lm_data_config, lm_mixture_data_config
from marin.speedrun.speedrun import Author, SpeedrunConfig, SpeedrunResultsConfig, speedrun_results
from marin.utilities.wandb_utils import WANDB_ENTITY, WANDB_PROJECT

logger = logging.getLogger("ray")

# ---------------------------------------------------------------------------
# Shared experiment knobs (mirrors the dense baseline for flop matching)
# ---------------------------------------------------------------------------
SEQ_LEN = 2048
DEFAULT_GLOBAL_BATCH_SIZE = 64
DEFAULT_TOKEN_TARGET = 40_000_000_000  # 40B tokens
COMPOSITE_TOKEN_TARGET = 100_000_000_000  # 100B tokens
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.1
STEPS_PER_EVAL = 5000
STEPS_PER_EXPORT = 20_000
TPU_TYPE = "v5p-64"
DEFAULT_PROFILER_START_STEP = 3
DEFAULT_PROFILER_NUM_STEPS = 20

MODEL_OLMOE_1B7B = "olmoe_1b7b"
MODEL_OLMOE_1B7B_BILINEAR = "olmoe_1b7b_bilinear"
MODEL_MIXTRAL_8X7B = "mixtral_8x7b"
MODEL_LLAMA_1_4B = "llama_1_4b"
MODEL_LLAMA_1_4B_BILINEAR = "llama_1_4b_bilinear"
MODEL_OPTIONS = (
    MODEL_OLMOE_1B7B,
    MODEL_OLMOE_1B7B_BILINEAR,
    MODEL_MIXTRAL_8X7B,
    MODEL_LLAMA_1_4B,
    MODEL_LLAMA_1_4B_BILINEAR,
)
DEFAULT_MODEL = MODEL_OLMOE_1B7B

OLMOE_1B7B_REFERENCE_CHECKPOINT = "allenai/OLMoE-1B-7B-0125"

_EVAL_SUITES: dict[str, tuple | None] = {
    "none": None,
    "core": CORE_TASKS,
    "core_plus_mmlu": CORE_TASKS_PLUS_MMLU,
    "core_plus_leaderboard": CORE_TASKS_PLUS_LEADERBOARD,
}


@dataclass(frozen=True)
class LevanterEvalHarnessStepConfig:
    """Config for running Levanter's eval harness on a Levanter (non-HF) checkpoint."""

    model_name: str
    model_config: object
    tokenizer: str
    checkpoint_root: str
    evals: tuple
    max_eval_instances: int | None
    output_path: str
    apply_chat_template: bool = False
    wandb_group: str | None = None


def run_levanter_checkpoint_eval_harness(config: LevanterEvalHarnessStepConfig) -> None:
    checkpoint_path = discover_latest_checkpoint(config.checkpoint_root)
    if checkpoint_path is None:
        raise ValueError(f"No checkpoints found under {config.checkpoint_root}")

    trainer_config = TrainerConfig(
        tracker=WandbConfig(project="marin", tags=["eval_harness"], name=config.model_name, group=config.wandb_group),
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


def _supports_hf_export(model_config: object) -> bool:
    activation = getattr(model_config, "activation_function", None)
    return activation != ActivationFunctionEnum.linear


def _build_olmoe_1b7b_config(seq_len: int) -> MixtralConfig:
    """OLMoE-style MoE config (inspired by AllenAI OLMoE 1B-7B)."""
    return MixtralConfig(
        seq_len=seq_len,
        hidden_dim=2048,
        intermediate_dim=1024,
        num_layers=16,
        num_heads=16,
        num_kv_heads=8,
        n_routed_experts=64,
        num_experts_per_tok=8,
        layer_norm_epsilon=1e-5,
        gradient_checkpointing=True,
        scan_layers=True,
        use_gmm=True,
        cross_entropy_block_size=32000,
        reference_checkpoint=OLMOE_1B7B_REFERENCE_CHECKPOINT,
        tokenizer=OLMOE_1B7B_REFERENCE_CHECKPOINT,
    )


def _build_mixtral_8x7b_config(seq_len: int) -> MixtralConfig:
    """Mixtral 8x7B config (8 experts, 2 routed/token), aligned with MaxText's model geometry."""
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
        # MaxText's high-MFU recipe uses Megablox (dropless) MoE kernels; for us this maps most closely to
        # `use_gmm=True`.
        use_gmm=True,
        cross_entropy_block_size=32000,
        lbl_coef=None,
        rzl_coef=None,
    )


def build_model_config(*, model: str, seq_len: int):
    if model == MODEL_OLMOE_1B7B:
        return _build_olmoe_1b7b_config(seq_len)
    if model == MODEL_OLMOE_1B7B_BILINEAR:
        # Bilinear expert MLP variant: remove the gate nonlinearity (SwiGLU -> (W1 x) * (W3 x)).
        # Note: HF export isn't supported for this activation, so HF exports/evals are disabled by default.
        return dataclasses.replace(_build_olmoe_1b7b_config(seq_len), activation_function=ActivationFunctionEnum.linear)
    if model == MODEL_MIXTRAL_8X7B:
        return _build_mixtral_8x7b_config(seq_len)
    if model == MODEL_LLAMA_1_4B:
        return dataclasses.replace(llama_1_4b, max_seq_len=seq_len)
    if model == MODEL_LLAMA_1_4B_BILINEAR:
        # Bilinear MLP variant: remove the gate nonlinearity (SwiGLU -> (W1 x) * (W3 x)).
        # Note: HF export isn't supported for this activation, so we disable exports below.
        return dataclasses.replace(llama_1_4b, max_seq_len=seq_len, activation_function=ActivationFunctionEnum.linear)
    raise ValueError(f"Unknown model preset {model!r}. Options: {MODEL_OPTIONS}.")


nemotron_cc_steps = tokenize_nemotron(tokenizer=llama3_tokenizer)
nemotron_cc_mixture = lm_mixture_data_config(
    components=nemotron_cc_steps,
    weights=NEMOTRON_WEIGHTS,
    permutation_type="linear",
)

fineweb_edu_subcache_10B_llama3 = download_pretokenized_cache(
    "fineweb-edu-10B-llama3",
    "marin-community/fineweb-edu-pretokenized-10B",
    llama3_tokenizer,
)


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return {k: v / total for k, v in weights.items()}


# Composite mixture ratios (approx token targets):
# - Nemotron-CC: 40B tokens  -> 0.4
# - FineWeb-Edu: 10B tokens  -> 0.1
# - DCLM mix:    50B tokens  -> 0.5
_NEMOTRON_SHARE = 0.4
_FINEWEB_SHARE = 0.1
_DCLM_SHARE = 0.5

nemotron_dclm_fineweb_components = {
    **nemotron_cc_steps,
    **dclm_components_llama3,
    "fineweb_edu_10b": fineweb_edu_subcache_10B_llama3,
}
nemotron_dclm_fineweb_weights = {
    **{k: v * _NEMOTRON_SHARE for k, v in _normalize_weights(NEMOTRON_WEIGHTS).items()},
    **{k: v * _DCLM_SHARE for k, v in _normalize_weights(DCLM_MIXTURE_WEIGHTS).items()},
    "fineweb_edu_10b": _FINEWEB_SHARE,
}
nemotron_dclm_fineweb_mixture = lm_mixture_data_config(
    components=nemotron_dclm_fineweb_components,
    weights=nemotron_dclm_fineweb_weights,
    permutation_type="linear",
)

DATASET_OPTIONS = {
    "nemotron_cc": nemotron_cc_mixture,
    "dclm": dclm_mixture_config_llama3,
    "dclm_baseline_only": dclm_baseline_only_mixture_config_llama3,
    "fineweb_edu_10b": fineweb_edu_subcache_10B,
    "nemotron_dclm_fineweb_10b": nemotron_dclm_fineweb_mixture,
}
DEFAULT_DATASET = "nemotron_cc"


def nemotron_only_speedrun(
    name: str,
    config: SpeedrunConfig,
    tags: list[str] | None = None,
    override_output_path: str | None = None,
    *,
    append_ici_ag_pipelining_flags: bool = False,
    append_async_collective_permute_flag: bool = False,
    wandb_name: str | None = None,
    wandb_group: str | None = None,
    single_checkpoint: bool = False,
    checkpoint_save_minutes: int = 60,
    use_default_validation: bool = False,
    eval_suite: str = "none",
    eval_suite_mode: str = "post_train",
    eval_tpu_type: str | None = None,
    max_eval_instances: int | None = None,
) -> Sequence[ExecutorStep]:
    """Clone of default_speedrun that skips Paloma validation datasets."""

    logger.info(f"Running nemotron-only speedrun {name}")
    config.print_run_info()

    if eval_suite_mode not in ("post_train", "during_train", "both"):
        raise ValueError("eval_suite_mode must be one of: post_train, during_train, both")

    run_tags = ["speedrun"] + (tags or [])
    train_config = dataclasses.replace(config.train_config, data_seed=42)

    if isinstance(config.tokenized_dataset, (InputName, ExecutorStep)):
        pretraining_data = lm_data_config(
            training_set=config.tokenized_dataset,
            validation_sets=None,
            permutation_type="linear",
        )
    else:
        pretraining_data = config.tokenized_dataset

    suite_evals = None
    if eval_suite != "none":
        if eval_suite not in _EVAL_SUITES:
            raise ValueError(f"Unknown eval suite {eval_suite!r}. Options: {sorted(_EVAL_SUITES.keys())}")
        suite_evals = _EVAL_SUITES[eval_suite]

    eval_harness_tasks = ()
    if eval_suite_mode in ("during_train", "both") and suite_evals:
        eval_harness_tasks = suite_evals

    train_step = default_train(
        name=f"speedrun/{name}",
        tokenized=pretraining_data,
        model_config=config.model_config,
        train_config=train_config,
        tags=run_tags,
        eval_harness_tasks=eval_harness_tasks,
        wandb_name=wandb_name or name,
        wandb_group=wandb_group,
        override_output_path=override_output_path,
        use_default_validation=use_default_validation,
        checkpointer_save_interval=timedelta(minutes=checkpoint_save_minutes),
        checkpointer_keep=[] if single_checkpoint else None,
    )

    if append_ici_ag_pipelining_flags or append_async_collective_permute_flag:
        base_env = load_config().env_for_accel(train_config.resources.device.variant)
        base_libtpu = base_env.get("LIBTPU_INIT_ARGS", "")

        extra_flags: list[str] = []
        if append_ici_ag_pipelining_flags:
            extra_flags.extend(
                [
                    "--xla_should_allow_loop_variant_parameter_in_chain=enabled",
                    "--xla_should_add_loop_invariant_op_in_chain=enabled",
                    "--xla_tpu_enable_ici_ag_pipelining=true",
                ]
            )
        if append_async_collective_permute_flag:
            extra_flags.append("--xla_enable_async_collective_permute=true")

        combined = base_libtpu
        for flag in extra_flags:
            if flag not in combined:
                combined = f"{combined} {flag}".strip() if combined else flag

        env_vars = dict(getattr(train_step.config, "env_vars", None) or {})
        env_vars["LIBTPU_INIT_ARGS"] = combined
        train_step = dataclasses.replace(train_step, config=dataclasses.replace(train_step.config, env_vars=env_vars))

    wandb_entity = WANDB_ENTITY
    wandb_project = WANDB_PROJECT
    wandb_run_id = None

    trainer_cfg = getattr(train_step.config, "train_config", None)
    trainer = getattr(trainer_cfg, "trainer", None) if trainer_cfg else None
    tracker = getattr(trainer, "tracker", None) if trainer else None
    if tracker:
        wandb_entity = tracker.entity or WANDB_ENTITY
        wandb_project = tracker.project or WANDB_PROJECT

    if override_output_path:
        wandb_run_id = override_output_path.split("/")[-1]
    else:
        wandb_run_id = train_step  # resolved to the actual output path by the executor

    results_step = ExecutorStep(
        name=f"speedrun/{name}-speedrun_results",
        description=f"compute and store metrics and stats for the speedrun {name}.",
        fn=speedrun_results,
        config=SpeedrunResultsConfig(
            wandb_run_id=wandb_run_id,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            speedrun_config=config,
            output_path=output_path_of(train_step, "speedrun_results.json"),
        ),
    )

    steps: list[ExecutorStep] = [train_step, results_step]

    if eval_suite_mode in ("post_train", "both") and eval_suite != "none":
        if suite_evals is None:
            return steps

        train_tpu_type = getattr(getattr(train_config.resources, "device", None), "variant", None)
        if train_tpu_type is None:
            raise ValueError(f"Cannot infer TPU type from resources: {train_config.resources!r}")
        eval_resources = ResourceConfig.with_tpu(eval_tpu_type or train_tpu_type)
        hf_export_enabled = train_config.steps_per_hf_export != -1
        if _supports_hf_export(config.model_config) and hf_export_enabled:
            steps.append(
                default_eval(
                    step=train_step,
                    resource_config=eval_resources,
                    evals=suite_evals,  # type: ignore[arg-type]
                    max_eval_instances=max_eval_instances,
                    apply_chat_template=False,
                    wandb_name=f"{name}_eval_{eval_suite}",
                    wandb_group=wandb_group,
                )
            )
        else:
            # Run Levanter eval harness directly on the latest Levanter checkpoint.
            tokenizer = getattr(config.model_config, "tokenizer", None) or llama3_tokenizer
            steps.append(
                ExecutorStep(
                    name=f"evaluation/levanter_eval_harness/{name}/{eval_suite}",
                    fn=run_levanter_checkpoint_eval_harness,
                    config=LevanterEvalHarnessStepConfig(
                        model_name=f"{name}_{eval_suite}",
                        model_config=config.model_config,
                        tokenizer=tokenizer,
                        checkpoint_root=train_step / "checkpoints",
                        evals=suite_evals,  # type: ignore[arg-type]
                        max_eval_instances=max_eval_instances,
                        output_path=output_path_of(train_step, f"eval_harness/{eval_suite}"),
                        wandb_group=wandb_group,
                    ),
                    resources=eval_resources,
                    pip_dependency_groups=["tpu", "eval"],
                )
            )

    return steps


def make_speedrun_config(
    *,
    model: str,
    global_batch_size: int,
    num_train_steps: int,
    profiler: bool,
    profiler_start_step: int,
    profiler_num_steps: int,
    dataset_name: str,
    seq_len: int,
    tpu_type: str,
) -> SpeedrunConfig:
    tokenized_dataset = DATASET_OPTIONS[dataset_name]
    model_config = build_model_config(model=model, seq_len=seq_len)
    # `steps_per_export` controls how often we *keep* Levanter checkpoints.
    # It must be positive: negative values interact badly with `every=...` schedules (e.g. `step % -1 == 0`),
    # causing checkpoint/HF export hooks to fire every step.
    steps_per_export = STEPS_PER_EXPORT

    # HF export uses a separate knob. For the bilinear Llama variant we intentionally disable HF export because
    # there is no valid HF config/model_type mapping for `ActivationFunctionEnum.linear`.
    steps_per_hf_export: int | None = None
    if isinstance(getattr(model_config, "activation_function", None), ActivationFunctionEnum):
        if model_config.activation_function == ActivationFunctionEnum.linear:
            steps_per_hf_export = -1
    return SpeedrunConfig(
        author=Author(
            name="Marin Team",
            affiliation="Marin Project",
            url=None,
        ),
        description=(f"{model} speedrun on {dataset_name}."),
        model_config=model_config,
        train_config=SimpleTrainConfig(
            resources=ResourceConfig.with_tpu(tpu_type=tpu_type),
            train_batch_size=global_batch_size,
            num_train_steps=num_train_steps,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            steps_per_eval=STEPS_PER_EVAL,
            steps_per_export=steps_per_export,
            steps_per_hf_export=steps_per_hf_export,
            profiler=profiler,
            profiler_start_step=profiler_start_step,
            profiler_num_steps=profiler_num_steps,
        ),
        tokenized_dataset=tokenized_dataset,
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="OLMoE Nemotron-CC speedrun launcher.")
    parser.add_argument(
        "--model",
        choices=MODEL_OPTIONS,
        default=DEFAULT_MODEL,
        help=f"Which model preset to run (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--num-train-steps",
        type=int,
        default=None,
        help="Number of training steps to run (default: computed from --token-target, --global-batch-size, --seq-len).",
    )
    parser.add_argument(
        "--token-target",
        type=int,
        default=None,
        help=(
            "Total token budget used to compute a default --num-train-steps when that flag is omitted. "
            f"Defaults to {DEFAULT_TOKEN_TARGET} for single-corpus runs and {COMPOSITE_TOKEN_TARGET} for the composite "
            "mixture."
        ),
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=DEFAULT_GLOBAL_BATCH_SIZE,
        help="Override the global batch size (default 64).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable the JAX profiler (writes traces under ./logs/<run_id>/profiler).",
    )
    parser.add_argument(
        "--profile-start-step",
        type=int,
        default=DEFAULT_PROFILER_START_STEP,
        help=f"Step to start profiling (default {DEFAULT_PROFILER_START_STEP}).",
    )
    parser.add_argument(
        "--profile-num-steps",
        type=int,
        default=DEFAULT_PROFILER_NUM_STEPS,
        help=f"Number of steps to capture once profiling starts (default {DEFAULT_PROFILER_NUM_STEPS}).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=SEQ_LEN,
        help=f"Sequence length (default {SEQ_LEN}).",
    )
    parser.add_argument(
        "--tpu-type",
        type=str,
        default=TPU_TYPE,
        help=f"TPU type for ResourceConfig.with_tpu (default {TPU_TYPE}).",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_OPTIONS.keys(),
        default=DEFAULT_DATASET,
        help=f"Which tokenized dataset to train on (default: {DEFAULT_DATASET}).",
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
        help="How often to save temporary checkpoints when --single-checkpoint is set (default: 60).",
    )
    parser.add_argument(
        "--run-suffix",
        type=str,
        default=None,
        help="Optional override for the executor run suffix/output path.",
    )
    parser.add_argument(
        "--eval-suite",
        choices=sorted(_EVAL_SUITES.keys()),
        default="none",
        help="Eval-harness suite name (see --eval-suite-mode).",
    )
    parser.add_argument(
        "--eval-suite-mode",
        choices=("post_train", "during_train", "both"),
        default="post_train",
        help="When to run eval-harness: post_train (default), during_train, or both.",
    )
    parser.add_argument(
        "--eval-tpu-type",
        type=str,
        default=None,
        help="Optional TPU type for the eval step (default: same as --tpu-type).",
    )
    parser.add_argument(
        "--max-eval-instances",
        type=int,
        default=None,
        help="Optional cap on max eval instances per task (default: no cap).",
    )
    parser.add_argument(
        "--use-default-validation",
        action="store_true",
        help=(
            "Enable Levanter-native validation losses (e.g. Paloma + uncheatable) during training. "
            "By default this launcher skips them."
        ),
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Optional W&B group name (overrides WANDB_GROUP env var when set).",
    )
    parser.add_argument(
        "--append-ici-ag-pipelining-flags",
        action="store_true",
        help=(
            "Append the ICI all-gather pipelining XLA flags onto the baseline LIBTPU_INIT_ARGS from .levanter.yaml "
            "(useful for quick A/B without editing cluster defaults)."
        ),
    )
    parser.add_argument(
        "--append-async-collective-permute-flag",
        action="store_true",
        help=(
            "Append --xla_enable_async_collective_permute=true onto the baseline LIBTPU_INIT_ARGS from .levanter.yaml "
            "(MaxText commonly enables this; can matter for MoE routing/all-to-all-heavy workloads)."
        ),
    )
    return parser.parse_known_args()


def _steps_for_token_target(token_target: int, global_batch_size: int, seq_len: int) -> int:
    return math.ceil(token_target / (global_batch_size * seq_len))


# Keep a default config available for scripts that import this module.
speedrun_config = make_speedrun_config(
    model=DEFAULT_MODEL,
    global_batch_size=DEFAULT_GLOBAL_BATCH_SIZE,
    num_train_steps=_steps_for_token_target(DEFAULT_TOKEN_TARGET, DEFAULT_GLOBAL_BATCH_SIZE, SEQ_LEN),
    profiler=False,
    profiler_start_step=DEFAULT_PROFILER_START_STEP,
    profiler_num_steps=DEFAULT_PROFILER_NUM_STEPS,
    dataset_name=DEFAULT_DATASET,
    seq_len=SEQ_LEN,
    tpu_type=TPU_TYPE,
)


if __name__ == "__main__":
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]
    token_target = args.token_target
    if token_target is None:
        token_target = COMPOSITE_TOKEN_TARGET if args.dataset == "nemotron_dclm_fineweb_10b" else DEFAULT_TOKEN_TARGET
    num_train_steps = (
        args.num_train_steps
        if args.num_train_steps is not None
        else _steps_for_token_target(token_target, args.global_batch_size, args.seq_len)
    )
    run_config = make_speedrun_config(
        model=args.model,
        global_batch_size=args.global_batch_size,
        num_train_steps=num_train_steps,
        profiler=args.profile,
        profiler_start_step=args.profile_start_step,
        profiler_num_steps=args.profile_num_steps,
        dataset_name=args.dataset,
        seq_len=args.seq_len,
        tpu_type=args.tpu_type,
    )
    logger.info("Launching Nemotron speedrun.")
    effective_tokens = run_config.train_config.train_batch_size * args.seq_len * run_config.train_config.num_train_steps
    logger.info(
        "Settings: dataset=%s, batch=%s, seq_len=%s, steps=%s (~%.2fB tokens)",
        args.dataset,
        run_config.train_config.train_batch_size,
        args.seq_len,
        run_config.train_config.num_train_steps,
        effective_tokens / 1e9,
    )
    logger.info("Model preset: %s", args.model)
    if hasattr(run_config.model_config, "use_gmm"):
        logger.info("Model config flags: use_gmm=%s", run_config.model_config.use_gmm)
    if args.profile:
        logger.info(
            "Profiler enabled: start_step=%s num_steps=%s",
            run_config.train_config.profiler_start_step,
            run_config.train_config.profiler_num_steps,
        )
    run_suffix = (
        args.run_suffix
        or f"{args.model}_nemotron_40b_{args.tpu_type}_bs{args.global_batch_size}_{args.dataset}_seq{args.seq_len}"
    )
    if args.profile:
        run_suffix += f"_profile_s{args.profile_start_step}_n{args.profile_num_steps}"
    logger.info("LIBTPU_INIT_ARGS=%s", os.environ.get("LIBTPU_INIT_ARGS", "<unset>"))
    executor_main(
        steps=nemotron_only_speedrun(
            run_suffix,
            run_config,
            append_ici_ag_pipelining_flags=args.append_ici_ag_pipelining_flags,
            append_async_collective_permute_flag=args.append_async_collective_permute_flag,
            eval_suite=args.eval_suite,
            eval_suite_mode=args.eval_suite_mode,
            eval_tpu_type=args.eval_tpu_type,
            max_eval_instances=args.max_eval_instances,
            single_checkpoint=args.single_checkpoint,
            checkpoint_save_minutes=args.checkpoint_save_minutes,
            use_default_validation=args.use_default_validation,
            wandb_group=args.wandb_group if args.wandb_group is not None else os.environ.get("WANDB_GROUP"),
        )
    )
