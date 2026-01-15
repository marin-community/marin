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
import argparse
import dataclasses
import logging
import math
import os
import sys
from collections.abc import Sequence

from experiments.defaults import default_train
from experiments.pretraining_datasets import NEMOTRON_WEIGHTS, tokenize_nemotron
from experiments.pretraining_datasets.dclm import dclm_mixture_config_llama3
from experiments.llama import llama3_tokenizer
from experiments.speedrun.custom_mixtral import MixtralConfig
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from levanter.infra.cli_helpers import load_config
from marin.execution import step, deferred, ExecutorStep, executor_main, StepRef
from marin.processing.tokenize import lm_data_config, lm_mixture_data_config
from marin.speedrun.speedrun import Author, SpeedrunConfig, SpeedrunResultsConfig, speedrun_results as _speedrun_results
from marin.utilities.wandb_utils import WANDB_ENTITY, WANDB_PROJECT

speedrun_results = deferred(_speedrun_results)

logger = logging.getLogger("ray")

# ---------------------------------------------------------------------------
# Shared experiment knobs (mirrors the dense baseline for flop matching)
# ---------------------------------------------------------------------------
SEQ_LEN = 2048
DEFAULT_GLOBAL_BATCH_SIZE = 64
TOKEN_TARGET = 40_000_000_000  # 40B tokens
NUM_TRAIN_STEPS = math.ceil(TOKEN_TARGET / (DEFAULT_GLOBAL_BATCH_SIZE * SEQ_LEN))
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.1
STEPS_PER_EVAL = 5000
STEPS_PER_EXPORT = 20_000
TPU_TYPE = "v5p-64"
DEFAULT_PROFILER_START_STEP = 3
DEFAULT_PROFILER_NUM_STEPS = 20

MODEL_OLMOE_1B7B = "olmoe_1b7b"
MODEL_MIXTRAL_8X7B = "mixtral_8x7b"
MODEL_OPTIONS = (MODEL_OLMOE_1B7B, MODEL_MIXTRAL_8X7B)
DEFAULT_MODEL = MODEL_OLMOE_1B7B

OLMOE_1B7B_REFERENCE_CHECKPOINT = "allenai/OLMoE-1B-7B-0125"


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


def build_model_config(*, model: str, seq_len: int) -> MixtralConfig:
    if model == MODEL_OLMOE_1B7B:
        return _build_olmoe_1b7b_config(seq_len)
    if model == MODEL_MIXTRAL_8X7B:
        return _build_mixtral_8x7b_config(seq_len)
    raise ValueError(f"Unknown model preset {model!r}. Options: {MODEL_OPTIONS}.")


nemotron_cc_steps = tokenize_nemotron(tokenizer=llama3_tokenizer)
nemotron_cc_mixture = lm_mixture_data_config(
    components=nemotron_cc_steps,
    weights=NEMOTRON_WEIGHTS,
    permutation_type="linear",
)

DATASET_OPTIONS = {
    "nemotron_cc": nemotron_cc_mixture,
    "dclm": dclm_mixture_config_llama3,
}
DEFAULT_DATASET = "nemotron_cc"


@step(name="speedrun/{name}-speedrun_results")
def _results_step_impl(
    name: str,
    wandb_run_id: str | ExecutorStep,
    wandb_entity: str,
    wandb_project: str,
    config: SpeedrunConfig,
    train_step: ExecutorStep,
) -> ExecutorStep:
    return speedrun_results(
        SpeedrunResultsConfig(
            wandb_run_id=wandb_run_id,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            speedrun_config=config,
            output_path=train_step / "speedrun_results.json",
        )
    )


def nemotron_only_speedrun(
    name: str,
    config: SpeedrunConfig,
    tags: list[str] | None = None,
    override_output_path: str | None = None,
    *,
    append_ici_ag_pipelining_flags: bool = False,
    append_async_collective_permute_flag: bool = False,
) -> Sequence[ExecutorStep]:
    """Clone of default_speedrun that skips Paloma validation datasets."""

    logger.info(f"Running nemotron-only speedrun {name}")
    config.print_run_info()

    run_tags = ["speedrun"] + (tags or [])
    train_config = dataclasses.replace(config.train_config, data_seed=42)

    if isinstance(config.tokenized_dataset, (StepRef, ExecutorStep)):
        pretraining_data = lm_data_config(
            training_set=config.tokenized_dataset,
            validation_sets=[],
            permutation_type="linear",
        )
    else:
        pretraining_data = config.tokenized_dataset

    train_step = default_train(
        name=f"speedrun/{name}",
        tokenized=pretraining_data,
        model_config=config.model_config,
        train_config=train_config,
        tags=run_tags,
        eval_harness_tasks=None,
        override_output_path=override_output_path,
        use_default_validation=False,
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

    return [
        train_step,
        _results_step_impl(
            name=name,
            wandb_run_id=wandb_run_id,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            config=config,
            train_step=train_step,
        ),
    ]


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
            steps_per_export=STEPS_PER_EXPORT,
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
        default=NUM_TRAIN_STEPS,
        help=f"Number of training steps to run (default {NUM_TRAIN_STEPS}, i.e. ~40B tokens).",
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
        help="Which tokenized dataset to train on (default: dclm).",
    )
    parser.add_argument(
        "--run-suffix",
        type=str,
        default=None,
        help="Optional override for the executor run suffix/output path.",
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


# Keep a default config available for scripts that import this module.
speedrun_config = make_speedrun_config(
    model=DEFAULT_MODEL,
    global_batch_size=DEFAULT_GLOBAL_BATCH_SIZE,
    num_train_steps=NUM_TRAIN_STEPS,
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
    if args.num_train_steps == NUM_TRAIN_STEPS and (
        args.global_batch_size != DEFAULT_GLOBAL_BATCH_SIZE or args.seq_len != SEQ_LEN
    ):
        num_train_steps = math.ceil(TOKEN_TARGET / (args.global_batch_size * args.seq_len))
    else:
        num_train_steps = args.num_train_steps
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
    logger.info("Launching MoE Nemotron speedrun.")
    logger.info(
        "Settings: dataset=%s, batch=%s, seq_len=%s, steps=%s (~%.2fB tokens)",
        args.dataset,
        run_config.train_config.train_batch_size,
        run_config.model_config.seq_len,
        run_config.train_config.num_train_steps,
        TOKEN_TARGET / 1e9,
    )
    logger.info("Model preset: %s", args.model)
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
        )
    )
