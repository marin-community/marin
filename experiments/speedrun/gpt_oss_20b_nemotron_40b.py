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
GPT-OSS-20B (MoE) Nemotron-CC speedrun launcher.

This file mirrors `experiments/speedrun/olmoe_1b7b_nemotron_40b.py`, but with a GPT-OSS-20B-inspired MoE
configuration (as published in the Hugging Face `openai/gpt-oss-20b` config).

Notes / approximations:
- We tokenize Nemotron-CC with the Llama 3.1 tokenizer used elsewhere in Marin speedruns. GPT-OSS uses a different
  tokenizer (o200k_harmony); using Llama 3 lets this run with our existing pretokenized dataset recipes.
- GPT-OSS alternates sliding-window and full attention and supports very long context (128k). Levanter's
  `experiments/speedrun/custom_mixtral.py` model is a single attention mode and we set `seq_len` to the training
  sequence length you pass (`--seq-len`).
- The HF config specifies `head_dim=64` with `hidden_size=2880` and `num_attention_heads=64`. Our MixtralConfig
  derives head size as `hidden_dim // num_heads` and the RoPE implementation requires an even head size. We therefore
  bump `hidden_dim` to 3072 so `head_size = 3072 // 64 = 48`.
"""

# nodryrun
import argparse
import dataclasses
import logging
import math
import os
import sys
from collections.abc import Sequence

from experiments.defaults import default_train
from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets import tokenize_nemotron
from experiments.pretraining_datasets.dclm import dclm_mixture_config_llama3
from experiments.speedrun.custom_mixtral import MixtralConfig
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, InputName, ExecutorMainConfig, executor_main, output_path_of
from marin.processing.tokenize import lm_data_config, lm_mixture_data_config
from marin.speedrun.speedrun import Author, SpeedrunConfig, SpeedrunResultsConfig, speedrun_results
from marin.utilities.wandb_utils import WANDB_ENTITY, WANDB_PROJECT

logger = logging.getLogger("ray")

# ---------------------------------------------------------------------------
# Shared experiment knobs (mirrors the dense baseline for flop matching)
# ---------------------------------------------------------------------------

SEQ_LEN = 2048
DEFAULT_GLOBAL_BATCH_SIZE = 64
TOKEN_TARGET = 40_000_000_000  # 40B tokens
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.1
STEPS_PER_EVAL = 5000
STEPS_PER_EXPORT = 20_000
TPU_TYPE = "v5p-64"
DEFAULT_PROFILER_START_STEP = 3
DEFAULT_PROFILER_NUM_STEPS = 20

# ---------------------------------------------------------------------------
# Dataset options (shared with other speedruns)
# ---------------------------------------------------------------------------

nemotron_cc_steps = tokenize_nemotron(tokenizer=llama3_tokenizer)
nemotron_cc_mixture = lm_mixture_data_config(
    components={"nemotron_cc/hq_actual": nemotron_cc_steps["nemotron_cc/hq_actual"]},
    weights={"nemotron_cc/hq_actual": 1.0},
    permutation_type="linear",
)

DATASET_OPTIONS = {
    "nemotron_cc": nemotron_cc_mixture,
    "dclm": dclm_mixture_config_llama3,
}
DEFAULT_DATASET = "nemotron_cc"


def _num_train_steps_for_token_target(*, token_target: int, global_batch_size: int, seq_len: int) -> int:
    return math.ceil(token_target / (global_batch_size * seq_len))


def _build_gpt_oss_20b_config(seq_len: int) -> MixtralConfig:
    # Based on HF config for openai/gpt-oss-20b (approximate mapping to our MixtralConfig).
    # https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json
    return MixtralConfig(
        seq_len=seq_len,
        hidden_dim=3072,
        intermediate_dim=9216,
        num_layers=24,
        num_heads=64,
        num_kv_heads=16,
        n_routed_experts=32,
        num_experts_per_tok=4,
        layer_norm_epsilon=1e-6,
        gradient_checkpointing=True,
        scan_layers=True,
        use_gmm=True,
        cross_entropy_block_size=32000,
        flash_attention_block_size=1024,
        reference_checkpoint=None,
        # Needed so Levanter can construct an HFCheckpointConverter for export, even when training from scratch.
        tokenizer=llama3_tokenizer,
    )


def make_speedrun_config(
    *,
    dataset_name: str,
    seq_len: int,
    tpu_type: str,
    global_batch_size: int,
    num_train_steps: int,
    profiler: bool,
    profiler_start_step: int,
    profiler_num_steps: int,
) -> SpeedrunConfig:
    cfg = _build_gpt_oss_20b_config(seq_len=seq_len)

    resources = ResourceConfig.with_tpu(tpu_type)
    train_config = SimpleTrainConfig(
        resources=resources,
        train_batch_size=global_batch_size,
        train_seq_len=seq_len,
        num_train_steps=num_train_steps,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=1.0,
        steps_per_eval=STEPS_PER_EVAL,
        steps_per_hf_export=STEPS_PER_EXPORT,
        profiler=profiler,
        profiler_start_step=profiler_start_step,
        profiler_num_steps=profiler_num_steps,
    )

    return SpeedrunConfig(
        author=Author(name="TODO", affiliation="TODO", url=""),
        description="GPT-OSS-20B-inspired MoE speedrun on nemotron_cc.",
        model_config=cfg,
        train_config=train_config,
        tokenized_dataset=DATASET_OPTIONS[dataset_name],
    )


def nemotron_only_speedrun(
    name: str,
    config: SpeedrunConfig,
    tags: list[str] | None = None,
    override_output_path: str | None = None,
) -> Sequence[ExecutorStep]:
    """Clone of default_speedrun that skips Paloma validation datasets."""

    logger.info(f"Running nemotron-only speedrun {name}")
    config.print_run_info()

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

    return [train_step, results_step]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT-OSS-20B-inspired MoE speedrun on Nemotron-40B.")
    parser.add_argument(
        "--dataset",
        choices=DATASET_OPTIONS.keys(),
        default=DEFAULT_DATASET,
        help=f"Which dataset mixture to train on (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument("--tpu-type", default=TPU_TYPE, help=f"TPU slice type (default: {TPU_TYPE}).")
    parser.add_argument("--global-batch-size", type=int, default=DEFAULT_GLOBAL_BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument(
        "--token-target",
        type=int,
        default=TOKEN_TARGET,
        help="Target total tokens (used to derive --num-train-steps when it is not provided).",
    )
    parser.add_argument(
        "--num-train-steps",
        type=int,
        default=None,
        help=(
            "Override the number of training steps (otherwise computed from --token-target, --global-batch-size, "
            "and --seq-len)."
        ),
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-start-step", type=int, default=DEFAULT_PROFILER_START_STEP)
    parser.add_argument("--profile-num-steps", type=int, default=DEFAULT_PROFILER_NUM_STEPS)
    parser.add_argument("--run-suffix", default=None)
    # Executor controls (so this script can be run under ray_run without draccus CLI conflicts).
    parser.add_argument("--prefix", default=os.getenv("MARIN_PREFIX"))
    parser.add_argument("--executor-info-base-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-run-failed", action="store_true")
    parser.add_argument("--run-only", nargs="*", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    # log LIBTPU_INIT_ARGS for debugging (helps confirm TPU flags are being passed through)
    if "LIBTPU_INIT_ARGS" in os.environ:
        logger.info(f"LIBTPU_INIT_ARGS={os.environ['LIBTPU_INIT_ARGS']}")

    num_train_steps = args.num_train_steps
    if num_train_steps is None:
        num_train_steps = _num_train_steps_for_token_target(
            token_target=args.token_target,
            global_batch_size=args.global_batch_size,
            seq_len=args.seq_len,
        )

    cfg = make_speedrun_config(
        dataset_name=args.dataset,
        seq_len=args.seq_len,
        tpu_type=args.tpu_type,
        global_batch_size=args.global_batch_size,
        num_train_steps=num_train_steps,
        profiler=args.profile,
        profiler_start_step=args.profile_start_step,
        profiler_num_steps=args.profile_num_steps,
    )

    run_suffix = (
        args.run_suffix
        if args.run_suffix
        else f"gpt_oss_20b_nemotron40b_{args.tpu_type}_bs{args.global_batch_size}_seq{args.seq_len}"
    )
    if args.profile:
        run_suffix += f"_profile_s{args.profile_start_step}_n{args.profile_num_steps}"

    steps = nemotron_only_speedrun(run_suffix, cfg)
    executor_cfg = ExecutorMainConfig(
        prefix=args.prefix,
        executor_info_base_path=args.executor_info_base_path,
        dry_run=args.dry_run,
        force_run_failed=args.force_run_failed,
        run_only=args.run_only,
    )
    executor_main.__wrapped__(executor_cfg, steps=steps, description="GPT-OSS-20B MoE speedrun on nemotron_cc.")


if __name__ == "__main__":
    main(sys.argv[1:])
