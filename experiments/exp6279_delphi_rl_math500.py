# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# nodryrun because vLLM is not installed by default

"""Delphi RL MATH-500 scaling-law probe launcher (issue #6279).

Runs the standard 100-step MATH-500 RL probe against checkpoints from the
Delphi K=0.20 midtraining ladder. Two kinds of starting points are supported:

- cold-start SFT checkpoints, e.g. ``--checkpoint 9e19-p33m67-magpie_lr1e5``:
  Delphi midtrained models put through the cold-start instruction SFT
  calibration grid (HF ``laion/delphi-*-coldstart-*``). These cover priority
  rows 1/5/6 of the issue plan.
- raw midtrained endpoints for RL-zero, e.g. ``--checkpoint 9e19-p33m67``:
  the best-endpoint checkpoints from the issue table (priority rows 3/4).
  These are HF exports under ``checkpoints/`` in the us-east5 Marin bucket,
  so the run must be launched in us-east5.

The MATH-500 envelope (MathEnv prompt format, boxed-answer reward contract,
sampling settings, RLOO loss) is shared with ``llama_3_8b_rl_math500.py`` so
probes stay comparable across model scales, mixes, and SFT recipes.
"""

import argparse
import datetime
import logging
import os

from levanter.models.qwen import Qwen3Config
from marin.execution.executor import executor_main
from marin.execution.types import InputName
from marin.rl.rl_experiment_utils import (
    ModelConfig,
    RLExperimentConfig,
    config_class_path,
    executor_main_config_for_rl_experiment,
    make_rl_step,
)

from experiments.llama_3_8b_rl_math500 import (
    DEFAULT_CHECKPOINTER_SAVE_INTERVAL,
    DEFAULT_EVAL_FREQUENCY,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_N_PROMPTS,
    DEFAULT_NUM_ROLLOUT_WORKERS,
    DEFAULT_TPU_TYPE,
    DEFAULT_TPU_WORKER_RAM,
    build_math500_curriculum,
    default_math500_rl_loss,
)
from experiments.models import ModelConfig as HfHubModelConfig
from experiments.models import download_model_step

logger = logging.getLogger(__name__)

PROJECT_NAME = "delphi_rl_scaling"

# Issue protocol: 100-step probe; little improvement was seen past ~100 steps
# on the Llama 3.1 8B Instruct MATH-500 run (iris-rl-e4ms2-500).
DEFAULT_NUM_TRAIN_STEPS = 100
DEFAULT_LEARNING_RATE = 2e-6
DEFAULT_CHECKPOINT = "9e19-p33m67-magpie_lr1e5"

# (attempt suffix, endpoint step) for the K=0.20 lr83 sweep scales. All three
# mixes at a given scale share the same attempt and endpoint step.
_K020_LR83_ENDPOINTS: dict[str, tuple[str, int]] = {
    "3e18": ("a003", 7399),
    "9e18": ("a002", 8818),
    "2e19": ("a002", 10982),
    "3e19": ("a002", 7573),
    "9e19": ("a002", 8032),
    "2e20": ("a001", 11277),
    "3e20": ("a001", 7081),
}

# Best-endpoint midtrained checkpoints from the #6279 table, used for RL-zero.
# Paths are relative to the Marin prefix; the HF exports live in marin-us-east5.
DELPHI_MIDTRAIN_ENDPOINTS: dict[str, str] = {
    f"{scale}-{mix}": f"checkpoints/delphi-{scale}-{mix}-k0p20-lr83-{attempt}/hf/step-{step}"
    for scale, (attempt, step) in _K020_LR83_ENDPOINTS.items()
    for mix in ("p33m67", "p50m50", "p67m33")
} | {
    "1e21-p33m67": "checkpoints/delphi-1e21-p33m67-9p25b-lr0.67-9cf8da/hf/step-4410",
    "1e21-p50m50": "checkpoints/delphi-1e21-p50m50-9p25b-lr0.83-f9edd2/hf/step-4410",
    "1e21-p67m33": "checkpoints/delphi-1e21-p67m33-9p25b-lr0.67-ecbd27/hf/step-4410",
    "1e22-p33m67": "checkpoints/delphi-1e22-p33m67-32p07b-lr0.67-54770ae7/hf/step-7646",
    "1e22-p50m50": "checkpoints/delphi-1e22-p50m50-32p07b-lr0.5-ecfa99/hf/step-7646",
    "1e22-p67m33": "checkpoints/delphi-1e22-p67m33-32p07b-lr0.33-4e8cc7a7/hf/step-7646",
}

# Cold-start SFT winners from the calibration grid on the 9e19 p33m67 anchor:
# magpie_lr1e5 (math-strong) and wc386k_lr1e5 (math-weak) are both kept so the
# main experiments can separate the cold-start math prior from midtraining+RL.
DELPHI_COLDSTART_SFT: dict[str, HfHubModelConfig] = {
    "9e19-p33m67-magpie_lr1e5": HfHubModelConfig(
        hf_repo_id="laion/delphi-9e19-p33m67-coldstart-magpie_lr1e5",
        hf_revision="8be0758",
    ),
    "9e19-p33m67-wc386k_lr1e5": HfHubModelConfig(
        hf_repo_id="laion/delphi-9e19-p33m67-coldstart-wc386k_lr1e5",
        hf_revision="0b9538f",
    ),
}


def delphi_model_config(checkpoint: str) -> ModelConfig:
    """Build the RL model config for a Delphi checkpoint registry key.

    Delphi checkpoints are Qwen3-architecture HF exports that use the
    marin/Llama-3 tokenizer and chat template, so the model ``type`` (which
    only selects stop tokens) is ``llama``.
    """
    if checkpoint in DELPHI_COLDSTART_SFT:
        hub_config = DELPHI_COLDSTART_SFT[checkpoint]
        return ModelConfig(
            name=hub_config.hf_repo_id,
            type="llama",
            artifact=download_model_step(hub_config),
            config_class_path=config_class_path(Qwen3Config),
        )

    if checkpoint in DELPHI_MIDTRAIN_ENDPOINTS:
        path = DELPHI_MIDTRAIN_ENDPOINTS[checkpoint]
        run_name = path.split("/")[1]
        return ModelConfig(
            name=f"marin-community/{run_name}",
            type="llama",
            artifact=InputName.hardcoded(path),
            config_class_path=config_class_path(Qwen3Config),
        )

    known = sorted([*DELPHI_COLDSTART_SFT, *DELPHI_MIDTRAIN_ENDPOINTS])
    raise ValueError(f"Unknown Delphi checkpoint {checkpoint!r}. Known checkpoints: {known}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Delphi checkpoint registry key: a cold-start SFT model "
        f"({sorted(DELPHI_COLDSTART_SFT)}) or a '<scale>-<mix>' midtrained endpoint for RL-zero.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Stable logical run name for checkpoint/W&B resume across relaunches. "
        "If omitted, a fresh timestamped name is generated.",
    )
    parser.add_argument(
        "--num-train-steps",
        type=int,
        default=DEFAULT_NUM_TRAIN_STEPS,
        help="Number of trainer steps to run.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="RL trainer learning rate.",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=DEFAULT_N_PROMPTS,
        help="Number of prompts sampled per rollout batch.",
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=DEFAULT_EVAL_FREQUENCY,
        help="Full-eval cadence in completed trainer steps.",
    )
    parser.add_argument(
        "--checkpointer-save-interval",
        type=int,
        default=DEFAULT_CHECKPOINTER_SAVE_INTERVAL,
        help="Seconds between trainer checkpoint saves.",
    )
    parser.add_argument(
        "--num-rollout-workers",
        type=int,
        default=DEFAULT_NUM_ROLLOUT_WORKERS,
        help="Number of rollout worker jobs to launch.",
    )
    parser.add_argument(
        "--train-tpu-type",
        default=DEFAULT_TPU_TYPE,
        help="TPU type for the trainer job.",
    )
    parser.add_argument(
        "--inference-tpu-type",
        default=DEFAULT_TPU_TYPE,
        help="TPU type for rollout worker jobs.",
    )
    parser.add_argument(
        "--num-train-slices",
        type=int,
        default=1,
        help="Number of TPU slices for the trainer job.",
    )
    parser.add_argument(
        "--train-ram",
        default=DEFAULT_TPU_WORKER_RAM,
        help="Host RAM request for the trainer TPU job.",
    )
    parser.add_argument(
        "--inference-ram",
        default=DEFAULT_TPU_WORKER_RAM,
        help="Host RAM request for each rollout TPU job.",
    )
    parser.add_argument(
        "--zone",
        default=None,
        help="Optional concrete zone for trainer and rollout TPU jobs.",
    )
    parser.add_argument(
        "--inflight-weight-updates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable inflight rollout weight updates.",
    )
    parser.add_argument(
        "--project-name",
        default=PROJECT_NAME,
        help="W&B project name for trainer and rollout runs.",
    )
    return parser.parse_args()


def build_experiment_config(args: argparse.Namespace) -> RLExperimentConfig:
    model_config = delphi_model_config(args.checkpoint)
    tags = [
        "rl",
        "iris-rl",
        "executor",
        "math500",
        "delphi",
        args.checkpoint,
    ]

    return RLExperimentConfig(
        model_config=model_config,
        rl_loss=default_math500_rl_loss(),
        experiment_name_suffix=f"math500-{args.checkpoint}",
        project_name=args.project_name,
        tags=tags,
        num_train_steps=args.num_train_steps,
        checkpointer_save_interval=args.checkpointer_save_interval,
        train_batch_size=1024,
        per_device_parallelism=16,
        learning_rate=args.learning_rate,
        max_input_tokens=DEFAULT_MAX_INPUT_TOKENS,
        max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        n_prompts=args.n_prompts,
        n_generations_per_prompt=16,
        num_rollout_workers=args.num_rollout_workers,
        train_tpu_type=args.train_tpu_type,
        inference_tpu_type=args.inference_tpu_type,
        num_train_slices=args.num_train_slices,
        train_ram=args.train_ram,
        inference_ram=args.inference_ram,
        zone=args.zone,
        inflight_weight_updates=args.inflight_weight_updates,
        max_rollout_step_delay=1,
        weight_transfer_sync_interval_steps=1,
        max_weight_transfer_wait_time=0,
    )


def build_run_name(checkpoint: str, explicit_run_name: str | None) -> str:
    if explicit_run_name is not None:
        return explicit_run_name

    datestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"delphi-rl-{checkpoint}-{datestamp}"


def main() -> None:
    if os.getenv("CI") is not None:
        logger.info("Skipping experiment execution on CI environment.")
        return

    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    experiment_config = build_experiment_config(args)
    run_name = build_run_name(args.checkpoint, args.run_name)
    curriculum = build_math500_curriculum(run_name, experiment_config, args.eval_frequency)
    step = make_rl_step(
        name=run_name,
        config=experiment_config,
        curriculum=curriculum,
    )
    executor_config = executor_main_config_for_rl_experiment(experiment_config)

    logger.info(
        "Launching Delphi RL Math500 probe %s from checkpoint %s (train_tpu=%s, inference_tpu=%s, "
        "rollout_workers=%d, zone=%s, inflight=%s, executor_prefix=%s)",
        run_name,
        args.checkpoint,
        experiment_config.train_tpu_type,
        experiment_config.inference_tpu_type,
        experiment_config.num_rollout_workers,
        experiment_config.zone,
        experiment_config.inflight_weight_updates,
        executor_config.prefix,
    )

    executor_main(
        executor_config,
        steps=[step],
        description=f"Delphi RL MATH-500 scaling probe (#6279): {args.checkpoint}",
    )


if __name__ == "__main__":
    main()
