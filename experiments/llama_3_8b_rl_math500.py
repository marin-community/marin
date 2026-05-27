# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# nodryrun because vLLM is not installed by default

"""Canonical executor-backed Iris RL Math500 launcher for Llama 3.1 8B.

This is the clean executor-first replacement for the older ``exp2039`` RL script.
It keeps the current production-like Math500 envelope, uses the executor-managed
regional model artifact, and routes launch details through the shared RL
experiment utilities so executor and direct probes stay aligned.
"""

import argparse
import datetime
import logging
import os

from levanter.checkpoint import CheckpointDebugConfig
from levanter.models.llama import LlamaConfig
from marin.execution.executor import executor_main
from marin.rl.curriculum import CurriculumConfig, EvalSamplingParams, LessonConfig, SamplingParams
from marin.rl.environments import EnvConfig
from marin.rl.kl_regularization import KLConfig, KLMode
from marin.rl.rl_experiment_utils import (
    ModelConfig,
    RLExperimentConfig,
    config_class_path,
    executor_main_config_for_rl_experiment,
    get_stop_tokens,
    make_rl_step,
)
from marin.rl.rl_losses import RLOOLoss

from experiments.models import llama_3_1_8b_instruct

logger = logging.getLogger(__name__)

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
PROJECT_NAME = "marin_iris_rl_debug"
DEFAULT_EXPERIMENT_NAME_SUFFIX = "math500"
DEFAULT_SEED = 42
DEFAULT_NUM_TRAIN_STEPS = 500
DEFAULT_CHECKPOINTER_SAVE_INTERVAL = 600
DEFAULT_N_PROMPTS = 64
DEFAULT_EVAL_FREQUENCY = 1
DEFAULT_NUM_ROLLOUT_WORKERS = 2
DEFAULT_TPU_TYPE = "v5p-8"
DEFAULT_TPU_WORKER_RAM = "400g"
DEFAULT_MAX_INPUT_TOKENS = 1024
DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_EVAL_N_EXAMPLES = 500

LLAMA_3_1_8B_INSTRUCT = ModelConfig(
    name=MODEL_NAME,
    type="llama",
    artifact=llama_3_1_8b_instruct,
    config_class_path=config_class_path(LlamaConfig),
)


def _default_rl_loss() -> RLOOLoss:
    """RLOO + importance-sampling hybrid.

    RLOO (Reward Leave-One-Out, Ahmadian et al. 2024, arXiv:2402.14740) provides
    the variance-reduced REINFORCE baseline. PPO-style clipped importance
    sampling (Schulman et al. 2017) handles the off-policy correction needed
    when rollouts arrive with bounded staleness via the replay buffer.

    Clip-higher (epsilon_low=0.2, epsilon_high=0.28) follows DAPO
    (Yu et al. 2025, arXiv:2503.14476) to slow PPO-clip-induced entropy
    collapse.

    KL is intentionally OFF: with clip-higher providing the trust region and
    Iris weight-transfer keeping rollouts close to on-policy, the held-out
    decay we care about is more cheaply fought via clip + dynamic sampling.
    Re-enable KL only after the per-token aggregator in
    ``marin.rl.kl_regularization.masked_response_mean`` is fixed to match the
    token-level loss form (see rl_blog logbook F5).
    """
    return RLOOLoss(
        kl=KLConfig(mode=KLMode.NONE, beta=0.0),
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.28,
        # IS hybrid: ratio is computed against the sampler's stored logprobs so
        # that PPO clipping actually fires for stale tokens (synchronous=True
        # forced ratio == 1, which made clip-* dead code).
        synchronous=False,
        do_trainer_inference_mismatch_importance_sampling=True,
        tis_importance_sampling_ratio_max=2.0,
        do_overlong_filtering=True,
        vocab_tile_size=32064,
        # log_policy_entropy=True is the cleaner entropy diagnostic but is
        # currently incompatible with vocab_tile_size (RLOOLoss raises a
        # ValueError when both are set). Disabling vocab_tile_size would
        # materialize a [batch, seq, vocab] logits tensor (~67 GiB/chip on
        # v5p-8 at batch=1024/seq=512/vocab=128k), which OOMs. Keep the
        # tiling and rely on policy_entropy / current_entropy (NLL of
        # sampled tokens) as a noisier-but-correlated proxy. Re-enable once
        # chunked entropy is implemented (see rl_blog logbook F3 follow-up).
        log_policy_entropy=False,
    )


def _region_from_zone(zone: str | None) -> str | None:
    if zone is None:
        return None
    region, separator, _zone_suffix = zone.rpartition("-")
    if not separator or not region:
        raise ValueError(f"Invalid GCP zone: {zone}")
    return region


def build_math500_curriculum(run_id: str, config: RLExperimentConfig, eval_frequency: int) -> CurriculumConfig:
    sampling_params = SamplingParams(
        temperature=1.0,
        n_prompts=config.n_prompts,
        n_generations_per_prompt=config.n_generations_per_prompt,
        max_output_tokens=config.max_output_tokens,
        top_k=config.inference_top_k,
        stop_tokens=None,
    )

    return CurriculumConfig(
        lessons={
            "math_full": LessonConfig(
                lesson_id="math_full",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.math_env.MathEnv",
                    env_args={"seed": 42},
                ),
                dependencies=[],
                sampling_params=sampling_params,
            ),
        },
        eval_frequency=eval_frequency,
        eval_sampling_params=[
            EvalSamplingParams(
                name="pass1_greedy",
                n_examples=DEFAULT_EVAL_N_EXAMPLES,
                n_generations=1,
                temperature=0.0,
                top_k=-1,
                max_output_tokens=config.max_output_tokens,
                stop_tokens=get_stop_tokens(config.model_config.type),
                update_curriculum_stats=True,
            ),
            EvalSamplingParams(
                name="pass16_sample",
                n_examples=DEFAULT_EVAL_N_EXAMPLES,
                n_generations=config.n_generations_per_prompt,
                temperature=1.0,
                top_k=config.inference_top_k,
                max_output_tokens=config.max_output_tokens,
                stop_tokens=get_stop_tokens(config.model_config.type),
                update_curriculum_stats=False,
            ),
        ],
        micro_eval_frequency=None,
        actor_name=f"curriculum-{run_id}",
        eval_n_examples=DEFAULT_EVAL_N_EXAMPLES,
        max_seq_len=config.max_input_tokens + config.max_output_tokens,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-name",
        default=None,
        help="Stable logical run name for checkpoint/W&B resume across relaunches. "
        "If omitted, a fresh timestamped name is generated.",
    )
    parser.add_argument(
        "--experiment-name-suffix",
        default=DEFAULT_EXPERIMENT_NAME_SUFFIX,
        help="Suffix used in the executor step name, checkpoint path, and W&B runs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base seed propagated to trainer, replay sampling, rollout sampling, and vLLM engine.",
    )
    parser.add_argument(
        "--num-train-steps",
        type=int,
        default=DEFAULT_NUM_TRAIN_STEPS,
        help="Number of trainer steps to run.",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=DEFAULT_N_PROMPTS,
        help="Number of prompts sampled per rollout batch.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Maximum generated response tokens for train and eval rollouts.",
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
        "--delete-previous-temporary-checkpoint-after-save",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete the previous temporary checkpoint after a successful new save. "
        "Defaults to false to match the current clean 500-step baseline.",
    )
    parser.add_argument(
        "--debug-checkpointer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable verbose trainer-side checkpoint diagnostics.",
    )
    parser.add_argument(
        "--debug-checkpointer-log-interval",
        type=float,
        default=60.0,
        help="Seconds between checkpoint progress logs when debug checkpointer is enabled.",
    )
    parser.add_argument(
        "--debug-checkpointer-dump-stacks-after",
        type=float,
        default=60.0,
        help="Dump Python thread stacks after this many seconds in one checkpoint phase when "
        "debug checkpointer is enabled.",
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
        "--launcher-region",
        default=None,
        help="Concrete region for executor output paths and RL child resource constraints. "
        "Defaults to the region implied by --zone when a zone is provided.",
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
    parser.add_argument(
        "--override-output-path",
        default=None,
        help="Concrete executor output path to reuse when recovering an interrupted RL run. "
        "The run will fail fast if the saved RL config guard does not match the current config.",
    )
    return parser.parse_args()


def build_experiment_config(args: argparse.Namespace) -> RLExperimentConfig:
    tags = [
        "rl",
        "iris-rl",
        "executor",
        "math500",
        args.experiment_name_suffix,
        LLAMA_3_1_8B_INSTRUCT.safe_name,
    ]

    return RLExperimentConfig(
        model_config=LLAMA_3_1_8B_INSTRUCT,
        rl_loss=_default_rl_loss(),
        experiment_name_suffix=args.experiment_name_suffix,
        project_name=args.project_name,
        tags=tags,
        seed=args.seed,
        num_train_steps=args.num_train_steps,
        checkpointer_save_interval=args.checkpointer_save_interval,
        delete_previous_temporary_checkpoint_after_save=args.delete_previous_temporary_checkpoint_after_save,
        checkpoint_debug=CheckpointDebugConfig(
            enabled=args.debug_checkpointer,
            log_interval=args.debug_checkpointer_log_interval,
            dump_stacks_after=args.debug_checkpointer_dump_stacks_after,
        ),
        train_batch_size=1024,
        per_device_parallelism=16,
        # Was 2e-6 before the F1 fix to compute_dapo_loss. That bug silently
        # divided the loss by batch_size (1024), so the effective LR was
        # ~2e-9. After the fix, the canonical token-level normalization
        # restores ~1024x the per-step gradient magnitude. Starting from a
        # conservative 5e-7 (≈ DAPO paper's 1e-6 / 2) and letting clip-higher
        # bound per-step drift; sweep up to 1e-6 if training is too slow.
        learning_rate=5e-7,
        max_input_tokens=DEFAULT_MAX_INPUT_TOKENS,
        max_output_tokens=args.max_output_tokens,
        n_prompts=args.n_prompts,
        n_generations_per_prompt=16,
        num_rollout_workers=args.num_rollout_workers,
        train_tpu_type=args.train_tpu_type,
        inference_tpu_type=args.inference_tpu_type,
        num_train_slices=args.num_train_slices,
        train_ram=args.train_ram,
        inference_ram=args.inference_ram,
        launcher_region=args.launcher_region or _region_from_zone(args.zone),
        zone=args.zone,
        inflight_weight_updates=args.inflight_weight_updates,
        max_rollout_step_delay=1,
        weight_transfer_sync_interval_steps=1,
        max_weight_transfer_wait_time=0,
    )


def build_run_name(config: RLExperimentConfig, explicit_run_name: str | None) -> str:
    if explicit_run_name is not None:
        return explicit_run_name

    datestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_base_name = config.model_config.name.split("/")[-1].lower().replace("-instruct", "i")
    return f"{model_base_name}-{config.experiment_name_suffix}-{datestamp}"


def main() -> None:
    if os.getenv("CI") is not None:
        logger.info("Skipping experiment execution on CI environment.")
        return

    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    experiment_config = build_experiment_config(args)
    run_name = build_run_name(experiment_config, args.run_name)
    curriculum = build_math500_curriculum(run_name, experiment_config, args.eval_frequency)
    step = make_rl_step(
        name=run_name,
        config=experiment_config,
        curriculum=curriculum,
    )
    if args.override_output_path is not None:
        step = step.with_output_path(args.override_output_path)
    executor_config = executor_main_config_for_rl_experiment(experiment_config)

    logger.info(
        "Launching executor RL Math500 run %s (train_tpu=%s, inference_tpu=%s, rollout_workers=%d, "
        "train_ram=%s, inference_ram=%s, launcher_region=%s, zone=%s, inflight=%s, executor_prefix=%s, "
        "override_output_path=%s)",
        run_name,
        experiment_config.train_tpu_type,
        experiment_config.inference_tpu_type,
        experiment_config.num_rollout_workers,
        experiment_config.train_ram,
        experiment_config.inference_ram,
        experiment_config.launcher_region,
        experiment_config.zone,
        experiment_config.inflight_weight_updates,
        executor_config.prefix,
        args.override_output_path,
    )

    executor_main(
        executor_config,
        steps=[step],
        description="Executor-backed Iris RL Math500 run for Llama 3.1 8B",
    )


if __name__ == "__main__":
    main()
