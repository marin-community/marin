# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# nodryrun because vLLM is not installed by default

"""Alternating multi-host RL on a single TPU pod.

User-facing entrypoint with explicit CLI modes:
  - controller:     orchestrates the full alternating RL loop
  - sampling-host:  runs on one TPU host during the sampling phase
  - materialize:    converts raw rollouts into materialized training batches
  - train-phase:    runs full-pod Levanter training over materialized batches
  - export-only:    recovery mode to export a policy without rerunning training
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alternating RL MATH-500")
    parser.add_argument(
        "--mode",
        choices=["controller", "sampling-host", "materialize", "train-phase", "export-only"],
        required=True,
        help="Execution mode.",
    )

    # Controller args
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--shared-root", default=None, help="Shared gs:// root for all run artifacts.")
    parser.add_argument("--tpu-name", default=None)
    parser.add_argument("--tpu-type", default="v6e-16")
    parser.add_argument("--zone", default="europe-west4-a")
    parser.add_argument("--project", default="hai-gcp-models")
    parser.add_argument("--image", default=None, help="Docker image tag or digest.")
    parser.add_argument("--steps-per-phase", type=int, default=80)
    parser.add_argument("--max-phases", type=int, default=None)
    parser.add_argument("--global-batch-size", type=int, default=1024)
    parser.add_argument("--num-train-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--capacity-type", default="on-demand")

    # Sampling-host args
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--host-ordinal", type=int, default=0)

    # Materialize args
    parser.add_argument("--output-dir", default=None)

    # Train-phase args
    parser.add_argument("--run-state-path", default=None)

    # Model
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--initial-checkpoint", default=None, help="Initial HF or Levanter checkpoint.")

    # Common
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    return parser.parse_args()


def _run_controller(args: argparse.Namespace):
    from levanter.checkpoint import CheckpointerConfig
    from levanter.models.llama import LlamaConfig
    from levanter.tracker.wandb import WandbConfig
    from levanter.trainer import TrainerConfig

    from marin.rl.alternating.config import AlternatingRLConfig
    from marin.rl.alternating.controller import run_controller
    from marin.rl.curriculum import CurriculumConfig, LessonConfig, SamplingParams
    from marin.rl.environments import EnvConfig
    from marin.rl.rl_losses import RLOOLoss

    if not args.run_id:
        raise ValueError("--run-id is required for controller mode")
    if not args.shared_root:
        raise ValueError("--shared-root is required for controller mode")
    if not args.image:
        raise ValueError("--image is required for controller mode")

    tpu_name = args.tpu_name or args.run_id

    # Build curriculum
    lessons = {
        "math_full": LessonConfig(
            lesson_id="math_full",
            env_config=EnvConfig(
                env_class="marin.rl.environments.math_env.MathEnv",
                env_args={"seed": args.seed},
            ),
            dependencies=[],
            sampling_params=SamplingParams(
                temperature=1.0,
                n_prompts=64,
                n_generations_per_prompt=16,
                max_output_tokens=1024,
                top_k=4096,
            ),
        ),
    }
    curriculum = CurriculumConfig(
        lessons=lessons,
        eval_frequency=10,
        micro_eval_frequency=9999999,
        actor_name=f"curriculum-{args.run_id}",
        eval_n_examples=500,
        max_seq_len=2048,
    )

    loss = RLOOLoss(
        kl_coef=0.0,
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.28,
        synchronous=True,
        do_trainer_inference_mismatch_importance_sampling=True,
        tis_importance_sampling_ratio_max=2.0,
        do_overlong_filtering=True,
        vocab_tile_size=32064,
    )

    trainer_config = TrainerConfig(
        num_train_steps=args.num_train_steps,
        seed=args.seed,
        checkpointer=CheckpointerConfig(
            base_path=f"{args.shared_root}/levanter_checkpoints",
            save_interval=600,
        ),
        tracker=[
            WandbConfig(
                project="marin_post_training",
                tags=["rl", "math", "alternating"],
            ),
        ],
    )

    initial_checkpoint = args.initial_checkpoint or args.model_name

    config = AlternatingRLConfig(
        run_id=args.run_id,
        shared_root=args.shared_root,
        tpu_name=tpu_name,
        tpu_type=args.tpu_type,
        zone=args.zone,
        project=args.project,
        model_name_or_path=args.model_name,
        model_config_class=LlamaConfig,
        tokenizer=args.model_name,
        initial_checkpoint=initial_checkpoint,
        trainer=trainer_config,
        loss=loss,
        curriculum=curriculum,
        steps_per_phase=args.steps_per_phase,
        global_batch_size=args.global_batch_size,
        seed=args.seed,
        max_phases=args.max_phases,
        image=args.image,
        learning_rate=args.learning_rate,
        capacity_type=args.capacity_type,
    )

    run_controller(config)


def _run_sampling_host(args: argparse.Namespace):
    from marin.rl.alternating.sampling_host import SamplingHostConfig, run_sampling_host
    from marin.rl.alternating.state import SamplingManifest, read_json_from_path
    from marin.rl.curriculum import CurriculumConfig, LessonConfig, SamplingParams
    from marin.rl.environments import EnvConfig

    if not args.manifest_path:
        raise ValueError("--manifest-path is required for sampling-host mode")

    # Read manifest to get topology info
    manifest = SamplingManifest.from_json(read_json_from_path(args.manifest_path))

    # Build curriculum config (same as controller)
    lessons = {
        "math_full": LessonConfig(
            lesson_id="math_full",
            env_config=EnvConfig(
                env_class="marin.rl.environments.math_env.MathEnv",
                env_args={"seed": args.seed},
            ),
            dependencies=[],
            sampling_params=SamplingParams(
                temperature=1.0,
                n_prompts=64,
                n_generations_per_prompt=16,
                max_output_tokens=1024,
                top_k=4096,
            ),
        ),
    }
    curriculum_config = CurriculumConfig(
        lessons=lessons,
        eval_frequency=10,
        micro_eval_frequency=9999999,
        eval_n_examples=500,
        max_seq_len=2048,
    )

    config = SamplingHostConfig(
        manifest_path=args.manifest_path,
        host_ordinal=args.host_ordinal,
        curriculum_config=curriculum_config,
        model_name=args.model_name,
        max_model_len=2048,
        tensor_parallel_size=manifest.local_tensor_parallel_size,
        seed=args.seed,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", force=True)
    run_sampling_host(config)


def _run_materialize(args: argparse.Namespace):
    from marin.rl.alternating.materializer import MaterializerConfig, run_materializer
    from marin.rl.rl_losses import RLOOLoss

    if not args.manifest_path:
        raise ValueError("--manifest-path is required for materialize mode")
    if not args.output_dir:
        raise ValueError("--output-dir is required for materialize mode")

    loss = RLOOLoss(
        kl_coef=0.0,
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.28,
        synchronous=True,
        do_trainer_inference_mismatch_importance_sampling=True,
        tis_importance_sampling_ratio_max=2.0,
        do_overlong_filtering=True,
        vocab_tile_size=32064,
    )

    config = MaterializerConfig(
        sampling_manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        loss_module=loss,
        tokenizer_name=args.model_name,
        steps_per_phase=args.steps_per_phase,
        global_batch_size=args.global_batch_size,
        max_seq_len=2048,
        seed=args.seed,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", force=True)
    run_materializer(config)


def _run_train_phase(args: argparse.Namespace):
    from levanter.checkpoint import CheckpointerConfig
    from levanter.models.llama import LlamaConfig
    from levanter.optim import AdamConfig
    from levanter.trainer import TrainerConfig

    from marin.rl.alternating.state import AlternatingRunState, read_json_from_path
    from marin.rl.alternating.training_phase import TrainingPhaseConfig, run_training_phase
    from marin.rl.rl_losses import RLOOLoss

    if not args.manifest_path:
        raise ValueError("--manifest-path is required for train-phase mode")
    if not args.run_state_path:
        raise ValueError("--run-state-path is required for train-phase mode")

    run_state = AlternatingRunState.from_json(read_json_from_path(args.run_state_path))

    loss = RLOOLoss(
        kl_coef=0.0,
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.28,
        synchronous=True,
        do_trainer_inference_mismatch_importance_sampling=True,
        tis_importance_sampling_ratio_max=2.0,
        do_overlong_filtering=True,
        vocab_tile_size=32064,
    )

    # Figure out checkpoint base path
    checkpoint_base = run_state.current_levanter_checkpoint_path
    shared_root = os.path.dirname(os.path.dirname(args.run_state_path))  # state/run_state.json -> root

    trainer_config = TrainerConfig(
        num_train_steps=args.num_train_steps,
        seed=args.seed,
        checkpointer=CheckpointerConfig(
            base_path=checkpoint_base or f"{shared_root}/levanter_checkpoints/phase_{run_state.phase_id:04d}",
            save_interval=600,
        ),
    )

    optimizer = AdamConfig(
        learning_rate=args.learning_rate,
        weight_decay=0.0,
    )

    # Build model config from HF
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(args.model_name)
    model_config = LlamaConfig.from_hf_config(hf_config)

    next_policy_version = run_state.policy_version + 1
    policy_output_dir = f"{shared_root}/policies/policy_{next_policy_version:04d}"

    config = TrainingPhaseConfig(
        materialized_manifest_path=args.manifest_path,
        run_state_path=args.run_state_path,
        model_config=model_config,
        model_config_class=LlamaConfig,
        trainer=trainer_config,
        optimizer=optimizer,
        loss=loss,
        tokenizer_name=args.model_name,
        model_name=args.model_name,
        initial_checkpoint=run_state.current_policy_path if run_state.current_levanter_checkpoint_path is None else None,
        seed=args.seed,
        export_policy_after_training=True,
        policy_output_dir=policy_output_dir,
    )

    run_training_phase(config)


def _run_export_only(args: argparse.Namespace):
    """Recovery mode: export a policy from the latest checkpoint."""
    from marin.rl.alternating.state import AlternatingRunState, read_json_from_path

    if not args.run_state_path:
        raise ValueError("--run-state-path is required for export-only mode")

    run_state = AlternatingRunState.from_json(read_json_from_path(args.run_state_path))
    logger.info(
        "Export-only mode: phase=%d, policy_version=%d, checkpoint=%s",
        run_state.phase_id,
        run_state.policy_version,
        run_state.current_levanter_checkpoint_path,
    )

    # This would require loading the full mesh, so it's essentially the same
    # as running a train-phase with 0 steps + export. Delegate to train-phase.
    logger.info("Export-only is implemented by running train-phase with the latest checkpoint.")
    logger.info("Use --mode train-phase with appropriate args instead.")
    sys.exit(1)


def main():
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment.")
        return

    args = _parse_args()

    try:
        if args.mode == "controller":
            _run_controller(args)
        elif args.mode == "sampling-host":
            _run_sampling_host(args)
        elif args.mode == "materialize":
            _run_materialize(args)
        elif args.mode == "train-phase":
            _run_train_phase(args)
        elif args.mode == "export-only":
            _run_export_only(args)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
    except Exception:
        logger.exception("FATAL: %s mode crashed", args.mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
