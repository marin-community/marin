# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression probe E4: direct + GCS + production-like RL launch.

This preserves the direct coordinator topology while restoring the heavy 500-step
production-like envelope. It uses the cached regional GCS model artifact rather
than live Hugging Face bootstrap.
"""

import argparse
import datetime
import logging
from dataclasses import replace

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.layers.attention import AttentionBackend
from levanter.models.llama import LlamaConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

from marin.rl.curriculum import CurriculumConfig, LessonConfig
from marin.rl.curriculum import SamplingParams as CurriculumSamplingParams
from marin.rl.environments import EnvConfig
from marin.rl.environments.inference_ctx import VLLMSamplingConfig, vLLMInferenceContextConfig
from marin.rl.orchestration import _run_rl_coordinator
from marin.rl.replay_buffer import ReplayBufferConfig
from marin.rl.rl_job import RLJobConfig, RunConfig, TrainParams
from marin.rl.rl_losses import RLOOLoss
from marin.rl.rollout_storage import RolloutStorageConfig, StorageType
from marin.rl.rollout_worker import RolloutTrackerConfig
from marin.rl.weight_transfer import WeightTransferConfig, WeightTransferMode

logger = logging.getLogger(__name__)

CANONICAL_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_PATH = "gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f"
MARIN_PREFIX = "gs://marin-us-central1"
DEFAULT_SUFFIX = "e4p"
DEFAULT_NUM_TRAIN_STEPS = 500
PROD_TPU_WORKER_RAM = "400g"
DEFAULT_N_PROMPTS = 64
DEFAULT_EVAL_FREQUENCY = 1
DEFAULT_REGION = "us-central1"
DEFAULT_NUM_ROLLOUT_WORKERS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-name-suffix",
        default=DEFAULT_SUFFIX,
        help="Short suffix used in run ids and W&B names.",
    )
    parser.add_argument(
        "--num-train-steps",
        type=int,
        default=DEFAULT_NUM_TRAIN_STEPS,
        help="Number of trainer steps for the direct + GCS + prod probe.",
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
        help="Full-eval cadence in rollout-worker iterations.",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help="Concrete region for trainer and rollout TPU jobs.",
    )
    parser.add_argument(
        "--inflight-weight-updates",
        action="store_true",
        help="Enable inflight rollout weight updates.",
    )
    parser.add_argument(
        "--num-rollout-workers",
        type=int,
        default=DEFAULT_NUM_ROLLOUT_WORKERS,
        help="Number of rollout worker jobs to launch.",
    )
    parser.add_argument(
        "--kv-cache-metrics",
        action="store_true",
        help="Enable vLLM KV-cache metrics on rollout workers.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Stable run name for checkpoint and W&B resume across preemption retries. "
        "When set, checkpoints and W&B use this name so retries resume progress. "
        "If not set, generates a fresh timestamp-based name (no resume on retry).",
    )
    parser.add_argument(
        "--debug-checkpointer",
        action="store_true",
        help="Enable verbose trainer-side checkpoint diagnostics for debugging checkpoint failures.",
    )
    parser.add_argument(
        "--debug-checkpointer-log-interval",
        type=float,
        default=60.0,
        help="Seconds between checkpoint progress logs when --debug-checkpointer is enabled.",
    )
    parser.add_argument(
        "--debug-checkpointer-dump-stacks-after",
        type=float,
        default=None,
        help="If set, dump Python thread stacks after this many seconds in one checkpoint phase.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    datestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Stable name: survives preemption retries. Used for checkpoints, W&B, rollout storage.
    # Instance name: unique per coordinator invocation. Used for Iris child job names and actors.
    if args.run_name:
        stable_name = args.run_name
        instance_name = f"{stable_name}-{datestamp}"
    else:
        stable_name = f"{args.experiment_name_suffix}-{datestamp}"
        instance_name = stable_name
    name = stable_name

    converter = HFCheckpointConverter(
        LlamaConfig,
        reference_checkpoint=MODEL_PATH,
        tokenizer=MODEL_PATH,
    )
    hf_config = converter.default_hf_config
    model_config = replace(
        LlamaConfig.from_hf_config(hf_config),
        max_seq_len=2048,
        tokenizer=MODEL_PATH,
        attn_backend=AttentionBackend.SPLASH,
    )

    curriculum = CurriculumConfig(
        lessons={
            "math_full": LessonConfig(
                lesson_id="math_full",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.math_env.MathEnv",
                    env_args={"seed": 42},
                ),
                dependencies=[],
                sampling_params=CurriculumSamplingParams(
                    temperature=1.0,
                    n_prompts=args.n_prompts,
                    n_generations_per_prompt=16,
                    max_output_tokens=1024,
                    top_k=4096,
                ),
            ),
        },
        eval_frequency=args.eval_frequency,
        micro_eval_frequency=None,
        actor_name=f"curriculum-{instance_name}",
        eval_n_examples=500,
        max_seq_len=2048,
    )

    job_config = RLJobConfig(
        model=model_config,
        vocab_size=hf_config.vocab_size,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="marin_iris_rl_debug",
                name=name,
                tags=["rl", "iris-debug", "regression", "e4", "direct-gcs-prod"],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=1024,
            per_device_parallelism=16,
            num_train_steps=args.num_train_steps,
            steps_per_eval=100,
            checkpointer=CheckpointerConfig(
                base_path=f"{MARIN_PREFIX}/checkpoints/{name}",
                save_interval=datetime.timedelta(seconds=600),
                debug_checkpointer=args.debug_checkpointer,
                debug_checkpointer_log_interval=args.debug_checkpointer_log_interval,
                debug_checkpointer_dump_stacks_after=args.debug_checkpointer_dump_stacks_after,
            ),
            mesh=MeshConfig(
                axes={"context": 1, "model": 1},
                shared_mapping={"mlp": "model", "heads": "model", "position": "context"},
            ),
        ),
        train_params=TrainParams(
            optimizer=AdamConfig(learning_rate=2e-6, lr_schedule="constant"),
            rl_loss=RLOOLoss(
                kl_coef=0.0,
                clip_epsilon_low=0.2,
                clip_epsilon_high=0.28,
                synchronous=True,
                do_trainer_inference_mismatch_importance_sampling=True,
                tis_importance_sampling_ratio_max=2.0,
                do_overlong_filtering=True,
                vocab_tile_size=32064,
            ),
            replay_buffer=ReplayBufferConfig(capacity=4096, alpha=3.0, max_samples=1, max_rollout_step_delay=1),
        ),
        curriculum=curriculum,
        tokenizer=MODEL_PATH,
        inference_type="vllm",
        inference_config=vLLMInferenceContextConfig(
            model_name=MODEL_PATH,
            canonical_model_name=CANONICAL_MODEL_NAME,
            max_model_len=2048,
            tensor_parallel_size=4,
            gpu_memory_utilization=0.90,
            sampling_params=VLLMSamplingConfig(
                temperature=1.0,
                n=8,
                max_tokens=1024,
                stop=["<|eot_id|>"],
                include_stop_str_in_output=True,
                logprobs=1,
                top_k=4096,
            ),
            load_format="runai_streamer",
            kv_cache_metrics=args.kv_cache_metrics,
        ),
        initial_checkpoint=MODEL_PATH,
        rollout_storage=RolloutStorageConfig(
            storage_type=StorageType.FILE,
            path=f"{MARIN_PREFIX}/rollouts/{name}",
        ),
        weight_transfer=WeightTransferConfig(
            mode=WeightTransferMode.ARROW_FLIGHT,
            sync_interval_steps=1,
            max_weight_transfer_wait_time=0,
            coordinator_name=f"wt-coord-{instance_name}",
        ),
        inflight_weight_updates=args.inflight_weight_updates,
        run_id=name,
        instance_id=instance_name if instance_name != name else None,
        log_freq=1,
        run_config=RunConfig(
            train_tpu_type="v5p-8",
            num_rollout_workers=args.num_rollout_workers,
            inference_tpu_type="v5p-8",
            train_ram=PROD_TPU_WORKER_RAM,
            inference_ram=PROD_TPU_WORKER_RAM,
            regions=[args.region],
        ),
        rollout_tracker=RolloutTrackerConfig(
            project="marin_iris_rl_debug",
            name=f"{instance_name}-rollout",
            tags=["rl", "iris-debug", "regression", "e4", "rollout"],
        ),
        pip_dependency_groups=["vllm", "math"],
    )

    logger.info(
        "Running E4 direct + GCS + prod probe: %s (instance=%s) "
        "(n_prompts=%d, eval_frequency=%d, inflight=%s, rollout_workers=%d, "
        "region=%s, kv_cache_metrics=%s)",
        name,
        instance_name,
        args.n_prompts,
        args.eval_frequency,
        args.inflight_weight_updates,
        args.num_rollout_workers,
        args.region,
        args.kv_cache_metrics,
    )
    _run_rl_coordinator(job_config)


if __name__ == "__main__":
    main()
