# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Region-aware RL regression probe: derives GCS paths from --region.

Drop-in replacement for exp_iris_rl_regression_direct_gcs_prod.py that does not
hardcode us-central1 paths.  MODEL_PATH, MARIN_PREFIX, and all artifact
locations are derived from the --region flag so the script works in any region
that has a corresponding gs://marin-<region> bucket with the model cached.
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
from marin.rl.weight_transfer import ArrowFlightExportStrategy, WeightTransferConfig, WeightTransferMode

logger = logging.getLogger(__name__)

CANONICAL_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_SUBPATH = "models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f"
DEFAULT_SUFFIX = "e4p"
DEFAULT_NUM_TRAIN_STEPS = 500
PROD_TPU_WORKER_RAM = "400g"
DEFAULT_N_PROMPTS = 64
DEFAULT_EVAL_FREQUENCY = 1
DEFAULT_REGION = "us-central1"
DEFAULT_NUM_ROLLOUT_WORKERS = 2
DEFAULT_TPU_TYPE = "v5p-8"
DEFAULT_ROLLOUT_TENSOR_PARALLEL_SIZE = 4
DEFAULT_ROLLOUT_GPU_MEMORY_UTILIZATION = 0.90
DEFAULT_ROLLOUT_MAX_MODEL_LEN = 2048
DEFAULT_TRAINER_MODEL_AXIS_SIZE = 1
DEFAULT_TRAINER_CONTEXT_AXIS_SIZE = 1


def _marin_prefix(region: str) -> str:
    return f"gs://marin-{region}"


def _model_path(region: str) -> str:
    return f"{_marin_prefix(region)}/{MODEL_SUBPATH}"


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
        help="Number of trainer steps.",
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
        help="Region for trainer/rollout TPU jobs AND GCS artifact paths.",
    )
    parser.add_argument(
        "--zone",
        default=None,
        help="Optional zone for trainer and rollout TPU jobs.",
    )
    parser.add_argument(
        "--tpu-type",
        default=DEFAULT_TPU_TYPE,
        help="TPU type for rollout workers (and trainer unless --train-tpu-type is set).",
    )
    parser.add_argument(
        "--train-tpu-type",
        default=None,
        help="TPU type for trainer. Defaults to --tpu-type if not set.",
    )
    parser.add_argument(
        "--train-ram",
        default=PROD_TPU_WORKER_RAM,
        help="Host RAM request for the trainer TPU job.",
    )
    parser.add_argument(
        "--inference-ram",
        default=PROD_TPU_WORKER_RAM,
        help="Host RAM request for each rollout TPU job.",
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
        "--rollout-tensor-parallel-size",
        type=int,
        default=DEFAULT_ROLLOUT_TENSOR_PARALLEL_SIZE,
        help="vLLM tensor-parallel size for rollout workers.",
    )
    parser.add_argument(
        "--rollout-gpu-memory-utilization",
        type=float,
        default=DEFAULT_ROLLOUT_GPU_MEMORY_UTILIZATION,
        help="vLLM gpu_memory_utilization for rollout workers.",
    )
    parser.add_argument(
        "--rollout-max-model-len",
        type=int,
        default=DEFAULT_ROLLOUT_MAX_MODEL_LEN,
        help="vLLM max_model_len for rollout workers.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Stable run name for checkpoint and W&B resume across preemption retries.",
    )
    parser.add_argument(
        "--debug-checkpointer",
        action="store_true",
        help="Enable verbose trainer-side checkpoint diagnostics.",
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
        default=60.0,
        help="Dump Python thread stacks after this many seconds in one checkpoint phase when "
        "--debug-checkpointer is enabled.",
    )
    parser.add_argument(
        "--delete-previous-temporary-checkpoint-after-save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete the prior temporary checkpoint after a new checkpoint commits successfully. "
        "Disable to isolate checkpoint save cost from previous-temp cleanup.",
    )
    parser.add_argument(
        "--debug-weight-transfer",
        action="store_true",
        help="Enable verbose Arrow Flight transfer diagnostics and debug worker env vars.",
    )
    parser.add_argument(
        "--weight-transfer-export-strategy",
        default=ArrowFlightExportStrategy.TREE_JIT.value,
        choices=[strategy.value for strategy in ArrowFlightExportStrategy],
        help="How the trainer prepares weights for Arrow Flight serving.",
    )
    parser.add_argument(
        "--trainer-model-axis-size",
        type=int,
        default=DEFAULT_TRAINER_MODEL_AXIS_SIZE,
        help="Trainer mesh size for the model axis.",
    )
    parser.add_argument(
        "--trainer-context-axis-size",
        type=int,
        default=DEFAULT_TRAINER_CONTEXT_AXIS_SIZE,
        help="Trainer mesh size for the context axis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    datestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    region = args.region
    marin_prefix = _marin_prefix(region)
    model_path = _model_path(region)
    train_tpu_type = args.train_tpu_type or args.tpu_type

    if args.run_name:
        stable_name = args.run_name
        instance_name = f"{stable_name}-{datestamp}"
    else:
        stable_name = f"{args.experiment_name_suffix}-{datestamp}"
        instance_name = stable_name
    name = stable_name

    converter = HFCheckpointConverter(
        LlamaConfig,
        reference_checkpoint=model_path,
        tokenizer=model_path,
    )
    hf_config = converter.default_hf_config
    model_config = replace(
        LlamaConfig.from_hf_config(hf_config),
        max_seq_len=2048,
        tokenizer=model_path,
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
                tags=["rl", "iris-debug", "regression", "e4", "direct-gcs-prod", region],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=1024,
            per_device_parallelism=16,
            num_train_steps=args.num_train_steps,
            steps_per_eval=100,
            checkpointer=CheckpointerConfig(
                base_path=f"{marin_prefix}/checkpoints/{name}",
                save_interval=datetime.timedelta(seconds=600),
                delete_previous_temporary_checkpoint_after_save=args.delete_previous_temporary_checkpoint_after_save,
                debug_checkpointer=args.debug_checkpointer,
                debug_checkpointer_log_interval=args.debug_checkpointer_log_interval,
                debug_checkpointer_dump_stacks_after=args.debug_checkpointer_dump_stacks_after,
            ),
            mesh=MeshConfig(
                axes={"context": args.trainer_context_axis_size, "model": args.trainer_model_axis_size},
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
        tokenizer=model_path,
        inference_type="vllm",
        inference_config=vLLMInferenceContextConfig(
            model_name=model_path,
            canonical_model_name=CANONICAL_MODEL_NAME,
            max_model_len=args.rollout_max_model_len,
            tensor_parallel_size=args.rollout_tensor_parallel_size,
            gpu_memory_utilization=args.rollout_gpu_memory_utilization,
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
        initial_checkpoint=model_path,
        rollout_storage=RolloutStorageConfig(
            storage_type=StorageType.FILE,
            path=f"{marin_prefix}/rollouts/{name}",
        ),
        weight_transfer=WeightTransferConfig(
            mode=WeightTransferMode.ARROW_FLIGHT,
            sync_interval_steps=1,
            max_weight_transfer_wait_time=0,
            coordinator_name=f"wt-coord-{instance_name}",
            export_strategy=ArrowFlightExportStrategy(args.weight_transfer_export_strategy),
            debug_weight_transfer=args.debug_weight_transfer,
        ),
        inflight_weight_updates=args.inflight_weight_updates,
        run_id=name,
        instance_id=instance_name if instance_name != name else None,
        log_freq=1,
        run_config=RunConfig(
            train_tpu_type=train_tpu_type,
            num_rollout_workers=args.num_rollout_workers,
            inference_tpu_type=args.tpu_type,
            train_ram=args.train_ram,
            inference_ram=args.inference_ram,
            regions=[region],
            zone=args.zone,
        ),
        rollout_tracker=RolloutTrackerConfig(
            project="marin_iris_rl_debug",
            name=f"{instance_name}-rollout",
            tags=["rl", "iris-debug", "regression", "e4", "rollout", region],
        ),
        pip_dependency_groups=["vllm", "math"],
    )

    logger.info(
        "Running E4 region-aware probe: %s (instance=%s) "
        "(region=%s, zone=%s, train_tpu=%s, rollout_tpu=%s, n_prompts=%d, eval_frequency=%d, "
        "inflight=%s, rollout_workers=%d, train_ram=%s, inference_ram=%s, rollout_tp=%d, "
        "rollout_gpu_mem=%.2f, rollout_max_model_len=%d, kv_cache_metrics=%s, "
        "trainer_model_axis=%d, trainer_context_axis=%d, transfer_debug=%s, transfer_strategy=%s, "
        "model=%s, prefix=%s)",
        name,
        instance_name,
        region,
        args.zone,
        train_tpu_type,
        args.tpu_type,
        args.n_prompts,
        args.eval_frequency,
        args.inflight_weight_updates,
        args.num_rollout_workers,
        args.train_ram,
        args.inference_ram,
        args.rollout_tensor_parallel_size,
        args.rollout_gpu_memory_utilization,
        args.rollout_max_model_len,
        args.kv_cache_metrics,
        args.trainer_model_axis_size,
        args.trainer_context_axis_size,
        args.debug_weight_transfer,
        args.weight_transfer_export_strategy,
        model_path,
        marin_prefix,
    )
    _run_rl_coordinator(job_config)


if __name__ == "__main__":
    main()
