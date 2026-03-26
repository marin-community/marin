# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression probe E2: direct + GCS + small RL launch.

This preserves the old successful topology:
- no executor_main wrapper
- coordinator runs directly in the outer Iris job
- model bootstrap comes from the regional GCS model artifact
- tiny 5-step envelope
"""

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


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    datestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"direct-gcs-small-{datestamp}"

    converter = HFCheckpointConverter(
        LlamaConfig,
        reference_checkpoint=MODEL_PATH,
        tokenizer=MODEL_PATH,
    )
    hf_config = converter.default_hf_config
    model_config = replace(
        LlamaConfig.from_hf_config(hf_config),
        max_seq_len=1024,
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
                    n_prompts=4,
                    n_generations_per_prompt=16,
                    max_output_tokens=512,
                    top_k=4096,
                ),
            ),
        },
        eval_frequency=5,
        micro_eval_frequency=None,
        actor_name=f"curriculum-{name}",
        eval_n_examples=10,
        max_seq_len=1024,
    )

    job_config = RLJobConfig(
        model=model_config,
        vocab_size=hf_config.vocab_size,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="marin_iris_rl_debug",
                name=name,
                tags=["rl", "iris-debug", "regression", "e2", "direct-gcs-small"],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=64,
            per_device_parallelism=16,
            num_train_steps=5,
            steps_per_eval=100,
            checkpointer=CheckpointerConfig(
                base_path=f"{MARIN_PREFIX}/checkpoints/{name}",
                save_interval=datetime.timedelta(seconds=600),
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
            replay_buffer=ReplayBufferConfig(capacity=4096, alpha=3.0, max_samples=1, max_rollout_step_delay=0),
        ),
        curriculum=curriculum,
        tokenizer=MODEL_PATH,
        inference_type="vllm",
        inference_config=vLLMInferenceContextConfig(
            model_name=MODEL_PATH,
            canonical_model_name=CANONICAL_MODEL_NAME,
            max_model_len=1024,
            tensor_parallel_size=4,
            gpu_memory_utilization=0.90,
            sampling_params=VLLMSamplingConfig(
                temperature=1.0,
                n=8,
                max_tokens=512,
                stop=["<|eot_id|>"],
                include_stop_str_in_output=True,
                logprobs=1,
                top_k=4096,
            ),
            load_format="runai_streamer",
        ),
        initial_checkpoint=MODEL_PATH,
        rollout_storage=RolloutStorageConfig(
            storage_type=StorageType.FILE,
            path=f"{MARIN_PREFIX}/rollouts/{name}",
        ),
        weight_transfer=WeightTransferConfig(
            mode=WeightTransferMode.ARROW_FLIGHT,
            sync_interval_steps=1,
            max_weight_transfer_wait_time=300,
            coordinator_name=f"wt-coord-{name}",
        ),
        run_id=name,
        log_freq=1,
        run_config=RunConfig(
            train_tpu_type="v5p-8",
            num_rollout_workers=1,
            inference_tpu_type="v5p-8",
        ),
        rollout_tracker=RolloutTrackerConfig(
            project="marin_iris_rl_debug",
            name=f"{name}-rollout",
            tags=["rl", "iris-debug", "regression", "e2", "rollout"],
        ),
        pip_dependency_groups=["vllm", "math"],
    )

    logger.info("Running E2 direct + GCS + small probe: %s", name)
    _run_rl_coordinator(job_config)


if __name__ == "__main__":
    main()
