# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct Iris RL submission — bypasses executor_main entirely.

Calls submit_rl_job() directly so the coordinator and workers all run
on TPU workers with vllm available. No CPU-only orchestration layer.
"""

import datetime
import logging

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.layers.attention import AttentionBackend
from levanter.models.llama import LlamaConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from transformers import AutoConfig
from vllm import SamplingParams

from marin.rl.curriculum import CurriculumConfig, LessonConfig
from marin.rl.curriculum import SamplingParams as CurriculumSamplingParams
from marin.rl.environments import EnvConfig
from marin.rl.environments.inference_ctx import vLLMInferenceContextConfig
from marin.rl.replay_buffer import ReplayBufferConfig
from marin.rl.rl_job import RLJobConfig, RunConfig, TrainParams
from marin.rl.rl_losses import RLOOLoss
from marin.rl.rollout_storage import RolloutStorageConfig, StorageType
from marin.rl.rollout_worker import RolloutTrackerConfig
from marin.rl.weight_transfer import WeightTransferConfig, WeightTransferMode

logger = logging.getLogger(__name__)

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


def main():
    logging.basicConfig(level=logging.INFO)
    datestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"iris-rl-direct-{datestamp}"

    hf_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config = LlamaConfig.from_hf_config(hf_config)
    import dataclasses as dc

    model_config = dc.replace(
        model_config,
        max_seq_len=1024,
        tokenizer=MODEL_NAME,
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
        micro_eval_frequency=9999999,
        actor_name=f"curriculum-{name}",
        eval_n_examples=10,
        max_seq_len=1024,
    )

    job_config = RLJobConfig(
        model=model_config,
        vocab_size=hf_config.vocab_size,
        trainer=TrainerConfig(
            tracker=WandbConfig(project="marin_iris_rl_debug", name=name, tags=["rl", "iris-debug"]),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=64,
            per_device_parallelism=16,
            num_train_steps=5,
            steps_per_eval=100,
            checkpointer=CheckpointerConfig(
                base_path=f"gs://marin-us-central2/checkpoints/{name}",
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
        tokenizer=MODEL_NAME,
        inference_type="vllm",
        inference_config=vLLMInferenceContextConfig(
            model_name=MODEL_NAME,
            max_model_len=1024,
            tensor_parallel_size=4,
            gpu_memory_utilization=0.90,
            sampling_params=SamplingParams(
                temperature=1.0,
                n=8,
                max_tokens=512,
                stop=["<|eot_id|>"],
                include_stop_str_in_output=True,
                logprobs=1,
                top_k=4096,
            ),
            load_format="dummy",
        ),
        initial_checkpoint=MODEL_NAME,
        rollout_storage=RolloutStorageConfig(
            storage_type=StorageType.FILE,
            path=f"gs://marin-us-central2/rollouts/{name}",
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
            train_tpu_type="v6e-4",
            num_rollout_workers=1,
            inference_tpu_type="v6e-4",
        ),
        rollout_tracker=RolloutTrackerConfig(
            project="marin_iris_rl_debug",
            name=f"{name}-rollout",
            tags=["rl", "iris-debug", "rollout"],
        ),
        pip_dependency_groups=["vllm", "math"],
    )

    # Run coordinator directly in this process (not as a child job).
    # This saves one TPU slot — the outer job IS the coordinator.
    from marin.rl.orchestration import _run_rl_coordinator

    logger.info("Running RL coordinator directly for: %s", name)
    _run_rl_coordinator(job_config)


if __name__ == "__main__":
    main()
