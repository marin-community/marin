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
Testing Prime Intellect environment with GSM8K dataset.
"""

import dataclasses
import datetime
import logging
import os
from dataclasses import dataclass

import jmp
import ray
from levanter.checkpoint import CheckpointerConfig
from levanter.distributed import RayConfig
from levanter.inference.engine import InferenceEngineConfig
from levanter.inference.openai import InferenceServerConfig
from levanter.infra.ray_tpu import run_on_pod_ray
from levanter.models.qwen import Qwen3Config
from levanter.optim import AdamConfig
from levanter.tracker.tensorboard import TensorboardConfig
from levanter.trainer import TrainerConfig
from ray.runtime_env import RuntimeEnv
from transformers import AutoConfig, AutoTokenizer

from marin.execution.executor import (
    ExecutorStep,
    OutputName,
    executor_main,
)
from marin.post_training.rollout_storage import RolloutStorageConfig, StorageType
from marin.post_training.rollout_worker import RolloutWorker, RolloutWorkerConfig
from marin.post_training.train_worker import ReplayBufferConfig, TrainWorker, TrainWorkerConfig
from marin.post_training.weight_transfer import WeightTransferConfig
from marin.resources import TpuPodConfig
from marin.training.training import (
    _add_run_env_variables,
)
from marin.utils import remove_tpu_lockfile_on_exit

logger = logging.getLogger(__name__)

ENVIRONMENT_SPEC = "prime_intellect:env_id=primeintellect/gsm8k,env_args={num_train_examples=-1,num_eval_examples=-1}"
# MODEL_NAME = "meta-llama/Llama-3.2-8B-Instruct"
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_TYPE = "qwen3"
WANDB_PROJECT = f"rl_testing_{MODEL_NAME.split('/')[-1].lower()}"
MODEL_TOKENIZER = MODEL_NAME
MODEL_CHECKPOINT = MODEL_NAME
MAX_INPUT_TOKENS = 128
MAX_OUTPUT_TOKENS = 128
RUN_ID = f"test-{MODEL_NAME.split('/')[-1]}-{ENVIRONMENT_SPEC.replace(':', '_').replace('=', '_')}"


def stop_tokens(tokenizer_name: str):
    """Infer the stop tokens from the given tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return [tokenizer.eos_token_id]


@dataclass(frozen=True)
class RLTrainConfig:
    """Configuration for async RL training on pods."""

    rollout_worker_config: RolloutWorkerConfig
    train_worker_config: TrainWorkerConfig
    inference_tpu_type: str
    train_tpu_type: str
    num_inference_workers: int = 1
    num_train_slices: int = 1


@ray.remote
def run_rl_training_on_pod(config: RLTrainConfig):
    """
    Run async RL training with separate inference and training workers.
    """
    env = {}
    env = _add_run_env_variables(env)

    if "JAX_COMPILATION_CACHE_DIR" not in env:
        marin_prefix = os.environ.get("MARIN_PREFIX")
        if marin_prefix:
            env["JAX_COMPILATION_CACHE_DIR"] = os.path.join(marin_prefix, "compilation-cache")
            logger.info(f"JAX compilation cache enabled at: {env['JAX_COMPILATION_CACHE_DIR']}")
        else:
            logger.warning("MARIN_PREFIX environment variable not set. JAX compilation cache will not be configured.")

    # Use the default env when running on the driver (Ray doesn't support otherwise.)
    # runtime_env = ray_deps.build_runtime_env_for_packages(extra=["tpu", "post_training"])
    runtime_env = RuntimeEnv()

    train_pod_config = TpuPodConfig(tpu_type=config.train_tpu_type, runtime_env=runtime_env)
    train_hw_config = train_pod_config.with_env_vars(env)

    train_kwargs = train_hw_config.as_remote_kwargs()
    train_kwargs["max_calls"] = 1

    @ray.remote(**train_kwargs)
    def train_worker_task():
        with remove_tpu_lockfile_on_exit():
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
            # Print jax & TPU configuration
            import jax

            logging.error(f"JAX configuration: {jax.__version__}")
            worker = TrainWorker(
                config=config.train_worker_config,
            )
            worker.train()

    rollout_pod_config = TpuPodConfig(tpu_type=config.inference_tpu_type, runtime_env=runtime_env)
    rollout_hw_config = rollout_pod_config.with_env_vars(env)

    rollout_kwargs = rollout_hw_config.as_remote_kwargs()
    rollout_kwargs["max_calls"] = 1

    @ray.remote(**rollout_kwargs)
    def inference_worker_task():
        with remove_tpu_lockfile_on_exit():
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
            worker = RolloutWorker(
                config=config.rollout_worker_config,
            )
            worker.run()

    train_tasks = []
    train_tasks.append(
        run_on_pod_ray.remote(
            train_worker_task,
            config.train_tpu_type,
            num_slices=config.num_train_slices,
            max_retries_failure=10,
            max_retries_preemption=10,
        )
    )

    inference_tasks = []
    for _ in range(config.num_inference_workers):
        inference_tasks.append(
            run_on_pod_ray.remote(
                inference_worker_task,
                config.inference_tpu_type,
                num_slices=1,
                max_retries_failure=10,
                max_retries_preemption=10,
            )
        )

    return ray.get(inference_tasks + train_tasks)


def rl_train(name: str) -> ExecutorStep:
    hf_config = AutoConfig.from_pretrained(MODEL_NAME)
    config = Qwen3Config.from_hf_config(hf_config)

    # Adjust the max sequence length of the model to reduce memory usage.
    model_config = dataclasses.replace(config, seq_len=MAX_INPUT_TOKENS + MAX_OUTPUT_TOKENS, tokenizer=MODEL_TOKENIZER)

    trainer_config = TrainerConfig(
        tracker=TensorboardConfig(
            logdir=OutputName("tblogs"),
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=256,
        num_train_steps=50000,
        steps_per_eval=100,
        checkpointer=CheckpointerConfig(
            base_path=OutputName("checkpoints"),
            save_interval=datetime.timedelta(seconds=600),
        ),
        tensor_parallel_axes=["mlp", "kv_head"],
        fsdp_axis="embed",
        batch_axis="batch",
        ray=RayConfig(auto_start_cluster=False),
    )

    opt_config = AdamConfig(
        learning_rate=1e-3,
        weight_decay=1e-3,
        warmup=10,
        lr_schedule="constant",
    )

    inference_server_config = InferenceServerConfig(
        model=model_config,
        # Turn on tensor parallelism for inference
        trainer=dataclasses.replace(trainer_config, tensor_parallel_axes=4),
        hf_checkpoint=MODEL_CHECKPOINT,
        tokenizer=MODEL_TOKENIZER,
        temperature=1.0,
        service=InferenceEngineConfig(
            max_seqs=16,
            max_pages_per_seq=32,
            page_size=32,
            max_seqs_in_prefill=16,
        ),
    )

    rollout_storage = RolloutStorageConfig(
        storage_type=StorageType.FILE,
        path=OutputName("rollouts"),
    )
    weight_transfer = WeightTransferConfig(
        sync_interval_steps=25,
        poll_interval_seconds=10,
        checkpoint_dir=OutputName("policy_checkpoints"),
        max_checkpoints=5,
    )

    train_worker = TrainWorkerConfig(
        rollout_storage=rollout_storage,
        weight_transfer=weight_transfer,
        model=model_config,
        trainer=trainer_config,
        optimizer=opt_config,
        replay_buffer=ReplayBufferConfig(
            capacity=4096,
            alpha=3,
        ),
        kl_coef=0.001,
        initial_checkpoint=MODEL_NAME,
        run_id=RUN_ID,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_TOKENIZER)

    rollout_worker = RolloutWorkerConfig(
        trainer=trainer_config,
        inference_server_config=inference_server_config,
        model=model_config,
        environment_spec=ENVIRONMENT_SPEC,
        max_input_length=MAX_INPUT_TOKENS,
        max_output_length=MAX_OUTPUT_TOKENS,
        pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
        n_prompts_per_step=16,
        n_generations=4,
        temperature=0.7,
        log_freq=5,
        max_rollouts=100000,
        stop_tokens=stop_tokens(MODEL_TOKENIZER),
        initial_checkpoint=MODEL_NAME,
        weight_transfer=weight_transfer,
        rollout_storage=rollout_storage,
        run_id=RUN_ID,
        env_id="primeintellect/gsm8k",
        env_args={"num_train_examples": -1, "num_eval_examples": -1},
    )

    config = RLTrainConfig(
        rollout_worker_config=rollout_worker,
        train_worker_config=train_worker,
        inference_tpu_type="v5litepod-4",
        train_tpu_type="v5litepod-128",
        num_inference_workers=1,
        num_train_slices=1,
    )

    return ExecutorStep(
        name=f"rl_testing/{name}",
        description=f"Async RL training: {name}",
        fn=run_rl_training_on_pod,
        config=config,
        pip_dependency_groups=["post_training"],
    )


def main():
    import time

    nonce = int(time.time())

    experiments = [
        rl_train(name=f"qwen3-gsm8k-rl-test-{nonce}"),
    ]

    executor_main(
        steps=experiments,
        description="Async RL GSM8K training experiments",
    )


if __name__ == "__main__":
    main()
