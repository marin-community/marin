#!/usr/bin/env python3
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

import argparse
import dataclasses
import datetime
import logging
import subprocess
from pathlib import Path

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.compat.hf_checkpoints import (
    HFCheckpointConverter,
    HFCompatConfig,
)
from levanter.distributed import RayConfig
from levanter.inference.openai import InferenceServerConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from transformers import AutoConfig, AutoTokenizer

from marin.rl.environments import EnvConfig
from marin.rl.replay_buffer import ReplayBufferConfig
from marin.rl.rollout_storage import RolloutStorageConfig, StorageType
from marin.rl.rollout_worker import RolloutWorker, RolloutWorkerConfig
from marin.rl.train_worker import TrainWorker, TrainWorkerConfig
from marin.rl.weight_transfer import WeightTransferConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Test configuration
WANDB_PROJECT = "llama_small_rl_test"

SCRIPT_PATH = "src/marin/post_training/scripts/test_llama_small.py"
PREFIX = "gs://marin-eu-west4"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_TOKENIZER = MODEL_NAME
MODEL_CHECKPOINT = MODEL_NAME
ENVIRONMENT_CONFIG = EnvConfig(
    env_class="marin.rl.environments.mock_env.MockEnv", env_args={"task_type": "number_comparison", "seed": 42}
)
ENV_NAME = "number_comparison"
RUN_ID = f"006-{MODEL_NAME.split('/')[-1]}-{ENV_NAME}"
CHECKPOINT_DIR = f"{PREFIX}/rl_checkpoints/llama_small_test/checkpoints/{ENV_NAME}/run_{RUN_ID}"
ROLLOUT_QUEUE_PATH = f"{PREFIX}/rl_checkpoints/llama_small_test/rollout_queue/{ENV_NAME}/run_{RUN_ID}"
MAX_INPUT_TOKENS = 32
MAX_OUTPUT_TOKENS = 32


def stop_tokens(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    stop_tokens = tokenizer.eos_token_id
    print("STOP", tokenizer.eos_token, stop_tokens)
    return [stop_tokens]


def llama_small_config() -> HFCompatConfig:
    hf_config = AutoConfig.from_pretrained(MODEL_NAME)
    hf_converter = HFCheckpointConverter.from_hf(MODEL_NAME)
    lev_config = hf_converter.config_from_hf_config(hf_config)
    return dataclasses.replace(lev_config, seq_len=MAX_INPUT_TOKENS + MAX_OUTPUT_TOKENS, tokenizer=MODEL_TOKENIZER)


def llama_small_trainer_config(output_dir: str) -> TrainerConfig:
    return TrainerConfig(
        tracker=WandbConfig(
            project=WANDB_PROJECT,
            mode="shared",
            tags=["rl", ENVIRONMENT_CONFIG.env_class.split(".")[-1], "llama_small"],
            # N.B. run_id is set by the individual workers.
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=8,
        num_train_steps=10000,
        steps_per_eval=100,
        checkpointer=CheckpointerConfig(
            base_path=output_dir,
            save_interval=datetime.timedelta(seconds=600),
        ),
        tensor_parallel_axes=["mlp", "heads"],
        fsdp_axis="embed",
        batch_axis="batch",
        ray=RayConfig(auto_start_cluster=False),
    )


def llama_small_optimizer_config() -> AdamConfig:
    return AdamConfig(
        learning_rate=1e-6,
        # don't overwhelm the learning signal
        weight_decay=1e-5,
        warmup=10,
        lr_schedule="cosine",
    )


def llama_small_inference_server_config(output_dir: str) -> InferenceServerConfig:
    from levanter.inference.engine import InferenceEngineConfig

    return InferenceServerConfig(
        model=llama_small_config(),
        trainer=llama_small_trainer_config(output_dir),
        hf_checkpoint=MODEL_CHECKPOINT,
        tokenizer=MODEL_TOKENIZER,
        temperature=1.0,
        service=InferenceEngineConfig(
            max_seqs=8,
            max_pages_per_seq=32,
            page_size=128,
            max_seqs_in_prefill=4,
        ),
    )


def llama_small_training_worker_config(output_dir: str, run_id: str) -> TrainWorkerConfig:

    rollout_storage = RolloutStorageConfig(
        storage_type=StorageType.FILE,
        path=ROLLOUT_QUEUE_PATH,
    )
    weight_transfer = WeightTransferConfig(
        sync_interval_steps=10,
        poll_interval_seconds=1,
        checkpoint_dir=f"{output_dir}/policy_checkpoints",
        max_checkpoints=5,
    )

    return TrainWorkerConfig(
        rollout_storage=rollout_storage,
        model=llama_small_config(),
        trainer=llama_small_trainer_config(output_dir),
        optimizer=llama_small_optimizer_config(),
        replay_buffer=ReplayBufferConfig(
            capacity=4096,
            alpha=3,
        ),
        kl_coef=0.01,
        initial_checkpoint=MODEL_NAME,
        weight_transfer=weight_transfer,
        run_id=run_id,
    )


def llama_small_rollout_worker_config(output_dir: str, run_id: str) -> RolloutWorkerConfig:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TOKENIZER)

    rollout_storage = RolloutStorageConfig(
        storage_type=StorageType.FILE,
        path=ROLLOUT_QUEUE_PATH,
    )
    weight_transfer = WeightTransferConfig(
        sync_interval_steps=10,
        poll_interval_seconds=1,
        checkpoint_dir=f"{output_dir}/policy_checkpoints",
        max_checkpoints=5,
    )

    return RolloutWorkerConfig(
        trainer=llama_small_trainer_config(output_dir),
        inference_server_config=llama_small_inference_server_config(output_dir),
        model=llama_small_config(),
        environment_spec=ENVIRONMENT_CONFIG,
        rollout_storage=rollout_storage,
        max_input_length=MAX_INPUT_TOKENS,
        max_output_length=MAX_OUTPUT_TOKENS,
        pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
        n_prompts_per_step=16,
        n_generations=8,
        temperature=0.7,
        log_freq=10,
        max_rollouts=100000,
        stop_tokens=stop_tokens(MODEL_NAME),
        initial_checkpoint=MODEL_NAME,
        weight_transfer=weight_transfer,
        run_id=run_id,
    )


def cleanup():
    subprocess.run("sudo --non-interactive rm -f /tmp/libtpu_lockfile", shell=True, check=False)

    if Path("/dev/vfio/0").exists():
        subprocess.run(
            "sudo --non-interactive lsof -t /dev/vfio/* | xargs -r sudo kill -9",
            shell=True,
            check=False,
        )

    if Path("/dev/accel0").exists():
        subprocess.run(
            "sudo --non-interactive lsof -t /dev/accel* | xargs -r sudo kill -9",
            shell=True,
            check=False,
        )


def run_inference_mode(args):
    """Run in inference worker mode."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("rollout_worker")

    logger.info("Starting inference worker mode...")

    # cleanup()
    worker_config = llama_small_rollout_worker_config(CHECKPOINT_DIR, str(RUN_ID))
    worker = RolloutWorker(
        config=worker_config,
    )

    worker.run()
    logger.info("Inference worker completed")


def run_training_mode(args):
    """Run in training worker mode."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("training_worker")

    logger.info("Starting training worker mode...")
    cleanup()

    worker_config = llama_small_training_worker_config(CHECKPOINT_DIR, str(RUN_ID))
    worker = TrainWorker(
        config=worker_config,
    )

    worker.train()
    logger.info("Training worker completed")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
    )
    args = parser.parse_args()

    if args.mode == "rollout":
        logger.info("Running in rollout mode")
        run_inference_mode(args)
    elif args.mode == "training":
        logger.info("Running in training mode")
        run_training_mode(args)
    elif args.mode == "config":
        print("Rollout worker config:")
        print(llama_small_rollout_worker_config(CHECKPOINT_DIR, str(RUN_ID)))
        print("Training worker config:")
        print(llama_small_training_worker_config(CHECKPOINT_DIR, str(RUN_ID)))
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
