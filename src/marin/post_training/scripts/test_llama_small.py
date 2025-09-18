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

"""
Test script for Llama-3.2-1B-Instruct RL training with mock environment.

Supports three modes:
- driver (default): Launches inference and training workers via ray_run
- inference: Runs as an inference worker
- training: Runs as a training worker
"""

import argparse
import dataclasses
import datetime
import logging
import os
import subprocess
import threading
import time
from pathlib import Path

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.distributed import RayConfig
from levanter.inference.openai import InferenceServerConfig
from levanter.models.llama import LlamaConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from marin.cluster.ray import ray_dashboard
from marin.post_training.rollout_storage import FileRolloutReader, FileRolloutWriter
from marin.post_training.rollout_worker import RolloutWorker, RolloutWorkerConfig
from marin.post_training.train_worker import TrainWorker, TrainWorkerConfig
from marin.post_training.weight_transfer_manager import WeightTransferConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Test configuration
WANDB_PROJECT = "llama_small_rl_test"

STOP_TOKENS = [
    [524, 9399],
    [694, 9399],
    [4005, 9399],
    [6199, 9399],
    [8217, 9399],
    [9169, 9399],
    [12817, 9399],
    [19203, 9399],
    [20264, 9399],
    [22246, 9399],
    [27147, 9399],
    [128001],
]


SCRIPT_PATH = "src/marin/post_training/scripts/test_llama_small.py"
PREFIX = "gs://marin-us-central2"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_TOKENIZER = MODEL_NAME
# Use HuggingFace checkpoint directly for this test
MODEL_CHECKPOINT = MODEL_NAME
CHECKPOINT_DIR = f"{PREFIX}/rl_checkpoints/llama_small_test/checkpoints"
ROLLOUT_QUEUE_PATH = f"{PREFIX}/rl_checkpoints/llama_small_test/rollout_queue"
# Default cluster config for dashboard
DEFAULT_CLUSTER_CONFIG = "infra/marin-us-central2.yaml"


def llama_small_config() -> LlamaConfig:
    """Create LlamaConfig for Llama-3.2-1B-Instruct with reduced memory usage."""
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(MODEL_NAME)
    config = LlamaConfig.from_hf_config(hf_config)
    return dataclasses.replace(config, seq_len=512, tokenizer=MODEL_TOKENIZER)


def llama_small_trainer_config(output_dir: str) -> TrainerConfig:
    """Create TrainerConfig for Llama-3.2-1B training with reduced memory usage."""
    return TrainerConfig(
        tracker=WandbConfig(
            project=WANDB_PROJECT,
            mode="disabled",  # Disable for testing
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=8,
        num_train_steps=100,
        steps_per_eval=10,
        checkpointer=CheckpointerConfig(
            base_path=str(Path(output_dir)),
            save_interval=datetime.timedelta(seconds=60),
        ),
        tensor_parallel_axes=["mlp", "heads"],
        fsdp_axis="embed",
        batch_axis="batch",
        ray=RayConfig(auto_start_cluster=False),
    )


def llama_small_optimizer_config() -> AdamConfig:
    """Create optimizer configuration for Llama-3.2-1B."""
    return AdamConfig(
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup=10,
        lr_schedule="cosine",
    )


def llama_small_inference_server_config(output_dir: str) -> InferenceServerConfig:
    """Create inference server configuration for Llama-3.2-1B with reduced memory usage."""
    from levanter.compat.hf_checkpoints import RepoRef
    from levanter.inference.engine import InferenceEngineConfig

    return InferenceServerConfig(
        model=llama_small_config(),
        trainer=llama_small_trainer_config(output_dir),
        tokenizer=MODEL_TOKENIZER,
        hf_checkpoint=RepoRef(MODEL_NAME),
        max_new_tokens=32,  # Reduced for memory savings
        temperature=1.0,
        service=InferenceEngineConfig(
            max_seqs=8,  # Reduced from default 16
            max_pages_per_seq=16,  # Reduced from default 64 (16*128 = 2048 tokens max)
            page_size=128,  # Keep default
            max_seqs_in_prefill=4,  # Reduced from default 16
        ),
    )


def llama_small_training_worker_config(rollout_reader, output_dir: str) -> TrainWorkerConfig:
    """Create training worker configuration for Llama-3.2-1B."""
    return TrainWorkerConfig(
        rollout_reader=rollout_reader,
        model=llama_small_config(),
        trainer=llama_small_trainer_config(output_dir),
        optimizer=llama_small_optimizer_config(),
        kl_coef=1e-4,
        weight_transfer=WeightTransferConfig(
            sync_interval_steps=10,
            poll_interval_seconds=1,
            checkpoint_dir=str(Path(output_dir) / "policy_checkpoints"),
            max_checkpoints=5,
        ),
    )


def llama_small_rollout_worker_config(rollout_writer, output_dir: str) -> RolloutWorkerConfig:
    """Create inference worker configuration for Llama-3.2-1B."""
    # Create mock environment with Llama tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_TOKENIZER)

    return RolloutWorkerConfig(
        trainer=llama_small_trainer_config(output_dir),
        inference_server_config=llama_small_inference_server_config(output_dir),
        model=llama_small_config(),
        environment_spec="mock_env:task_type=simple_addition",
        rollout_writer=rollout_writer,
        max_input_length=32,
        max_output_length=32,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        n_prompts_per_step=8,
        n_generations=4,
        temperature=0.7,
        log_freq=10,
        rollout_batch_size=16,
        max_rollouts=10,
        weight_transfer=WeightTransferConfig(
            sync_interval_steps=10,
            poll_interval_seconds=1,
            checkpoint_dir=str(Path(output_dir) / "policy_checkpoints"),
            max_checkpoints=5,
        ),
    )


def run_inference_mode(args):
    """Run in inference worker mode."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("rollout_worker")

    logger.info("Starting inference worker mode...")

    subprocess.run("sudo --non-interactive rm -f /tmp/libtpu_lockfile", shell=True, check=False)

    rollout_writer = FileRolloutWriter(ROLLOUT_QUEUE_PATH)
    worker_config = llama_small_rollout_worker_config(rollout_writer, "/tmp/inference_checkpoint")
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

    subprocess.run("sudo --non-interactive rm -f /tmp/libtpu_lockfile", shell=True, check=False)
    rollout_reader = FileRolloutReader(ROLLOUT_QUEUE_PATH)
    worker_config = llama_small_training_worker_config(rollout_reader, CHECKPOINT_DIR)
    worker = TrainWorker(
        config=worker_config,
    )

    worker.train()
    logger.info("Training worker completed")


def tail_logs(proc, prefix):
    """Tail logs from a process with a prefix."""
    while proc.poll() is None:
        line = proc.stdout.readline()
        if line:
            print(f"[{prefix}] {line.rstrip()}")
        else:
            time.sleep(0.1)

    # Get remaining output
    remaining = proc.stdout.read()
    if remaining:
        for line in remaining.splitlines():
            print(f"[{prefix}] {line}")


def run_driver_mode():
    """Run in driver mode - launch inference and training workers via ray_run."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("driver")

    logger.info("Starting driver mode...")

    logger.info(f"Rollout queue path: {ROLLOUT_QUEUE_PATH}")

    # Build ray_run commands
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")

    inference_cmd = [
        "uv",
        "run",
        "src/marin/run/ray_run.py",
        "-e",
        "WANDB_API_KEY",
        wandb_key,
        "-e",
        "HF_TOKEN",
        hf_token,
        "--auto-stop",
        "--extra=tpu,post_training",
        "--entrypoint-resources",
        '{"TPU-v4-8-head":1, "TPU":4}',
        "--",
        "bash",
        "-c",
        f"python {SCRIPT_PATH} --mode inference; sudo rm -f /tmp/libtpu_lockfile*",
    ]

    training_cmd = [
        "uv",
        "run",
        "src/marin/run/ray_run.py",
        "-e",
        "WANDB_API_KEY",
        wandb_key,
        "-e",
        "HF_TOKEN",
        hf_token,
        "--auto-stop",
        "--extra=tpu,post_training",
        "--entrypoint-resources",
        '{"TPU-v4-8-head":1, "TPU":4}',
        "--",
        "bash",
        "-c",
        f"python {SCRIPT_PATH} --mode training; sudo rm -f /tmp/libtpu_lockfile*",
    ]

    logger.info("Ray address: " + os.environ.get("RAY_ADDRESS", "Not set"))

    # Launch processes
    logger.info("Launching inference worker... with command: " + " ".join(inference_cmd))
    inference_proc = subprocess.Popen(
        inference_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=os.environ,
    )

    # It will surprise no one that Ray gets confused if jobs schedule simultaneously
    time.sleep(5)

    logger.info("Launching training worker with command: " + " ".join(training_cmd))
    training_proc = subprocess.Popen(
        training_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=os.environ,
    )

    def monitor_processes():
        # Start log tailing threads
        inference_thread = threading.Thread(target=tail_logs, args=(inference_proc, "INFERENCE"))
        training_thread = threading.Thread(target=tail_logs, args=(training_proc, "TRAINING"))

        inference_thread.start()
        training_thread.start()

        while True:
            # Check if processes are still running
            inference_finished = inference_proc.poll() is not None
            training_finished = training_proc.poll() is not None

            if inference_finished:
                logger.info("Inference worker has completed")
                break
            if training_finished:
                logger.info("Training worker has completed")
                break

            time.sleep(5)

    try:
        monitor_processes()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, terminating workers...")
    except Exception as e:
        logger.error(f"Error occurred: {e}")

    # Clean shutdown
    logger.info("Terminating workers...")
    inference_proc.send_signal(subprocess.signal.SIGINT)
    training_proc.send_signal(subprocess.signal.SIGINT)
    time.sleep(5)
    if inference_proc.poll() is None:
        inference_proc.terminate()
    if training_proc.poll() is None:
        training_proc.terminate()
    inference_proc.wait()
    training_proc.wait()
    return 0


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Llama-3.2-1B-Instruct RL training with mock environment")
    parser.add_argument(
        "--mode",
        choices=["driver", "inference", "training"],
        default="driver",
        help="Execution mode (default: driver)",
    )
    parser.add_argument(
        "--cluster-config",
        default=DEFAULT_CLUSTER_CONFIG,
        help=f"Ray cluster config file (default: {DEFAULT_CLUSTER_CONFIG})",
    )
    args = parser.parse_args()

    if args.mode == "inference":
        logger.info("Running in inference mode")
        run_inference_mode(args)
    elif args.mode == "training":
        logger.info("Running in training mode")
        run_training_mode(args)
    else:
        logger.info("Running in driver mode")
        with ray_dashboard(args.cluster_config) as dashboard_port:
            logger.info(f"Ray dashboard started on port {dashboard_port}")
            return run_driver_mode()


if __name__ == "__main__":
    main()
