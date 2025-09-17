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
Test script for mock environment RL training.

Supports three modes:
- driver (default): Launches inference and training workers via ray_run
- inference: Runs as an inference worker
- training: Runs as a training worker
"""

import argparse
import logging
import os
import subprocess
import threading
import time

from marin.post_training.inference_worker import InferenceWorker
from marin.post_training.rollout_storage import FileRolloutReader, FileRolloutWriter
from marin.post_training.train_worker import TrainingWorker
from marin.post_training.training_config import (
    CheckpointerConfigData,
    DistributedConfig,
    GenerationConfig,
    LoggingConfig,
    ModelConfig,
    ModelOverrideConfig,
    ModelPathsConfig,
    OptimizerConfig,
    TokenizerOverrideConfig,
    TrainingConfig,
    TrainingHyperparameters,
    WeightTransferConfig,
    WeightTransferMode,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Test configuration
WANDB_PROJECT = "mock_env_rl_test"

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


SCRIPT_PATH = "src/marin/post_training/scripts/test_mock_env.py"
PREFIX = "gs://marin-us-central2"
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
MODEL_TOKENIZER = MODEL_NAME
MODEL_PARAMS = f"{PREFIX}/rl_checkpoints/base/{MODEL_NAME}/params.msgpack"
MODEL_CONFIG = f"{PREFIX}/rl_checkpoints/base/{MODEL_NAME}/config.json"
CHECKPOINT_DIR = f"{PREFIX}/rl_checkpoints/mock_env_test/checkpoints"
ROLLOUT_QUEUE_PATH = f"{PREFIX}/rl_checkpoints/mock_env_test/rollout_queue"


def mock_eval_config() -> TrainingConfig:
    """Create training configuration for single node test."""
    model_paths_config = ModelPathsConfig(
        params=MODEL_PARAMS,
        tokenizer=MODEL_TOKENIZER,
        default_config_name=None,
        config=MODEL_CONFIG,
    )

    optim_config = OptimizerConfig(
        init_lr=1e-5,
        end_lr=1e-6,
        lr=1e-5,
        lr_warmup_steps=10,
        lr_decay_steps=100,
        b1=0.9,
        b2=0.95,
        clip_gradient=1.0,
        weight_decay=0.01,
        bf16_momentum=False,
        multiply_by_parameter_scale=False,
        weight_decay_exclusions=[],
        schedule="cos",
        grad_accum_steps=1,
    )

    generation_config = GenerationConfig(
        max_output_length=129,
        temperature=1.0,
        stop_tokens=STOP_TOKENS,
        n_generations=8,
    )

    test_generation_config = GenerationConfig(
        max_output_length=129,
        temperature=0.0,
        stop_tokens=STOP_TOKENS,
        n_generations=8,
    )

    # Model override configuration for testing
    model_config_override = ModelOverrideConfig(
        max_sequence_length=512,
        initializer_range=0.02,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        remat_block="nothing_saveable",
    )

    checkpointer_config = CheckpointerConfigData(
        save_optimizer_state=False,
        save_float_dtype="bf16",
        save_model_freq=5,  # Save frequently for testing
    )

    weight_transfer_config = WeightTransferConfig(
        mode=WeightTransferMode.GCS_CHECKPOINT,
        sync_interval_steps=10,
        poll_interval_seconds=5.0,
        transfer_timeout=30.0,
        checkpoint_dir=CHECKPOINT_DIR,
        max_checkpoints=3,
    )

    jax_distributed_config = {
        "initialize_jax_distributed": True,
    }
    sharding = [1, 4, 1, 1]

    return TrainingConfig(
        model=ModelConfig(
            model_paths=model_paths_config,
            inference_param_dtype="bf16",
            inference_activation_dtype="bf16",
            training_param_dtype="fp32",
            training_activation_dtype="bf16",
            model_config_override=model_config_override,
            tokenizer_override=TokenizerOverrideConfig(),
            train_attention_kernel_config="default:{}",
            prefill_attention_kernel_config="default:{}",
            generate_attention_kernel_config="default:{}",
            # train_attention_kernel_config='splash:{"block_size": 256}',
            # prefill_attention_kernel_config='splash:{"block_size": 256}',
            # generate_attention_kernel_config=(
            #     'paged:{"page_size": 256, "pages_per_compute_block": 1, "inline_seq_dim": true, "use_int8": false}'
            # ),
        ),
        hyperparameters=TrainingHyperparameters(
            num_train_steps=10000,
            max_input_length=64,
            max_output_length=65,
            train_bsize=4,
            decode_bsize=8,
            prefill_bsize=8,
            reference_logprobs_bsize=8,
            n_prompts_per_step=8,
            optim_config=optim_config,
            pad_token_id=2,
            kl_coef=1e-4,
        ),
        logging=LoggingConfig(
            log_freq=5,
            num_eval_examples=2,
            wandb_project=WANDB_PROJECT,
            save_initial_checkpoint=True,
            log_initial_step=True,
            max_checkpoints=None,
            online=False,  # Disable wandb for testing
            enable=False,
            prefix="test_single_node",
            prefix_to_id=True,
            experiment_id="test_single_node",
        ),
        distributed=DistributedConfig(
            train_sharding=sharding,
            inference_sharding=sharding,
            physical_axis_splitting=False,
            jax_distributed_initialize_config=jax_distributed_config,
        ),
        generation_config=generation_config,
        test_generation_config=test_generation_config,
        output_dir=CHECKPOINT_DIR,
        checkpoint=checkpointer_config,
        weight_transfer=weight_transfer_config,
    )


def run_inference_mode(args):
    """Run in inference worker mode."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("inference_worker")

    logger.info("Starting inference worker mode...")

    subprocess.run("sudo --non-interactive rm -f /tmp/libtpu_lockfile", shell=True, check=False)

    rollout_writer = FileRolloutWriter(ROLLOUT_QUEUE_PATH)
    worker = InferenceWorker(
        training_config=mock_eval_config(),
        environment_spec="mock:task_type=count",
        rollout_writer=rollout_writer,
        rollout_batch_size=2,
        max_rollouts=100,
        coordinator=None,
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
    worker = TrainingWorker(
        training_config=mock_eval_config(),
        rollout_reader=rollout_reader,
        coordinator=None,
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

    # Launch processes
    logger.info("Launching inference worker... with command: " + " ".join(inference_cmd))
    inference_proc = subprocess.Popen(
        inference_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
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
    )

    def monitor_processes():
        # Start log tailing threads
        inference_thread = threading.Thread(target=tail_logs, args=(inference_proc, "INFERENCE"))
        training_thread = threading.Thread(target=tail_logs, args=(training_proc, "TRAINING"))

        inference_thread.start()
        training_thread.start()

        while True:
            # Check if processes are still running
            if inference_proc.poll() is not None:
                logger.info("Inference worker has completed")
                break
            if training_proc.poll() is not None:
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
    inference_proc.terminate()
    training_proc.terminate()
    inference_proc.wait()
    training_proc.wait()
    return 0


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Mock environment RL training")
    parser.add_argument(
        "--mode",
        choices=["driver", "inference", "training"],
        default="driver",
        help="Execution mode (default: driver)",
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
        return run_driver_mode()


if __name__ == "__main__":
    main()
