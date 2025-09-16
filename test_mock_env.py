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
import asyncio
import logging
import os
import subprocess
import time

from marin.post_training.inference_worker import InferenceWorker
from marin.post_training.rollout_storage import FileRolloutReader, FileRolloutWriter
from marin.post_training.train_worker import TrainingWorker
from marin.post_training.training_config import (
    CheckpointerConfigData,
    DistributedConfig,
    EnvironmentConfig,
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


def mock_eval_config(
    checkpoint_dir: str,
) -> TrainingConfig:
    """Create training configuration for single node test."""
    model_paths_config = ModelPathsConfig(
        params=None,  # Will initialize randomly
        tokenizer="meta-llama/Llama-3.2-1B",
        default_config_name="test_1m",
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
        max_output_length=16,
        temperature=1.0,
        stop_tokens=STOP_TOKENS,
        n_generations=8,
    )

    test_generation_config = GenerationConfig(
        max_output_length=16,
        temperature=0.0,
        stop_tokens=STOP_TOKENS,
        n_generations=8,
    )

    # Model override configuration for testing
    model_config_override = ModelOverrideConfig(
        max_sequence_length=128,  # Short sequences for testing
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
        sync_interval_steps=2,
        poll_interval_seconds=5.0,
        transfer_timeout=30.0,
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=3,
    )

    environment_config = EnvironmentConfig(
        train_environments_path="environments_test.json",
        test_environments_path="environments_test.json",
    )

    jax_distributed_config = {
        "initialize_jax_distributed": True,
        "process_id": 0,
    }
    sharding = [4, 1, 1, 1]

    return TrainingConfig(
        model=ModelConfig(
            model_paths=model_paths_config,
            inference_param_dtype="bf16",
            inference_activation_dtype="bf16",
            training_param_dtype="fp32",
            training_activation_dtype="bf16",
            model_config_override=model_config_override,
            tokenizer_override=TokenizerOverrideConfig(),
            train_attention_kernel_config='splash:{"block_size": 256}',
            prefill_attention_kernel_config='splash:{"block_size": 256}',
            generate_attention_kernel_config=(
                'paged:{"page_size": 256, "pages_per_compute_block": 1, "inline_seq_dim": true, "use_int8": false}'
            ),
        ),
        hyperparameters=TrainingHyperparameters(
            num_train_steps=10000,
            max_input_length=128,
            max_output_length=128,
            train_bsize=2,
            decode_bsize=4,
            prefill_bsize=4,
            reference_logprobs_bsize=4,
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
        environment=environment_config,
        distributed=DistributedConfig(
            train_sharding=sharding,
            inference_sharding=sharding,
            physical_axis_splitting=False,
            jax_distributed_initialize_config=jax_distributed_config,
        ),
        generation_config=generation_config,
        test_generation_config=test_generation_config,
        output_dir=checkpoint_dir,
        checkpoint=checkpointer_config,
        weight_transfer=weight_transfer_config,
    )


def run_inference_mode(args):
    """Run in inference worker mode."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("inference_worker")

    logger.info("Starting inference worker mode...")

    subprocess.run("sudo rm -f /tmp/libtpu_lockfile", shell=True, check=False)

    rollout_writer = FileRolloutWriter(args.rollout_queue_path)
    worker = InferenceWorker(
        training_config=mock_eval_config(args.checkpoint_dir),
        environment_spec="mock_env:count",
        rollout_writer=rollout_writer,
        rollout_batch_size=2,
        max_rollouts=15,
        coordinator=None,
    )

    worker.run()
    logger.info("Inference worker completed")


def run_training_mode(args):
    """Run in training worker mode."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("training_worker")

    logger.info("Starting training worker mode...")
    subprocess.run("sudo rm -f /tmp/libtpu_lockfile", shell=True, check=False)
    rollout_reader = FileRolloutReader(args.rollout_queue_path)
    worker = TrainingWorker(
        training_config=mock_eval_config(args.checkpoint_dir),
        rollout_reader=rollout_reader,
        coordinator=None,
    )

    worker.train()
    logger.info("Training worker completed")


async def tail_logs(proc, prefix):
    """Tail logs from a process with a prefix."""
    while proc.poll() is None:
        line = proc.stdout.readline()
        if line:
            print(f"[{prefix}] {line.rstrip()}")
        else:
            await asyncio.sleep(0.1)

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

    rollout_queue_path = "gs://marin-us-central2/rl_checkpoints/rollout_queue/mock_test"
    checkpoint_dir = "gs://marin-us-central2/rl_checkpoints/checkpoints/mock_test"

    logger.info(f"Rollout queue path: {rollout_queue_path}")

    script_path = "test_mock_env.py"

    # Build ray_run commands
    wandb_key = os.environ.get("WANDB_API_KEY", "")

    inference_cmd = [
        "uv",
        "run",
        "src/marin/run/ray_run.py",
        "-e",
        "WANDB_API_KEY",
        wandb_key,
        "--auto-stop",
        "--extra=tpu,post_training",
        "--entrypoint-resources",
        '{"TPU-v4-8-head":1, "TPU":4}',
        "--",
        "bash",
        "-c",
        f"python {script_path} --mode inference "
        f"--rollout-queue-path {rollout_queue_path} --checkpoint-dir={checkpoint_dir}; sudo rm -f /tmp/libtpu_lockfile*",
    ]

    training_cmd = [
        "uv",
        "run",
        "src/marin/run/ray_run.py",
        "-e",
        "WANDB_API_KEY",
        wandb_key,
        "--auto-stop",
        "--extra=tpu,post_training",
        "--entrypoint-resources",
        '{"TPU-v4-8-head":1, "TPU":4}',
        "--",
        "bash",
        "-c",
        f"python {script_path} --mode training "
        f"--rollout-queue-path {rollout_queue_path} --checkpoint-dir={checkpoint_dir}; sudo rm -f /tmp/libtpu_lockfile*",
    ]

    # Launch processes
    logger.info("Launching inference worker...")
    inference_proc = subprocess.Popen(
        inference_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    logger.info("Launching training worker...")
    training_proc = subprocess.Popen(
        training_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    start_time = time.time()
    max_runtime = 300  # 5 minutes max

    async def monitor_processes():
        tasks = [
            asyncio.create_task(tail_logs(inference_proc, "INFERENCE")),
            asyncio.create_task(tail_logs(training_proc, "TRAINING")),
        ]

        while time.time() - start_time < max_runtime:
            # Check if processes are still running
            if inference_proc.poll() is not None and training_proc.poll() is not None:
                logger.info("Both workers have completed")
                break

            await asyncio.sleep(5)
            elapsed = time.time() - start_time
            if int(elapsed) % 30 == 0:  # Log status every 30 seconds
                logger.info(
                    f"[{elapsed:.0f}s] Status: " f"inference={inference_proc.poll()}, training={training_proc.poll()}"
                )

        # Cancel tasks
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    # Run monitoring
    asyncio.run(monitor_processes())

    # Clean shutdown
    if inference_proc.poll() is None:
        logger.info("Terminating inference worker...")
        inference_proc.terminate()
        inference_proc.wait(timeout=10)

    if training_proc.poll() is None:
        logger.info("Terminating training worker...")
        training_proc.terminate()
        training_proc.wait(timeout=10)

    # Check results
    training_success = training_proc.returncode == 0
    inference_exit = inference_proc.returncode

    logger.info("=== Test Results ===")
    logger.info(f"Training process exit code: {training_proc.returncode}")
    logger.info(f"Inference process exit code: {inference_exit}")
    logger.info(f"Test duration: {time.time() - start_time:.1f}s")

    logger.info("✓ Mock RL training test completed successfully!")
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
    parser.add_argument("--rollout-queue-path", help="Path to rollout queue directory")
    parser.add_argument("--checkpoint-dir", help="Directory for checkpoints")
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
