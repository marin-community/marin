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

"""Main training script for Kelp tree diffusion models.

Supports training at different scales with various datasets.

Usage:
    # Toy model on toy data (laptop)
    uv run python experiments/kelp/train.py --preset laptop --dataset toy

    # Medium model on quine data (GPU)
    uv run python experiments/kelp/train.py --preset single_gpu --dataset quine

    # Transfer from Marin 8b
    uv run python experiments/kelp/train.py --preset tpu_v5p_8 --dataset stack_edu --transfer

For ExecutorStep-based training, use the executor_main entry point.
"""

import argparse
import logging
import sys
from dataclasses import dataclass

from experiments.kelp.model.presets import PRESETS, get_preset
from experiments.kelp.training.train import TrainingConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KelpTrainingConfig:
    """Full configuration for Kelp training."""

    preset: str
    """Model preset name."""

    dataset: str
    """Dataset name: 'toy', 'quine', 'stack_edu'."""

    output_path: str
    """Output path for checkpoints and logs."""

    total_steps: int = 10000
    """Total training steps."""

    learning_rate: float | None = None
    """Learning rate (uses preset default if None)."""

    batch_size: int | None = None
    """Batch size (uses preset default if None)."""

    transfer_from: str | None = None
    """Path to checkpoint for transfer learning."""

    seed: int = 42
    """Random seed."""

    log_interval: int = 10
    """Steps between logging."""

    checkpoint_interval: int = 1000
    """Steps between checkpoints."""


def run_kelp_training(config: KelpTrainingConfig) -> dict:
    """Run Kelp training.

    This function is called by the executor framework.

    Args:
        config: Training configuration.

    Returns:
        Dictionary with training results.
    """
    from jax import random

    preset = get_preset(config.preset)
    model_config = preset.config
    lr = config.learning_rate or preset.learning_rate
    batch_size = config.batch_size or preset.batch_size

    training_config = TrainingConfig(
        model=model_config,
        learning_rate=lr,
        total_steps=config.total_steps,
        batch_size=batch_size,
        log_interval=config.log_interval,
        checkpoint_interval=config.checkpoint_interval,
        output_dir=config.output_path,
        seed=config.seed,
    )

    key = random.PRNGKey(config.seed)

    if config.dataset == "toy":
        from experiments.kelp.train_toy import SimpleTokenizer, create_toy_data_iter

        tokenizer = SimpleTokenizer(vocab_size=model_config.vocab_size)
        data_iter = create_toy_data_iter(
            batch_size=batch_size,
            max_seq_len=model_config.max_seq_len,
            tokenizer=tokenizer,
            key=key,
        )
    elif config.dataset == "quine":
        from experiments.kelp.data.quine_dataset import create_quine_data_iter

        data_iter = create_quine_data_iter(
            batch_size=batch_size,
            max_seq_len=model_config.max_seq_len,
            key=key,
        )
    elif config.dataset == "stack_edu":
        from experiments.kelp.data.stack_edu import create_stack_edu_data_iter

        data_iter = create_stack_edu_data_iter(
            batch_size=batch_size,
            max_seq_len=model_config.max_seq_len,
            key=key,
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    initial_params = None
    if config.transfer_from:
        logger.info(f"Loading weights from {config.transfer_from}")
        from experiments.kelp.transfer.adapt import transfer_ar_to_tree_diffusion

        initial_params = transfer_ar_to_tree_diffusion(
            config.transfer_from,
            model_config,
        )

    from experiments.kelp.training.train import train

    model = train(training_config, data_iter, initial_params=initial_params)

    return {
        "status": "completed",
        "output_path": config.output_path,
        "total_steps": config.total_steps,
    }


def kelp_training_step(
    preset: str = "laptop",
    dataset: str = "toy",
    total_steps: int = 10000,
    transfer_from: str | None = None,
    name_suffix: str = "",
) -> ExecutorStep:
    """Create an ExecutorStep for Kelp training.

    Args:
        preset: Model preset name.
        dataset: Dataset name.
        total_steps: Total training steps.
        transfer_from: Optional path for transfer learning.
        name_suffix: Optional suffix for step name.

    Returns:
        ExecutorStep for training.
    """
    step_name = f"kelp/train/{preset}/{dataset}"
    if name_suffix:
        step_name = f"{step_name}/{name_suffix}"
    if transfer_from:
        step_name = f"{step_name}/transfer"

    return ExecutorStep(
        name=step_name,
        fn=run_kelp_training,
        config=KelpTrainingConfig(
            preset=preset,
            dataset=dataset,
            output_path=this_output_path(),
            total_steps=total_steps,
            transfer_from=transfer_from,
        ),
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a Kelp tree diffusion model")
    parser.add_argument("--preset", type=str, default="laptop", choices=list(PRESETS.keys()), help="Model preset")
    parser.add_argument("--dataset", type=str, default="toy", choices=["toy", "quine", "stack_edu"], help="Dataset")
    parser.add_argument("--steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (uses preset default if not set)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (uses preset default if not set)")
    parser.add_argument("--transfer", type=str, default=None, help="Path to checkpoint for transfer learning")
    parser.add_argument("--output-dir", type=str, default="checkpoints/kelp", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-executor", action="store_true", help="Run via executor framework")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.use_executor:
        step = kelp_training_step(
            preset=args.preset,
            dataset=args.dataset,
            total_steps=args.steps,
            transfer_from=args.transfer,
        )
        executor_main(steps=[step])
    else:
        config = KelpTrainingConfig(
            preset=args.preset,
            dataset=args.dataset,
            output_path=args.output_dir,
            total_steps=args.steps,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            transfer_from=args.transfer,
            seed=args.seed,
        )

        run_kelp_training(config)


if __name__ == "__main__":
    main()
