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

"""Main training script for Kelp tree diffusion edit models.

Trains an AR edit-prediction model on Python source code using the tree
diffusion pipeline: corrupt programs via AST subtree replacement, compute
TreeDiff edit paths, and train the model to predict single edits.

Usage:
    # Train on toy corpus (laptop)
    uv run python experiments/kelp/train.py --preset toy --steps 1000

    # Train overnight on CPU
    uv run python experiments/kelp/train.py --preset overnight_cpu --steps 30000

    # With W&B logging
    uv run python experiments/kelp/train.py --preset laptop --wandb-project kelp
"""

import argparse
import logging
import sys

from experiments.kelp.model.presets import PRESETS, get_preset
from experiments.kelp.tree.subtree_bank import SubtreeBank
from experiments.kelp.tree.tokenizer import TreeDiffusionTokenizer
from experiments.kelp.tree.train import (
    EditTrainingConfig,
    create_edit_data_iter,
    train_edit_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# Toy corpus for quick iteration.
TOY_CORPUS = [
    "def add(a, b):\n    return a + b\n",
    "def sub(a, b):\n    return a - b\n",
    "def mul(a, b):\n    return a * b\n",
    "def div(a, b):\n    return a / b\n",
    "def neg(x):\n    return -x\n",
    "def square(x):\n    return x * x\n",
    "def double(x):\n    return x + x\n",
    "def is_positive(x):\n    return x > 0\n",
    "def is_zero(x):\n    return x == 0\n",
    "def identity(x):\n    return x\n",
    "def abs_val(x):\n    if x < 0:\n        return -x\n    return x\n",
    "def max_val(a, b):\n    if a > b:\n        return a\n    return b\n",
    "def min_val(a, b):\n    if a < b:\n        return a\n    return b\n",
    "def clamp(x, lo, hi):\n    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x\n",
    "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n - 1) + fib(n - 2)\n",
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a Kelp tree diffusion edit model")
    parser.add_argument("--preset", type=str, default="toy", choices=list(PRESETS.keys()), help="Model preset")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (uses preset default if not set)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (uses preset default if not set)")
    parser.add_argument("--output-dir", type=str, default="checkpoints/kelp-edit", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-interval", type=int, default=10, help="Steps between logging")
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project name (enables W&B logging)")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument(
        "--corpus-file",
        type=str,
        default=None,
        help="Path to corpus file (one program per entry, separated by blank lines)",
    )
    return parser.parse_args()


def load_corpus(path: str) -> list[str]:
    """Load a corpus from a file.

    Programs are separated by blank lines. Each program is a contiguous
    block of non-empty lines.
    """
    programs = []
    current: list[str] = []

    with open(path) as f:
        for line in f:
            if line.strip() == "" and current:
                programs.append("\n".join(current) + "\n")
                current = []
            elif line.strip():
                current.append(line.rstrip())

    if current:
        programs.append("\n".join(current) + "\n")

    return programs


def main():
    """Main entry point."""
    args = parse_args()

    preset = get_preset(args.preset)
    model_config = preset.config
    lr = args.lr or preset.learning_rate
    batch_size = args.batch_size or preset.batch_size

    # Load corpus.
    if args.corpus_file:
        corpus = load_corpus(args.corpus_file)
        logger.info(f"Loaded {len(corpus)} programs from {args.corpus_file}")
    else:
        corpus = TOY_CORPUS
        logger.info(f"Using toy corpus ({len(corpus)} programs)")

    # Build subtree bank and tokenizer.
    bank = SubtreeBank.from_corpus(corpus)
    tokenizer = TreeDiffusionTokenizer(max_seq_len=model_config.max_seq_len)
    logger.info(f"Subtree bank: {bank.total_entries} entries across {len(bank.entries)} node types")

    # Override model config vocab_size to match tokenizer.
    from dataclasses import replace

    model_config = replace(model_config, vocab_size=tokenizer.vocab_size)

    train_cfg = EditTrainingConfig(
        model=model_config,
        max_seq_len=model_config.max_seq_len,
        learning_rate=lr,
        total_steps=args.steps,
        batch_size=batch_size,
        log_interval=args.log_interval,
        output_dir=args.output_dir,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    data_iter = create_edit_data_iter(
        corpus=corpus,
        bank=bank,
        tokenizer=tokenizer,
        config=train_cfg,
        seed=args.seed,
    )

    logger.info(f"Training config: {train_cfg}")
    logger.info(f"Model config: {model_config}")

    params = train_edit_model(
        config=train_cfg,
        data_iter=data_iter,
    )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
