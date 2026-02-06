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

"""Train a toy tree diffusion model on the toy dataset.

This script is designed for fast iteration on a laptop (CPU).
It trains a small model on the hardcoded toy Python functions.

Usage:
    uv run python experiments/kelp/train_toy.py
    uv run python experiments/kelp/train_toy.py --steps 100 --batch-size 4
"""

import argparse
import logging
import sys
from collections.abc import Iterator

import jax
import jax.numpy as jnp
from jax import random

from experiments.kelp.data.toy_dataset import TOY_PROGRAMS
from experiments.kelp.model.config import TreeDiffusionConfig
from experiments.kelp.tokenizer import SimpleTokenizer
from experiments.kelp.training.train import TrainingConfig, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a toy tree diffusion model")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--max-seq-len", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--diffusion-steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument("--output-dir", type=str, default="checkpoints/kelp-toy", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-interval", type=int, default=10, help="Steps between logging")
    parser.add_argument("--checkpoint-interval", type=int, default=500, help="Steps between checkpoints")
    return parser.parse_args()


def create_toy_data_iter(
    batch_size: int,
    max_seq_len: int,
    tokenizer: SimpleTokenizer,
    key: jax.Array,
) -> Iterator[dict[str, jax.Array]]:
    """Create an iterator over toy dataset batches.

    Args:
        batch_size: Batch size.
        max_seq_len: Maximum sequence length.
        tokenizer: Tokenizer.
        key: PRNG key.

    Yields:
        Batches with 'tokens' and 'prefix_len' keys.
    """
    encoded_examples = []
    for prog in TOY_PROGRAMS:
        prompt_ids = tokenizer.encode(prog.prompt)
        code_ids = tokenizer.encode(prog.full_code)

        if len(code_ids) <= max_seq_len:
            encoded_examples.append(
                {
                    "tokens": code_ids,
                    "prefix_len": len(prompt_ids),
                }
            )

    logger.info(f"Created {len(encoded_examples)} examples from toy dataset")

    num_examples = len(encoded_examples)

    while True:
        key, shuffle_key = random.split(key)
        indices = random.permutation(shuffle_key, jnp.arange(num_examples))

        for i in range(0, num_examples, batch_size):
            batch_indices = indices[i : i + batch_size]

            if len(batch_indices) < batch_size:
                key, extra_key = random.split(key)
                extra_indices = random.randint(extra_key, (batch_size - len(batch_indices),), 0, num_examples)
                batch_indices = jnp.concatenate([batch_indices, extra_indices])

            batch_tokens = []
            batch_prefix_lens = []

            for idx in batch_indices:
                example = encoded_examples[int(idx)]
                tokens = example["tokens"]

                padded = tokens + [tokenizer.pad_token_id] * (max_seq_len - len(tokens))
                padded = padded[:max_seq_len]

                batch_tokens.append(padded)
                batch_prefix_lens.append(min(example["prefix_len"], max_seq_len))

            yield {
                "tokens": jnp.array(batch_tokens),
                "prefix_len": jnp.array(batch_prefix_lens),
            }


def main():
    """Main training function."""
    args = parse_args()

    logger.info("Starting toy tree diffusion training")
    logger.info(f"Configuration: {args}")

    vocab_size = 256
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)

    model_config = TreeDiffusionConfig(
        vocab_size=vocab_size,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.hidden_dim * 4,
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=args.heads,
        max_seq_len=args.max_seq_len,
        num_diffusion_steps=args.diffusion_steps,
        prefix_max_len=args.max_seq_len // 2,
    )

    training_config = TrainingConfig(
        model=model_config,
        learning_rate=args.lr,
        total_steps=args.steps,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    key = random.PRNGKey(args.seed)
    data_iter = create_toy_data_iter(
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        tokenizer=tokenizer,
        key=key,
    )

    model = train(training_config, data_iter)

    logger.info("Training complete!")

    logger.info("\nGenerating sample outputs...")
    sample_prompts = [prog.prompt for prog in TOY_PROGRAMS[:3]]

    for i, prompt in enumerate(sample_prompts):
        logger.info(f"\nPrompt {i + 1}:")
        logger.info(prompt)

        prefix_ids = jnp.array(tokenizer.encode(prompt))
        generated = model.sample(
            prefix=prefix_ids,
            max_iterations=args.diffusion_steps,
            temperature=0.8,
            tokenizer=tokenizer,
            target_len=args.max_seq_len,
        )

        logger.info("Generated:")
        logger.info(generated)


if __name__ == "__main__":
    main()
