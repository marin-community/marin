#!/usr/bin/env python3
# Copyright 2026 The Marin Authors
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
Local training script for tree diffusion model.

This script is optimized for running on a laptop (CPU or Apple Silicon MPS).
It uses a simple training loop without complex vmaps to ensure compatibility.

Usage:
    uv run python experiments/kelp/train_local.py

Or with custom settings:
    uv run python experiments/kelp/train_local.py --num_steps=500 --batch_size=4
"""

import argparse
import logging
import random
import sys
import time
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax

import haliax as hax

from experiments.kelp.edit_path import create_training_example
from experiments.kelp.python_grammar import PythonNodeVocab, PythonValueVocab
from experiments.kelp.toy_dataset import get_toy_programs
from experiments.kelp.tree_diffusion import TreeDiffusionConfig, TreeDiffusionModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class LocalTrainConfig:
    """Configuration for local training."""

    # Model (small for laptop)
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    mlp_dim: int = 256

    # Training
    batch_size: int = 8
    learning_rate: float = 3e-4
    num_steps: int = 1000
    log_every: int = 10
    warmup_steps: int = 100

    # Data
    max_nodes: int = 128  # Smaller for speed
    max_children: int = 8
    max_value_len: int = 16
    s_max: int = 3  # Max corruption steps

    # Misc
    seed: int = 42


def create_single_example(
    code: str,
    num_corruption_steps: int,
    node_vocab: PythonNodeVocab,
    value_vocab: PythonValueVocab,
    config: LocalTrainConfig,
) -> dict[str, np.ndarray] | None:
    """Create a single training example as numpy arrays."""
    example = create_training_example(
        code,
        num_corruption_steps,
        node_vocab,
        value_vocab,
        max_nodes=config.max_nodes,
        max_children=config.max_children,
        max_value_len=config.max_value_len,
    )
    if example is None:
        return None

    return {
        "node_types": example.corrupted_tensors.node_types,
        "node_values": example.corrupted_tensors.node_values,
        "depth": example.corrupted_tensors.depth,
        "mask": example.corrupted_tensors.node_mask,
        "edit_location": example.edit_location,
        "replacement_type": example.replacement_type,
        "replacement_value": example.replacement_value,
    }


def create_batch(
    programs: list[str],
    node_vocab: PythonNodeVocab,
    value_vocab: PythonValueVocab,
    config: LocalTrainConfig,
) -> dict[str, jnp.ndarray]:
    """Create a batch of training examples."""
    examples = []
    attempts = 0
    max_attempts = config.batch_size * 3

    while len(examples) < config.batch_size and attempts < max_attempts:
        code = random.choice(programs)
        num_steps = random.randint(1, config.s_max)
        example = create_single_example(code, num_steps, node_vocab, value_vocab, config)
        if example is not None:
            examples.append(example)
        attempts += 1

    if len(examples) < config.batch_size:
        # Pad with duplicates if needed
        while len(examples) < config.batch_size:
            examples.append(examples[0])

    # Stack into batch
    return {
        "node_types": jnp.stack([e["node_types"] for e in examples]),
        "node_values": jnp.stack([e["node_values"] for e in examples]),
        "depth": jnp.stack([e["depth"] for e in examples]),
        "mask": jnp.stack([e["mask"] for e in examples]),
        "edit_location": jnp.array([e["edit_location"] for e in examples]),
        "replacement_type": jnp.array([e["replacement_type"] for e in examples]),
        "replacement_value": jnp.stack([e["replacement_value"] for e in examples]),
    }


def compute_loss_single(
    model: TreeDiffusionModel,
    node_types: jnp.ndarray,
    node_values: jnp.ndarray,
    depth: jnp.ndarray,
    mask: jnp.ndarray,
    edit_location: jnp.ndarray,
    replacement_type: jnp.ndarray,
    replacement_value: jnp.ndarray,
) -> jnp.ndarray:
    """Compute loss for a single example (JAX-compatible, no Python ints)."""
    config = model.config

    # Convert to NamedArrays
    node_types_na = hax.named(node_types, (config.Nodes,))
    node_values_na = hax.named(node_values, (config.Nodes, config.ValueLen))
    depth_na = hax.named(depth, (config.Nodes,))
    mask_na = hax.named(mask, (config.Nodes,))

    # Forward pass
    location_logits, type_logits, value_logits = model(
        node_types_na, node_values_na, depth_na, mask=mask_na, key=None
    )

    # Location loss (cross-entropy) - use one-hot for indexing
    loc_log_probs = hax.nn.log_softmax(location_logits, axis=config.Nodes).array
    loc_one_hot = jax.nn.one_hot(edit_location, config.max_nodes)
    loc_loss = -jnp.sum(loc_log_probs * loc_one_hot)

    # Type loss
    type_log_probs = hax.nn.log_softmax(type_logits, axis=config.NodeVocab).array
    type_one_hot = jax.nn.one_hot(replacement_type, config.node_vocab_size)
    type_loss = -jnp.sum(type_log_probs * type_one_hot)

    # Value loss (average over positions) - use gather
    value_log_probs = hax.nn.log_softmax(value_logits, axis=config.ValueVocab).array
    # value_log_probs: (max_value_len, value_vocab_size)
    # replacement_value: (max_value_len,)
    # Use vmap to gather the correct log prob at each position
    def get_log_prob(log_probs_row, target_idx):
        return log_probs_row[target_idx]

    value_log_probs_selected = jax.vmap(get_log_prob)(value_log_probs, replacement_value)
    value_loss = -jnp.mean(value_log_probs_selected)

    return loc_loss + type_loss + value_loss


def compute_batch_loss(model: TreeDiffusionModel, batch: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Compute average loss over a batch using vmap."""

    def single_loss(node_types, node_values, depth, mask, edit_loc, repl_type, repl_value):
        return compute_loss_single(
            model, node_types, node_values, depth, mask, edit_loc, repl_type, repl_value
        )

    # vmap over batch dimension
    losses = jax.vmap(single_loss)(
        batch["node_types"],
        batch["node_values"],
        batch["depth"],
        batch["mask"],
        batch["edit_location"],
        batch["replacement_type"],
        batch["replacement_value"],
    )

    return jnp.mean(losses)


@eqx.filter_jit
def train_step(
    model: TreeDiffusionModel,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    batch: dict[str, jnp.ndarray],
) -> tuple[TreeDiffusionModel, optax.OptState, jnp.ndarray]:
    """Single training step with gradient computation."""

    def loss_fn(model):
        return compute_batch_loss(model, batch)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss


def train(config: LocalTrainConfig) -> tuple[TreeDiffusionModel, list[float]]:
    """Train the model and return it with loss history."""
    logger.info("=" * 60)
    logger.info("Kelp Tree Diffusion - Local Training")
    logger.info("=" * 60)

    # Check JAX backend
    backend = jax.default_backend()
    logger.info(f"JAX backend: {backend}")
    logger.info(f"JAX devices: {jax.devices()}")

    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    key = jrandom.PRNGKey(config.seed)

    # Create vocabularies
    node_vocab = PythonNodeVocab()
    value_vocab = PythonValueVocab()
    logger.info(f"Node vocab size: {node_vocab.vocab_size}")
    logger.info(f"Value vocab size: {value_vocab.vocab_size}")

    # Create model
    model_config = TreeDiffusionConfig(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        mlp_dim=config.mlp_dim,
        max_nodes=config.max_nodes,
        max_children=config.max_children,
        max_value_len=config.max_value_len,
        node_vocab_size=node_vocab.vocab_size,
        value_vocab_size=value_vocab.vocab_size,
        s_max=config.s_max,
    )

    key, model_key = jrandom.split(key)
    model = TreeDiffusionModel.init(model_config, key=model_key)

    # Count parameters
    num_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    logger.info(f"Model parameters: {num_params:,}")

    # Create optimizer with warmup
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.num_steps,
        end_value=config.learning_rate * 0.1,
    )
    optimizer = optax.adamw(schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Load programs
    programs = get_toy_programs()
    logger.info(f"Loaded {len(programs)} toy programs")

    # Training loop
    logger.info("-" * 60)
    logger.info(f"Training for {config.num_steps} steps...")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info("-" * 60)

    loss_history = []
    start_time = time.time()

    for step in range(config.num_steps):
        # Create batch
        batch = create_batch(programs, node_vocab, value_vocab, config)

        # Train step
        model, opt_state, loss = train_step(model, opt_state, optimizer, batch)
        loss_val = float(loss)
        loss_history.append(loss_val)

        # Log progress
        if step % config.log_every == 0 or step == config.num_steps - 1:
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0

            # Compute running average
            window = min(50, len(loss_history))
            avg_loss = sum(loss_history[-window:]) / window

            logger.info(
                f"Step {step:5d}/{config.num_steps} | "
                f"Loss: {loss_val:7.4f} | "
                f"Avg(50): {avg_loss:7.4f} | "
                f"Speed: {steps_per_sec:.1f} steps/s"
            )

    total_time = time.time() - start_time
    logger.info("-" * 60)
    logger.info(f"Training complete in {total_time:.1f}s")
    logger.info(f"Final loss: {loss_history[-1]:.4f}")
    logger.info(f"Best loss: {min(loss_history):.4f}")

    # Check if loss decreased
    initial_avg = sum(loss_history[:10]) / 10
    final_avg = sum(loss_history[-10:]) / 10
    if final_avg < initial_avg:
        logger.info(f"Loss decreased from {initial_avg:.4f} to {final_avg:.4f}")
    else:
        logger.warning(f"Loss did NOT decrease: {initial_avg:.4f} -> {final_avg:.4f}")

    return model, loss_history


def main():
    parser = argparse.ArgumentParser(description="Train Kelp tree diffusion model locally")
    parser.add_argument("--num_steps", type=int, default=500, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--log_every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = LocalTrainConfig(
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        log_every=args.log_every,
        seed=args.seed,
    )

    try:
        model, loss_history = train(config)
        logger.info("Done!")
        return 0
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1


if __name__ == "__main__":
    sys.exit(main())
