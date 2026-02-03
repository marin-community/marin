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
Training script for Kelp on Stack-Edu Python dataset.

This script trains the tree diffusion model on real Python code from
the Stack-Edu dataset (filtered for Python files).

Usage:
    # Local training with cached data:
    uv run python experiments/kelp/train_stackedu.py --num_steps=1000

    # Force reload from GCS:
    uv run python experiments/kelp/train_stackedu.py --no_cache

    # Use more data:
    uv run python experiments/kelp/train_stackedu.py --max_files=20 --max_functions=20000
"""

import argparse
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax

import haliax as hax

from experiments.kelp.datasets import StackEduPythonDataset, load_stackv2_edu_python
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
class StackEduTrainConfig:
    """Configuration for Stack-Edu training."""

    # Model
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    mlp_dim: int = 512

    # Training
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_steps: int = 5000
    log_every: int = 50
    save_every: int = 1000
    warmup_steps: int = 500

    # Data
    max_nodes: int = 128
    max_children: int = 16
    max_value_len: int = 32
    s_max: int = 5  # Max corruption steps

    # Dataset
    max_files: int = 10  # GCS files to load
    max_functions: int = 10000  # Functions to cache
    use_cache: bool = True
    fallback_to_toy: bool = True  # Fall back to toy dataset if GCS fails

    # Output
    output_dir: str = "outputs/kelp"
    seed: int = 42


def get_cache_path(config: StackEduTrainConfig) -> str:
    """Get cache path for dataset."""
    cache_dir = Path(config.output_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / f"stackedu_python_{config.max_files}f_{config.max_functions}n.json")


def load_dataset(config: StackEduTrainConfig) -> list[str]:
    """Load training programs from Stack-Edu or fallback to toy."""
    cache_path = get_cache_path(config) if config.use_cache else None

    try:
        logger.info("Loading Stack-Edu Python dataset...")
        dataset = load_stackv2_edu_python(
            max_files=config.max_files,
            max_functions=config.max_functions,
            cache_path=cache_path,
        )
        dataset.load()
        programs = dataset.get_programs()

        if len(programs) > 0:
            logger.info(f"Loaded {len(programs)} functions from Stack-Edu")
            return programs
        else:
            logger.warning("Stack-Edu returned no programs")
            raise ValueError("No programs loaded")

    except Exception as e:
        if config.fallback_to_toy:
            logger.warning(f"Failed to load Stack-Edu ({e}), falling back to toy dataset")
            programs = get_toy_programs()
            logger.info(f"Loaded {len(programs)} toy programs as fallback")
            return programs
        else:
            raise


def create_single_example(
    code: str,
    num_corruption_steps: int,
    node_vocab: PythonNodeVocab,
    value_vocab: PythonValueVocab,
    config: StackEduTrainConfig,
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
    config: StackEduTrainConfig,
) -> dict[str, jnp.ndarray]:
    """Create a batch of training examples."""
    examples = []
    attempts = 0
    max_attempts = config.batch_size * 5

    while len(examples) < config.batch_size and attempts < max_attempts:
        code = random.choice(programs)
        num_steps = random.randint(1, config.s_max)
        example = create_single_example(code, num_steps, node_vocab, value_vocab, config)
        if example is not None:
            examples.append(example)
        attempts += 1

    if len(examples) < config.batch_size:
        while len(examples) < config.batch_size:
            examples.append(examples[0])

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
    """Compute loss for a single example."""
    config = model.config

    node_types_na = hax.named(node_types, (config.Nodes,))
    node_values_na = hax.named(node_values, (config.Nodes, config.ValueLen))
    depth_na = hax.named(depth, (config.Nodes,))
    mask_na = hax.named(mask, (config.Nodes,))

    location_logits, type_logits, value_logits = model(
        node_types_na, node_values_na, depth_na, mask=mask_na, key=None
    )

    # Location loss
    loc_log_probs = hax.nn.log_softmax(location_logits, axis=config.Nodes).array
    loc_one_hot = jax.nn.one_hot(edit_location, config.max_nodes)
    loc_loss = -jnp.sum(loc_log_probs * loc_one_hot)

    # Type loss
    type_log_probs = hax.nn.log_softmax(type_logits, axis=config.NodeVocab).array
    type_one_hot = jax.nn.one_hot(replacement_type, config.node_vocab_size)
    type_loss = -jnp.sum(type_log_probs * type_one_hot)

    # Value loss
    value_log_probs = hax.nn.log_softmax(value_logits, axis=config.ValueVocab).array

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
    """Single training step."""

    def loss_fn(model):
        return compute_batch_loss(model, batch)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss


def save_checkpoint(model: TreeDiffusionModel, path: str, step: int):
    """Save model checkpoint."""
    import pickle

    checkpoint = {
        "step": step,
        "model": eqx.filter(model, eqx.is_array),
        "config": model.config,
    }
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
    logger.info(f"Saved checkpoint to {path}")


def train(config: StackEduTrainConfig) -> tuple[TreeDiffusionModel, list[float]]:
    """Train the model on Stack-Edu Python."""
    logger.info("=" * 60)
    logger.info("Kelp Tree Diffusion - Stack-Edu Python Training")
    logger.info("=" * 60)

    # Check JAX backend
    backend = jax.default_backend()
    logger.info(f"JAX backend: {backend}")
    logger.info(f"JAX devices: {jax.devices()}")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    key = jrandom.PRNGKey(config.seed)

    # Create vocabularies
    node_vocab = PythonNodeVocab()
    value_vocab = PythonValueVocab()
    logger.info(f"Node vocab size: {node_vocab.vocab_size}")
    logger.info(f"Value vocab size: {value_vocab.vocab_size}")

    # Load dataset
    programs = load_dataset(config)

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

    # Create optimizer with warmup capped to avoid negative decay steps
    effective_warmup = min(config.warmup_steps, config.num_steps // 2)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=effective_warmup,
        decay_steps=config.num_steps,
        end_value=config.learning_rate * 0.1,
    )
    logger.info(f"Using warmup steps: {effective_warmup} (requested: {config.warmup_steps})")
    optimizer = optax.adamw(schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Training loop
    logger.info("-" * 60)
    logger.info(f"Training for {config.num_steps} steps on {len(programs)} programs")
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

            window = min(50, len(loss_history))
            avg_loss = sum(loss_history[-window:]) / window

            logger.info(
                f"Step {step:5d}/{config.num_steps} | "
                f"Loss: {loss_val:7.4f} | "
                f"Avg(50): {avg_loss:7.4f} | "
                f"Speed: {steps_per_sec:.1f} steps/s"
            )

        # Save checkpoint
        if config.save_every > 0 and step > 0 and step % config.save_every == 0:
            ckpt_path = os.path.join(config.output_dir, f"checkpoint_{step}.pkl")
            save_checkpoint(model, ckpt_path, step)

    total_time = time.time() - start_time
    logger.info("-" * 60)
    logger.info(f"Training complete in {total_time:.1f}s")
    logger.info(f"Final loss: {loss_history[-1]:.4f}")
    logger.info(f"Best loss: {min(loss_history):.4f}")

    # Save final checkpoint
    final_path = os.path.join(config.output_dir, "checkpoint_final.pkl")
    save_checkpoint(model, final_path, config.num_steps)

    # Check if loss decreased
    initial_avg = sum(loss_history[:10]) / min(10, len(loss_history))
    final_avg = sum(loss_history[-10:]) / min(10, len(loss_history))
    if final_avg < initial_avg:
        logger.info(f"Loss decreased from {initial_avg:.4f} to {final_avg:.4f}")
    else:
        logger.warning(f"Loss did NOT decrease: {initial_avg:.4f} -> {final_avg:.4f}")

    return model, loss_history


def main():
    parser = argparse.ArgumentParser(description="Train Kelp on Stack-Edu Python")
    parser.add_argument("--num_steps", type=int, default=2000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Transformer layers")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--log_every", type=int, default=50, help="Log every N steps")
    parser.add_argument("--save_every", type=int, default=1000, help="Save every N steps")
    parser.add_argument("--max_files", type=int, default=10, help="Max GCS files to load")
    parser.add_argument("--max_functions", type=int, default=10000, help="Max functions to cache")
    parser.add_argument("--no_cache", action="store_true", help="Disable caching")
    parser.add_argument("--output_dir", type=str, default="outputs/kelp", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = StackEduTrainConfig(
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        log_every=args.log_every,
        save_every=args.save_every,
        max_files=args.max_files,
        max_functions=args.max_functions,
        use_cache=not args.no_cache,
        output_dir=args.output_dir,
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
