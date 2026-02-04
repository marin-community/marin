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
Training script comparing Marin-initialized vs random-initialized tree diffusion.

This script trains two models:
1. Random initialization (baseline)
2. Marin 8B transfer initialization (experimental)

And compares their convergence speed and final performance.

Usage:
    # Compare on toy dataset:
    uv run python experiments/kelp/train_transfer.py --num_steps=500

    # Compare on Stack-Edu:
    uv run python experiments/kelp/train_transfer.py --num_steps=2000 --use_stackedu
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
class TransferExperimentConfig:
    """Configuration for transfer experiment."""

    # Model
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    mlp_dim: int = 512

    # Training
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_steps: int = 1000
    log_every: int = 50
    warmup_steps: int = 100

    # Data
    max_nodes: int = 128
    max_children: int = 16
    max_value_len: int = 32
    s_max: int = 5

    # Transfer
    projection_method: str = "random"  # "random" or "pca"
    use_marin_transfer: bool = True

    # Dataset
    use_stackedu: bool = False
    max_files: int = 5
    max_functions: int = 5000

    # Misc
    seed: int = 42


def load_programs(config: TransferExperimentConfig) -> list[str]:
    """Load training programs."""
    if config.use_stackedu:
        try:
            from experiments.kelp.data_pipeline import load_stackv2_edu_python

            logger.info("Loading Stack-Edu Python dataset...")
            dataset = load_stackv2_edu_python(
                max_files=config.max_files,
                max_functions=config.max_functions,
            )
            dataset.load()
            programs = dataset.get_programs()
            if programs:
                logger.info(f"Loaded {len(programs)} programs from Stack-Edu")
                return programs
        except Exception as e:
            logger.warning(f"Failed to load Stack-Edu: {e}")

    programs = get_toy_programs()
    logger.info(f"Using {len(programs)} toy programs")
    return programs


def create_batch(
    programs: list[str],
    node_vocab: PythonNodeVocab,
    value_vocab: PythonValueVocab,
    config: TransferExperimentConfig,
) -> dict[str, jnp.ndarray]:
    """Create a training batch."""
    examples = []
    attempts = 0
    max_attempts = config.batch_size * 5

    while len(examples) < config.batch_size and attempts < max_attempts:
        code = random.choice(programs)
        num_steps = random.randint(1, config.s_max)

        example = create_training_example(
            code,
            num_steps,
            node_vocab,
            value_vocab,
            max_nodes=config.max_nodes,
            max_children=config.max_children,
            max_value_len=config.max_value_len,
        )
        if example is not None:
            examples.append({
                "node_types": example.corrupted_tensors.node_types,
                "node_values": example.corrupted_tensors.node_values,
                "depth": example.corrupted_tensors.depth,
                "mask": example.corrupted_tensors.node_mask,
                "edit_location": example.edit_location,
                "replacement_type": example.replacement_type,
                "replacement_value": example.replacement_value,
            })
        attempts += 1

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
    """Compute average loss over a batch."""

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


def train_model(
    model: TreeDiffusionModel,
    programs: list[str],
    node_vocab: PythonNodeVocab,
    value_vocab: PythonValueVocab,
    config: TransferExperimentConfig,
    model_name: str,
) -> list[float]:
    """Train a model and return loss history."""
    # Create optimizer
    effective_warmup = min(config.warmup_steps, config.num_steps // 2)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=effective_warmup,
        decay_steps=config.num_steps,
        end_value=config.learning_rate * 0.1,
    )
    optimizer = optax.adamw(schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    loss_history = []
    start_time = time.time()

    for step in range(config.num_steps):
        batch = create_batch(programs, node_vocab, value_vocab, config)
        model, opt_state, loss = train_step(model, opt_state, optimizer, batch)
        loss_val = float(loss)
        loss_history.append(loss_val)

        if step % config.log_every == 0 or step == config.num_steps - 1:
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            window = min(50, len(loss_history))
            avg_loss = sum(loss_history[-window:]) / window

            logger.info(
                f"[{model_name}] Step {step:5d}/{config.num_steps} | "
                f"Loss: {loss_val:7.4f} | Avg: {avg_loss:7.4f} | "
                f"Speed: {steps_per_sec:.1f} steps/s"
            )

    return loss_history


def run_experiment(config: TransferExperimentConfig):
    """Run the transfer comparison experiment."""
    logger.info("=" * 70)
    logger.info("Kelp AR Transfer Experiment: Marin 8B vs Random Initialization")
    logger.info("=" * 70)

    backend = jax.default_backend()
    logger.info(f"JAX backend: {backend}")

    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Create vocabularies
    node_vocab = PythonNodeVocab()
    value_vocab = PythonValueVocab()
    logger.info(f"Node vocab: {node_vocab.vocab_size}, Value vocab: {value_vocab.vocab_size}")

    # Load programs
    programs = load_programs(config)

    # Model config
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

    # ============================================================
    # Train random-initialized model (baseline)
    # ============================================================
    logger.info("-" * 70)
    logger.info("Training RANDOM-initialized model (baseline)...")
    logger.info("-" * 70)

    key = jrandom.PRNGKey(config.seed)
    key, model_key = jrandom.split(key)
    random_model = TreeDiffusionModel.init(model_config, key=model_key)

    num_params = sum(x.size for x in jax.tree.leaves(eqx.filter(random_model, eqx.is_array)))
    logger.info(f"Model parameters: {num_params:,}")

    random_history = train_model(
        random_model, programs, node_vocab, value_vocab, config, "RANDOM"
    )

    # ============================================================
    # Train Marin-initialized model (experimental)
    # ============================================================
    if config.use_marin_transfer:
        logger.info("-" * 70)
        logger.info("Training MARIN-initialized model (transfer)...")
        logger.info("-" * 70)

        try:
            from experiments.kelp.ar_transfer import (
                TransferConfig,
                initialize_model_from_marin,
            )

            transfer_config = TransferConfig(
                target_hidden_dim=config.hidden_dim,
                projection_method=config.projection_method,
                seed=config.seed,
            )

            key, transfer_key = jrandom.split(key)
            marin_model = initialize_model_from_marin(
                model_config,
                node_vocab,
                value_vocab,
                transfer_config,
                transfer_key,
            )

            marin_history = train_model(
                marin_model, programs, node_vocab, value_vocab, config, "MARIN"
            )

        except ImportError as e:
            logger.warning(f"Could not initialize from Marin (missing transformers?): {e}")
            logger.warning("Install with: pip install transformers")
            marin_history = None
        except Exception as e:
            logger.error(f"Marin transfer failed: {e}")
            marin_history = None
    else:
        marin_history = None

    # ============================================================
    # Compare results
    # ============================================================
    logger.info("=" * 70)
    logger.info("EXPERIMENT RESULTS")
    logger.info("=" * 70)

    def summarize(name: str, history: list[float]):
        if not history:
            return
        initial = sum(history[:10]) / min(10, len(history))
        final = sum(history[-10:]) / min(10, len(history))
        best = min(history)
        logger.info(f"{name}:")
        logger.info(f"  Initial loss (avg first 10): {initial:.4f}")
        logger.info(f"  Final loss (avg last 10):    {final:.4f}")
        logger.info(f"  Best loss:                   {best:.4f}")
        logger.info(f"  Improvement:                 {initial - final:.4f} ({(initial - final) / initial * 100:.1f}%)")

    summarize("RANDOM init", random_history)
    if marin_history:
        summarize("MARIN init", marin_history)

        # Compare
        logger.info("-" * 70)
        random_final = sum(random_history[-10:]) / 10
        marin_final = sum(marin_history[-10:]) / 10

        if marin_final < random_final:
            improvement = (random_final - marin_final) / random_final * 100
            logger.info(f"MARIN transfer IMPROVED final loss by {improvement:.1f}%")
        else:
            degradation = (marin_final - random_final) / random_final * 100
            logger.info(f"MARIN transfer DEGRADED final loss by {degradation:.1f}%")

    return random_history, marin_history


def main():
    parser = argparse.ArgumentParser(description="Compare Marin vs Random initialization")
    parser.add_argument("--num_steps", type=int, default=500, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Transformer layers")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--log_every", type=int, default=50, help="Log every N steps")
    parser.add_argument("--projection_method", type=str, default="random", help="Projection method")
    parser.add_argument("--use_stackedu", action="store_true", help="Use Stack-Edu data")
    parser.add_argument("--no_marin", action="store_true", help="Skip Marin transfer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = TransferExperimentConfig(
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        log_every=args.log_every,
        projection_method=args.projection_method,
        use_stackedu=args.use_stackedu,
        use_marin_transfer=not args.no_marin,
        seed=args.seed,
    )

    try:
        run_experiment(config)
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 1


if __name__ == "__main__":
    sys.exit(main())
