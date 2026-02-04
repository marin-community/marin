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
Training script for conditional tree diffusion.

Trains a tree diffusion model conditioned on function signatures and docstrings.
The model learns to generate function bodies given the signature/docstring.

Usage:
    # Train on toy dataset:
    uv run python experiments/kelp/train_conditional.py --num_steps=500

    # Train on Stack-Edu (requires GCS auth):
    uv run python experiments/kelp/train_conditional.py --use_stackedu --num_steps=2000
"""

import argparse
import logging
import os
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

from experiments.kelp.conditioning import (
    CONDITION_VOCAB_SIZE,
    ConditionalTrainingExample,
    analyze_conditioning_coverage,
    create_condition_mask,
    create_conditional_training_example,
    extract_function_condition,
    tokenize_condition,
)
from experiments.kelp.python_grammar import PythonNodeVocab, PythonValueVocab
from experiments.kelp.toy_dataset import get_toy_programs
from experiments.kelp.train_stackedu import save_checkpoint
from experiments.kelp.tree_diffusion import TreeDiffusionConfig, TreeDiffusionModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class ConditionalTrainConfig:
    """Configuration for conditional training."""

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
    max_cond_len: int = 128
    s_max: int = 5

    # Dataset
    use_stackedu: bool = False
    max_files: int = 5
    max_functions: int = 5000
    require_docstring: bool = False  # Only use examples with docstrings

    # Misc
    seed: int = 42
    output_dir: str = "outputs/kelp_conditional"
    save_every: int = 1000


def load_programs(config: ConditionalTrainConfig) -> list[str]:
    """Load training programs."""
    if config.use_stackedu:
        try:
            from experiments.kelp.datasets import load_stackv2_edu_python

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


def filter_programs_with_docstrings(programs: list[str]) -> list[str]:
    """Filter to only programs with docstrings."""
    filtered = []
    for code in programs:
        cond = extract_function_condition(code)
        if cond is not None and cond.has_docstring:
            filtered.append(code)
    return filtered


def create_conditional_batch(
    programs: list[str],
    node_vocab: PythonNodeVocab,
    value_vocab: PythonValueVocab,
    config: ConditionalTrainConfig,
) -> dict[str, jnp.ndarray] | None:
    """Create a batch of conditional training examples."""
    examples = []
    attempts = 0
    max_attempts = config.batch_size * 10

    while len(examples) < config.batch_size and attempts < max_attempts:
        code = random.choice(programs)
        num_steps = random.randint(1, config.s_max)

        example = create_conditional_training_example(
            code,
            num_steps,
            node_vocab,
            value_vocab,
            max_nodes=config.max_nodes,
            max_children=config.max_children,
            max_value_len=config.max_value_len,
            max_cond_len=config.max_cond_len,
        )

        if example is not None:
            # Optionally require docstring
            if config.require_docstring and not example.has_docstring:
                attempts += 1
                continue
            examples.append(example)

        attempts += 1

    if len(examples) < config.batch_size:
        # Pad with duplicates if needed
        while len(examples) < config.batch_size and examples:
            examples.append(examples[0])

    if not examples:
        return None

    return {
        "node_types": jnp.stack([e.corrupted_tensors.node_types for e in examples]),
        "node_values": jnp.stack([e.corrupted_tensors.node_values for e in examples]),
        "depth": jnp.stack([e.corrupted_tensors.depth for e in examples]),
        "mask": jnp.stack([e.corrupted_tensors.node_mask for e in examples]),
        "edit_location": jnp.array([e.edit_location for e in examples]),
        "replacement_type": jnp.array([e.replacement_type for e in examples]),
        "replacement_value": jnp.stack([e.replacement_value for e in examples]),
        "condition_tokens": jnp.stack([e.condition_tokens for e in examples]),
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
    condition_tokens: jnp.ndarray,
) -> jnp.ndarray:
    """Compute loss for a single conditional example."""
    config = model.config

    # Convert to NamedArrays
    node_types_na = hax.named(node_types, (config.Nodes,))
    node_values_na = hax.named(node_values, (config.Nodes, config.ValueLen))
    depth_na = hax.named(depth, (config.Nodes,))
    mask_na = hax.named(mask, (config.Nodes,))

    # Conditioning
    condition_tokens_na = hax.named(condition_tokens, (config.CondLen,))
    condition_mask = create_condition_mask(condition_tokens)
    condition_mask_na = hax.named(condition_mask, (config.CondLen,))

    # Forward pass with conditioning
    location_logits, type_logits, value_logits = model(
        node_types_na,
        node_values_na,
        depth_na,
        mask=mask_na,
        condition_tokens=condition_tokens_na,
        condition_mask=condition_mask_na,
        key=None,
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
    """Compute average loss over a conditional batch."""

    def single_loss(
        node_types, node_values, depth, mask, edit_loc, repl_type, repl_value, cond_tokens
    ):
        return compute_loss_single(
            model,
            node_types,
            node_values,
            depth,
            mask,
            edit_loc,
            repl_type,
            repl_value,
            cond_tokens,
        )

    losses = jax.vmap(single_loss)(
        batch["node_types"],
        batch["node_values"],
        batch["depth"],
        batch["mask"],
        batch["edit_location"],
        batch["replacement_type"],
        batch["replacement_value"],
        batch["condition_tokens"],
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


def train(config: ConditionalTrainConfig) -> tuple[TreeDiffusionModel, list[float]]:
    """Train conditional tree diffusion model."""
    logger.info("=" * 70)
    logger.info("Kelp Conditional Tree Diffusion Training")
    logger.info("=" * 70)

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
    logger.info(f"Node vocab: {node_vocab.vocab_size}, Value vocab: {value_vocab.vocab_size}")
    logger.info(f"Condition vocab: {CONDITION_VOCAB_SIZE}")

    # Load programs
    programs = load_programs(config)

    # Analyze conditioning coverage
    stats = analyze_conditioning_coverage(programs)
    logger.info(f"Conditioning coverage:")
    logger.info(f"  With docstring: {stats['with_docstring']}/{stats['total_programs']} ({stats['docstring_rate']:.1%})")
    logger.info(f"  With type hints: {stats['with_type_hints']}/{stats['total_programs']} ({stats['type_hint_rate']:.1%})")

    # Optionally filter to only programs with docstrings
    if config.require_docstring:
        programs = filter_programs_with_docstrings(programs)
        logger.info(f"Filtered to {len(programs)} programs with docstrings")

    if len(programs) == 0:
        logger.error("No valid programs found!")
        return None, []

    # Create model with conditioning enabled
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
        # Conditioning
        use_conditioning=True,
        condition_vocab_size=CONDITION_VOCAB_SIZE,
        max_condition_len=config.max_cond_len,
    )

    key, model_key = jrandom.split(key)
    model = TreeDiffusionModel.init(model_config, key=model_key)

    # Count parameters
    num_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    logger.info(f"Model parameters: {num_params:,}")

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

    # Training loop
    logger.info("-" * 70)
    logger.info(f"Training for {config.num_steps} steps on {len(programs)} programs")
    logger.info(f"Batch size: {config.batch_size}, Conditioning: ENABLED")
    logger.info("-" * 70)

    loss_history = []
    start_time = time.time()

    for step in range(config.num_steps):
        # Create batch
        batch = create_conditional_batch(programs, node_vocab, value_vocab, config)
        if batch is None:
            logger.warning(f"Step {step}: Could not create batch, skipping")
            continue

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
    logger.info("-" * 70)
    logger.info(f"Training complete in {total_time:.1f}s")
    logger.info(f"Final loss: {loss_history[-1]:.4f}")
    logger.info(f"Best loss: {min(loss_history):.4f}")

    # Save final checkpoint
    final_path = os.path.join(config.output_dir, "checkpoint_final.pkl")
    save_checkpoint(model, final_path, config.num_steps)

    # Check if loss decreased
    if len(loss_history) >= 20:
        initial_avg = sum(loss_history[:10]) / 10
        final_avg = sum(loss_history[-10:]) / 10
        if final_avg < initial_avg:
            logger.info(f"Loss decreased from {initial_avg:.4f} to {final_avg:.4f}")
        else:
            logger.warning(f"Loss did NOT decrease: {initial_avg:.4f} -> {final_avg:.4f}")

    return model, loss_history


def main():
    parser = argparse.ArgumentParser(description="Train conditional tree diffusion")
    parser.add_argument("--num_steps", type=int, default=500, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Transformer layers")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--log_every", type=int, default=50, help="Log every N steps")
    parser.add_argument("--use_stackedu", action="store_true", help="Use Stack-Edu data")
    parser.add_argument("--require_docstring", action="store_true", help="Only use examples with docstrings")
    parser.add_argument("--output_dir", type=str, default="outputs/kelp", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = ConditionalTrainConfig(
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        log_every=args.log_every,
        use_stackedu=args.use_stackedu,
        require_docstring=args.require_docstring,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    try:
        model, loss_history = train(config)
        if model is not None:
            logger.info("Done!")
            return 0
        else:
            return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 1


if __name__ == "__main__":
    sys.exit(main())
