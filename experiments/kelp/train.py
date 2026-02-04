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
Unified training script for Kelp tree diffusion.

Supports:
- Toy dataset or Stack-Edu Python
- Conditional or unconditional training
- Random or Marin transfer initialization

Usage:
    # Basic training on toy dataset:
    uv run python experiments/kelp/train.py --num_steps=500

    # Conditional training (learns from docstrings):
    uv run python experiments/kelp/train.py --conditioning --num_steps=500

    # Train on Stack-Edu data:
    uv run python experiments/kelp/train.py --dataset=stackedu --num_steps=2000

    # Transfer learning from Marin 8B:
    uv run python experiments/kelp/train.py --transfer=marin --num_steps=1000
"""

import argparse
import logging
import os
import pickle
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

from experiments.kelp.conditioning import (
    CONDITION_VOCAB_SIZE,
    analyze_conditioning_coverage,
    create_condition_mask,
    create_conditional_training_example,
    extract_function_condition,
)
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
class TrainConfig:
    """Unified configuration for Kelp training."""

    # Model architecture
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    mlp_dim: int = 512

    # Training
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_steps: int = 1000
    log_every: int = 50
    save_every: int = 1000
    warmup_steps: int = 100

    # Data dimensions
    max_nodes: int = 128
    max_children: int = 16
    max_value_len: int = 32
    max_cond_len: int = 128
    s_max: int = 5  # Max corruption steps

    # Dataset: "toy" or "stackedu"
    dataset: str = "toy"
    max_files: int = 10  # For Stack-Edu
    max_functions: int = 10000  # For Stack-Edu
    use_cache: bool = True

    # Conditioning
    conditioning: bool = False
    require_docstring: bool = False

    # Transfer learning: "random" or "marin"
    transfer: str = "random"

    # Output
    output_dir: str = "outputs/kelp"
    seed: int = 42


# ============================================================================
# Data Loading
# ============================================================================


def load_programs(config: TrainConfig) -> list[str]:
    """Load training programs based on config."""
    if config.dataset == "stackedu":
        try:
            from experiments.kelp.data_pipeline import load_stackv2_edu_python

            logger.info("Loading Stack-Edu Python dataset...")
            cache_path = None
            if config.use_cache:
                cache_dir = Path(config.output_dir) / "cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path = str(cache_dir / f"stackedu_{config.max_files}f_{config.max_functions}n.json")

            dataset = load_stackv2_edu_python(
                max_files=config.max_files,
                max_functions=config.max_functions,
                cache_path=cache_path,
            )
            dataset.load()
            programs = dataset.get_programs()

            if programs:
                logger.info(f"Loaded {len(programs)} programs from Stack-Edu")
                return programs
            else:
                logger.warning("Stack-Edu returned no programs, falling back to toy")
        except Exception as e:
            logger.warning(f"Failed to load Stack-Edu ({e}), falling back to toy")

    # Default: toy dataset
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


# ============================================================================
# Batch Creation
# ============================================================================


def create_batch(
    programs: list[str],
    node_vocab: PythonNodeVocab,
    value_vocab: PythonValueVocab,
    config: TrainConfig,
) -> dict[str, jnp.ndarray] | None:
    """Create a batch of training examples."""
    examples = []
    attempts = 0
    max_attempts = config.batch_size * 10

    while len(examples) < config.batch_size and attempts < max_attempts:
        code = random.choice(programs)
        num_steps = random.randint(1, config.s_max)

        if config.conditioning:
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
                if config.require_docstring and not example.has_docstring:
                    attempts += 1
                    continue
                examples.append({
                    "node_types": example.corrupted_tensors.node_types,
                    "node_values": example.corrupted_tensors.node_values,
                    "depth": example.corrupted_tensors.depth,
                    "mask": example.corrupted_tensors.node_mask,
                    "edit_location": example.edit_location,
                    "replacement_type": example.replacement_type,
                    "replacement_value": example.replacement_value,
                    "condition_tokens": example.condition_tokens,
                })
        else:
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

    # Pad with duplicates if needed
    if len(examples) < config.batch_size and examples:
        while len(examples) < config.batch_size:
            examples.append(examples[0])

    if not examples:
        return None

    batch = {
        "node_types": jnp.stack([e["node_types"] for e in examples]),
        "node_values": jnp.stack([e["node_values"] for e in examples]),
        "depth": jnp.stack([e["depth"] for e in examples]),
        "mask": jnp.stack([e["mask"] for e in examples]),
        "edit_location": jnp.array([e["edit_location"] for e in examples]),
        "replacement_type": jnp.array([e["replacement_type"] for e in examples]),
        "replacement_value": jnp.stack([e["replacement_value"] for e in examples]),
    }

    if config.conditioning:
        batch["condition_tokens"] = jnp.stack([e["condition_tokens"] for e in examples])

    return batch


# ============================================================================
# Loss Computation
# ============================================================================


def compute_loss_single(
    model: TreeDiffusionModel,
    node_types: jnp.ndarray,
    node_values: jnp.ndarray,
    depth: jnp.ndarray,
    mask: jnp.ndarray,
    edit_location: jnp.ndarray,
    replacement_type: jnp.ndarray,
    replacement_value: jnp.ndarray,
    condition_tokens: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute loss for a single example."""
    cfg = model.config

    # Convert to NamedArrays
    node_types_na = hax.named(node_types, (cfg.Nodes,))
    node_values_na = hax.named(node_values, (cfg.Nodes, cfg.ValueLen))
    depth_na = hax.named(depth, (cfg.Nodes,))
    mask_na = hax.named(mask, (cfg.Nodes,))

    # Handle conditioning
    cond_tokens_na = None
    cond_mask_na = None
    if condition_tokens is not None and cfg.use_conditioning:
        cond_tokens_na = hax.named(condition_tokens, (cfg.CondLen,))
        cond_mask = create_condition_mask(condition_tokens)
        cond_mask_na = hax.named(cond_mask, (cfg.CondLen,))

    # Forward pass
    location_logits, type_logits, value_logits = model(
        node_types_na,
        node_values_na,
        depth_na,
        mask=mask_na,
        condition_tokens=cond_tokens_na,
        condition_mask=cond_mask_na,
        key=None,
    )

    # Location loss
    loc_log_probs = hax.nn.log_softmax(location_logits, axis=cfg.Nodes).array
    loc_one_hot = jax.nn.one_hot(edit_location, cfg.max_nodes)
    loc_loss = -jnp.sum(loc_log_probs * loc_one_hot)

    # Type loss
    type_log_probs = hax.nn.log_softmax(type_logits, axis=cfg.NodeVocab).array
    type_one_hot = jax.nn.one_hot(replacement_type, cfg.node_vocab_size)
    type_loss = -jnp.sum(type_log_probs * type_one_hot)

    # Value loss
    value_log_probs = hax.nn.log_softmax(value_logits, axis=cfg.ValueVocab).array

    def get_log_prob(log_probs_row, target_idx):
        return log_probs_row[target_idx]

    value_log_probs_selected = jax.vmap(get_log_prob)(value_log_probs, replacement_value)
    value_loss = -jnp.mean(value_log_probs_selected)

    return loc_loss + type_loss + value_loss


def compute_batch_loss(
    model: TreeDiffusionModel,
    batch: dict[str, jnp.ndarray],
    use_conditioning: bool,
) -> jnp.ndarray:
    """Compute average loss over a batch."""
    if use_conditioning:

        def single_loss(node_types, node_values, depth, mask, edit_loc, repl_type, repl_value, cond_tokens):
            return compute_loss_single(
                model, node_types, node_values, depth, mask, edit_loc, repl_type, repl_value, cond_tokens
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
    else:

        def single_loss(node_types, node_values, depth, mask, edit_loc, repl_type, repl_value):
            return compute_loss_single(
                model, node_types, node_values, depth, mask, edit_loc, repl_type, repl_value, None
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


# ============================================================================
# Checkpointing
# ============================================================================


def save_checkpoint(model: TreeDiffusionModel, path: str, step: int):
    """Save model checkpoint."""
    arrays, structure = eqx.partition(model, eqx.is_array)
    checkpoint = {
        "step": step,
        "arrays": jax.tree.map(lambda x: np.array(x), arrays),
        "structure": structure,
        "config": model.config,
    }
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(path: str) -> tuple[TreeDiffusionModel, int]:
    """Load model checkpoint."""
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)

    step = checkpoint["step"]

    if "arrays" in checkpoint and "structure" in checkpoint:
        # New format
        arrays = jax.tree.map(lambda x: jnp.array(x), checkpoint["arrays"])
        model = eqx.combine(arrays, checkpoint["structure"])
    else:
        # Old format fallback
        config = checkpoint["config"]
        key = jrandom.PRNGKey(0)
        model = TreeDiffusionModel.init(config, key=key)
        saved_arrays = checkpoint["model"]

        def replace_arrays(model_leaf, saved_leaf):
            if eqx.is_array(model_leaf):
                return jnp.array(saved_leaf)
            return model_leaf

        model = jax.tree.map(replace_arrays, model, saved_arrays, is_leaf=eqx.is_array)

    logger.info(f"Loaded checkpoint from {path} (step {step})")
    return model, step


# ============================================================================
# Training
# ============================================================================


def create_model(config: TrainConfig, node_vocab: PythonNodeVocab, value_vocab: PythonValueVocab, key: jax.Array):
    """Create model with optional transfer initialization."""
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
        use_conditioning=config.conditioning,
        condition_vocab_size=CONDITION_VOCAB_SIZE if config.conditioning else 256,
        max_condition_len=config.max_cond_len,
    )

    if config.transfer == "marin":
        try:
            from experiments.kelp.ar_transfer import TransferConfig, initialize_model_from_marin

            logger.info("Initializing from Marin 8B transfer...")
            transfer_config = TransferConfig(target_hidden_dim=config.hidden_dim)
            model = initialize_model_from_marin(
                model_config,
                node_vocab,
                value_vocab,
                transfer_config,
                key,
            )
            logger.info("Transfer initialization complete")
            return model
        except Exception as e:
            logger.warning(f"Transfer initialization failed ({e}), using random init")

    return TreeDiffusionModel.init(model_config, key=key)


def train(config: TrainConfig) -> tuple[TreeDiffusionModel, list[float]]:
    """Main training function."""
    logger.info("=" * 70)
    logger.info("Kelp Tree Diffusion Training")
    logger.info("=" * 70)

    # Log configuration
    logger.info(f"JAX backend: {jax.default_backend()}")
    logger.info(f"Dataset: {config.dataset}")
    logger.info(f"Conditioning: {config.conditioning}")
    logger.info(f"Transfer: {config.transfer}")

    os.makedirs(config.output_dir, exist_ok=True)

    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    key = jrandom.PRNGKey(config.seed)

    # Create vocabularies
    node_vocab = PythonNodeVocab()
    value_vocab = PythonValueVocab()
    logger.info(f"Node vocab: {node_vocab.vocab_size}, Value vocab: {value_vocab.vocab_size}")

    # Load programs
    programs = load_programs(config)

    if config.conditioning:
        stats = analyze_conditioning_coverage(programs)
        logger.info(f"Docstring coverage: {stats['with_docstring']}/{stats['total_programs']} ({stats['docstring_rate']:.1%})")

        if config.require_docstring:
            programs = filter_programs_with_docstrings(programs)
            logger.info(f"Filtered to {len(programs)} programs with docstrings")

    if not programs:
        logger.error("No valid programs found!")
        return None, []

    # Create model
    key, model_key = jrandom.split(key)
    model = create_model(config, node_vocab, value_vocab, model_key)

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

    # JIT compile train step
    @eqx.filter_jit
    def train_step(model, opt_state, batch):
        def loss_fn(m):
            return compute_batch_loss(m, batch, config.conditioning)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, new_opt_state = optimizer.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss

    # Training loop
    logger.info("-" * 70)
    logger.info(f"Training for {config.num_steps} steps on {len(programs)} programs")
    logger.info("-" * 70)

    loss_history = []
    start_time = time.time()

    for step in range(config.num_steps):
        batch = create_batch(programs, node_vocab, value_vocab, config)
        if batch is None:
            logger.warning(f"Step {step}: Could not create batch")
            continue

        model, opt_state, loss = train_step(model, opt_state, batch)
        loss_val = float(loss)
        loss_history.append(loss_val)

        if step % config.log_every == 0 or step == config.num_steps - 1:
            elapsed = time.time() - start_time
            speed = (step + 1) / elapsed if elapsed > 0 else 0
            window = min(50, len(loss_history))
            avg_loss = sum(loss_history[-window:]) / window
            logger.info(f"Step {step:5d}/{config.num_steps} | Loss: {loss_val:7.4f} | Avg: {avg_loss:7.4f} | {speed:.1f} steps/s")

        if config.save_every > 0 and step > 0 and step % config.save_every == 0:
            save_checkpoint(model, os.path.join(config.output_dir, f"checkpoint_{step}.pkl"), step)

    # Final checkpoint
    total_time = time.time() - start_time
    logger.info("-" * 70)
    logger.info(f"Training complete in {total_time:.1f}s")
    logger.info(f"Final loss: {loss_history[-1]:.4f}")

    save_checkpoint(model, os.path.join(config.output_dir, "checkpoint_final.pkl"), config.num_steps)

    return model, loss_history


def main():
    parser = argparse.ArgumentParser(description="Train Kelp tree diffusion model")

    # Dataset
    parser.add_argument("--dataset", type=str, default="toy", choices=["toy", "stackedu"])
    parser.add_argument("--max_files", type=int, default=10)
    parser.add_argument("--max_functions", type=int, default=10000)
    parser.add_argument("--no_cache", action="store_true")

    # Model
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)

    # Training
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=1000)

    # Conditioning
    parser.add_argument("--conditioning", action="store_true")
    parser.add_argument("--require_docstring", action="store_true")

    # Transfer
    parser.add_argument("--transfer", type=str, default="random", choices=["random", "marin"])

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/kelp")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config = TrainConfig(
        dataset=args.dataset,
        max_files=args.max_files,
        max_functions=args.max_functions,
        use_cache=not args.no_cache,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        log_every=args.log_every,
        save_every=args.save_every,
        conditioning=args.conditioning,
        require_docstring=args.require_docstring,
        transfer=args.transfer,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    try:
        model, _ = train(config)
        return 0 if model is not None else 1
    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 1


if __name__ == "__main__":
    sys.exit(main())
