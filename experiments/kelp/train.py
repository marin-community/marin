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
Training loop for tree diffusion model.

This module provides a simple training loop for prototyping the tree
diffusion model on the toy dataset.
"""

import logging
import random
from dataclasses import dataclass
from collections.abc import Iterator

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
from jaxtyping import PRNGKeyArray

import haliax as hax
from haliax import NamedArray

from experiments.kelp.edit_path import batch_training_examples, create_training_example
from experiments.kelp.python_grammar import PythonNodeVocab, PythonValueVocab
from experiments.kelp.toy_dataset import get_toy_programs
from experiments.kelp.tree_diffusion import TreeDiffusionConfig, TreeDiffusionModel

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Configuration for training."""

    # Model config
    hidden_dim: int = 256
    num_layers: int = 4  # Smaller for prototyping
    num_heads: int = 8
    mlp_dim: int = 512

    # Training
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_steps: int = 10000
    log_every: int = 100
    eval_every: int = 1000

    # Data
    max_nodes: int = 256
    max_children: int = 16
    max_value_len: int = 32
    s_max: int = 5  # Max corruption steps

    # Misc
    seed: int = 42


def create_model_config(
    train_config: TrainConfig, node_vocab: PythonNodeVocab, value_vocab: PythonValueVocab
) -> TreeDiffusionConfig:
    """Create model config from training config."""
    return TreeDiffusionConfig(
        hidden_dim=train_config.hidden_dim,
        num_layers=train_config.num_layers,
        num_heads=train_config.num_heads,
        mlp_dim=train_config.mlp_dim,
        max_nodes=train_config.max_nodes,
        max_children=train_config.max_children,
        max_value_len=train_config.max_value_len,
        node_vocab_size=node_vocab.vocab_size,
        value_vocab_size=value_vocab.vocab_size,
        s_max=train_config.s_max,
    )


def create_data_iterator(
    programs: list[str],
    node_vocab: PythonNodeVocab,
    value_vocab: PythonValueVocab,
    config: TrainConfig,
    key: PRNGKeyArray,
) -> Iterator[dict[str, np.ndarray]]:
    """Create an iterator that yields training batches.

    For each batch:
    1. Sample batch_size programs
    2. For each program, sample num_corruption_steps ~ Uniform[1, s_max]
    3. Create training examples
    4. Batch and yield
    """
    while True:
        examples = []
        for _ in range(config.batch_size):
            # Sample a program
            code = random.choice(programs)

            # Sample corruption steps
            num_steps = random.randint(1, config.s_max)

            # Create training example
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
                examples.append(example)

        if len(examples) < config.batch_size // 2:
            # Not enough valid examples, skip this batch
            continue

        # Batch examples
        batch = batch_training_examples(examples)
        yield batch


def numpy_to_haliax(batch: dict[str, np.ndarray], config: TreeDiffusionConfig) -> dict[str, NamedArray]:
    """Convert numpy batch to Haliax NamedArrays."""
    Batch = hax.Axis("batch", batch["corrupted_node_types"].shape[0])

    return {
        "corrupted_node_types": hax.named(jnp.array(batch["corrupted_node_types"]), (Batch, config.Nodes)),
        "corrupted_node_values": hax.named(
            jnp.array(batch["corrupted_node_values"]), (Batch, config.Nodes, config.ValueLen)
        ),
        "corrupted_depth": hax.named(jnp.array(batch["corrupted_depth"]), (Batch, config.Nodes)),
        "corrupted_node_mask": hax.named(jnp.array(batch["corrupted_node_mask"]), (Batch, config.Nodes)),
        "edit_location": hax.named(jnp.array(batch["edit_location"]), (Batch,)),
        "replacement_type": hax.named(jnp.array(batch["replacement_type"]), (Batch,)),
        "replacement_value": hax.named(jnp.array(batch["replacement_value"]), (Batch, config.ValueLen)),
    }


@eqx.filter_jit
def train_step(
    model: TreeDiffusionModel,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    batch: dict[str, NamedArray],
    key: PRNGKeyArray,
) -> tuple[TreeDiffusionModel, optax.OptState, dict[str, jnp.ndarray]]:
    """Single training step."""

    def loss_fn(model):
        # Vmap over batch dimension
        Batch = list(batch.values())[0].axes[0]

        def single_forward(node_types, node_values, depth, mask, edit_loc, repl_type, repl_value, k):
            location_logits, type_logits, value_logits = model(node_types, node_values, depth, mask=mask, key=k)

            # Location loss
            loc_loss = -hax.nn.log_softmax(location_logits, axis=model.config.Nodes)[model.config.Nodes, edit_loc]

            # Type loss
            type_loss = -hax.nn.log_softmax(type_logits, axis=model.config.NodeVocab)[model.config.NodeVocab, repl_type]

            # Value loss (average over positions)
            value_log_probs = hax.nn.log_softmax(value_logits, axis=model.config.ValueVocab)
            # Index into value_log_probs for each position
            value_losses = []
            for pos in range(model.config.max_value_len):
                target_id = repl_value[model.config.ValueLen, pos]
                value_losses.append(-value_log_probs[model.config.ValueLen, pos][model.config.ValueVocab, target_id])
            value_loss = sum(value_losses) / len(value_losses)

            return loc_loss + type_loss + value_loss

        # Generate keys for each example in batch
        keys = jrandom.split(key, batch["corrupted_node_types"].axis_size(Batch))
        keys = hax.named(keys, (Batch, hax.Axis("key", 2)))

        losses = hax.vmap(single_forward, Batch)(
            batch["corrupted_node_types"],
            batch["corrupted_node_values"],
            batch["corrupted_depth"],
            batch["corrupted_node_mask"],
            batch["edit_location"],
            batch["replacement_type"],
            batch["replacement_value"],
            keys,
        )

        return hax.mean(losses).scalar()

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)

    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    metrics = {"loss": loss}
    return model, opt_state, metrics


def train(
    config: TrainConfig | None = None,
    output_dir: str | None = None,
) -> TreeDiffusionModel:
    """Train tree diffusion model on toy dataset.

    Args:
        config: Training configuration
        output_dir: Directory to save checkpoints (optional)

    Returns:
        Trained model
    """
    if config is None:
        config = TrainConfig()

    # Set random seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    key = jrandom.PRNGKey(config.seed)

    # Create vocabularies
    node_vocab = PythonNodeVocab()
    value_vocab = PythonValueVocab()

    # Create model config and model
    model_config = create_model_config(config, node_vocab, value_vocab)
    key, model_key = jrandom.split(key)
    model = TreeDiffusionModel.init(model_config, key=model_key)

    # Create optimizer
    optimizer = optax.adamw(config.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Load toy programs
    programs = get_toy_programs()
    logger.info(f"Loaded {len(programs)} toy programs")

    # Create data iterator
    key, data_key = jrandom.split(key)
    data_iter = create_data_iterator(programs, node_vocab, value_vocab, config, data_key)

    # Training loop
    logger.info("Starting training...")
    for step in range(config.num_steps):
        # Get batch
        batch_np = next(data_iter)
        batch = numpy_to_haliax(batch_np, model_config)

        # Train step
        key, step_key = jrandom.split(key)
        model, opt_state, metrics = train_step(model, opt_state, optimizer, batch, step_key)

        # Log
        if step % config.log_every == 0:
            loss = float(metrics["loss"])
            logger.info(f"Step {step}: loss = {loss:.4f}")

        # Evaluate
        if step % config.eval_every == 0 and step > 0:
            logger.info(f"Step {step}: Running evaluation...")
            # Simple evaluation: check if model generates valid Python
            key, eval_key = jrandom.split(key)
            # TODO: Add proper evaluation

    logger.info("Training complete!")
    return model


def main():
    """Main entry point for training."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    config = TrainConfig(
        num_steps=1000,  # Quick test
        log_every=50,
        eval_every=500,
        batch_size=8,  # Small for testing
        num_layers=2,  # Small model for testing
    )

    model = train(config)
    logger.info("Done!")


if __name__ == "__main__":
    main()
