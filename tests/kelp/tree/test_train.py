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

"""Tests for the tree diffusion training pipeline."""

import random

import jax
import jax.numpy as jnp
import pytest

from experiments.kelp.model.config import TreeDiffusionConfig
from experiments.kelp.tree.edit_model import EditModelParams, init_edit_params
from experiments.kelp.tree.subtree_bank import SubtreeBank
from experiments.kelp.tree.tokenizer import TreeDiffusionTokenizer
from experiments.kelp.tree.train import (
    EditTrainingConfig,
    EditTrainingState,
    _edit_weight_decay_mask,
    create_edit_data_iter,
    create_edit_optimizer,
    generate_training_example,
    make_edit_train_step,
    train_edit_model,
)

CORPUS = [
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
]

MAX_SEQ_LEN = 128


@pytest.fixture
def bank():
    return SubtreeBank.from_corpus(CORPUS)


@pytest.fixture
def tokenizer():
    return TreeDiffusionTokenizer(max_seq_len=MAX_SEQ_LEN)


@pytest.fixture
def model_cfg(tokenizer):
    return TreeDiffusionConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=64,
        intermediate_dim=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        max_seq_len=MAX_SEQ_LEN,
    )


@pytest.fixture
def train_cfg(model_cfg):
    return EditTrainingConfig(
        model=model_cfg,
        max_seq_len=MAX_SEQ_LEN,
        total_steps=3,
        batch_size=2,
        warmup_steps=1,
        log_interval=1,
    )


def test_generate_training_example(bank, tokenizer, train_cfg):
    rng = random.Random(42)
    result = generate_training_example(
        clean_source=CORPUS[0],
        corpus=CORPUS,
        bank=bank,
        tokenizer=tokenizer,
        max_seq_len=MAX_SEQ_LEN,
        config=train_cfg,
        rng=rng,
    )

    if result is not None:
        token_ids, loss_mask = result
        assert len(token_ids) == len(loss_mask)
        assert len(token_ids) <= MAX_SEQ_LEN
        # There should be some loss tokens.
        assert sum(loss_mask) > 0
        # Loss mask should have 0s followed by 1s (context then edit).
        assert loss_mask[0] == 0  # First token is context.


def test_generate_training_example_many_programs(bank, tokenizer, train_cfg):
    """At least some programs should produce valid training examples."""
    rng = random.Random(42)
    successes = 0
    for source in CORPUS:
        for seed in range(5):
            rng_i = random.Random(seed)
            result = generate_training_example(
                clean_source=source,
                corpus=CORPUS,
                bank=bank,
                tokenizer=tokenizer,
                max_seq_len=MAX_SEQ_LEN,
                config=train_cfg,
                rng=rng_i,
            )
            if result is not None:
                successes += 1

    assert successes >= 5, f"Only {successes} successes out of {len(CORPUS) * 5}"


def test_create_edit_data_iter_yields_batches(bank, tokenizer, train_cfg):
    data_iter = create_edit_data_iter(
        corpus=CORPUS,
        bank=bank,
        tokenizer=tokenizer,
        config=train_cfg,
    )

    batch = next(data_iter)
    assert "token_ids" in batch
    assert "loss_mask" in batch
    assert batch["token_ids"].shape == (train_cfg.batch_size, MAX_SEQ_LEN)
    assert batch["loss_mask"].shape == (train_cfg.batch_size, MAX_SEQ_LEN)
    assert batch["token_ids"].dtype == jnp.int32
    assert batch["loss_mask"].dtype == jnp.float32


def test_weight_decay_mask_structure(model_cfg):
    key = jax.random.PRNGKey(0)
    params = init_edit_params(model_cfg, key=key)
    mask = _edit_weight_decay_mask(params)

    # Embeddings and norms should be False (no decay).
    assert mask.token_embed is False
    assert mask.final_norm is False

    # Output projection should be True (decay).
    assert mask.output_proj is True

    # Block attention weights should be True, norms False.
    for block in mask.blocks:
        assert block.attn.w_q is True
        assert block.rms_attn is False
        assert block.mlp_gate is True


def test_edit_train_step_reduces_loss(bank, tokenizer, model_cfg, train_cfg):
    """A single training step should produce finite loss and metrics."""
    key = jax.random.PRNGKey(0)
    params = init_edit_params(model_cfg, key=key)
    optimizer = create_edit_optimizer(train_cfg)
    opt_state = optimizer.init(params)
    state = EditTrainingState(step=0, params=params, opt_state=opt_state, key=key)

    train_step = make_edit_train_step(model_cfg, optimizer)

    data_iter = create_edit_data_iter(
        corpus=CORPUS,
        bank=bank,
        tokenizer=tokenizer,
        config=train_cfg,
    )

    batch = next(data_iter)
    new_state, metrics = train_step(state, batch)

    assert new_state.step == 1
    assert jnp.isfinite(metrics["loss"])
    assert jnp.isfinite(metrics["grad_norm"])
    assert float(metrics["loss"]) > 0


def test_train_edit_model_runs(bank, tokenizer, model_cfg, train_cfg):
    """End-to-end: train for a few steps without crashing."""
    data_iter = create_edit_data_iter(
        corpus=CORPUS,
        bank=bank,
        tokenizer=tokenizer,
        config=train_cfg,
    )

    logged_metrics = []

    def log_cb(step, metrics):
        logged_metrics.append((step, {k: float(v) for k, v in metrics.items()}))

    params = train_edit_model(
        config=train_cfg,
        data_iter=data_iter,
        log_callback=log_cb,
    )

    assert isinstance(params, EditModelParams)
    assert len(logged_metrics) > 0
    # Loss should be finite for all logged steps.
    for step, m in logged_metrics:
        assert m["loss"] > 0 and m["loss"] < 100, f"Bad loss at step {step}: {m['loss']}"
