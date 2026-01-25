# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import jax.numpy as jnp
import jax.random as jrandom
import pytest
from transformers import AutoTokenizer

import haliax as hax

from levanter.data.text import DpoExample, PreferenceChatProcessor, PreferencePairDataset
from levanter.main.train_simpo import _average_logp, simpo_loss_from_logps
from levanter.metrics import Metric
from levanter.models.gpt2 import Gpt2Config
from levanter.models.lm_model import LmExample
from levanter.store.cache import SerialCacheWriter
from levanter.utils.tree_utils import inference_mode

MODEL_NAME = "stanford-crfm/marin-tokenizer"


@pytest.fixture(scope="module")
def tokenizer_path() -> Path:
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        return Path(tokenizer.name_or_path)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Could not load tokenizer {MODEL_NAME}: {exc}", allow_module_level=True)
        raise NotImplementedError("unreachable")


def _load_tokenizer(tokenizer_path: Path):
    return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)


def test_preference_chat_processor_skips_invalid_rows(tokenizer_path: Path):
    tokenizer = _load_tokenizer(tokenizer_path)
    processor = PreferenceChatProcessor(tokenizer)

    batch = [
        {"chosen": [], "rejected": []},
        {
            "chosen": [
                {"role": "user", "content": "Say hi."},
                {"role": "assistant", "content": "Hi!"},
            ],
            "rejected": [
                {"role": "user", "content": "Say hi."},
                {"role": "assistant", "content": "No."},
            ],
        },
    ]

    result = processor(batch)
    assert len(result) == 1


def test_simpo_loss_decreases_with_margin():
    Batch = hax.Axis("batch", 2)
    avg_rejected = hax.zeros(Batch)

    avg_small = hax.named(jnp.array([0.1, 0.2], dtype=jnp.float32), Batch)
    avg_large = hax.named(jnp.array([1.0, 1.2], dtype=jnp.float32), Batch)

    loss_small, _ = simpo_loss_from_logps(avg_small, avg_rejected, beta=1.0, gamma_beta_ratio=0.0)
    loss_large, _ = simpo_loss_from_logps(avg_large, avg_rejected, beta=1.0, gamma_beta_ratio=0.0)

    assert float(loss_large) < float(loss_small)


def test_simpo_metrics_are_explicit_metrics():
    Batch = hax.Axis("batch", 2)
    avg_rejected = hax.zeros(Batch)
    avg_chosen = hax.named(jnp.array([0.4, 0.8], dtype=jnp.float32), Batch)

    _, metrics = simpo_loss_from_logps(avg_chosen, avg_rejected, beta=1.0, gamma_beta_ratio=0.0)

    assert isinstance(metrics["simpo_loss"], Metric)
    assert isinstance(metrics["simpo_margin"], Metric)
    assert isinstance(metrics["simpo_accuracy"], Metric)


def test_average_logp_passes_key_for_dropout():
    config = Gpt2Config(max_seq_len=8, hidden_dim=16, num_layers=1, num_heads=2, embed_pdrop=0.1)
    Vocab = hax.Axis("vocab", 32)
    model = config.build(Vocab, key=jrandom.PRNGKey(0))
    model = inference_mode(model, False)

    Pos = hax.Axis("position", 8)
    tokens = hax.named(jnp.arange(Pos.size, dtype=jnp.int32) % Vocab.size, Pos)
    example = LmExample.causal(tokens)

    out = _average_logp(model, example, key=jrandom.PRNGKey(1))
    assert isinstance(out, hax.NamedArray)


def test_preference_chat_processor_outputs_masks(tokenizer_path: Path):
    tokenizer = _load_tokenizer(tokenizer_path)
    processor = PreferenceChatProcessor(tokenizer)

    batch = [
        {
            "chosen": [
                {"role": "user", "content": "Hi there."},
                {"role": "assistant", "content": "Hello!"},
            ],
            "rejected": [
                {"role": "user", "content": "Hi there."},
                {"role": "assistant", "content": "No."},
            ],
        }
    ]

    result = processor(batch)
    assert len(result) == 1

    row = result[0]
    assert row["chosen_input_ids"].shape == row["chosen_assistant_masks"].shape
    assert row["rejected_input_ids"].shape == row["rejected_assistant_masks"].shape
    assert row["chosen_assistant_masks"].sum() > 0
    assert row["rejected_assistant_masks"].sum() > 0


def test_preference_pair_dataset_builds_example(tokenizer_path: Path):
    tokenizer = _load_tokenizer(tokenizer_path)
    processor = PreferenceChatProcessor(tokenizer)

    batch = [
        {
            "chosen": [
                {"role": "user", "content": "Hello."},
                {"role": "assistant", "content": "Hi!"},
            ],
            "rejected": [
                {"role": "user", "content": "Hello."},
                {"role": "assistant", "content": "Nope."},
            ],
        }
    ]

    processed = processor(batch)
    with tempfile.TemporaryDirectory() as tmpdir:
        with SerialCacheWriter(tmpdir, processor.output_exemplar) as writer:
            writer.write_batch(processed)

        cache = writer.result()
        Pos = hax.Axis("position", 128)
        dataset = PreferencePairDataset(cache, Pos, max_segments_per_example=1, slice_strategy="raise")
        example = dataset.as_sync_dataset()[0]

        assert isinstance(example, DpoExample)
        assert example.chosen.tokens.axes == (Pos,)
        assert example.rejected.tokens.axes == (Pos,)
        assert example.chosen.loss_weight.array.sum() > 0
        assert example.rejected.loss_weight.array.sum() > 0
