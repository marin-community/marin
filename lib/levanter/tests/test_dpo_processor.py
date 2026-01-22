# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pytest
from transformers import AutoTokenizer

import haliax as hax

from levanter.data.text import DpoExample, PreferenceChatProcessor, PreferencePairDataset
from levanter.store.cache import SerialCacheWriter


MODEL_NAME = "stanford-crfm/marin-tokenizer"


@pytest.fixture(scope="module")
def tokenizer_path() -> Path:
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        return Path(tokenizer.name_or_path)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Could not load tokenizer {MODEL_NAME}: {e}", allow_module_level=True)
        raise NotImplementedError("unreachable")


def load_tokenizer(tokenizer_path: Path):
    return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)


def test_preference_chat_processor_outputs_masks(tokenizer_path: Path):
    tokenizer = load_tokenizer(tokenizer_path)
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
    tokenizer = load_tokenizer(tokenizer_path)
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
