# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from haliax import Axis

from levanter.data.text.datasets import DatasetComponent, dataset_for_component
from levanter.data.text.formats import SerializedChatLmDatasetFormat
from levanter.store.cache import SerialCacheWriter
from levanter.utils.thread_utils import blocking_wait


def _cached_transcript_dataset(tmp_path, documents, *, pack, length, block_attention=False, lookahead_window=None):
    rows = [{"input_ids": np.asarray(tokens, dtype=np.int32)} for tokens in documents]
    with SerialCacheWriter(str(tmp_path), rows[0]) as writer:
        writer.write_batch(rows)
    component = DatasetComponent(
        cache_dir=str(tmp_path),
        pack=pack,
        pack_lookahead_window=lookahead_window,
        format=SerializedChatLmDatasetFormat(message_start_id=10, assistant_prefix=(10, 3, 11, 12)),
    )
    return dataset_for_component(
        component,
        Axis("position", length),
        writer.result(),
        eos_id=None,
        block_cross_document_attention=block_attention,
    )


@pytest.mark.parametrize(
    ("pack", "block_attention", "length", "loss_positions"),
    [
        (False, False, 32, [9, 10, 11, 22, 23]),
        (True, True, 32, [9, 10, 11, 24, 25]),
        (True, False, 32, [9, 10, 11, 24, 25]),
    ],
    ids=["single", "packed-blocked", "packed-unblocked"],
)
def test_cached_transcript_masks_and_attention(tmp_path, pack, block_attention, length, loss_positions):
    first_turn = [10, 2, 11, 12, 20, 13, 10, 3, 11, 12, 21, 22, 13]
    tool_and_reply = [10, 4, 11, 12, 23, 13, 10, 3, 11, 12, 24, 13]
    # A root or text token before the next packed header cannot inherit assistant loss.
    documents = [first_turn, [1, 20, *tool_and_reply]] if pack else [first_turn + tool_and_reply]
    dataset = _cached_transcript_dataset(
        tmp_path, documents, pack=pack, length=length, block_attention=block_attention
    )
    example = blocking_wait(dataset.get_batch([0]))[0]
    tokens = [token for document in documents for token in document][:length]
    np.testing.assert_array_equal(np.asarray(example.tokens)[: len(tokens)], tokens)
    np.testing.assert_array_equal(np.flatnonzero(example.loss_weight), loss_positions)
    if pack and block_attention:
        segments = [0] * len(documents[0]) + [1] * len(documents[1])
        np.testing.assert_array_equal(np.asarray(example.attn_mask.segment_ids[0])[: len(tokens)], segments)
    elif pack:
        assert example.attn_mask.segment_ids is None


def test_overlong_unpacked_transcript_raises(tmp_path):
    transcript = [10, 3, 11, 12, 21, 22, 23, 24, 13]
    with pytest.raises(ValueError, match="exceeds maximum allowed length"):
        dataset = _cached_transcript_dataset(tmp_path, [transcript], pack=False, length=8)
        blocking_wait(dataset.get_batch([0]))


def test_packed_transcript_uses_lookahead_window(tmp_path):
    first = [10, 3, 11, 12, 21, 22, 23, 24, 13]
    second = [10, 3, 11, 12, 31, 32, 33, 34, 13]
    short = [10, 2, 11, 13]
    dataset = _cached_transcript_dataset(tmp_path, [first, second, short], pack=True, length=13, lookahead_window=3)
    example = blocking_wait(dataset.get_batch([0]))[0]
    np.testing.assert_array_equal(np.asarray(example.tokens), first + short)
