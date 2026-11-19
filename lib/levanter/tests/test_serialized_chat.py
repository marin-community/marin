# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from haliax import Axis

from levanter.data.text.datasets import DatasetComponent, dataset_for_component
from levanter.data.text.formats import SerializedChatLmDatasetFormat
from levanter.store.cache import SerialCacheWriter
from levanter.utils.thread_utils import blocking_wait


def test_cached_transcript_masks_prompts_tools_and_headers_without_changing_tokens(tmp_path):
    # Message start=10, header end=11, newline=12, turn end=13;
    # user=2, assistant=3, tool=4. Both assistant turns must retain loss.
    tokens = np.asarray(
        [10, 2, 11, 12, 20, 13, 10, 3, 11, 12, 21, 22, 13, 10, 4, 11, 12, 23, 13, 10, 3, 11, 12, 24, 13],
        dtype=np.int32,
    )
    with SerialCacheWriter(str(tmp_path), {"input_ids": np.zeros(0, dtype=np.int32)}) as writer:
        writer.write_batch([{"input_ids": tokens}])
    component = DatasetComponent(
        cache_dir=str(tmp_path),
        pack=False,
        format=SerializedChatLmDatasetFormat(message_start_id=10, assistant_prefix=(10, 3, 11, 12)),
    )
    dataset = dataset_for_component(
        component, Axis("position", 32), writer.result(), eos_id=None, block_cross_document_attention=False
    )
    example = blocking_wait(dataset.get_batch([0]))[0]
    np.testing.assert_array_equal(np.asarray(example.tokens)[: len(tokens)], tokens)
    # Weights index the prediction's predecessor, including each assistant EOS.
    np.testing.assert_array_equal(np.flatnonzero(example.loss_weight), [9, 10, 11, 22, 23])

    # Left truncation is explicit for the single-transcript view. It must not
    # label a tool observation as assistant output or predict beyond the window.
    truncated = dataset_for_component(
        component, Axis("position", 18), writer.result(), eos_id=None, block_cross_document_attention=False
    )
    short = blocking_wait(truncated.get_batch([0]))[0]
    np.testing.assert_array_equal(np.asarray(short.tokens), tokens[:18])
    np.testing.assert_array_equal(np.flatnonzero(short.loss_weight), [9, 10, 11])


@pytest.mark.parametrize("block_cross_document_attention", [True, False])
def test_packed_native_transcripts_preserve_document_masks_and_attention_boundaries(
    tmp_path, block_cross_document_attention
):
    documents = [
        [10, 2, 11, 12, 20, 13, 10, 3, 11, 12, 21, 22, 13],
        # BOS/text before the next header must not inherit assistant loss.
        [1, 20, 10, 4, 11, 12, 23, 13, 10, 3, 11, 12, 24, 13],
    ]
    with SerialCacheWriter(str(tmp_path), {"input_ids": np.zeros(0, dtype=np.int32)}) as writer:
        writer.write_batch([{"input_ids": np.asarray(tokens, dtype=np.int32)} for tokens in documents])
    component = DatasetComponent(
        cache_dir=str(tmp_path),
        pack=True,
        format=SerializedChatLmDatasetFormat(message_start_id=10, assistant_prefix=(10, 3, 11, 12)),
    )
    packed = dataset_for_component(
        component,
        Axis("position", 32),
        writer.result(),
        eos_id=None,
        block_cross_document_attention=block_cross_document_attention,
    )
    actual = blocking_wait(packed.get_batch([0]))[0]
    np.testing.assert_array_equal(np.flatnonzero(actual.loss_weight), [9, 10, 11, 24, 25])
    tokens = documents[0] + documents[1]
    np.testing.assert_array_equal(np.asarray(actual.tokens)[: len(tokens)], tokens)
    if block_cross_document_attention:
        segments = [0] * len(documents[0]) + [1] * len(documents[1])
        np.testing.assert_array_equal(np.asarray(actual.attn_mask.segment_ids[0])[: len(tokens)], segments)
    else:
        assert actual.attn_mask.segment_ids is None
