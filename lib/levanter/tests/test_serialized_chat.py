# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
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
