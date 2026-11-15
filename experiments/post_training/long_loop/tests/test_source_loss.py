# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import draccus
import numpy as np
import pytest
from haliax import Axis
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, dataset_for_component
from levanter.data.text.formats import PrebuiltLmDatasetFormat
from levanter.store.cache import SerialCacheWriter
from levanter.utils.thread_utils import blocking_wait

from experiments.post_training.long_loop.sft_accounting import sample_origin, sequence_statistics
from experiments.post_training.long_loop.source_loss import (
    prepared_supervision_data,
    source_loss_metadata,
)


def test_pinned_source_modes_preserve_chat_and_full_token_weights_and_accounting(tmp_path):
    sources = {}
    components = {}
    caches = {}
    expected = {"chat": [0, 1, 1, 0], "replay": [1, 1, 1, 0]}
    for name, weights in expected.items():
        directory = tmp_path / name
        with SerialCacheWriter(
            str(directory), {"input_ids": np.empty(0, dtype=np.int32), "loss_weights": np.empty(0, dtype=np.float32)}
        ) as writer:
            writer.write_batch(
                [{"input_ids": np.arange(4, dtype=np.int32), "loss_weights": np.asarray(weights, dtype=np.float32)}]
            )
        components[name] = DatasetComponent(
            cache_dir=str(directory), format=PrebuiltLmDatasetFormat(loss_weights_key="loss_weights")
        )
        caches[name] = writer.result()
        sources[name] = {
            "loss_mode": "chat_assistant_only" if name == "chat" else "full_token",
            "loss_weights_key": "loss_weights",
            "loss_weights_indexing": "prediction",
            "teacher_only": False,
            "source_revision": "frozen-cache-v1",
        }
    path = tmp_path / "supervision.json"
    raw = json.dumps({"version": 1, "sources": sources}).encode()
    path.write_bytes(raw)
    metadata = source_loss_metadata(str(path), hashlib.sha256(raw).hexdigest())
    data = prepared_supervision_data(LmDataConfig(components=components), metadata, packing_cap=256, lookahead=128)
    data = draccus.decode(LmDataConfig, json.loads(json.dumps(draccus.encode(data))))
    rows = []
    for name, component in data.components.items():
        dataset = dataset_for_component(
            component, Axis("position", 8), caches[name], eos_id=None, block_cross_document_attention=True
        )
        example = blocking_wait(dataset.get_batch([0]))[0]
        np.testing.assert_array_equal(example.loss_weight, expected[name] + [0] * 4)
        origin = blocking_wait(sample_origin(dataset, 0, loss_mode=metadata[name].loss_mode))
        assert origin.is_sft == (name == "chat")
        with pytest.raises(ValueError, match="explicit source loss mode"):
            blocking_wait(sample_origin(dataset, 0))
        rows.append(
            {
                "real_tokens": 4,
                "original_tokens": 4,
                "loss_tokens": sum(expected[name]),
                "is_sft": origin.is_sft,
                "documents": [{"real_tokens": 4, "original_tokens": 4}],
            }
        )
    stats = sequence_statistics(rows, 8, 2, 1.0, max_segments=256)
    assert stats["pad_fraction"] == 0.5
    assert stats["loss_bearing_assistant_tokens"] == 2
    assert stats["replay"]["loss_tokens"] == 3
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        source_loss_metadata(str(path), "0" * 64)
    with pytest.raises(ValueError, match="exactly one explicit"):
        prepared_supervision_data(data, {"chat": metadata["chat"]}, packing_cap=256, lookahead=128)
