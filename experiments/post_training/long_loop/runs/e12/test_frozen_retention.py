# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import io
import json

import numpy as np
import pytest
from haliax import Axis

from experiments.post_training.long_loop.runs.e12.frozen_retention import FrozenRetentionDataset
from experiments.post_training.long_loop.runs.e12.mixed_data import E12DataConfig


def test_frozen_fields_and_checksum(tmp_path):
    fields = {
        "input_ids": np.arange(8, dtype="<i4"),
        "loss_weights": np.asarray([0, 1, 1, 0, 1, 0, 0, 0], dtype="<f4"),
        "q_segment_ids": np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype="<i4"),
        "kv_segment_ids": np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype="<i4"),
    }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **fields)
    raw = buffer.getvalue()
    (tmp_path / "row.npz").write_bytes(raw)
    record = {
        "index": 0,
        "file": "row.npz",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "field_hashes": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in fields.items()},
        "has_segment_ids": True,
        "is_causal": True,
        "sliding_window": None,
        "tags": ["unit"],
        "target_tokens": 3.0,
    }
    dataset = FrozenRetentionDataset(str(tmp_path), [record])
    row = asyncio.run(dataset.get_batch([0]))[0]
    np.testing.assert_array_equal(row.tokens, fields["input_ids"])
    np.testing.assert_array_equal(row.loss_weight, fields["loss_weights"])
    for x, k in zip(row.attn_mask.segment_ids, ["q_segment_ids", "kv_segment_ids"], strict=True):
        np.testing.assert_array_equal(x, fields[k])
    payload = (json.dumps([record]) + "\n").encode()
    (tmp_path / "manifest.json").write_bytes(payload)
    config = E12DataConfig(
        retention_root=str(tmp_path),
        retention_manifest=str(tmp_path / "manifest.json"),
        retention_manifest_sha256=hashlib.sha256(payload).hexdigest(),
    )
    sets = config.tagged_eval_sets(Axis("position", 8))
    assert len(sets) == 1 and sets[0][1] == ["unit"]
    (tmp_path / "row.npz").write_bytes(raw + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        asyncio.run(dataset.get_batch([0]))
