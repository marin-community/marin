# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
from levanter.store.jagged_array import JaggedArrayStore

from experiments.sft_cold_start.replay_subset import ReplaySubsetConfig, prepare_replay_subset


def test_replay_subset_preserves_whole_documents_in_disjoint_length_streams(tmp_path):
    source = str(tmp_path / "source")
    documents = [np.full(length, token, dtype=np.int32) for token, length in enumerate((3, 65537, 8, 65538, 2))]
    store = JaggedArrayStore.open(source + "/input_ids", mode="w", dtype=np.int32)
    store.extend(documents)
    output = str(tmp_path / "output")
    paths = prepare_replay_subset(ReplaySubsetConfig((source,), output, 5, 65537, "test-tokenizer"))
    for name, expected in (("lte_64k", [documents[0], documents[2]]), ("gt_64k", [documents[1]])):
        actual = JaggedArrayStore.open(paths[name] + "/input_ids", mode="r", dtype=np.int32)
        assert len(actual) == len(expected)
        for got, want in zip(actual.get_batch_sync(list(range(len(actual)))), expected, strict=True):
            np.testing.assert_array_equal(got, want)
        ledger = json.loads((tmp_path / "output" / name / "shard_ledger.json").read_text())
        assert ledger["is_finished"]
        assert ledger["field_counts"]["input_ids"] == sum(len(doc) for doc in expected)
    audit = json.loads((tmp_path / "output" / "selection.json").read_text())
    assert audit[0]["short_tokens"] == 11
    assert audit[0]["long_tokens"] == 65537
