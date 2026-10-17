# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combining prepared domains must not train on old batch padding."""

import gzip
import json

from experiments.post_training.mixed_transfer.combine import combine_rows


def test_rebatch_removes_old_padding_and_preserves_supervision():
    first = {"input_ids": [1, 2, 3, 4], "loss_weights": [0, 1, 1, 0]}
    second = {"input_ids": [5, 6, 7, 8], "loss_weights": [0, 0, 1, 0]}
    empty = {"input_ids": [128009] * 4, "loss_weights": [0] * 4}
    sources = []
    for row in (first, second):
        compressed = gzip.compress((json.dumps(row) + "\n" + json.dumps(empty) + "\n").encode())
        sources.append(
            (
                compressed,
                {
                    "max_sequence_tokens": 4,
                    "batch_size": 2,
                    "training_rows": 1,
                    "padding_rows": 1,
                    "supervised_tokens": sum(row["loss_weights"]),
                },
            )
        )
    combined, padding = combine_rows(sources)
    assert padding == 0
    assert [json.loads(line) for line in gzip.decompress(combined).splitlines()] == [first, second]


def test_rebatch_adds_only_zero_loss_rows_to_complete_mixed_batch():
    row = {"input_ids": [1, 2, 3, 4], "loss_weights": [0, 1, 1, 0]}
    old_pad = {"input_ids": [128489] * 4, "loss_weights": [0] * 4}
    compressed = gzip.compress((json.dumps(row) + "\n" + (json.dumps(old_pad) + "\n") * 3).encode())
    manifest = {
        "max_sequence_tokens": 4,
        "batch_size": 4,
        "training_rows": 1,
        "padding_rows": 3,
        "supervised_tokens": 2,
    }
    combined, padding = combine_rows([(compressed, manifest), (compressed, manifest)])
    rows = [json.loads(line) for line in gzip.decompress(combined).splitlines()]
    assert padding == 2
    assert rows[:2] == [row, row]
    assert rows[2:] == [{"input_ids": [128009] * 4, "loss_weights": [0] * 4}] * 2
    assert sum(sum(value["loss_weights"]) for value in rows) == 4
