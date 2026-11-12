# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from itertools import pairwise

import pytest
from levanter.data.dataset import ListAsyncDataset

from experiments.post_training.long_loop.sft_runtime.insert_dataset import InsertedDataset


def read(dataset, indices):
    return asyncio.run(dataset.get_batch(indices))


def test_insert_keeps_base_order_and_spreads_rows_evenly():
    base = ListAsyncDataset([f"b{i}" for i in range(100)])
    insert = ListAsyncDataset([f"c{i}" for i in range(10)])
    stream = read(InsertedDataset(base, insert, rows=7, total=40), list(range(50)))
    inserted = [item for item in stream[:40] if item.startswith("c")]
    assert inserted == [f"c{i}" for i in range(7)]
    # Without the insert, the same positions read b0, b1, ...; with it the base order is unchanged.
    assert [item for item in stream if item.startswith("b")] == [f"b{i}" for i in range(43)]
    slots = [i for i, item in enumerate(stream) if item.startswith("c")]
    assert max(b - a for a, b in pairwise(slots)) <= 40 // 7 + 1
    assert all(item.startswith("b") for item in stream[40:])


def test_insert_batches_match_single_reads():
    dataset = InsertedDataset(
        ListAsyncDataset(list(range(1000))), ListAsyncDataset(list(range(-53, 0))), rows=53, total=512
    )
    indices = [511, 3, 0, 200, 512, 47]
    assert read(dataset, indices) == [read(dataset, [i])[0] for i in indices]
    assert sum(1 for i in range(512) if dataset.source(i)[0]) == 53


@pytest.mark.parametrize("rows", [0, 513])
def test_insert_rows_must_fit(rows):
    with pytest.raises(ValueError, match="Insert rows"):
        InsertedDataset(ListAsyncDataset([0]), ListAsyncDataset([0]), rows=rows, total=512)
