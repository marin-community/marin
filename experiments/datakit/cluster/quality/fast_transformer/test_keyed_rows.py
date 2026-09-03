# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The keyed match between two co-partitioned shards, and what it refuses."""

import numpy as np
import pytest

from experiments.datakit.cluster.quality.fast_transformer.keyed_rows import KeyedRows


def keyed(ids: list[str], values: list[float]) -> KeyedRows:
    return KeyedRows.from_arrays(np.array(ids, dtype=object), np.array(values, dtype=np.float32))


def test_rows_are_matched_on_key_not_stored_order():
    side = keyed(["d", "b", "a", "c"], [4.0, 2.0, 1.0, 3.0])
    claimed = np.zeros(len(side), dtype=bool)

    rows = side.rows_for(np.array(["a", "b", "c", "d"], dtype=object), claimed, "shard")

    assert side.values[rows].tolist() == [1.0, 2.0, 3.0, 4.0]
    side.require_all_claimed(claimed, 4, "shard")


def test_a_duplicate_id_claims_its_whole_run_once():
    """Byte-identical documents share an id and a value; each copy gets its own row.

    Both copies on the walked side take the run's first row, and the run is
    claimed as a whole, so neither an unclaimed duplicate nor a double count is
    left behind.
    """
    side = keyed(["dup", "solo", "dup"], [0.5, 0.75, 0.5])
    claimed = np.zeros(len(side), dtype=bool)

    first = side.rows_for(np.array(["dup", "solo"], dtype=object), claimed, "shard")
    second = side.rows_for(np.array(["dup"], dtype=object), claimed, "shard")

    assert side.values[first].tolist() == [0.5, 0.75]
    assert side.values[second].tolist() == [0.5]
    side.require_all_claimed(claimed, 3, "shard")


def test_a_document_without_a_row_fails_the_shard():
    side = keyed(["a"], [1.0])
    with pytest.raises(ValueError, match="have no row"):
        side.rows_for(np.array(["a", "z"], dtype=object), np.zeros(1, dtype=bool), "shard")


def test_a_row_no_document_claims_fails_the_shard():
    side = keyed(["a", "b"], [1.0, 2.0])
    claimed = np.zeros(2, dtype=bool)
    side.rows_for(np.array(["a"], dtype=object), claimed, "shard")
    with pytest.raises(ValueError, match="belong to no document"):
        side.require_all_claimed(claimed, 1, "shard")


def test_more_documents_than_rows_fails_the_shard():
    # Two documents sharing an id against one row: every row is claimed, but the
    # sides still disagree on how many documents there are.
    side = keyed(["a"], [1.0])
    claimed = np.zeros(1, dtype=bool)
    side.rows_for(np.array(["a", "a"], dtype=object), claimed, "shard")
    with pytest.raises(ValueError, match="2 documents against 1 rows"):
        side.require_all_claimed(claimed, 2, "shard")
