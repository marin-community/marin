# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Match one co-partitioned shard's rows to another's on ``id``.

Two attribute shards of one source hold the same documents, but their row order
is only guaranteed to agree when both were written in the normalized shard's
order, and one Harrier leaf is a repartition that was not. So a consumer that
walks one shard in order looks the other side up by key: the side is read whole,
its ids sorted once, and every document claims its rows by binary search.

The claim is what makes the match checkable. A document that finds no row means
the sides came from different normalize runs; a row no document claims means the
same the other way round. Both are reported by the caller after the walk.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl
from rigging.filesystem.storage_path import StoragePath


@dataclass(frozen=True)
class KeyedRows:
    """One shard's rows of ``column``, addressable by ``id``."""

    lookup: np.ndarray
    """The shard's ids, sorted."""
    order: np.ndarray
    """Rank in ``lookup`` -> row in ``values``."""
    values: np.ndarray
    """The column in stored row order: 1-D for a scalar column, 2-D for an array column."""

    @classmethod
    def from_arrays(cls, ids: np.ndarray, values: np.ndarray) -> "KeyedRows":
        order = np.argsort(ids, kind="stable")
        return cls(lookup=ids[order], order=order, values=values)

    def __len__(self) -> int:
        return len(self.lookup)

    def rows_for(self, doc_ids: np.ndarray, claimed: np.ndarray, where: str) -> np.ndarray:
        """Return the stored row of each document, marking every row of its id claimed.

        ``claimed`` is one bool per rank of ``lookup`` that the caller keeps across
        batches; a duplicate id claims its whole run of rows at once, so a duplicate
        on the other side cannot claim a row twice, and the caller can tell at the
        end whether any row was never claimed at all.
        """
        lo = np.searchsorted(self.lookup, doc_ids, side="left")
        hi = np.searchsorted(self.lookup, doc_ids, side="right")
        missing = hi <= lo
        if missing.any():
            examples = doc_ids[missing][:3].tolist()
            raise ValueError(
                f"{where}: {int(missing.sum())} of {len(doc_ids)} documents have no row, e.g. {examples}; "
                f"the two sides did not come from one normalize run"
            )
        counts = hi - lo
        starts = np.cumsum(counts) - counts
        ranks = np.repeat(lo, counts) + (np.arange(int(counts.sum())) - np.repeat(starts, counts))
        claimed[ranks] = True
        return self.order[lo]

    def require_all_claimed(self, claimed: np.ndarray, documents: int, where: str) -> None:
        """Fail unless every row was claimed and the row count equals ``documents``."""
        unclaimed = int((~claimed).sum())
        if unclaimed:
            examples = self.lookup[~claimed][:3].tolist()
            raise ValueError(
                f"{where}: {unclaimed} of {len(self)} rows belong to no document, e.g. {examples}; "
                f"the two sides did not come from one normalize run"
            )
        if documents != len(self):
            raise ValueError(
                f"{where}: {documents} documents against {len(self)} rows; the two sides did not come from "
                f"one normalize run"
            )


def read_keyed_rows(path: str, column: str) -> KeyedRows:
    """Read ``id`` and ``column`` from one parquet shard, whole."""
    # polars types a fixed-width list column as an Array with no offsets buffer,
    # so the int32 offset ceiling that fails a whole-column pyarrow read of the
    # largest Harrier shards (2,682,446 documents x 1,024 values > 2^31-1) does
    # not apply, and to_numpy hands back one contiguous [n, width] block.
    with StoragePath(path).open("rb") as fh:
        frame = pl.read_parquet(fh, columns=["id", column])
    return KeyedRows.from_arrays(frame.get_column("id").to_numpy(), frame.get_column(column).to_numpy())
