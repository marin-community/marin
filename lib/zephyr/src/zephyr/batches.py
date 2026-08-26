# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Arrow batch interfaces used at Zephyr stage boundaries."""

from collections.abc import Iterable, Iterator
from typing import Protocol, runtime_checkable

import pyarrow as pa


@runtime_checkable
class ArrowBatch(Protocol):
    """A columnar batch exportable through Arrow's PyCapsule stream interface."""

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object: ...


def iter_record_batches(items: Iterable[object]) -> Iterator[pa.RecordBatch]:
    """Canonicalize Arrow-exportable stage items to RecordBatches."""
    for item in items:
        if not isinstance(item, ArrowBatch):
            raise TypeError(
                "group_by with zephyr.expr.col(...) requires Arrow-exportable batch items "
                f"implementing __arrow_c_stream__; got {type(item).__name__}. Return "
                "pyarrow.RecordBatch or another Arrow-compatible batch."
            )
        yield from pa.RecordBatchReader.from_stream(item)
