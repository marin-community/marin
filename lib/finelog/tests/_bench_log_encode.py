# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Microbenchmark for caller-side log row encoding.

Compares the legacy per-row Arrow construction path (one ``RecordBatch`` per
row + ``pa.concat_batches`` at flush) against the current path that builds a
single ``pa.RecordBatch`` for the entire batch.

Run with::

    uv run python -m finelog.tests._bench_log_encode
"""

from __future__ import annotations

import io
import time
import types
from typing import Any

import pyarrow as pa
import pyarrow.ipc as paipc
from finelog.client.log_client import _make_stats_batch_encoder
from finelog.store.log_namespace import LOG_REGISTERED_SCHEMA
from finelog.store.schema import Schema, schema_to_arrow

N = 20_000


def _make_rows(n: int) -> list[types.SimpleNamespace]:
    return [
        types.SimpleNamespace(
            key="key",
            source="stdout",
            data=f"line {i}: this is a moderately long log message representative of real workloads",
            epoch_ms=1_700_000_000_000 + i,
            level=2,
        )
        for i in range(n)
    ]


def _legacy_row_to_record_batch(row: Any, arrow_schema: pa.Schema, schema: Schema) -> pa.RecordBatch:
    """Reproduce the pre-fix per-row encoder for the benchmark only."""
    columns: list[pa.Array] = []
    for col, field in zip(schema.columns, arrow_schema, strict=True):
        value = getattr(row, col.name, None)
        raw = [None] if value is None else [value]
        columns.append(pa.array(raw, type=field.type, from_pandas=False))
    return pa.RecordBatch.from_arrays(columns, schema=arrow_schema)


def bench_per_row_legacy(rows: list[object]) -> tuple[float, float, float]:
    arrow_schema = schema_to_arrow(LOG_REGISTERED_SCHEMA)

    t0 = time.perf_counter()
    payloads = [_legacy_row_to_record_batch(r, arrow_schema, LOG_REGISTERED_SCHEMA) for r in rows]
    t_encode = time.perf_counter() - t0

    t0 = time.perf_counter()
    combined = pa.concat_batches(payloads)
    t_concat = time.perf_counter() - t0

    t0 = time.perf_counter()
    sink = io.BytesIO()
    with paipc.new_stream(sink, combined.schema) as writer:
        writer.write_batch(combined)
    _ = sink.getvalue()
    t_ipc = time.perf_counter() - t0

    return t_encode, t_concat, t_ipc


def bench_per_batch_current(rows: list[object]) -> tuple[float, float]:
    """Build a single RecordBatch for the whole batch in one pass."""
    arrow_schema = schema_to_arrow(LOG_REGISTERED_SCHEMA)
    encoder = _make_stats_batch_encoder(arrow_schema, LOG_REGISTERED_SCHEMA)

    t0 = time.perf_counter()
    batch = encoder(rows)
    t_encode = time.perf_counter() - t0

    t0 = time.perf_counter()
    sink = io.BytesIO()
    with paipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    _ = sink.getvalue()
    t_ipc = time.perf_counter() - t0

    return t_encode, t_ipc


def main() -> None:
    rows = _make_rows(N)
    print(f"benchmark with N={N} log rows")

    encode, concat, ipc = bench_per_row_legacy(rows)
    print(
        f"  per-row encode (legacy):   encode={encode:.4f}s  concat={concat:.4f}s  ipc={ipc:.4f}s  "
        f"total={(encode + concat + ipc):.4f}s"
    )

    encode_b, ipc_b = bench_per_batch_current(rows)
    print(f"  per-batch encode (current):  encode={encode_b:.4f}s  ipc={ipc_b:.4f}s  total={(encode_b + ipc_b):.4f}s")


if __name__ == "__main__":
    main()
