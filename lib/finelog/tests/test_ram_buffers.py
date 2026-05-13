# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pyarrow as pa
from finelog.store.log_namespace import RamBuffers, _SealedBuffer
from finelog.store.schema import IMPLICIT_SEQ_COLUMN, schema_to_arrow, with_implicit_seq

from tests.conftest import _worker_schema


def _ns_arrow_schema() -> pa.Schema:
    return schema_to_arrow(with_implicit_seq(_worker_schema()))


def _make_chunk(arrow_schema: pa.Schema, first_seq: int, num_rows: int) -> pa.Table:
    return pa.table(
        [
            pa.array(range(first_seq, first_seq + num_rows), type=pa.int64()),
            pa.array([f"w-{i}" for i in range(num_rows)], type=pa.string()),
            pa.array(range(num_rows), type=pa.int64()),
            pa.array(range(num_rows), type=pa.int64()),
        ],
        schema=arrow_schema,
    )


def test_sealed_buffer_caches_nbytes_and_num_rows():
    arrow_schema = _ns_arrow_schema()
    table = _make_chunk(arrow_schema, first_seq=1, num_rows=5)
    sealed = _SealedBuffer(table=table, min_seq=1, max_seq=5)
    assert sealed.nbytes == table.nbytes
    assert sealed.num_rows == table.num_rows


def test_ram_bytes_reads_cached_flushing_nbytes_not_table():
    arrow_schema = _ns_arrow_schema()
    buffers = RamBuffers(arrow_schema=arrow_schema, next_seq=1)
    buffers.append_table(_make_chunk(arrow_schema, first_seq=1, num_rows=3))
    pre_seal_bytes = buffers.ram_bytes()
    pre_seal_rows = buffers.ram_rows()

    sealed = buffers.seal()
    assert sealed is not None
    assert buffers.ram_bytes() == pre_seal_bytes
    assert buffers.ram_rows() == pre_seal_rows

    # Mutate the cached scalars to confirm ram_bytes / ram_rows read them,
    # not table.nbytes / table.num_rows on every call.
    sealed.nbytes = 999_999_999
    sealed.num_rows = 42_424_242
    assert buffers.ram_bytes() == 999_999_999
    assert buffers.ram_rows() == 42_424_242


def test_restore_flush_preserves_accounting():
    arrow_schema = _ns_arrow_schema()
    buffers = RamBuffers(arrow_schema=arrow_schema, next_seq=1)
    buffers.append_table(_make_chunk(arrow_schema, first_seq=1, num_rows=4))
    expected_bytes = buffers.ram_bytes()
    expected_rows = buffers.ram_rows()

    buffers.seal()
    buffers.restore_flush()

    assert buffers.ram_bytes() == expected_bytes
    assert buffers.ram_rows() == expected_rows


def test_implicit_seq_column_present():
    arrow_schema = _ns_arrow_schema()
    assert IMPLICIT_SEQ_COLUMN in arrow_schema.names
