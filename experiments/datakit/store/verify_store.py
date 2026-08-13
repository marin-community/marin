# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify that tokenized Parquet documents match materialized cache documents."""

import argparse
import dataclasses
import glob
import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from levanter.store.cache import TreeCache

_BATCH_ROWS = 8192
_DATABASE_BATCH_ROWS = 4096
_EXEMPLAR = {"input_ids": np.zeros(0, dtype=np.int32)}


@dataclasses.dataclass(frozen=True)
class VerificationMismatch:
    fingerprint: str
    input_count: int
    output_count: int
    token_count: int
    sample_id: str | None


@dataclasses.dataclass(frozen=True)
class VerificationResult:
    input_documents: int
    output_documents: int
    input_tokens: int
    output_tokens: int
    missing_documents: int
    extra_documents: int
    mismatches: list[VerificationMismatch]

    @property
    def matches(self) -> bool:
        return self.missing_documents == 0 and self.extra_documents == 0


def _document_fingerprint(tokens: np.ndarray) -> bytes:
    canonical = np.ascontiguousarray(tokens, dtype="<i4").reshape(-1)
    digest = hashlib.sha256()
    digest.update(len(canonical).to_bytes(8, byteorder="little"))
    digest.update(memoryview(canonical).cast("B"))
    return digest.digest()


def _iter_parquet_documents(paths: Sequence[str], row_limit: int | None) -> Iterator[tuple[str, np.ndarray]]:
    remaining = row_limit
    for path in paths:
        parquet = pq.ParquetFile(path)
        columns = ["id", "input_ids"]
        has_chunks = "chunk_index" in parquet.schema_arrow.names
        if has_chunks:
            columns.insert(1, "chunk_index")

        document_id: str | None = None
        chunks: list[np.ndarray] = []
        for batch in parquet.iter_batches(batch_size=_BATCH_ROWS, columns=columns, use_threads=True):
            ids = batch.column("id").to_pylist()
            input_ids = batch.column("input_ids")
            chunk_indices = batch.column("chunk_index").to_pylist() if has_chunks else None
            for index, row_id in enumerate(ids):
                tokens = input_ids[index].values.to_numpy(zero_copy_only=False)
                if not has_chunks:
                    yield row_id, tokens
                    if remaining is not None:
                        remaining -= 1
                        if remaining == 0:
                            return
                    continue

                assert chunk_indices is not None
                chunk_index = chunk_indices[index]
                if chunk_index == 0:
                    if chunks:
                        assert document_id is not None
                        yield document_id, chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
                        if remaining is not None:
                            remaining -= 1
                            if remaining == 0:
                                return
                    document_id, chunks = row_id, []
                elif row_id != document_id or chunk_index != len(chunks):
                    raise ValueError(
                        f"{path}: chunk {chunk_index} of {row_id} follows chunk {len(chunks) - 1} of {document_id}"
                    )
                chunks.append(tokens)
        if chunks:
            assert document_id is not None
            yield document_id, chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
            if remaining is not None:
                remaining -= 1
                if remaining == 0:
                    return

    if remaining is not None and remaining > 0:
        assert row_limit is not None
        raise ValueError(f"row_limit exceeds the {row_limit - remaining} documents in the input")


def discover_cache_paths(output_root: str) -> list[str]:
    """Return the shallowest cache directories below a local output root."""
    root = Path(output_root)
    ledgers = sorted(root.rglob("shard_ledger.json"))
    if not ledgers:
        raise ValueError(f"No shard_ledger.json files found below {output_root}")
    depths = [len(ledger.parent.relative_to(root).parts) for ledger in ledgers]
    minimum_depth = min(depths)
    return [str(ledger.parent) for ledger, depth in zip(ledgers, depths, strict=True) if depth == minimum_depth]


def _record_documents(
    connection: sqlite3.Connection,
    documents: Iterable[tuple[str | None, np.ndarray]],
    *,
    input_side: bool,
) -> tuple[int, int]:
    count_column = "input_count" if input_side else "output_count"
    statement = f"""
        INSERT INTO documents (fingerprint, input_count, output_count, token_count, sample_id)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(fingerprint) DO UPDATE SET {count_column} = {count_column} + 1
    """
    rows: list[tuple[bytes, int, int, int, str | None]] = []
    total_documents = 0
    total_tokens = 0
    for document_id, tokens in documents:
        token_count = int(np.asarray(tokens).size)
        rows.append(
            (
                _document_fingerprint(tokens),
                1 if input_side else 0,
                0 if input_side else 1,
                token_count,
                document_id,
            )
        )
        total_documents += 1
        total_tokens += token_count
        if len(rows) == _DATABASE_BATCH_ROWS:
            connection.executemany(statement, rows)
            rows.clear()
    if rows:
        connection.executemany(statement, rows)
    connection.commit()
    return total_documents, total_tokens


def _iter_cache_documents(cache_paths: Sequence[str]) -> Iterator[tuple[None, np.ndarray]]:
    for path in cache_paths:
        cache = TreeCache.load(path, _EXEMPLAR)
        for document in cache:
            yield None, np.asarray(document["input_ids"])


def verify_store(
    input_paths: Sequence[str],
    cache_paths: Sequence[str],
    *,
    row_limit: int | None = None,
    database_path: str | None = None,
    mismatch_limit: int = 20,
) -> VerificationResult:
    """Compare input and output document-token multisets exactly."""
    if not input_paths:
        raise ValueError("At least one input Parquet path is required")
    if not cache_paths:
        raise ValueError("At least one output cache path is required")
    if row_limit is not None and row_limit < 1:
        raise ValueError(f"row_limit must be positive, got {row_limit}")
    if mismatch_limit < 1:
        raise ValueError(f"mismatch_limit must be positive, got {mismatch_limit}")

    with tempfile.TemporaryDirectory(prefix="datakit-verify-") as temporary_dir:
        resolved_database_path = database_path or os.path.join(temporary_dir, "documents.sqlite")
        with sqlite3.connect(resolved_database_path) as connection:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                """
                CREATE TABLE documents (
                    fingerprint BLOB PRIMARY KEY,
                    input_count INTEGER NOT NULL,
                    output_count INTEGER NOT NULL,
                    token_count INTEGER NOT NULL,
                    sample_id TEXT
                ) WITHOUT ROWID
                """
            )
            input_documents, input_tokens = _record_documents(
                connection,
                _iter_parquet_documents(input_paths, row_limit),
                input_side=True,
            )
            output_documents, output_tokens = _record_documents(
                connection,
                _iter_cache_documents(cache_paths),
                input_side=False,
            )
            missing_documents, extra_documents = connection.execute(
                """
                SELECT
                    COALESCE(SUM(MAX(input_count - output_count, 0)), 0),
                    COALESCE(SUM(MAX(output_count - input_count, 0)), 0)
                FROM documents
                """
            ).fetchone()
            mismatch_rows = connection.execute(
                """
                SELECT hex(fingerprint), input_count, output_count, token_count, sample_id
                FROM documents
                WHERE input_count != output_count
                ORDER BY ABS(input_count - output_count) DESC, fingerprint
                LIMIT ?
                """,
                (mismatch_limit,),
            ).fetchall()

    return VerificationResult(
        input_documents=input_documents,
        output_documents=output_documents,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        missing_documents=missing_documents,
        extra_documents=extra_documents,
        mismatches=[VerificationMismatch(*row) for row in mismatch_rows],
    )


def _resolve_inputs(patterns: Sequence[str]) -> list[str]:
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    if not paths:
        raise ValueError(f"Input patterns matched no files: {patterns}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Local Parquet path or glob; repeatable")
    parser.add_argument("--output-root", required=True, help="Root containing logical output caches")
    parser.add_argument("--row-limit", type=int, help="Compare only the first N input documents")
    parser.add_argument("--database", help="Keep the SQLite reconciliation database at this path")
    parser.add_argument("--mismatch-limit", type=int, default=20)
    args = parser.parse_args()

    result = verify_store(
        _resolve_inputs(args.input),
        discover_cache_paths(args.output_root),
        row_limit=args.row_limit,
        database_path=args.database,
        mismatch_limit=args.mismatch_limit,
    )
    print(json.dumps(dataclasses.asdict(result) | {"matches": result.matches}, indent=2))
    if not result.matches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
