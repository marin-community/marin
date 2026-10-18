# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
from collections.abc import Callable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from zephyr.context import ZephyrContext

from experiments.grug.fast_track.corpus_sample import CorpusSource, RawCorpusPool
from experiments.grug.fast_track.ranked_pool import RangeTokenTotal


@pytest.fixture
def zephyr_context(tmp_path):
    client = LocalClient()
    context = ZephyrContext(
        client=client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name="fast-track-test",
    )
    yield context
    context.shutdown()
    client.shutdown(wait=True)


@pytest.fixture
def write_raw_pool(tmp_path) -> Callable[..., RawCorpusPool]:
    """Return a writer that stores rank-ordered rows as a two-shard raw pool."""

    def write(rows: Sequence[dict], *, name: str = "raw", sources: tuple[CorpusSource, ...] = ()) -> RawCorpusPool:
        middle = len(rows) // 2
        ranges = []
        for range_key, range_rows in enumerate((rows[:middle], rows[middle:])):
            path = tmp_path / f"{name}-{range_key}.parquet"
            pq.write_table(pa.Table.from_pylist(list(range_rows)), path)
            ranges.append(
                RangeTokenTotal(range_key, str(path), len(range_rows), sum(row["token_count"] for row in range_rows))
            )
        total_tokens = sum(item.tokens for item in ranges)
        return RawCorpusPool(
            tokenizer="test-tokenizer",
            tokenizer_hash="a" * 64,
            sources=sources,
            seed=19,
            requested_tokens=total_tokens,
            actual_tokens=total_tokens,
            documents=len(rows),
            shards=tuple(item.path for item in ranges),
            range_totals=tuple(ranges),
            source_tokens={"source": total_tokens},
            manifest_path=str(tmp_path / f"{name}-manifest.json"),
        )

    return write


@pytest.fixture
def raw_pool(tmp_path, write_raw_pool) -> RawCorpusPool:
    """Twelve 5-token documents ``text-<i>`` with locators into one normalized shard and its Harrier vectors."""
    normalized_path = tmp_path / "normalized.parquet"
    rows = [
        {
            "source": "source",
            "id": str(index),
            "sample_rank": hashlib.sha256(f"seed:{index}".encode()).hexdigest(),
            "duplicate_group": hashlib.sha256(f"text-{index}".encode()).hexdigest(),
            "text": f"text-{index}",
            "normalized_shard": str(normalized_path),
            "normalized_row": index,
            "input_ids": [index + 1] * 5,
            "token_count": 5,
        }
        for index in range(12)
    ]
    rows.sort(key=lambda row: (row["sample_rank"], row["source"], row["id"]))
    pq.write_table(
        pa.Table.from_pylist([{"id": str(index), "text": f"text-{index}"} for index in range(12)]), normalized_path
    )
    embedding_path = tmp_path / "harrier" / "normalized.parquet"
    embedding_path.parent.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [{"id": str(index), "embedding": np.full(1024, index + 1, dtype=np.int8)} for index in range(12)],
            schema=pa.schema([("id", pa.string()), ("embedding", pa.list_(pa.int8(), 1024))]),
        ),
        embedding_path,
    )
    return write_raw_pool(rows, sources=(CorpusSource("source", str(tmp_path), 60),))
