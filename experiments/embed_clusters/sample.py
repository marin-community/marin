# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample the first N documents from a datakit-normalized source.

Reads normalized parquet shards row-group at a time via ``pyarrow.parquet``
(native GCS byte-range reads) and writes a single ``samples.parquet`` with
``id`` + ``text``. Deterministic first-N — relies on the upstream sort to
provide enough natural mixing; swap to reservoir sampling here if the
clusters look biased toward shard order.
"""

import logging
import os

import pyarrow.parquet as pq
from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import Artifact
from marin.utils import fsspec_glob
from zephyr.writers import write_parquet_file

logger = logging.getLogger(__name__)


def sample_normalized(
    output_path: str,
    normalized_path: str,
    n_docs: int,
) -> None:
    """Take the first ``n_docs`` records from a normalized source's main output."""
    normalized = Artifact.from_path(normalized_path, NormalizedData)
    shards = sorted(fsspec_glob(f"{normalized.main_output_dir.rstrip('/')}/**/*.parquet"))
    if not shards:
        raise RuntimeError(f"No parquet shards under {normalized.main_output_dir}")

    docs: list[dict] = []
    for shard in shards:
        if len(docs) >= n_docs:
            break
        pf = pq.ParquetFile(shard)
        for i in range(pf.num_row_groups):
            if len(docs) >= n_docs:
                break
            rg = pf.read_row_group(i, columns=["id", "text"])
            need = n_docs - len(docs)
            if rg.num_rows > need:
                rg = rg.slice(0, need)
            for row in rg.to_pylist():
                docs.append({"id": row["id"], "text": row["text"]})

    logger.info("Sampled %d documents across %d shards", len(docs), len(shards))
    write_parquet_file(docs, os.path.join(output_path, "samples.parquet"))
