# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load the glm52-labels x harrier-embeddings join.

The ``glm52_labels_88k`` oracle labels joined against the embedded 50M sample:
80,897 rows, each carrying the label columns plus the stored 1024-d int8
harrier document embedding. Shared by the document-embedding experiment arms
(:mod:`embed_exp`) and the embedding-based domain typer (:mod:`domain_mlp`).
"""

import logging

import numpy as np
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath

logger = logging.getLogger(__name__)

DEFAULT_JOINED = (
    "s3://marin-us-east-02a/marin/user/muchanem/quality_v2/glm52_labels_88k-x-harrier-oss-v1-0.6b-50m-text-v1"
)
EMBED_DIM = 1024

JOINED_COLUMNS = [
    "id",
    "text",
    "embedding",
    "glm52_source",
    "glm52_content_type",
    "glm52_quality",
    "glm52_score_normalized",
]


def _walk_parquet(root: str, max_depth: int = 5) -> list[str]:
    """Every ``*.parquet`` under ``root``, via single-level globs only (a recursive
    glob HeadObjects the prefix, which the CW store answers with a 400). The join
    mirrors each source's own layout, so shard depth varies from 1 to 3."""
    shards: list[str] = []
    dirs = [root.rstrip("/")]
    for _ in range(max_depth):
        next_dirs: list[str] = []
        for d in dirs:
            for entry in sorted(str(m) for m in StoragePath(f"{d}/*").glob()):
                if entry.endswith(".parquet"):
                    shards.append(entry)
                else:
                    # Descending into a non-directory just globs to nothing, so no
                    # name heuristic (source dirs like `numinamath-1.5` carry dots).
                    next_dirs.append(entry)
        dirs = next_dirs
        if not dirs:
            break
    return shards


def load_joined(joined_dir: str, columns: list[str] | None = None) -> dict[str, list]:
    """All joined label rows, deduplicated by id. ``columns`` must include ``id``."""
    columns = columns or JOINED_COLUMNS
    if "id" not in columns:
        raise ValueError("load_joined columns must include 'id' (rows are deduplicated by it)")
    root = joined_dir.rstrip("/")
    shards = _walk_parquet(f"{root}/outputs")
    if not shards:
        raise ValueError(f"no parquet shards under {root}/outputs/")
    out: dict[str, list] = {c: [] for c in columns}
    seen: set[str] = set()
    dupes = 0
    for shard in shards:
        with StoragePath(shard).open("rb") as fh:
            table = pq.ParquetFile(fh).read(columns=columns)
        rows = {c: table.column(c).to_pylist() for c in columns}
        for i, doc_id in enumerate(rows["id"]):
            if doc_id in seen:
                dupes += 1
                continue
            seen.add(doc_id)
            for c in columns:
                out[c].append(rows[c][i])
    logger.info("joined labels: %d rows from %d shards (%d duplicate ids dropped)", len(out["id"]), len(shards), dupes)
    return out


def embedding_matrix(raw: list) -> np.ndarray:
    """int8 rows -> float32, L2-normalized (recovers direction; drops the
    quantization scale, which carries no per-document information)."""
    x = np.asarray(raw, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-6)
