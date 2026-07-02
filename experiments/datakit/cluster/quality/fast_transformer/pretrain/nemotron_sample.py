# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample a quality-labelled slice of Nemotron-CC for pretraining.

Nemotron-CC ships its own quality buckets as separate datakit sources
(``nemotron_cc_v2/{high,medium_high,medium}_quality``). Those buckets are *free*
ordinal quality labels at massive scale -- the lever for the data-limited
plateau the 5.6k-doc oracle set hits. We sample ``per_bucket`` documents from
each bucket and write a parquet in the **same schema as the oracle scorer**
(``source, id, text, score_raw, score_normalized``) so the existing
:mod:`data` loader reads it unchanged. The bucket maps to an ordinal target:

    high_quality -> 1.0,  medium_high_quality -> 0.5,  medium_quality -> 0.0

Submit::

    python -m experiments.datakit.cluster.quality.fast_transformer.pretrain.nemotron_sample \\
        --out gs://marin-eu-west4/datakit/llm-quality-classifier/fast_transformer/nemotron-60k.parquet \\
        --per-bucket 20000
"""

import argparse
import logging
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq
from marin.datakit.normalize import NormalizedData
from marin.datakit.sources import all_sources
from marin.execution.artifact import Artifact
from rigging.filesystem import url_to_fs
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.v0.sample import _list_shards, _read_quota_from_shard

logger = logging.getLogger(__name__)

# Datakit Nemotron-CC quality buckets -> normalized ordinal target in [0, 1].
BUCKET_SCORE: dict[str, float] = {
    "nemotron_cc_v2/high_quality": 1.0,
    "nemotron_cc_v2/medium_high_quality": 0.5,
    "nemotron_cc_v2/medium_quality": 0.0,
}
MAX_DOC_CHARS = 32_000

_SCHEMA = pa.schema(
    [
        pa.field("source", pa.string()),
        pa.field("id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("score_raw", pa.int64()),
        pa.field("score_normalized", pa.float64()),
    ]
)


def _sample_bucket(source: str, per_bucket: int, seed: int) -> list[dict]:
    """Read ``per_bucket`` docs spread across several random shards of *source*."""
    src = all_sources()[source]
    nd = Artifact.from_path(src.normalized, NormalizedData)
    shards = _list_shards(nd.main_output_dir)
    if not shards:
        logger.warning("source %s: no shards", source)
        return []
    rng = random.Random(seed)
    rng.shuffle(shards)
    norm = BUCKET_SCORE[source]
    raw = round(norm * 4) + 1
    rows: list[dict] = []
    # Spread the quota over up to 8 shards so we don't read one giant shard whole.
    per_shard = max(1, per_bucket // min(8, len(shards)))
    for shard in shards:
        for rec in _read_quota_from_shard(shard, per_shard):
            text = rec["text"]
            rows.append(
                {
                    "source": source,
                    "id": str(rec["id"]),
                    "text": text[:MAX_DOC_CHARS],
                    "score_raw": raw,
                    "score_normalized": norm,
                }
            )
            if len(rows) >= per_bucket:
                logger.info("source %s: sampled %d docs", source, len(rows))
                return rows
    logger.info("source %s: sampled %d docs (exhausted shards)", source, len(rows))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output parquet (oracle schema)")
    parser.add_argument("--per-bucket", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    configure_logging(logging.INFO)
    rows: list[dict] = []
    for i, source in enumerate(sorted(BUCKET_SCORE)):
        rows.extend(_sample_bucket(source, args.per_bucket, args.seed + i))
    logger.info("total sampled: %d docs across %d buckets -> %s", len(rows), len(BUCKET_SCORE), args.out)

    table = pa.Table.from_pylist(rows, schema=_SCHEMA)
    fs, resolved = url_to_fs(args.out)
    parent = os.path.dirname(resolved)
    if parent:
        fs.mkdirs(parent, exist_ok=True)
    with fs.open(resolved, "wb") as fh:
        pq.write_table(table, fh, compression="zstd")
    logger.info("wrote %d rows", len(rows))


if __name__ == "__main__":
    main()
