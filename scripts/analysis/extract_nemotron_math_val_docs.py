# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract the nemotron math validation documents from normalized parquet.

Reads the val-doc manifest (shard -> row indices, built from the levanter
feistel permutation of the 4plus-2c5519 cache), selects those rows from
gs://marin-us-east5/normalized/nemotron_cc_math_v1/4plus_b05688a8/outputs/main/,
attaches xxh3_128 ids over `text`, and writes one parquet per shard.

Launch in us-east5 (intra-region reads only):

    uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
        --cpu 4 --memory 16GB --disk 20GB --priority interactive --extra cpu \
        --enable-extra-resources --preemptible --region us-east5 \
        --job-name extract-math-val-docs \
        -- python scripts/analysis/extract_nemotron_math_val_docs.py
"""

import json
import logging
from collections.abc import Iterator

import fsspec
import pyarrow.parquet as pq
import xxhash
from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext

logger = logging.getLogger(__name__)

NORMALIZED_ROOT = "gs://marin-us-east5/normalized/nemotron_cc_math_v1/4plus_b05688a8/outputs/main"
MANIFEST_URI = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/val_doc_manifest.json"
OUTPUT_PATH = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/val_docs"


def _extract_shard(item: dict) -> Iterator[dict]:
    shard, rows = item["shard"], sorted(item["rows"])
    with fsspec.open(f"{NORMALIZED_ROOT}/{shard}.parquet", "rb") as f:
        table = pq.read_table(f, columns=["text"])
    texts = table.column("text").take(rows).to_pylist()
    for row, text in zip(rows, texts, strict=True):
        yield {
            "shard": shard,
            "row": row,
            "id": xxhash.xxh3_128_hexdigest(text),
            "text": text,
        }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    with fsspec.open(MANIFEST_URI) as f:
        manifest = json.load(f)
    items = [{"shard": shard, "rows": rows} for shard, rows in sorted(manifest.items())]
    logger.info("Extracting %d docs from %d shards", sum(len(it["rows"]) for it in items), len(items))

    ctx = ZephyrContext(
        name="extract-math-val-docs",
        max_workers=64,
        resources=ResourceConfig(cpu=1, ram="4g", disk="5g"),
    )
    pipeline = (
        Dataset.from_list(items)
        .flat_map(_extract_shard)
        .write_parquet(f"{OUTPUT_PATH}/part-{{shard:05d}}-of-{{total:05d}}.parquet")
    )
    results = ctx.execute(pipeline)
    logger.info("Wrote %d output shards to %s", len(results.results), OUTPUT_PATH)


if __name__ == "__main__":
    main()
