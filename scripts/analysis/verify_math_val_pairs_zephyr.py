# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify LSH candidate pairs (val vs corpus) with exact Jaccard, in-region.

One Zephyr task per corpus parquet shard:
- load the shard's {id, text},
- keep rows whose id appears as `other_id` in the candidate pairs,
- compute 5-char shingle Jaccard against the val doc text,
- emit (val_id, other_id, jaccard) for pairs above threshold (plus all
  scores >= 0.5 for the distribution).

Launch:

    uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
        --cpu 4 --memory 16GB --disk 20GB --priority interactive --extra cpu \
        --enable-extra-resources --preemptible --region us-east5 \
        --job-name verify-math-val-pairs \
        -- python scripts/analysis/verify_math_val_pairs_zephyr.py
"""

import logging
import re
from collections import defaultdict
from collections.abc import Iterator

import pyarrow.parquet as pq
from fray import ResourceConfig
from marin.utils import fsspec_glob
from zephyr import Dataset, ZephyrContext, counters

logger = logging.getLogger(__name__)

SCRATCH = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup"
NORM = "gs://marin-us-east5/normalized/nemotron_cc_math_v1/4plus_b05688a8/outputs/main"
PAIRS = f"{SCRATCH}/val_candidate_pairs"
VAL_DOCS = f"{SCRATCH}/val_docs"
OUT = f"{SCRATCH}/verified_pairs"
MIN_REPORT_JACCARD = 0.5
_WS = re.compile(r"\s+")

_PAIRS: dict[str, list[str]] | None = None
_VAL_TEXT: dict[str, str] | None = None


def _load_state() -> tuple[dict[str, list[str]], dict[str, str]]:
    global _PAIRS, _VAL_TEXT
    if _PAIRS is None:
        by_other = defaultdict(set)
        for path in fsspec_glob(f"{PAIRS}/*.parquet"):
            t = pq.read_table(path, columns=["val_id", "other_id"])
            for val_id, other_id in zip(t.column("val_id").to_pylist(), t.column("other_id").to_pylist(), strict=True):
                by_other[other_id].add(val_id)
        _PAIRS = {k: sorted(v) for k, v in by_other.items()}
        val_text = {}
        for path in fsspec_glob(f"{VAL_DOCS}/*.parquet"):
            t = pq.read_table(path, columns=["id", "text"])
            val_text.update(zip(t.column("id").to_pylist(), t.column("text").to_pylist(), strict=True))
        _VAL_TEXT = val_text
    assert _PAIRS is not None and _VAL_TEXT is not None
    return _PAIRS, _VAL_TEXT


def _shingles(text: str, n: int = 5) -> set[str]:
    cleaned = _WS.sub(" ", text.lower()).strip()
    return {cleaned[i : i + n] for i in range(max(1, len(cleaned) - n + 1))}


def _verify_shard(shard_path: str) -> Iterator[dict]:
    pairs_by_other, val_text = _load_state()
    pf = pq.ParquetFile(shard_path)
    for batch in pf.iter_batches(columns=["id", "text"], batch_size=2048):
        for doc_id, text in zip(batch.column("id"), batch.column("text"), strict=True):
            id_str = doc_id.as_py()
            val_ids = pairs_by_other.get(id_str)
            if not val_ids:
                continue
            other_shingles = _shingles(text.as_py())
            counters.increment("verify/candidates")
            for val_id in val_ids:
                val_doc = val_text.get(val_id)
                if val_doc is None:
                    continue
                vs = _shingles(val_doc)
                union = len(vs | other_shingles)
                jac = len(vs & other_shingles) / union if union else 0.0
                if jac >= MIN_REPORT_JACCARD:
                    counters.increment("verify/reported")
                    yield {"val_id": val_id, "other_id": id_str, "jaccard": round(jac, 4)}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    shards = sorted(fsspec_glob(f"{NORM}/*.parquet"))
    ctx = ZephyrContext(
        name="verify-val-pairs",
        max_workers=231,
        resources=ResourceConfig(cpu=2, ram="16g", disk="10g"),
        coordinator_resources=ResourceConfig(cpu=2, ram="16g", disk="10g"),
    )
    pipeline = (
        Dataset.from_list(shards)
        .flat_map(_verify_shard)
        .write_parquet(f"{OUT}/verified-{{shard:05d}}-of-{{total:05d}}.parquet")
    )
    outcome = ctx.execute(pipeline)
    logger.info("counters: %s", dict(outcome.counters))


if __name__ == "__main__":
    main()
