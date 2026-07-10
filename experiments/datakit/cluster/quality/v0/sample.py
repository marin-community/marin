# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample documents from every active Datakit source.

For each source in :func:`marin.datakit.sources.all_sources` (minus the
standard ``safety_pt/*`` / ``climblab-ja`` carve-outs), allocates a
per-source quota using ``floor + extra * sqrt(tokens) / Σ sqrt(tokens)``,
opens one randomly-chosen shard of the normalized parquet output, reads
the first ``quota`` rows from it, and writes everything to a single
output parquet:

    source         : string  -- registry name (e.g. "cp/biodiversity")
    id             : string  -- canonical document id
    partition_id   : int64   -- source shard index
    text           : string  -- raw text, capped to MAX_DOC_CHARS

The floor + sqrt allocation compresses the ~1e6x per-source token-count
skew without starving small sources -- every active source contributes at
least ``floor_per_source`` documents, and the top sources contribute
~20-30x the floor rather than 1e6x. Picking one random shard per source
is unbiased because datakit normalize hash-partitions documents across
shards (every shard is a uniform random subset of its source).

Run from anywhere that can read ``gs://marin-eu-west4/datakit/...``:

    uv run python -m experiments.datakit.cluster.quality.v0.sample \\
        --output gs://marin-eu-west4/datakit/llm-quality-classifier/samples/train-n7000-seed42.parquet \\
        --total-size 7000 --floor-per-source 20 --seed 42
"""

import argparse
import logging
import math
import random
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq
from marin.datakit.normalize import NormalizedData
from marin.datakit.sources import all_sources
from marin.execution.artifact import read_artifact
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)


# Per-document text cap stored in the sample parquet. 32 KB is ~8x the
# oracle's 4 KB cut-off and ~8x train.py's default truncation; no
# downstream consumer reads more, so storing more just bloats memory
# during the final pa.Table.from_pylist conversion (which doubles peak
# usage for the duration of the alloc).
MAX_DOC_CHARS = 32_000

# Default per-source IO parallelism. The scaling cost is one
# decompressed parquet row group per concurrent worker -- and big
# sources (cp/stackv2_code, finepdfs, nemotron_cc_v2/*) have row groups
# of 200-500 MB uncompressed. 4 concurrent workers were enough to OOM a
# 2 GB driver; 1 worker keeps the peak bounded to a single row group
# (~500 MB worst case) and costs ~100 s wall-clock on 104 sources --
# acceptable for a once-per-run job. Override via --num-workers if the
# driver has more RAM headroom.
DEFAULT_NUM_WORKERS = 1

_OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("source", pa.string()),
        pa.field("id", pa.string()),
        pa.field("partition_id", pa.int64()),
        pa.field("text", pa.string()),
    ]
)


@dataclass(frozen=True)
class SourceQuota:
    name: str
    quota: int
    rough_tokens_b: float


def _active_sources() -> dict[str, float]:
    """Return ``{name: rough_token_count_b}`` for the active sources."""
    return {name: src.rough_token_count_b for name, src in all_sources().items()}


def compute_quotas(
    sources: dict[str, float],
    total_size: int,
    floor_per_source: int,
) -> list[SourceQuota]:
    """Allocate quotas via ``floor + extra ∝ sqrt(tokens)``.

    Each source first gets ``floor_per_source`` documents; the remainder
    of ``total_size`` is distributed in proportion to ``sqrt(tokens)``.
    The sqrt compresses the ~1e6x per-source skew without starving any
    one source. Floors larger than the largest sqrt slice keep this
    monotone in token count even at small ``total_size`` (when extra=0
    the result degenerates cleanly to uniform).
    """
    floor_total = floor_per_source * len(sources)
    if total_size < floor_total:
        raise ValueError(
            f"total_size={total_size} < floor_per_source*n_sources={floor_total}; "
            "lower floor_per_source or raise total_size"
        )
    extra_budget = total_size - floor_total

    sqrt_weights = {name: math.sqrt(max(tokens, 0.0)) for name, tokens in sources.items()}
    sqrt_sum = sum(sqrt_weights.values()) or 1.0

    quotas: list[SourceQuota] = []
    allocated_extra = 0
    items = sorted(sources.items())
    for i, (name, tokens) in enumerate(items):
        if i < len(items) - 1:
            extra = round(extra_budget * sqrt_weights[name] / sqrt_sum)
            allocated_extra += extra
        else:
            # Final source soaks up rounding remainder so quotas sum to total_size exactly.
            extra = extra_budget - allocated_extra
        quotas.append(SourceQuota(name=name, quota=floor_per_source + extra, rough_tokens_b=tokens))
    return quotas


def _list_shards(input_dir: str) -> list[str]:
    files: list[str] = []
    for root, _dirs, entries in StoragePath(input_dir).walk():
        for fname in entries:
            if not fname.endswith(".parquet") or fname.startswith("."):
                continue
            files.append(str(root / fname))
    files.sort()
    return files


def _read_quota_from_shard(shard_url: str, quota: int) -> Iterator[dict[str, object]]:
    """Yield up to ``quota`` rows from a single parquet shard.

    Reads row groups one at a time, breaking out as soon as ``quota``
    rows have been emitted. This bounds peak memory to one row group
    worth of text -- critical for large sources like cp/stackv2_code
    where a whole shard is multi-GB. Datakit normalize hash-partitions
    documents (both across and within shards), so the rows we see are
    still a uniform random subset of the source.
    """
    with StoragePath(shard_url).open("rb") as fh:
        pfile = pq.ParquetFile(fh)
        yielded = 0
        for rg_idx in range(pfile.num_row_groups):
            for batch in pfile.iter_batches(
                columns=["id", "partition_id", "text"],
                batch_size=256,
                row_groups=[rg_idx],
            ):
                ids = batch.column("id").to_pylist()
                pids = batch.column("partition_id").to_pylist()
                texts = batch.column("text").to_pylist()
                for doc_id, pid, text in zip(ids, pids, texts, strict=True):
                    if not text:
                        continue
                    yield {
                        "id": str(doc_id),
                        "partition_id": int(pid),
                        "text": text[:MAX_DOC_CHARS] if len(text) > MAX_DOC_CHARS else text,
                    }
                    yielded += 1
                    if yielded >= quota:
                        return
            if yielded >= quota:
                return


def _sample_one_source(quota: SourceQuota, seed: int) -> list[dict[str, object]]:
    src = all_sources()[quota.name]
    nd = read_artifact(src.normalized.output_path, NormalizedData)
    shards = _list_shards(nd.main_output_dir)
    if not shards:
        logger.warning("source %s: no shards under %s -- skipping", quota.name, nd.main_output_dir)
        return []

    rng = random.Random(seed)
    chosen = rng.choice(shards)
    rows = list(_read_quota_from_shard(chosen, quota.quota))
    for row in rows:
        row["source"] = quota.name
    if len(rows) < quota.quota:
        logger.warning(
            "source %s: only read %d / %d docs from %s (shard too small)",
            quota.name,
            len(rows),
            quota.quota,
            chosen,
        )
    else:
        logger.info("source %s: sampled %d docs from %s", quota.name, len(rows), chosen)
    return rows


def _per_source_seed(name: str, base_seed: int) -> int:
    return (hash((name, base_seed)) & 0x7FFF_FFFF) or 1


def sample(
    *,
    output_path: str,
    total_size: int,
    floor_per_source: int,
    seed: int,
    num_workers: int,
) -> None:
    sources = _active_sources()
    quotas = compute_quotas(sources, total_size=total_size, floor_per_source=floor_per_source)
    quotas.sort(key=lambda q: -q.quota)
    logger.info(
        "computed quotas for %d sources -- min=%d max=%d sum=%d",
        len(quotas),
        min(q.quota for q in quotas),
        max(q.quota for q in quotas),
        sum(q.quota for q in quotas),
    )
    for q in quotas[:5]:
        logger.info("  top: %s -> %d docs (%.2f B tokens)", q.name, q.quota, q.rough_tokens_b)
    for q in quotas[-5:]:
        logger.info("  bot: %s -> %d docs (%.4f B tokens)", q.name, q.quota, q.rough_tokens_b)

    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futs = {pool.submit(_sample_one_source, q, _per_source_seed(q.name, seed)): q for q in quotas}
        for fut in as_completed(futs):
            q = futs[fut]
            try:
                rows.extend(fut.result())
            except Exception:
                logger.exception("source %s: failed to sample", q.name)

    logger.info("sampled %d docs across %d sources -> %s", len(rows), len(quotas), output_path)
    table = pa.Table.from_pylist(rows, schema=_OUTPUT_SCHEMA)
    output = StoragePath(output_path)
    if output.parent.segments:
        output.parent.mkdirs()
    with output.open("wb") as fh:
        pq.write_table(table, fh, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="GCS path for the sampled parquet")
    parser.add_argument("--total-size", type=int, default=7000)
    parser.add_argument("--floor-per-source", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    args = parser.parse_args()

    configure_logging(logging.INFO)
    sample(
        output_path=args.output,
        total_size=args.total_size,
        floor_per_source=args.floor_per_source,
        seed=args.seed,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
