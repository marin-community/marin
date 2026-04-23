# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Post-normalize by-provenance sampler for the Datakit Testbed.

For each source, copy a subset of its *normalized* parquet shards.

Why post-normalize rather than pre:

* Normalize already targets ``target_partition_bytes`` per output shard, so
  the post-normalize shards are **uniform in size**. "First K by filename"
  becomes byte-fair by construction — no hash ordering needed.
* Normalize's ``group_by`` redistributes records across shards by
  ``hash(id)``, so output shards are content-decorrelated from input
  ordering. First-K is also content-fair.
* Row counts are exact (parquet footer), so in-shard row-level targets are
  possible later if we want finer granularity.

Design choices:

* **Deterministic, no RNG.** Sort by filename, take first K. Reproducible
  across ferry reruns.
* **Copy, not manifest.** GCS has no symlinks; a manifest would force a
  downstream API change. Intra-region GCS copy has no network egress — only
  the storage of the sampled subset.
* **Fraction per source** is computed upstream from ``rough_token_count_b``
  against ``RAW_TARGET_TOTAL_TOKENS_B`` via :func:`proportional_sample_fractions`.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from rigging.filesystem import url_to_fs

from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import Artifact
from marin.execution.step_spec import StepSpec
from marin.utils import fsspec_glob, fsspec_mkdirs

from marin.datakit.sources import DatakitSource

logger = logging.getLogger(__name__)

_COPY_PARALLELISM = 32


def proportional_sample_fractions(
    sources: Sequence[DatakitSource],
    target_total_tokens_b: float,
) -> dict[str, float]:
    """Per-source ``sample_fraction`` to hit ``target_total_tokens_b``.

    Each source's ``rough_token_count_b`` determines its share of the
    target; the fraction is ``target_share / its own count``, clamped to
    ``[0.0, 1.0]`` so a source whose target exceeds its known count
    simply contributes all of itself.
    """
    total_count = sum(s.rough_token_count_b for s in sources)
    return {
        src.name: min(1.0, target_total_tokens_b * (src.rough_token_count_b / total_count) / src.rough_token_count_b)
        for src in sources
    }


def _copy_shard(src: str, dst: str) -> int:
    """Copy a single file server-side. Both paths must share a backend."""
    src_fs, src_path = url_to_fs(src)
    dst_fs, dst_path = url_to_fs(dst)
    assert (
        src_fs.protocol == dst_fs.protocol
    ), f"sampler: src/dst filesystem mismatch: {src_fs.protocol!r} vs {dst_fs.protocol!r}. "
    parent = os.path.dirname(dst_path)
    if parent:
        fsspec_mkdirs(parent, exist_ok=True)
    src_fs.copy(src_path, dst_path)
    size = int(src_fs.size(src_path) or 0)
    assert size > 0, f"sampler: source shard has zero size: {src}"
    return size


def _sample_rows_within_shard(src: str, dst: str, sample_fraction: float) -> tuple[int, int]:
    """Read *src* parquet, take the first ``ceil(rows * sample_fraction)`` rows, write to *dst*.

    Returns ``(rows_in, rows_out)``. First-K is deterministic (no RNG) and
    matches the cross-shard "first K by filename" selection rule.
    """
    src_fs, src_path = url_to_fs(src)
    dst_fs, dst_path = url_to_fs(dst)
    parent = os.path.dirname(dst_path)
    if parent:
        fsspec_mkdirs(parent, exist_ok=True)

    with src_fs.open(src_path, "rb") as sf:
        pf = pq.ParquetFile(sf)
        rows_in = pf.metadata.num_rows
        rows_out = max(1, math.ceil(rows_in * sample_fraction))
        rows_out = min(rows_out, rows_in)
        table = pf.read().slice(0, rows_out)

    with dst_fs.open(dst_path, "wb") as df:
        pq.write_table(table, df)

    return rows_in, rows_out


def sample_normalized_shards(
    *,
    source: NormalizedData,
    output_path: str,
    sample_fraction: float,
) -> NormalizedData:
    """Copy the first ``K = ceil(N * sample_fraction)`` normalized shards or rows if single-shard.

    Shards are enumerated under ``source.main_output_dir``, sorted
    lexicographically, and the first ``K`` copied to ``{output_path}/outputs/main/``
    preserving the relative path so the co-partition invariant downstream is
    satisfied. Normalize writes ``{parent}/outputs/main/part-*.parquet`` — we
    mirror that layout.

    Args:
        source: Upstream normalize output (an ``Artifact.load(step, NormalizedData)``).
        output_path: Step output root; the new ``main_output_dir`` becomes
            ``{output_path}/outputs/main``.
        sample_fraction: Fraction of shards to keep, in ``(0.0, 1.0]``. Rounded
            up to at least 1 shard.

    Returns:
        A fresh ``NormalizedData`` pointing at the sampled directory. The
        ``dup_output_dir`` is passed through unchanged (it's normalize's
        exact-dedup side output — sampling doesn't touch it).

    Raises:
        ValueError: If ``sample_fraction`` is out of range or no shards found.
    """
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError(f"sample_fraction must be in (0.0, 1.0]; got {sample_fraction}")

    input_base = source.main_output_dir.rstrip("/")
    shards = sorted(fsspec_glob(f"{input_base}/**/*.parquet"))
    if not shards:
        raise ValueError(f"No parquet shards under {input_base}")

    total = len(shards)
    main_out = f"{output_path.rstrip('/')}/outputs/main"

    # Single-shard sources: "first K shards by filename" degenerates to copying
    # the whole shard regardless of sample_fraction. Fall back to row-level
    # sampling within the shard so small sources still honour the fraction.
    if total == 1 and sample_fraction < 1.0:
        src = shards[0]
        rel = os.path.relpath(src, input_base)
        dst = f"{main_out}/{rel}"
        rows_in, rows_out = _sample_rows_within_shard(src, dst, sample_fraction)
        logger.info(
            "sampler: single-shard source — sampled %d / %d rows (fraction=%.4f) from %s",
            rows_out,
            rows_in,
            sample_fraction,
            input_base,
        )
        return NormalizedData(
            main_output_dir=main_out,
            dup_output_dir=source.dup_output_dir,
            counters={
                "sampler/single_shard_rows_in": rows_in,
                "sampler/single_shard_rows_out": rows_out,
                "sampler/total_shards": 1,
            },
        )

    k = max(1, math.ceil(total * sample_fraction))
    k = min(k, total)
    selected = shards[:k]
    logger.info(
        "sampler: selecting %d / %d normalized shards (fraction=%.4f) from %s",
        k,
        total,
        sample_fraction,
        input_base,
    )

    tasks: list[tuple[str, str]] = []
    for shard in selected:
        rel = os.path.relpath(shard, input_base)
        tasks.append((shard, f"{main_out}/{rel}"))

    total_bytes = 0
    with ThreadPoolExecutor(max_workers=_COPY_PARALLELISM) as pool:
        for nbytes in pool.map(lambda args: _copy_shard(*args), tasks):
            total_bytes += nbytes
    logger.info("sampler: copied %d shards, %.1f GiB total", k, total_bytes / (1024**3))

    return NormalizedData(
        main_output_dir=main_out,
        dup_output_dir=source.dup_output_dir,
        counters={"sampler/selected_shards": k, "sampler/total_shards": total},
    )


def sample_normalized_shards_step(
    *,
    name: str,
    normalized: StepSpec,
    sample_fraction: float,
    override_output_path: str | None = None,
) -> StepSpec:
    """Create a StepSpec that samples ``normalized``'s output shards.

    The step loads the upstream ``NormalizedData`` artifact at execution time,
    so the sampled shard set reflects whatever normalize actually emitted.
    """
    return StepSpec(
        name=name,
        deps=[normalized],
        hash_attrs={"sample_fraction": sample_fraction},
        fn=lambda output_path: sample_normalized_shards(
            source=Artifact.load(normalized, NormalizedData),
            output_path=output_path,
            sample_fraction=sample_fraction,
        ),
        override_output_path=override_output_path,
    )
