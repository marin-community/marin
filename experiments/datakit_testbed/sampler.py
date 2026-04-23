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
* **Never reads row data.** Only lists shards (fsspec glob) and invokes the
  filesystem's own copy (server-side on GCS).
* **Fraction per source** is computed upstream from ``rough_token_count_b``
  against ``RAW_TARGET_TOTAL_TOKENS_B`` via :func:`proportional_sample_fractions`.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

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
    """Per-source ``sample_fraction`` to hit a target total across known sources.

    For each source with ``rough_token_count_b`` set, compute a fraction that
    makes its contribution proportional to its share of the known total. Sources
    with unknown counts get ``1.0`` (take everything) and a warning — we can't
    size them and refusing to run the ferry on them would be more surprising.

    Fractions are clamped to ``[0.0, 1.0]`` — if a source's target exceeds its
    known count it simply contributes all of itself.
    """
    known = [s for s in sources if s.rough_token_count_b is not None]
    unknown = [s for s in sources if s.rough_token_count_b is None]

    fractions: dict[str, float] = {}
    if not known:
        logger.warning(
            "sampler: no source has rough_token_count_b set; falling back to "
            "sample_fraction=1.0 for all %d source(s)",
            len(sources),
        )
        for src in sources:
            fractions[src.name] = 1.0
        return fractions

    known_sum = sum(s.rough_token_count_b for s in known)  # type: ignore[misc]
    for src in known:
        target = target_total_tokens_b * (src.rough_token_count_b / known_sum)  # type: ignore[operator]
        fractions[src.name] = min(1.0, target / src.rough_token_count_b)  # type: ignore[operator]
    for src in unknown:
        fractions[src.name] = 1.0

    if unknown:
        logger.warning(
            "sampler: %d source(s) missing rough_token_count_b, sampling 1.0 (take all): %s",
            len(unknown),
            sorted(s.name for s in unknown),
        )
    return fractions


def _copy_shard(src: str, dst: str) -> int:
    """Copy a single file. Server-side within a backend, streamed otherwise."""
    src_fs, src_path = url_to_fs(src)
    dst_fs, dst_path = url_to_fs(dst)
    parent = os.path.dirname(dst_path)
    if parent:
        fsspec_mkdirs(parent, exist_ok=True)
    if src_fs.protocol == dst_fs.protocol:
        src_fs.copy(src_path, dst_path)
        return int(src_fs.size(src_path) or 0)
    size = 0
    with src_fs.open(src_path, "rb") as sf, dst_fs.open(dst_path, "wb") as df:
        while True:
            chunk = sf.read(8 * 1024 * 1024)
            if not chunk:
                break
            df.write(chunk)
            size += len(chunk)
    return size


def sample_normalized_shards(
    *,
    source: NormalizedData,
    output_path: str,
    sample_fraction: float,
) -> NormalizedData:
    """Copy the first ``K = ceil(N * sample_fraction)`` normalized shards.

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

    main_out = f"{output_path.rstrip('/')}/outputs/main"
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
