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
from fray.v2 import ResourceConfig
from rigging.filesystem import url_to_fs

from marin.datakit.normalize import NormalizedData
from marin.datakit.sources import DatakitSource, all_sources
from marin.execution.artifact import Artifact
from marin.execution.remote import remote
from marin.execution.step_runner import check_cache
from marin.execution.step_spec import StepSpec
from marin.utils import fsspec_glob, fsspec_mkdirs

from experiments.datakit_testbed.settings import RAW_TARGET_TOTAL_TOKENS_B

_SAMPLE_REMOTE_RESOURCES = ResourceConfig(cpu=1, ram="5g")

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
    fractions = {
        src.name: min(1.0, target_total_tokens_b * (src.rough_token_count_b / total_count) / src.rough_token_count_b)
        for src in sources
    }
    clamped = sum(1 for f in fractions.values() if f >= 1.0)
    logger.info(
        "sampler: %d sources, total %.1fB tokens, target %.1fB → fractions range [%.4f, %.4f]"
        " (%d sources clamped to 1.0)",
        len(sources),
        total_count,
        target_total_tokens_b,
        min(fractions.values()),
        max(fractions.values()),
        clamped,
    )
    return fractions


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
    logger.info("sampler: row-level sampling %s → %s (fraction=%.4f)", src, dst, sample_fraction)
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
        logger.info("sampler: reading %d/%d rows from %s", rows_out, rows_in, src)
        table = pf.read().slice(0, rows_out)

    with dst_fs.open(dst_path, "wb") as df:
        pq.write_table(table, df)
    logger.info("sampler: wrote %d rows to %s (%d bytes)", rows_out, dst, dst_fs.size(dst_path) or 0)

    return rows_in, rows_out


def _parquet_num_rows(path: str) -> int:
    """Read the parquet footer at *path* and return its row count."""
    fs, resolved = url_to_fs(path)
    with fs.open(resolved, "rb") as f:
        return pq.ParquetFile(f).metadata.num_rows


def sample_normalized_shards(
    *,
    source: NormalizedData,
    output_path: str,
    sample_fraction: float,
) -> NormalizedData:
    """Sample ``source``'s normalized shards to a target row count.

    Probes a single shard for rows-per-file, then computes a total row
    budget ``target_rows = N * rows_per_file * sample_fraction``. Takes
    as many whole shards as fit in that budget (cheap server-side copy)
    and, if there's a row-level remainder, row-samples the next shard
    for the tail. When the budget fits inside one shard, skips the copy
    path entirely and row-samples a single file.

    Shards are enumerated under ``source.main_output_dir``, sorted
    lexicographically, and written to ``{output_path}/outputs/main/``
    preserving the relative path so the co-partition invariant
    downstream is satisfied.

    Args:
        source: Upstream normalize output.
        output_path: Step output root; the new ``main_output_dir``
            becomes ``{output_path}/outputs/main``.
        sample_fraction: Fraction of rows to keep, in ``(0.0, 1.0]``.

    Returns:
        A fresh ``NormalizedData`` pointing at the sampled directory.
        ``dup_output_dir`` is passed through unchanged.

    Raises:
        ValueError: If ``sample_fraction`` is out of range or no shards found.
    """
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError(f"sample_fraction must be in (0.0, 1.0]; got {sample_fraction}")

    input_base = source.main_output_dir.rstrip("/")
    logger.info("sampler: starting (fraction=%.4f) %s → %s", sample_fraction, input_base, output_path)
    shards = sorted(fsspec_glob(f"{input_base}/**/*.parquet"))
    if not shards:
        raise ValueError(f"No parquet shards under {input_base}")
    total = len(shards)
    main_out = f"{output_path.rstrip('/')}/outputs/main"
    logger.info("sampler: discovered %d parquet shards under %s", total, input_base)

    # Fraction==1.0 short-circuit: take everything as-is.
    if sample_fraction >= 1.0:
        logger.info("sampler: fraction=1.0, copying all %d shards", total)
        tasks = [(s, f"{main_out}/{os.path.relpath(s, input_base)}") for s in shards]
        total_bytes = 0
        with ThreadPoolExecutor(max_workers=_COPY_PARALLELISM) as pool:
            for nbytes in pool.map(lambda args: _copy_shard(*args), tasks):
                total_bytes += nbytes
        logger.info("sampler: copied %d shards, %.1f GiB total", total, total_bytes / (1024**3))
        return NormalizedData(
            main_output_dir=main_out,
            dup_output_dir=source.dup_output_dir,
            counters={"sampler/selected_shards": total, "sampler/total_shards": total},
        )

    # Probe one shard for rows-per-file; normalize's target_partition_bytes
    # keeps shards roughly uniform so this is a good estimate.
    rows_per_file = _parquet_num_rows(shards[0])
    est_total_rows = rows_per_file * total
    target_rows = max(1, math.ceil(est_total_rows * sample_fraction))
    logger.info(
        "sampler: probed rows/shard=%d from %s; est total=%d rows, target=%d rows",
        rows_per_file,
        shards[0],
        est_total_rows,
        target_rows,
    )

    # Case 1: target fits in one file — row-sample from the first shard only.
    if target_rows <= rows_per_file:
        src = shards[0]
        dst = f"{main_out}/{os.path.relpath(src, input_base)}"
        row_fraction = target_rows / rows_per_file
        rows_in, rows_out = _sample_rows_within_shard(src, dst, row_fraction)
        logger.info(
            "sampler: target fits in one shard — sampled %d / %d rows from %s",
            rows_out,
            rows_in,
            src,
        )
        return NormalizedData(
            main_output_dir=main_out,
            dup_output_dir=source.dup_output_dir,
            counters={
                "sampler/selected_shards": 1,
                "sampler/total_shards": total,
                "sampler/rows_out": rows_out,
                "sampler/target_rows": target_rows,
            },
        )

    # Case 2: target spans multiple shards — take whole shards for as much
    # as fits, row-sample the next shard for the remainder.
    n_whole = min(target_rows // rows_per_file, total)
    remainder = target_rows - n_whole * rows_per_file
    whole_shards = shards[:n_whole]
    logger.info(
        "sampler: %d whole shards + %d remainder rows (from shard #%d if any)",
        n_whole,
        remainder,
        n_whole,
    )

    tasks = [(s, f"{main_out}/{os.path.relpath(s, input_base)}") for s in whole_shards]
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=_COPY_PARALLELISM) as pool:
        for nbytes in pool.map(lambda args: _copy_shard(*args), tasks):
            total_bytes += nbytes
    logger.info("sampler: copied %d whole shards, %.1f GiB", n_whole, total_bytes / (1024**3))

    rows_out_total = n_whole * rows_per_file
    if remainder > 0 and n_whole < total:
        src = shards[n_whole]
        dst = f"{main_out}/{os.path.relpath(src, input_base)}"
        row_fraction = remainder / rows_per_file
        _, rows_out = _sample_rows_within_shard(src, dst, row_fraction)
        rows_out_total += rows_out

    return NormalizedData(
        main_output_dir=main_out,
        dup_output_dir=source.dup_output_dir,
        counters={
            "sampler/selected_shards": n_whole + (1 if remainder > 0 else 0),
            "sampler/total_shards": total,
            "sampler/rows_out": rows_out_total,
            "sampler/target_rows": target_rows,
        },
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
    Runs remotely via Fray with :data:`_SAMPLE_REMOTE_RESOURCES` so the
    entrypoint doesn't OOM when orchestrating many sources concurrently.
    """
    # Resolve the normalize output path once at factory time so the remote
    # fn's closure only captures strings + a float (cleanly picklable).
    normalized_path = normalized.output_path

    def sample(output_path: str) -> NormalizedData:
        return sample_normalized_shards(
            source=Artifact.load(normalized_path, NormalizedData),
            output_path=output_path,
            sample_fraction=sample_fraction,
        )

    return StepSpec(
        name=name,
        deps=[normalized],
        hash_attrs={"sample_fraction": sample_fraction},
        fn=remote(sample, resources=_SAMPLE_REMOTE_RESOURCES),
        override_output_path=override_output_path,
    )


def _sample_step_for(
    src: DatakitSource,
    normalized: StepSpec,
    sample_fraction: float,
    base: str,
) -> StepSpec:
    """Per-source post-normalize sampler. Copies first ceil(N * fraction) shards."""
    return sample_normalized_shards_step(
        name=f"datakit-testbed/{src.name}",
        normalized=normalized,
        sample_fraction=sample_fraction,
        override_output_path=f"{base}/{src.name}",
    )


def build_testbed_steps(
    run_id: str,
    sources: Sequence[DatakitSource] | None = None,
    target_total_tokens_b: float = RAW_TARGET_TOTAL_TOKENS_B,
) -> list[StepSpec]:
    """Build the full Datakit Testbed ferry DAG.

    Composes the canonical Datakit stages into one multi-source pipeline:
    ``<source.normalize_steps> ─► sample[source]``.

    Each :class:`DatakitSource` already carries its full
    ``(download, ..., normalize)`` :class:`StepSpec` chain; this function
    appends the testbed-specific sample stage on top of every source's
    terminal normalize step. Tokenize runs in the training executor graph
    (see :mod:`experiments.datakit_testbed.train`), not the ferry.

    Args:
        run_id: Per-run identifier; sample output paths land under
            ``datakit_testbed/{run_id}/...``. Normalize outputs land at
            canonical run-independent paths (``normalized/<name>-<hash>``)
            so they're reused across runs.
        sources: DatakitSource list to ferry. ``None`` auto-selects every
            entry from :func:`all_sources` whose normalize output is
            already cached on GCS, matching the run_source_sampling
            script. Pass an explicit list to bypass this check.
        target_total_tokens_b: Target total token count (in billions)
            across the sampled set. Drives per-source sample fractions
            via :func:`proportional_sample_fractions`. Default is
            :data:`RAW_TARGET_TOTAL_TOKENS_B`.

    Returns:
        Flat list of :class:`StepSpec` covering every normalize chain plus
        one sample step per source. Ready to hand to ``StepRunner().run()``.
    """
    if sources is None:
        # TODO (rav): remove the check_cache when ready?
        sources = tuple(s for s in all_sources().values() if check_cache(s.normalized.output_path))
    if not sources:
        raise ValueError("build_testbed_steps requires at least one source")

    base = f"datakit_testbed/{run_id}"
    fractions = proportional_sample_fractions(sources, target_total_tokens_b=target_total_tokens_b)

    # Walk each chain's transitive deps into the iterable — some sources
    # have a ``processed`` step whose own ``raw_download`` dep lives outside
    # ``normalize_steps``. Without that dep in the iterable, StepRunner's
    # dep-satisfaction loop stalls and the run dies with "Iterable
    # exhausted with unsatisfied dependencies". Dedup by ``output_path``
    # so shared family downloads (e.g. Nemotron) land once.
    seen: set[str] = set()
    all_steps: list[StepSpec] = []

    def _visit(step: StepSpec) -> None:
        if step.output_path in seen:
            return
        for dep in step.deps:
            _visit(dep)
        seen.add(step.output_path)
        all_steps.append(step)

    for src in sources:
        for step in src.normalize_steps:
            _visit(step)
        _visit(_sample_step_for(src, src.normalized, fractions[src.name], base))

    logger.info(
        "Built testbed DAG: %d sources, %d steps (normalize chains + sample), target %.0fB tokens",
        len(sources),
        len(all_steps),
        target_total_tokens_b,
    )
    return all_steps
