# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit Testbed negative-control variant — exact-row duplication.

Designed to validate the dedup arm by injecting a known-bad transformation:
take each per-source sampled parquet bucket and rewrite it so a configurable
fraction of rows are exact duplicates of other rows in the same bucket.

Per-shard recipe (rate ``r`` in ``[0.0, 1.0)``):

* Rows in the input shard: ``N``
* Unique rows kept (the "pool"): ``u = ceil((1 - r) * N)`` — the first ``u``
  rows in shard order (deterministic, matches the sampler's first-K rule).
* Output: write the pool repeatedly until ``N`` rows are emitted, truncating
  the final pass. The output bucket has the same row count as the input so
  the proportional mixture weights computed by ``weights_from_tokenized_bucket_stats``
  are unchanged relative to baseline — isolating the duplication effect from
  any mixture shift.

Resulting fraction of duplicate rows ≈ ``r``. Default ``DEFAULT_DUP_RATE = 0.5``
is a max-signal control; lower rates (e.g. 0.25) better mirror real
dedup-removed fractions.

Pipeline mirrors ``baseline.py``: ``ferry → duplicate → tokenize → train``.
"""

from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from fray import ResourceConfig
from rigging.filesystem import marin_prefix, url_to_fs
from rigging.log_setup import configure_logging

from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import Artifact
from marin.execution.executor import Executor, ExecutorMainConfig, ExecutorStep, executor_main
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.utils import fsspec_glob, fsspec_mkdirs

from experiments.datakit_testbed.mixture import weights_from_tokenized_bucket_stats
from experiments.datakit_testbed.sampler import build_testbed_steps
from experiments.datakit_testbed.settings import TESTBED_TOKENIZER
from experiments.datakit_testbed.train import run_testbed_config, testbed_tokenize

logger = logging.getLogger(__name__)

STAGING_PREFIX = "gs://marin-us-central1"
TARGET_TOTAL_TOKENS_B = 1000.0
MAX_STEP_CONCURRENCY = 20

DEFAULT_DUP_RATE = 0.5

_SAMPLE_STEP_PREFIX = "data/datakit/normalized/"
# Streaming algorithm caps memory at ~one row group per thread; 4 threads × a
# few hundred MB decoded = headroom inside 12g. Earlier 32-thread + materialized
# pool blew a 5g cap (Exit 137 OOM on every per-source duplicate task).
_DUP_REMOTE_RESOURCES = ResourceConfig(cpu=1, ram="12g")
_DUP_PARALLELISM = 4


def _part_name(idx: int, total: int) -> str:
    """``part-{idx}-of-{total}.parquet`` — matches normalize/sampler convention."""
    return f"part-{idx:05d}-of-{total:05d}.parquet"


def _duplicate_shard(src: str, dst: str, dup_rate: float) -> tuple[int, int]:
    """Read *src* parquet, write *dst* with same row count and ``dup_rate`` dups.

    Streams row groups one at a time — never holds more than a single decoded
    row group in memory. Achieves the desired duplication rate by re-opening
    the source and replaying the first ``unique_n`` rows on each pass until
    ``rows_in`` output rows are written. Per-pass IO is bounded by
    ``unique_n`` rows; total bytes read across all passes is roughly the
    input shard size (each output row is read from input exactly once).

    Returns ``(rows_in, rows_out)``. ``rows_out`` equals ``rows_in`` by
    construction.
    """
    src_fs, src_path = url_to_fs(src)
    dst_fs, dst_path = url_to_fs(dst)
    parent = os.path.dirname(dst_path)
    if parent:
        fsspec_mkdirs(parent, exist_ok=True)

    # Peek at metadata once to compute the per-pass budget and capture schema.
    with src_fs.open(src_path, "rb") as sf:
        pf = pq.ParquetFile(sf)
        rows_in = pf.metadata.num_rows
        schema = pf.schema_arrow

    unique_n = max(1, math.ceil((1.0 - dup_rate) * rows_in))
    unique_n = min(unique_n, rows_in)

    rows_out = 0
    with dst_fs.open(dst_path, "wb") as df, pq.ParquetWriter(df, schema) as writer:
        while rows_out < rows_in:
            with src_fs.open(src_path, "rb") as sf:
                pf = pq.ParquetFile(sf)
                rows_this_pass = 0
                for i in range(pf.num_row_groups):
                    if rows_this_pass >= unique_n or rows_out >= rows_in:
                        break
                    rg = pf.read_row_group(i)
                    if rows_this_pass + rg.num_rows > unique_n:
                        rg = rg.slice(0, unique_n - rows_this_pass)
                    output_remaining = rows_in - rows_out
                    if rg.num_rows > output_remaining:
                        rg = rg.slice(0, output_remaining)
                    writer.write_table(rg)
                    rows_out += rg.num_rows
                    rows_this_pass += rg.num_rows

    logger.info(
        "neg_control: %s → %s rows_in=%d unique_pool=%d rows_out=%d (dup_rate=%.2f)",
        src,
        dst,
        rows_in,
        unique_n,
        rows_out,
        dup_rate,
    )
    return rows_in, rows_out


def duplicate_normalized_shards(
    *,
    source: NormalizedData,
    output_path: str,
    dup_rate: float,
) -> NormalizedData:
    """Duplicate rows within each shard of ``source`` to inject ``dup_rate`` dups.

    Each input shard becomes one output shard with the same row count, written
    under ``{output_path}/outputs/main`` so the downstream tokenize glob picks
    it up unchanged.

    Args:
        source: Upstream sampler output (``NormalizedData`` with parquet shards).
        output_path: Step output root.
        dup_rate: Target fraction of duplicate rows in ``[0.0, 1.0)``.

    Returns:
        A fresh ``NormalizedData`` pointing at the duplicated directory.

    Raises:
        ValueError: If ``dup_rate`` is out of range or no shards found.
    """
    if not 0.0 <= dup_rate < 1.0:
        raise ValueError(f"dup_rate must be in [0.0, 1.0); got {dup_rate}")

    input_base = source.main_output_dir.rstrip("/")
    shards = sorted(fsspec_glob(f"{input_base}/**/*.parquet"))
    if not shards:
        raise ValueError(f"No parquet shards under {input_base}")
    total = len(shards)
    main_out = f"{output_path.rstrip('/')}/outputs/main"
    logger.info("neg_control: %d shards under %s, dup_rate=%.2f", total, input_base, dup_rate)

    tasks = [(s, f"{main_out}/{_part_name(i, total)}", dup_rate) for i, s in enumerate(shards)]
    rows_in_total = 0
    rows_out_total = 0
    with ThreadPoolExecutor(max_workers=_DUP_PARALLELISM) as pool:
        for ri, ro in pool.map(lambda args: _duplicate_shard(*args), tasks):
            rows_in_total += ri
            rows_out_total += ro

    logger.info(
        "neg_control: done %d shards rows_in=%d rows_out=%d (delta=%d)",
        total,
        rows_in_total,
        rows_out_total,
        rows_out_total - rows_in_total,
    )
    return NormalizedData(
        main_output_dir=main_out,
        dup_output_dir=source.dup_output_dir,
        counters={
            "neg_control/shards": total,
            "neg_control/rows_in": rows_in_total,
            "neg_control/rows_out": rows_out_total,
        },
    )


def _duplicate_step(src_name: str, sampled: StepSpec, dup_rate: float) -> StepSpec:
    """Per-source row-duplication step. Same shard count + row count as input."""
    sampled_path = sampled.output_path

    def duplicate(output_path: str) -> NormalizedData:
        return duplicate_normalized_shards(
            source=Artifact.load(sampled_path, NormalizedData),
            output_path=output_path,
            dup_rate=dup_rate,
        )

    return StepSpec(
        name=f"data/datakit/duplicated/{src_name}",
        deps=[sampled],
        hash_attrs={"dup_rate": dup_rate},
        fn=remote(duplicate, resources=_DUP_REMOTE_RESOURCES),
    )


def neg_control(
    steps: list[StepSpec],
    *,
    name: str,
    tokenizer: str,
    dup_rate: float = DEFAULT_DUP_RATE,
) -> ExecutorStep:
    """Assemble the negative-control training step off a testbed DAG.

    Pipeline:
      ``sample → duplicate(dup_rate) → tokenize → mixture-weighted train``

    Bucket sizes are preserved by the duplication step, so the mixture weights
    derived from per-bucket token counts match baseline's — this run differs
    from baseline only in the *content* (half repeated) of each bucket.
    """
    sampled_by_source = {
        s.name.removeprefix(_SAMPLE_STEP_PREFIX): s for s in steps if s.name.startswith(_SAMPLE_STEP_PREFIX)
    }
    if not sampled_by_source:
        raise ValueError("no sample steps found in the DAG (expected names under 'data/datakit/normalized/...')")

    duplicated_by_source = {
        src_name: _duplicate_step(src_name, sampled, dup_rate)
        for src_name, sampled in sampled_by_source.items()
    }

    logger.info(
        "neg_control variant %s: %d sources → duplicate(rate=%.2f) → tokenize → train",
        name,
        len(sampled_by_source),
        dup_rate,
    )

    StepRunner().run(list(duplicated_by_source.values()), max_concurrent=MAX_STEP_CONCURRENCY)

    tokenized_buckets = {
        src_name: testbed_tokenize(src_name, dup, tokenizer) for src_name, dup in duplicated_by_source.items()
    }
    prefix = marin_prefix()
    tokenize_executor = Executor(
        prefix=prefix,
        executor_info_base_path=os.path.join(prefix, "experiments"),
    )
    tokenize_executor.run(list(tokenized_buckets.values()), max_concurrent=MAX_STEP_CONCURRENCY)

    resolved_output_paths = {
        bucket_name: tokenize_executor.output_paths[step] for bucket_name, step in tokenized_buckets.items()
    }
    weights = weights_from_tokenized_bucket_stats(resolved_output_paths)
    return run_testbed_config(
        name=name,
        tokenized_buckets=tokenized_buckets,
        weights=weights,
        tokenizer=tokenizer,
    )


def main() -> None:
    """Entry-point: ferry → duplicate → tokenize → train."""
    os.environ["MARIN_PREFIX"] = STAGING_PREFIX

    tokenizer = TESTBED_TOKENIZER
    dup_rate = DEFAULT_DUP_RATE
    run_id = f"neg_control_dup{int(round(dup_rate * 100))}"

    testbed_steps = build_testbed_steps(target_total_tokens_b=TARGET_TOTAL_TOKENS_B)
    logger.info("Materializing %d ferry StepSpecs under %s", len(testbed_steps), STAGING_PREFIX)
    StepRunner().run(testbed_steps, max_concurrent=MAX_STEP_CONCURRENCY)

    training_step = neg_control(testbed_steps, name=run_id, tokenizer=tokenizer, dup_rate=dup_rate)
    executor_main(ExecutorMainConfig(), [training_step])


if __name__ == "__main__":
    configure_logging()
    main()
