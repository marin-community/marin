# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit Testbed negative-control duplication arm (#5310 / #8806).

Injects exact-row duplication while keeping per-shard row counts — and therefore
mixture weights — identical to baseline. For each input shard of ``N`` rows:

    unique_n = ceil(unique_fraction * N)
    output   = first ``unique_n`` rows, replayed from the start until ``N`` rows

``unique_fraction=0.50`` is the original #5310 recipe (``dup_rate = 0.50``).
``unique_fraction=1.0`` is a row-level identity. Training HPs are the same
Grug-MoE / MuonH protocol as :mod:`experiments.datakit.testbed.baseline`.

Submit in the staging region::

    uv run iris --cluster=marin job run --region us-central1 -- \\
        python experiments/datakit/testbed/neg_control.py --unique-fraction 0.25
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem.cluster_config import check_path_in_region, marin_prefix
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

from experiments.datakit.testbed.mixture import tokenized_bucket_weights_step
from experiments.datakit.testbed.sampler import build_testbed_steps
from experiments.datakit.testbed.settings import TESTBED_STAGING_PREFIX, TESTBED_STAGING_REGION, TESTBED_TOKENIZER
from experiments.datakit.testbed.train import run_testbed_config, testbed_tokenize
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.uncheatable import uncheatable_datasets

logger = logging.getLogger(__name__)

TARGET_TOTAL_TOKENS_B = 1000.0
MAX_STEP_CONCURRENCY = 20

_SAMPLE_STEP_PREFIX = "data/datakit/normalized/"
# After #5310 OOMed at 5g / 32 threads, the unique-pool pass uses a small pool
# and 12g so a mapper-sized shard's prefix can be re-read without materializing
# the whole file 32 ways at once.
_DUP_REMOTE_RESOURCES = ResourceConfig(cpu=1, ram="12g")
_DUP_PARALLELISM = 4

T = TypeVar("T")


def _check_unique_fraction(unique_fraction: float) -> None:
    if not 0.0 < unique_fraction <= 1.0:
        raise ValueError(f"unique_fraction must be in (0.0, 1.0]; got {unique_fraction}")


def unique_row_count(num_rows: int, unique_fraction: float) -> int:
    """``ceil(unique_fraction * N)`` — the unique-pool size for a shard of ``N`` rows."""
    _check_unique_fraction(unique_fraction)
    if num_rows < 0:
        raise ValueError(f"num_rows must be nonnegative; got {num_rows}")
    return math.ceil(unique_fraction * num_rows)


def first_k_replay(rows: Sequence[T], unique_fraction: float) -> list[T]:
    """Replay the first ``unique_n`` rows until the output length equals ``len(rows)``.

    ``dup_rate`` is ``1 - unique_fraction``. For ``unique_fraction=0.5`` this is
    the #5310 per-shard recipe: keep the first half (ceil), drop the tail, then
    replay the kept prefix until the original length is restored.
    """
    n = len(rows)
    unique_n = unique_row_count(n, unique_fraction)
    if n == 0:
        return []
    if unique_n == 0:
        raise ValueError(f"unique pool is empty for unique_fraction={unique_fraction} and N={n}")
    pool = list(rows[:unique_n])
    out: list[T] = []
    while len(out) < n:
        out.extend(pool[: n - len(out)])
    return out


def _append_prefix_rows(src: str, writer: pq.ParquetWriter, num_rows: int) -> None:
    """Append the first ``num_rows`` rows of ``src`` onto ``writer``."""
    if num_rows <= 0:
        return
    src_fs, src_path = url_to_fs(src)
    remaining = num_rows
    with src_fs.open(src_path, "rb") as sf:
        pf = pq.ParquetFile(sf)
        for i in range(pf.num_row_groups):
            if remaining <= 0:
                break
            rg = pf.read_row_group(i)
            if rg.num_rows > remaining:
                rg = rg.slice(0, remaining)
            writer.write_table(rg)
            remaining -= rg.num_rows
    if remaining != 0:
        raise ValueError(f"expected {num_rows} prefix rows from {src}, fell short by {remaining}")


def duplicate_parquet_shard(src: str, dst: str, unique_fraction: float) -> tuple[int, int]:
    """Rewrite ``src`` with first-K replay; return ``(rows_in, unique_n)``.

    Output has exactly ``rows_in`` rows. The unique pool is the file prefix of
    length ``unique_n``; later passes re-read that prefix rather than holding
    the whole shard.
    """
    src_fs, src_path = url_to_fs(src)
    with src_fs.open(src_path, "rb") as sf:
        pf = pq.ParquetFile(sf)
        rows_in = pf.metadata.num_rows
        schema = pf.schema_arrow
    unique_n = unique_row_count(rows_in, unique_fraction)

    dst_fs, dst_path = url_to_fs(dst)
    parent = os.path.dirname(dst_path)
    if parent:
        StoragePath(parent).mkdirs(exist_ok=True)

    remaining = rows_in
    with dst_fs.open(dst_path, "wb") as df, pq.ParquetWriter(df, schema) as writer:
        while remaining > 0:
            take = min(unique_n, remaining)
            _append_prefix_rows(src, writer, take)
            remaining -= take
    return rows_in, unique_n


def duplicate_normalized_shards(
    *,
    source: NormalizedData,
    output_path: str,
    unique_fraction: float,
) -> NormalizedData:
    """Apply first-K replay to every parquet shard under ``source.main_output_dir``.

    Relative shard names are preserved so tokenize still globs
    ``{output_path}/outputs/main/*.parquet``. Each output shard has the same
    row count as its input, so per-bucket token counts stay baseline-shaped.
    """
    _check_unique_fraction(unique_fraction)
    input_base = source.main_output_dir.rstrip("/")
    shards = sorted(str(m) for m in StoragePath(f"{input_base}/**/*.parquet").glob())
    if not shards:
        raise ValueError(f"No parquet shards under {input_base}")
    main_out = f"{output_path.rstrip('/')}/outputs/main"
    prefix_len = len(input_base)

    tasks: list[tuple[str, str]] = []
    for src in shards:
        rel = src[prefix_len:].lstrip("/")
        tasks.append((src, f"{main_out}/{rel}"))

    rows_in_total = 0
    unique_n_total = 0
    dup_rate = 1.0 - unique_fraction
    logger.info(
        "neg_control: unique_fraction=%.4f (dup_rate=%.4f) on %d shards %s → %s",
        unique_fraction,
        dup_rate,
        len(tasks),
        input_base,
        main_out,
    )
    with ThreadPoolExecutor(max_workers=_DUP_PARALLELISM) as pool:
        for rows_in, unique_n in pool.map(lambda args: duplicate_parquet_shard(*args, unique_fraction), tasks):
            rows_in_total += rows_in
            unique_n_total += unique_n

    return NormalizedData(
        main_output_dir=main_out,
        dup_output_dir=source.dup_output_dir,
        counters={
            "neg_control/shards": len(tasks),
            "neg_control/rows_in": rows_in_total,
            "neg_control/rows_out": rows_in_total,
            "neg_control/unique_rows": unique_n_total,
            "neg_control/unique_fraction": unique_fraction,
        },
    )


def duplicate_normalized_shards_step(
    *,
    name: str,
    sampled: StepSpec,
    unique_fraction: float,
) -> StepSpec:
    """StepSpec: first-K replay over one sampled source."""
    _check_unique_fraction(unique_fraction)
    sampled_path = sampled.output_path

    def duplicate(output_path: str) -> NormalizedData:
        return duplicate_normalized_shards(
            source=read_artifact(sampled_path, NormalizedData),
            output_path=output_path,
            unique_fraction=unique_fraction,
        )

    return StepSpec(
        name=name,
        deps=[sampled],
        hash_attrs={"unique_fraction": unique_fraction},
        fn=remote(duplicate, resources=_DUP_REMOTE_RESOURCES),
    )


def neg_control_run_id(unique_fraction: float) -> str:
    """Stable training-step / wandb name, e.g. ``neg_control_unique50``."""
    _check_unique_fraction(unique_fraction)
    return f"neg_control_unique{round(unique_fraction * 100):02d}"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Datakit Testbed first-K duplication arm")
    parser.add_argument(
        "--unique-fraction",
        type=float,
        required=True,
        help="Fraction of each shard kept as the unique pool in (0, 1]. "
        "0.50 reproduces #5310; 0.25/0.10/0.05 are the #8806 follow-ups.",
    )
    args = parser.parse_args(argv)
    unique_fraction: float = args.unique_fraction
    _check_unique_fraction(unique_fraction)

    os.environ.setdefault("MARIN_PREFIX", TESTBED_STAGING_PREFIX)
    check_path_in_region("MARIN_PREFIX", marin_prefix(), TESTBED_STAGING_REGION)

    tokenizer = TESTBED_TOKENIZER
    run_id = neg_control_run_id(unique_fraction)
    validation = [*paloma_datasets(tokenizer=tokenizer).values(), *uncheatable_datasets(tokenizer=tokenizer).values()]

    testbed_steps = build_testbed_steps(target_total_tokens_b=TARGET_TOTAL_TOKENS_B)
    sampled_by_source = {
        s.name.removeprefix(_SAMPLE_STEP_PREFIX): s for s in testbed_steps if s.name.startswith(_SAMPLE_STEP_PREFIX)
    }
    if not sampled_by_source:
        raise ValueError("no sample steps found in the testbed DAG")

    duplicated_by_source = {
        name: duplicate_normalized_shards_step(
            name=f"data/datakit/duplicated/{name}",
            sampled=sampled,
            unique_fraction=unique_fraction,
        )
        for name, sampled in sampled_by_source.items()
    }
    tokenized_buckets = {
        name: testbed_tokenize(name, duplicated, tokenizer) for name, duplicated in duplicated_by_source.items()
    }
    weights_step = tokenized_bucket_weights_step(run_id, tokenized_buckets)
    training_step = run_testbed_config(
        name=run_id,
        tokenized_buckets=tokenized_buckets,
        weights_step=weights_step,
        validation=validation,
        tokenizer=tokenizer,
    )

    logger.info(
        "Neg-control DAG: unique_fraction=%.4f dup_rate=%.4f, %d sources → duplicate → tokenize → weights → train",
        unique_fraction,
        1.0 - unique_fraction,
        len(sampled_by_source),
    )
    StepRunner().run([training_step], max_concurrent=MAX_STEP_CONCURRENCY)


if __name__ == "__main__":
    configure_logging()
    main()
