#!/usr/bin/env python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark row-native and chunk-native Parquet map/write pipelines on real text.

The corpus is derived from ten row groups (10,000 documents) of FineWeb-Edu
``sample/10BT`` at a pinned Hugging Face Parquet conversion commit. Larger
variants repeat the real documents while suffixing ``id`` so routing keys stay
distinct. Corpus preparation is outside the measured region and cached locally.

Each measured arm runs in a fresh subprocess. This isolates allocator state and
makes peak RSS comparable across implementations.

Examples:
    uv run python lib/zephyr/tests/benchmark_chunk_pipeline.py --rows 10000 --modes row
    uv run python lib/zephyr/tests/benchmark_chunk_pipeline.py --rows 100000 --modes row --repeat 5
"""

import gc
import hashlib
import json
import logging
import math
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import click
import fsspec
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SOURCE_COMMIT = "92cece42bcce787ee4af4619ab449fe48d86230d"
_SOURCE_URL = (
    "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/"
    f"{_SOURCE_COMMIT}/sample-10BT/train/0000.parquet"
)
_SOURCE_ROW_GROUPS = tuple(range(10))
_BASE_ROWS = 10_000
_COLUMNS = ("id", "text", "score")
_SCORE_THRESHOLD = 3.0
_DIGEST_BATCH_ROWS = 4096
_RSS_SAMPLE_INTERVAL = 0.005
_RESULT_PREFIX = "RESULT: "

_OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("score", pa.float64()),
        pa.field("text_chars", pa.int64()),
    ]
)


def _keep_row(row: dict[str, Any]) -> bool:
    return row["score"] >= _SCORE_THRESHOLD


def _enrich_row(row: dict[str, Any]) -> dict[str, Any]:
    text = row["text"]
    return {
        "id": row["id"],
        "text": text,
        "score": row["score"],
        "text_chars": len(text),
    }


def _row_pipeline(input_path: str, output_path: str) -> Dataset:
    return (
        Dataset.from_list([input_path])
        .load_parquet(columns=list(_COLUMNS))
        .filter(_keep_row)
        .map(_enrich_row)
        .write_parquet(output_path, schema=_OUTPUT_SCHEMA)
    )


def _enrich_batch(batch: pa.RecordBatch) -> pa.RecordBatch:
    filtered = batch.filter(pc.greater_equal(batch.column("score"), _SCORE_THRESHOLD))
    text_chars = pc.cast(pc.utf8_length(filtered.column("text")), pa.int64())
    return pa.RecordBatch.from_arrays(
        [
            filtered.column("id"),
            filtered.column("text"),
            filtered.column("score"),
            text_chars,
        ],
        schema=_OUTPUT_SCHEMA,
    )


def _batch_pipeline(input_path: str, output_path: str) -> Dataset:
    return (
        Dataset.from_list([input_path])
        .load_parquet(columns=list(_COLUMNS), batch_mode=True)
        .map_batches(_enrich_batch)
        .write_parquet(output_path, schema=_OUTPUT_SCHEMA)
    )


_PIPELINES: Mapping[str, Callable[[str, str], Dataset]] = MappingProxyType(
    {
        "row": _row_pipeline,
        "batch": _batch_pipeline,
    }
)


@dataclass(frozen=True)
class CorpusDigest:
    sha256: str
    row_count: int
    schema: pa.Schema


def _rss_bytes() -> int:
    return psutil.Process().memory_info().rss


@contextmanager
def _peak_rss_sampler() -> Iterator[list[int]]:
    stop = threading.Event()
    samples = [_rss_bytes()]

    def sample() -> None:
        while not stop.wait(_RSS_SAMPLE_INTERVAL):
            samples.append(_rss_bytes())

    thread = threading.Thread(target=sample, name="zephyr-chunk-benchmark-rss", daemon=True)
    thread.start()
    try:
        yield samples
    finally:
        stop.set()
        thread.join()
        samples.append(_rss_bytes())


def _corpus_digest(path: Path) -> CorpusDigest:
    """Hash fixed-size logical batches so digest ignores Parquet row-group layout."""
    parquet = pq.ParquetFile(path)
    digest = hashlib.sha256()
    pending: list[pa.Table] = []
    pending_rows = 0
    row_count = 0

    def consume(rows: int) -> None:
        nonlocal pending, pending_rows, row_count
        combined = pa.concat_tables(pending).combine_chunks()
        logical_batch = combined.slice(0, rows).to_batches()[0]
        digest.update(logical_batch.serialize().to_pybytes())
        row_count += rows
        remainder = combined.slice(rows)
        pending = [remainder] if len(remainder) else []
        pending_rows = len(remainder)

    for record_batch in parquet.iter_batches(batch_size=_DIGEST_BATCH_ROWS):
        pending.append(pa.Table.from_batches([record_batch]))
        pending_rows += len(record_batch)
        while pending_rows >= _DIGEST_BATCH_ROWS:
            consume(_DIGEST_BATCH_ROWS)

    if pending_rows:
        consume(pending_rows)
    return CorpusDigest(digest.hexdigest(), row_count, parquet.schema_arrow)


def _write_base_corpus(path: Path) -> dict[str, int]:
    logger.info("Reading %d pinned FineWeb-Edu row groups from %s", len(_SOURCE_ROW_GROUPS), _SOURCE_URL)
    compressed_bytes = 0
    uncompressed_bytes = 0
    rows = 0
    with fsspec.open(_SOURCE_URL, "rb").open() as source:
        parquet = pq.ParquetFile(source)
        projected_schema = pa.schema(
            [parquet.schema_arrow.field(column) for column in _COLUMNS],
            metadata=parquet.schema_arrow.metadata,
        )
        with pq.ParquetWriter(path, projected_schema) as writer:
            for row_group_idx in _SOURCE_ROW_GROUPS:
                metadata = parquet.metadata.row_group(row_group_idx)
                compressed_bytes += sum(
                    metadata.column(column_idx).total_compressed_size for column_idx in range(metadata.num_columns)
                )
                uncompressed_bytes += metadata.total_byte_size
                table = parquet.read_row_group(row_group_idx, columns=_COLUMNS)
                writer.write_table(table, row_group_size=len(table))
                rows += len(table)

    if rows != _BASE_ROWS:
        raise RuntimeError(f"Expected {_BASE_ROWS} source rows, got {rows}")
    return {
        "source_compressed_bytes_all_columns": compressed_bytes,
        "source_uncompressed_bytes_all_columns": uncompressed_bytes,
    }


def _write_scaled_corpus(base_path: Path, output_path: Path, target_rows: int) -> None:
    base = pq.ParquetFile(base_path)
    repetitions = math.ceil(target_rows / _BASE_ROWS)
    written = 0
    with pq.ParquetWriter(output_path, base.schema_arrow) as writer:
        for repetition in range(repetitions):
            suffix = pa.scalar(f":repeat-{repetition:06d}")
            for batch in base.iter_batches(batch_size=1000):
                remaining = target_rows - written
                if remaining <= 0:
                    return
                if len(batch) > remaining:
                    batch = batch.slice(0, remaining)
                ids = pc.binary_join_element_wise(batch.column("id"), suffix, pa.scalar(""))
                batch = batch.set_column(batch.schema.get_field_index("id"), "id", ids)
                writer.write_batch(batch)
                written += len(batch)


def _corpus_path(cache_dir: Path, rows: int) -> Path:
    if rows <= 0:
        raise ValueError(f"rows must be positive, got {rows}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_path = cache_dir / "fineweb-edu-10k.parquet"
    base_manifest_path = base_path.with_suffix(".manifest.json")
    if not base_path.exists():
        source_stats = _write_base_corpus(base_path)
        corpus_digest = _corpus_digest(base_path)
        base_manifest_path.write_text(
            json.dumps(
                {
                    "source_url": _SOURCE_URL,
                    "source_commit": _SOURCE_COMMIT,
                    "source_row_groups": _SOURCE_ROW_GROUPS,
                    "columns": _COLUMNS,
                    "rows": corpus_digest.row_count,
                    "schema": str(corpus_digest.schema),
                    "semantic_sha256": corpus_digest.sha256,
                    **source_stats,
                },
                indent=2,
            )
        )

    if rows == _BASE_ROWS:
        return base_path

    output_path = cache_dir / f"fineweb-edu-{rows}.parquet"
    manifest_path = output_path.with_suffix(".manifest.json")
    if output_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("rows") == rows and manifest.get("source_commit") == _SOURCE_COMMIT:
            return output_path

    logger.info("Building deterministic %s-row corpus from %s", f"{rows:,}", base_path)
    _write_scaled_corpus(base_path, output_path, rows)
    corpus_digest = _corpus_digest(output_path)
    if corpus_digest.row_count != rows:
        raise RuntimeError(f"Expected {rows} generated rows, got {corpus_digest.row_count}")
    manifest_path.write_text(
        json.dumps(
            {
                "source_commit": _SOURCE_COMMIT,
                "base_semantic_sha256": json.loads(base_manifest_path.read_text())["semantic_sha256"],
                "id_suffix": ":repeat-{repetition:06d}",
                "rows": corpus_digest.row_count,
                "schema": str(corpus_digest.schema),
                "semantic_sha256": corpus_digest.sha256,
            },
            indent=2,
        )
    )
    return output_path


def _measure(mode: str, input_path: Path, output_path: Path, chunk_prefix: Path) -> dict[str, Any]:
    pipeline = _PIPELINES[mode](str(input_path), str(output_path))
    client = LocalClient(max_threads=1)
    ctx = ZephyrContext(
        client=client,
        resources=ResourceConfig(cpu=1, ram="8g", disk="10g"),
        max_workers=1,
        chunk_storage_prefix=str(chunk_prefix),
        name=f"chunk-benchmark-{mode}",
    )

    gc.collect()
    pa.default_memory_pool().release_unused()
    rss_before = _rss_bytes()
    cpu_start = time.process_time()
    wall_start = time.monotonic()
    try:
        with _peak_rss_sampler() as rss_samples:
            result = ctx.execute(pipeline)
    finally:
        ctx.shutdown()
        client.shutdown(wait=True)
    wall_seconds = time.monotonic() - wall_start
    cpu_seconds = time.process_time() - cpu_start

    output_digest = _corpus_digest(output_path)
    input_rows = pq.ParquetFile(input_path).metadata.num_rows
    return {
        "mode": mode,
        "input_rows": input_rows,
        "output_rows": output_digest.row_count,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "rows_per_second": input_rows / wall_seconds,
        "rss_before_bytes": rss_before,
        "rss_peak_bytes": max(rss_samples),
        "rss_growth_peak_bytes": max(rss_samples) - rss_before,
        "input_bytes": input_path.stat().st_size,
        "output_bytes": output_path.stat().st_size,
        "output_schema": str(output_digest.schema),
        "output_semantic_sha256": output_digest.sha256,
        "counters": result.counters,
    }


def _run_child(script_path: Path, mode: str, input_path: Path, output_path: Path, chunk_prefix: Path) -> dict:
    command = [
        sys.executable,
        str(script_path),
        "--child-mode",
        mode,
        "--input-file",
        str(input_path),
        "--output-file",
        str(output_path),
        "--chunk-prefix",
        str(chunk_prefix),
    ]
    process = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    if process.returncode != 0:
        raise RuntimeError(f"Benchmark child failed for {mode}:\n{process.stderr}\n{process.stdout}")
    for line in reversed(process.stdout.splitlines()):
        if line.startswith(_RESULT_PREFIX):
            return json.loads(line.removeprefix(_RESULT_PREFIX))
    raise RuntimeError(f"Benchmark child produced no result for {mode}:\n{process.stdout}")


def _run_child_benchmark(
    mode: str,
    input_file: Path | None,
    output_file: Path | None,
    chunk_prefix: Path | None,
) -> None:
    if input_file is None or output_file is None or chunk_prefix is None:
        raise click.UsageError("Child mode requires --input-file, --output-file, and --chunk-prefix")
    result = _measure(mode, input_file, output_file, chunk_prefix)
    print(_RESULT_PREFIX + json.dumps(result, sort_keys=True))


def _run_parent_benchmark(rows: int, modes: str, repeat: int, cache_dir: Path, prepare_only: bool) -> None:
    input_path = _corpus_path(cache_dir, rows)
    logger.info("Corpus ready: %s (%s rows, %.2f GiB)", input_path, f"{rows:,}", input_path.stat().st_size / 2**30)
    if prepare_only:
        return

    selected = [mode.strip() for mode in modes.split(",") if mode.strip()]
    unknown = set(selected) - set(_PIPELINES)
    if unknown:
        raise click.BadParameter(
            f"Unknown modes: {sorted(unknown)}; available: {sorted(_PIPELINES)}", param_hint="modes"
        )
    if repeat <= 0:
        raise click.BadParameter("repeat must be positive", param_hint="repeat")

    results: list[dict[str, Any]] = []
    script_path = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="zephyr-chunk-benchmark-output-") as temp_dir:
        temp_path = Path(temp_dir)
        for repetition in range(repeat):
            for mode in selected:
                result = _run_child(
                    script_path,
                    mode,
                    input_path,
                    temp_path / f"{mode}-{repetition}.parquet",
                    temp_path / f"chunks-{mode}-{repetition}",
                )
                result["repetition"] = repetition
                results.append(result)
                print(_RESULT_PREFIX + json.dumps(result, sort_keys=True))

    digests = {result["output_semantic_sha256"] for result in results}
    schemas = {result["output_schema"] for result in results}
    row_counts = {result["output_rows"] for result in results}
    if len(digests) != 1 or len(schemas) != 1 or len(row_counts) != 1:
        raise RuntimeError("Benchmark modes disagree on output digest, schema, or row count")

    summary = {
        "rows": rows,
        "repeat": repeat,
        "modes": {
            mode: {
                "wall_seconds": [result["wall_seconds"] for result in results if result["mode"] == mode],
                "cpu_seconds": [result["cpu_seconds"] for result in results if result["mode"] == mode],
                "rss_growth_peak_bytes": [
                    result["rss_growth_peak_bytes"] for result in results if result["mode"] == mode
                ],
            }
            for mode in selected
        },
    }
    print("SUMMARY: " + json.dumps(summary, sort_keys=True))


@click.command()
@click.option("--rows", type=int, default=_BASE_ROWS, show_default=True)
@click.option("--modes", default="row", show_default=True, help="Comma-separated benchmark modes.")
@click.option("--repeat", type=int, default=5, show_default=True)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=Path("/tmp/zephyr-chunk-benchmark"),
    show_default=True,
)
@click.option("--prepare-only", is_flag=True, help="Prepare and validate the corpus without running a benchmark.")
@click.option("--child-mode", type=click.Choice(sorted(_PIPELINES)), hidden=True)
@click.option("--input-file", type=click.Path(path_type=Path), hidden=True)
@click.option("--output-file", type=click.Path(path_type=Path), hidden=True)
@click.option("--chunk-prefix", type=click.Path(path_type=Path), hidden=True)
def main(
    rows: int,
    modes: str,
    repeat: int,
    cache_dir: Path,
    prepare_only: bool,
    child_mode: str | None,
    input_file: Path | None,
    output_file: Path | None,
    chunk_prefix: Path | None,
) -> None:
    """Run the real-data chunk pipeline benchmark."""
    if child_mode is not None:
        _run_child_benchmark(child_mode, input_file, output_file, chunk_prefix)
        return
    _run_parent_benchmark(rows, modes, repeat, cache_dir, prepare_only)


if __name__ == "__main__":
    main()
