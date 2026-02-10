#!/usr/bin/env python
# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Zephyr deduplication pipeline (generate docs → map → deduplicate by simhash → write).

Usage:
    uv run python tests/benchmark_dedup_pipeline.py                           # Run benchmark with defaults
    uv run python tests/benchmark_dedup_pipeline.py benchmark --backends ray  # Benchmark specific backend
    uv run python tests/benchmark_dedup_pipeline.py write-input --output-dir /tmp/input  # Just generate input files
"""

import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import click
import psutil
from tqdm import tqdm
from zephyr import Dataset, ExecutionHint
from zephyr.execution import ZephyrContext
from zephyr.readers import load_file
from zephyr.writers import write_parquet_file

WORDS = """
the be to of and a in that have I it for not on with he as you do at this but his by from
they we say her she or an will my one all would there their what so up out if about who get
which go me when make can like time no just him know take people into year your good some
could them see other than then now look only come its over think also back after use two how
our work first well way even new want because any these give day most us data system process
compute memory network storage algorithm function variable method class object interface
protocol implementation performance optimization benchmark metric throughput latency
scalability distributed parallel concurrent asynchronous synchronous batch stream
""".split()


def generate_doc(doc_id: int, num_words: int = 1000) -> dict[str, Any]:
    """Generate synthetic document with body (num_words random words) and metadata."""
    words = [random.choice(WORDS) for _ in range(num_words)]
    body = " ".join(words)

    return {
        "doc_id": doc_id,
        "body": body,
        "meta1": random.randint(0, 1000),
        "meta2": random.choice(["A", "B", "C", "D", "E"]),
        "meta3": random.random(),
        "meta4": random.choice(["red", "green", "blue", "yellow"]),
        "meta5": random.randint(1000, 9999),
        "meta6": random.choice([True, False]),
        "meta7": random.uniform(0.0, 100.0),
        "meta8": f"category_{random.randint(1, 50)}",
        "meta9": random.choice(["alpha", "beta", "gamma", "delta"]),
        "meta10": doc_id % 100,
    }


def write_input_files(docs: Iterator[dict[str, Any]], input_dir: str, num_files: int = 10) -> list[str]:
    """Write documents to num_files files, returning file paths."""
    os.makedirs(input_dir, exist_ok=True)
    file_paths = []
    file_buffers = [[] for _ in range(num_files)]

    for idx, doc in tqdm(enumerate(docs)):
        file_idx = idx % num_files
        file_buffers[file_idx].append(doc)

    for i in range(num_files):
        file_path = os.path.join(input_dir, f"input-{i:05d}.parquet")
        file_paths.append(file_path)
        list(write_parquet_file(file_buffers[i], file_path))
        # list(write_vortex_file(file_buffers[i], file_path))

    return file_paths


def simhash(doc: dict[str, Any]) -> int:
    words = doc["body"].split()
    unique_words = sorted(set(words))
    return hash(" ".join(unique_words)) % 10000


def create_pipeline(input_dir: str, output_dir: str) -> Dataset:
    """Create benchmark pipeline: load → map → deduplicate by simhash → write."""
    return (
        Dataset.from_files(f"{input_dir}/*.parquet")
        .load_file()
        .map(
            lambda doc: {
                "simhash": simhash(doc),
            }
        )
        .group_by(
            key=lambda x: x["simhash"],
            reducer=lambda key, items: next(iter(items)),
        )
        .write_parquet(f"{output_dir}/output-{{shard:05d}}-of-{{total:05d}}.parquet")
    )


def count_docs(file_path: str) -> int:
    return len(list(load_file(file_path)))


def run_benchmark(
    backend_type: str,
    input_dir: str,
    num_docs: int,
    num_input_files: int,
) -> dict[str, Any]:
    """Run deduplication pipeline benchmark for specified backend, returning metrics dict."""
    print(f"\n{'=' * 70}")
    print(f"Backend: {backend_type.upper()}")
    print(f"{'=' * 70}")

    # Setup
    output_dir = tempfile.mkdtemp(prefix=f"zephyr_benchmark_{backend_type}_")
    process = psutil.Process(os.getpid())

    try:
        # Create pipeline
        print("Creating pipeline...")
        pipeline = create_pipeline(input_dir, output_dir)

        # Execute and measure
        print("Executing pipeline...")
        mem_before = process.memory_info().rss
        exec_start = time.time()
        with ZephyrContext(name="benchmark") as ctx:
            results = list(ctx.execute(pipeline, ExecutionHint()))
        exec_time = time.time() - exec_start
        mem_after = process.memory_info().rss

        # Calculate metrics
        throughput = num_docs / exec_time
        memory_mb = (mem_after - mem_before) / (1024 * 1024)

        # Count output documents
        print("Counting output documents...")
        output_docs = sum(count_docs(f) for f in results)
        dedup_ratio = (1 - output_docs / num_docs) * 100

        # Report results
        print("\nResults:")
        print("-" * 70)
        print(f"  Execution time:      {exec_time:>10.2f}s")
        print(f"  Throughput:          {throughput:>10,.0f} docs/sec")
        print(f"  Memory delta:        {memory_mb:>10.1f} MB")
        print(f"  Input documents:     {num_docs:>10,}")
        print(f"  Input files:         {num_input_files:>10,}")
        print(f"  Output documents:    {output_docs:>10,}")
        print(f"  Deduplication:       {dedup_ratio:>10.1f}%")
        print(f"  Output files:        {len(results):>10,}")
        print("=" * 70 + "\n")

        return {
            "backend": backend_type,
            "exec_time": exec_time,
            "throughput": throughput,
            "memory_mb": memory_mb,
            "output_docs": output_docs,
            "dedup_ratio": dedup_ratio,
            "output_files": len(results),
        }

    finally:
        # Cleanup output dir
        print(f"Cleaning up {output_dir}...")
        shutil.rmtree(output_dir, ignore_errors=True)


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Benchmark Zephyr deduplication pipeline."""
    if ctx.invoked_subcommand is None:
        # Default to benchmark command with default args
        ctx.invoke(benchmark)


@cli.command("write-input")
@click.option("--output-dir", type=click.Path(path_type=Path), required=True, help="Directory to write input files")
@click.option("--num-docs", type=int, default=1_000_000, help="Number of documents to generate")
@click.option("--words-per-doc", type=int, default=1000, help="Number of words per document")
@click.option("--num-input-files", type=int, default=10, help="Number of input parquet files to create")
def write_input(
    output_dir: Path,
    num_docs: int,
    words_per_doc: int,
    num_input_files: int,
) -> None:
    """Generate and write input files for benchmarking."""
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_input_files(num_docs, words_per_doc, num_input_files, str(output_dir))
    print(f"\nInput files written to: {output_dir}")


def setup_input_files(
    num_docs: int, words_per_doc: int, num_input_files: int, input_dir: str | None = None
) -> tuple[str, float]:
    """Generate input files and return directory path and generation time."""
    if input_dir is None:
        input_dir = tempfile.mkdtemp(prefix="zephyr_benchmark_input_")

    print(f"Generating {num_docs:,} documents and writing to {num_input_files} files...")
    gen_start = time.time()
    docs_generator = (generate_doc(i, words_per_doc) for i in range(num_docs))
    input_files = write_input_files(docs_generator, input_dir, num_input_files)

    dir_size = sum(os.path.getsize(f) for f in input_files)
    gen_time = time.time() - gen_start
    print(f"Total input size: {dir_size / (1024 * 1024):.2f} MB")
    print(f"Wrote {num_docs:,} docs to {len(input_files)} files in {gen_time:.2f}s")
    print(f"Input directory: {input_dir}")

    return input_dir, gen_time


@cli.command("benchmark")
@click.option(
    "--backends",
    multiple=True,
    type=click.Choice(["sync", "threadpool", "ray"]),
    default=["threadpool"],
    help="Backends to benchmark",
)
@click.option("--num-docs", type=int, default=1_000_000, help="Number of documents to generate")
@click.option("--words-per-doc", type=int, default=1000, help="Number of words per document")
@click.option("--num-input-files", type=int, default=10, help="Number of input JSONL files to create")
@click.option(
    "--input-dir", type=str, default=None, help="Use pre-generated input directory (skips generation and cleanup)"
)
@click.option("--profile", is_flag=True, help="Profile each backend with py-spy")
@click.option(
    "--profile-output",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for profile output (default: tests/profiles/)",
)
def benchmark(
    backends: tuple[str, ...] = ("threadpool",),
    num_docs: int = 1_000_000,
    words_per_doc: int = 1000,
    num_input_files: int = 10,
    input_dir: str | None = None,
    profile: bool = False,
    profile_output: Path | None = None,
) -> None:
    """Run benchmark on deduplication pipeline."""
    backends_list = list(backends) if backends else ["threadpool"]

    # Handle profiling mode - generate input once and subprocess with py-spy
    if profile:
        if profile_output is None:
            profile_output = Path(__file__).parent / "profiles"

        print("\nZephyr Benchmark Profiling with py-spy")
        print(f"Configuration: {num_docs:,} docs x {words_per_doc} words -> {num_input_files} files")
        print(f"Backends: {', '.join(backends_list)}")
        print(f"Profile output directory: {profile_output}\n")

        profile_output.mkdir(parents=True, exist_ok=True)

        # Generate input files once
        temp_input_dir, _ = setup_input_files(num_docs, words_per_doc, num_input_files)
        print()

        try:
            # Profile each backend
            for backend in backends_list:
                print(f"{'=' * 70}")
                print(f"Profiling Backend: {backend.upper()}")
                print(f"{'=' * 70}")

                env = os.environ.copy()
                if backend == "ray":
                    env["RAY_LOCAL_MODE"] = "1"
                    print("Using RAY_LOCAL_MODE=1 for single-process profiling")

                speedscope_file = profile_output / f"profile_{backend}.speedscope"

                pyspy_cmd = [
                    "sudo",
                    "py-spy",
                    "record",
                    "--format",
                    "speedscope",
                    "--output",
                    str(speedscope_file),
                    "--rate",
                    "100",
                    "--subprocesses",
                    "--",
                    sys.executable,
                    __file__,
                    "benchmark",
                    "--backends",
                    backend,
                    "--num-docs",
                    str(num_docs),
                    "--words-per-doc",
                    str(words_per_doc),
                    "--num-input-files",
                    str(num_input_files),
                    "--input-dir",
                    temp_input_dir,
                ]

                print(f"Running: sudo py-spy record ... --output {speedscope_file}")

                result = subprocess.run(pyspy_cmd, env=env)

                if result.returncode == 0:
                    print(f"Speedscope profile saved to {speedscope_file}\n")
                else:
                    print(f"py-spy failed with return code {result.returncode}\n")

            print("=" * 70)
            print("Profiling complete!")
            print("=" * 70)
            print(f"\nProfile files saved to: {profile_output}")
            print("\nTo view speedscope profiles:")
            print("  1. Visit https://www.speedscope.app/")
            print(f"  2. Upload .speedscope files from {profile_output}")

        finally:
            print(f"\nCleaning up input directory {temp_input_dir}...")
            shutil.rmtree(temp_input_dir, ignore_errors=True)

        return

    # Normal benchmark mode
    print("\nZephyr Deduplication Pipeline Benchmark")
    print(f"Configuration: {num_docs:,} docs x {words_per_doc} words -> {num_input_files} files")
    print(f"Backends: {', '.join(backends_list)}")

    # Use provided input_dir or generate new one
    should_cleanup = input_dir is None
    if should_cleanup:
        print()
        temp_input_dir, gen_time = setup_input_files(num_docs, words_per_doc, num_input_files)
    else:
        gen_time = 0.0
        temp_input_dir = input_dir
        print(f"Using existing input directory: {input_dir}")

    try:
        # Run benchmarks for each backend
        results = []
        for backend in backends_list:
            try:
                result = run_benchmark(backend, temp_input_dir, num_docs, num_input_files)
                result["gen_time"] = gen_time
                results.append(result)
            except Exception as e:
                print(f"\nERROR with backend '{backend}': {e}")
                traceback.print_exc()

        # Summary comparison
        if len(results) > 1:
            print("\nSummary Comparison:")
            print("=" * 70)
            print(f"{'Backend':<15} {'Time (s)':<12} {'Throughput':<15} {'Memory (MB)':<15}")
            print("-" * 70)
            for r in results:
                print(f"{r['backend']:<15} {r['exec_time']:<12.2f} {r['throughput']:<15,.0f} {r['memory_mb']:<15.1f}")
            print("-" * 70)
            if should_cleanup:
                print(f"Note: Generation time ({gen_time:.2f}s) was shared across all backends")
            print("=" * 70 + "\n")

    finally:
        # Only cleanup if we generated the input
        if should_cleanup:
            print(f"\nCleaning up input directory {temp_input_dir}...")
            shutil.rmtree(temp_input_dir, ignore_errors=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cli()
