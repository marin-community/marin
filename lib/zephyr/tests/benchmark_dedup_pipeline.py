#!/usr/bin/env python
# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Benchmark script for Zephyr deduplication pipeline.

Tests a realistic pipeline that:
1. Generates 1M documents with ~1000 words each
2. Maps over documents to split words
3. Groups by simhash to deduplicate
4. Writes output as JSONL

Usage:
    uv run python tests/benchmark_dedup_pipeline.py --backends ray threadpool
    uv run python tests/benchmark_dedup_pipeline.py --num-docs 100000
    uv run python tests/benchmark_dedup_pipeline.py --input-dir /path/to/input
    uv run python tests/benchmark_dedup_pipeline.py --profile --backends ray
"""

import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import click
import msgspec
import psutil
import zstandard as zstd
from zephyr import Dataset, create_backend

# Word list for generating synthetic text
WORDS = [
    "the",
    "be",
    "to",
    "of",
    "and",
    "a",
    "in",
    "that",
    "have",
    "I",
    "it",
    "for",
    "not",
    "on",
    "with",
    "he",
    "as",
    "you",
    "do",
    "at",
    "this",
    "but",
    "his",
    "by",
    "from",
    "they",
    "we",
    "say",
    "her",
    "she",
    "or",
    "an",
    "will",
    "my",
    "one",
    "all",
    "would",
    "there",
    "their",
    "what",
    "so",
    "up",
    "out",
    "if",
    "about",
    "who",
    "get",
    "which",
    "go",
    "me",
    "when",
    "make",
    "can",
    "like",
    "time",
    "no",
    "just",
    "him",
    "know",
    "take",
    "people",
    "into",
    "year",
    "your",
    "good",
    "some",
    "could",
    "them",
    "see",
    "other",
    "than",
    "then",
    "now",
    "look",
    "only",
    "come",
    "its",
    "over",
    "think",
    "also",
    "back",
    "after",
    "use",
    "two",
    "how",
    "our",
    "work",
    "first",
    "well",
    "way",
    "even",
    "new",
    "want",
    "because",
    "any",
    "these",
    "give",
    "day",
    "most",
    "us",
    "data",
    "system",
    "process",
    "compute",
    "memory",
    "network",
    "storage",
    "algorithm",
    "function",
    "variable",
    "method",
    "class",
    "object",
    "interface",
    "protocol",
    "implementation",
    "performance",
    "optimization",
    "benchmark",
    "metric",
    "throughput",
    "latency",
    "scalability",
    "distributed",
    "parallel",
    "concurrent",
    "asynchronous",
    "synchronous",
    "batch",
    "stream",
]


def generate_doc(doc_id: int, num_words: int = 1000) -> dict[str, Any]:
    """Generate a single synthetic document with body and metadata.

    Args:
        doc_id: Unique document identifier
        num_words: Number of words to generate in the body (default: 1000)

    Returns:
        Dictionary with body, simhash, and metadata fields
    """
    # Generate ~num_words random words
    words = [random.choice(WORDS) for _ in range(num_words)]
    body = " ".join(words)

    # Create simhash from unique words (creates natural duplicates)
    # Using modulo to control duplicate rate
    unique_words = sorted(set(words))
    simhash = hash(" ".join(unique_words)) % 10000

    return {
        "doc_id": doc_id,
        "body": body,
        "simhash": simhash,
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


def generate_documents(num_docs: int, words_per_doc: int = 1000) -> Iterator[dict[str, Any]]:
    """Generate synthetic documents lazily.

    Args:
        num_docs: Total number of documents to generate
        words_per_doc: Words per document body

    Yields:
        Synthetic document dictionaries
    """
    for i in range(num_docs):
        yield generate_doc(i, words_per_doc)


def write_input_files(docs: Iterator[dict[str, Any]], input_dir: str, num_files: int = 10) -> list[str]:
    """Write documents to N JSONL files for realistic file-based processing.

    Args:
        docs: Iterator of document dictionaries
        input_dir: Directory to write input files
        num_files: Number of files to create

    Returns:
        List of file paths created
    """
    os.makedirs(input_dir, exist_ok=True)
    file_paths = []
    file_buffers = [[] for _ in range(num_files)]

    # Collect documents for each file
    for idx, doc in enumerate(docs):
        file_idx = idx % num_files
        file_buffers[file_idx].append(doc)

    # Write each buffer to a zstd-compressed file
    cctx = zstd.ZstdCompressor(level=1)
    for i in range(num_files):
        file_path = os.path.join(input_dir, f"input-{i:05d}.jsonl.zst")
        file_paths.append(file_path)

        # Serialize all docs for this file
        data = b"".join(msgspec.json.encode(doc) + b"\n" for doc in file_buffers[i])

        # Compress and write
        with open(file_path, "wb") as f:
            f.write(cctx.compress(data))

    return file_paths


def create_pipeline(input_dir: str, output_dir: str) -> Dataset:
    """Create the benchmark pipeline.

    Pipeline steps:
    1. Read JSONL files from input directory
    2. Flat map: parse JSONL records
    3. Map: split body into words and add word count
    4. Group by simhash: deduplicate by taking first document per hash
    5. Write as compressed JSONL

    Args:
        input_dir: Directory containing input JSONL files
        output_dir: Directory to write output files

    Returns:
        Configured Dataset pipeline
    """
    from zephyr import load_jsonl

    return (
        Dataset.from_files(input_dir, "*.jsonl.zst")
        .flat_map(load_jsonl)
        .map(
            lambda doc: {
                **doc,
                "words": doc["body"].split(),
                "word_count": len(doc["body"].split()),
            }
        )
        .group_by(
            key=lambda x: x["simhash"],
            reducer=lambda key, items: next(iter(items)),  # Take first (deduplication)
        )
        .write_jsonl(f"{output_dir}/output-{{shard:05d}}-of-{{total:05d}}.jsonl.zst")
    )


def count_jsonl_docs(file_path: str) -> int:
    """Count number of documents in a JSONL file.

    Args:
        file_path: Path to JSONL file (may be gzipped or zstd compressed)

    Returns:
        Number of lines/documents in the file
    """
    import gzip
    import io

    if file_path.endswith(".gz"):
        with gzip.open(file_path, "rt") as f:
            return sum(1 for line in f if line.strip())
    elif file_path.endswith(".zst"):
        dctx = zstd.ZstdDecompressor()
        with open(file_path, "rb") as raw_f:
            with dctx.stream_reader(raw_f) as reader:
                text_f = io.TextIOWrapper(reader, encoding="utf-8")
                return sum(1 for line in text_f if line.strip())
    else:
        with open(file_path) as f:
            return sum(1 for line in f if line.strip())


def run_benchmark(
    backend_type: str,
    input_dir: str,
    num_docs: int,
    num_input_files: int,
) -> dict[str, Any]:
    """Run benchmark for a specific backend using pre-generated input files.

    Args:
        backend_type: Backend to use ("sync", "threadpool", or "ray")
        input_dir: Directory containing input JSONL files
        num_docs: Number of documents in the input files
        num_input_files: Number of input files

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'=' * 70}")
    print(f"Backend: {backend_type.upper()}")
    print(f"{'=' * 70}")

    # Setup
    output_dir = tempfile.mkdtemp(prefix=f"zephyr_benchmark_{backend_type}_")
    process = psutil.Process(os.getpid())

    try:
        # Create backend
        if backend_type == "sync":
            backend = create_backend("sync")
        elif backend_type == "threadpool":
            backend = create_backend("threadpool", max_parallelism=10)
        elif backend_type == "ray":
            backend = create_backend("ray", max_parallelism=10, memory="2GB")
        else:
            raise ValueError(f"Unknown backend: {backend_type}")

        # Create pipeline
        print("🔧 Creating pipeline...")
        pipeline = create_pipeline(input_dir, output_dir)

        # Execute and measure
        print("⚡ Executing pipeline...")
        mem_before = process.memory_info().rss
        exec_start = time.time()
        results = list(backend.execute(pipeline))
        exec_time = time.time() - exec_start
        mem_after = process.memory_info().rss

        # Calculate metrics
        throughput = num_docs / exec_time
        memory_mb = (mem_after - mem_before) / (1024 * 1024)

        # Count output documents
        print("📊 Counting output documents...")
        output_docs = sum(count_jsonl_docs(f) for f in results)
        dedup_ratio = (1 - output_docs / num_docs) * 100

        # Report results
        print(f"\n{'Results':^70}")
        print(f"{'-' * 70}")
        print(f"  Execution time:      {exec_time:>10.2f}s")
        print(f"  Throughput:          {throughput:>10,.0f} docs/sec")
        print(f"  Memory delta:        {memory_mb:>10.1f} MB")
        print(f"  Input documents:     {num_docs:>10,}")
        print(f"  Input files:         {num_input_files:>10,}")
        print(f"  Output documents:    {output_docs:>10,}")
        print(f"  Deduplication:       {dedup_ratio:>10.1f}%")
        print(f"  Output files:        {len(results):>10,}")
        print(f"{'=' * 70}\n")

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
        print(f"🧹 Cleaning up {output_dir}...")
        shutil.rmtree(output_dir, ignore_errors=True)


def setup_input_files(num_docs: int, words_per_doc: int, num_input_files: int) -> tuple[str, float]:
    """Generate input files and return directory path and generation time."""
    input_dir = tempfile.mkdtemp(prefix="zephyr_benchmark_input_")

    print(f"📝 Generating {num_docs:,} documents and writing to {num_input_files} files...")
    gen_start = time.time()
    docs_generator = generate_documents(num_docs, words_per_doc)
    input_files = write_input_files(docs_generator, input_dir, num_input_files)

    dir_size = sum(os.path.getsize(f) for f in input_files)
    print(f"   ✓ Total input size: {dir_size / (1024 * 1024):.2f} MB")

    gen_time = time.time() - gen_start
    print(f"   ✓ Wrote {num_docs:,} docs to {len(input_files)} files in {gen_time:.2f}s")
    print(f"   Input directory: {input_dir}")

    return input_dir, gen_time


@click.command()
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
def main(
    backends: tuple[str, ...],
    num_docs: int,
    words_per_doc: int,
    num_input_files: int,
    input_dir: str | None,
    profile: bool,
    profile_output: Path | None,
) -> None:
    """Benchmark Zephyr deduplication pipeline."""
    backends_list = list(backends) if backends else ["threadpool"]

    # Handle profiling mode - generate input once and subprocess with py-spy
    if profile:
        if profile_output is None:
            profile_output = Path(__file__).parent / "profiles"

        print("\n🔬 Zephyr Benchmark Profiling with py-spy")
        print(f"Configuration: {num_docs:,} docs x {words_per_doc} words -> {num_input_files} files")
        print(f"Backends: {', '.join(backends_list)}")
        print(f"📁 Profile output directory: {profile_output}\n")

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

                print(f"   Running: sudo py-spy record ... --output {speedscope_file}")

                result = subprocess.run(pyspy_cmd, env=env)

                if result.returncode == 0:
                    print(f"   ✓ Speedscope profile saved to {speedscope_file}\n")
                else:
                    print(f"   ✗ py-spy failed with return code {result.returncode}\n")

            print(f"{'=' * 70}")
            print("✓ Profiling complete!")
            print(f"{'=' * 70}")
            print(f"\nProfile files saved to: {profile_output}")
            print("\nTo view speedscope profiles:")
            print("  1. Visit https://www.speedscope.app/")
            print(f"  2. Upload .speedscope files from {profile_output}")

        finally:
            print(f"\n🧹 Cleaning up input directory {temp_input_dir}...")
            shutil.rmtree(temp_input_dir, ignore_errors=True)

        return

    # Normal benchmark mode
    print("\n🚀 Zephyr Deduplication Pipeline Benchmark")
    print(f"Configuration: {num_docs:,} docs x {words_per_doc} words -> {num_input_files} files")
    print(f"Backends: {', '.join(backends_list)}")

    # Use provided input_dir or generate new one
    should_cleanup = input_dir is None
    if should_cleanup:
        print()
        input_dir, gen_time = setup_input_files(num_docs, words_per_doc, num_input_files)
    else:
        gen_time = 0.0
        print(f"   Using existing input directory: {input_dir}")

    try:
        # Run benchmarks for each backend
        results = []
        for backend in backends_list:
            try:
                result = run_benchmark(backend, input_dir, num_docs, num_input_files)
                result["gen_time"] = gen_time
                results.append(result)
            except Exception as e:
                print(f"\n❌ ERROR with backend '{backend}': {e}")
                import traceback

                traceback.print_exc()

        # Summary comparison
        if len(results) > 1:
            print(f"\n{'Summary Comparison':^70}")
            print(f"{'=' * 70}")
            print(f"{'Backend':<15} {'Time (s)':<12} {'Throughput':<15} {'Memory (MB)':<15}")
            print(f"{'-' * 70}")
            for r in results:
                print(f"{r['backend']:<15} {r['exec_time']:<12.2f} {r['throughput']:<15,.0f} {r['memory_mb']:<15.1f}")
            print(f"{'-' * 70}")
            if should_cleanup:
                print(f"Note: Generation time ({gen_time:.2f}s) was shared across all backends")
            print(f"{'=' * 70}\n")

    finally:
        # Only cleanup if we generated the input
        if should_cleanup:
            print(f"\n🧹 Cleaning up input directory {input_dir}...")
            shutil.rmtree(input_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
