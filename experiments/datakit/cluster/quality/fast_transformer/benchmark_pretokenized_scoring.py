# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Core-sizing benchmark for scoring *pretokenized* documents with a fusion scorer.

``benchmark_scoring.py`` measures the deployed read -> tokenize -> forward loop.
This module answers the different question a large labeling run poses: given a
corpus that already arrives as token ids plus its document embeddings, how many
cores should a task hold, how many tasks fit on a node, and what saturates
first. Tokenization is deliberately out of the loop.

Five modes:

* ``prepare`` — tokenize a text sample once and write the pretokenized layout
  (``id`` string, ``ids`` int32[max_tokens], ``embedding`` int8[1024]) that the
  other modes read. Run once; the output is the fixture.
* ``forward`` — one pinned configuration: restrict the process to ``--cores``
  CPUs and sweep ``--batches``, reporting steady-state docs/s, docs/s/core, XLA
  compile time per shape, and peak RSS. ``--pin-when`` decides when the affinity
  mask is set, and the difference is not subtle: ``sched_setaffinity(0, ...)``
  binds the *calling thread*, and threads inherit the mask of the thread that
  created them. Pinning ``before`` JAX builds its CPU client therefore restricts
  every XLA worker thread it later spawns; pinning ``after`` binds only the main
  thread and leaves the already-running pool free on the whole node, which does
  not measure a core count at all. The reported ``threads`` count shows the pool
  stays node-sized either way — the pinned runs are restricted by the kernel,
  not by a smaller pool, so they may understate what an explicitly sized pool
  would reach.
* ``sweep`` — driver: runs ``forward`` as a subprocess once per core count so
  every point on the scaling curve is measured on the same node, in a fresh
  process with its own correctly-sized thread pool.
* ``pack`` — ``--tasks`` independent scorer processes on disjoint core sets,
  timed against a shared start time. One forward scales poorly across cores, so
  a node's real throughput comes from packing small tasks; this measures that
  sum, and the memory bandwidth those processes contend over.
* ``read`` — sustained read throughput over the pretokenized shards, optionally
  with ``--readers`` concurrent processes taking disjoint shard slices, which is
  how a node's LOTA cache and object-store bandwidth are made to contend.

Two model shapes are measured, selected by ``--fold-donor``. The trained
checkpoint carries a frozen ``[vocab, 640]`` donor table read through a learned
``[640, 256]`` projection. Deployment folds the two into one ``[vocab, 256]``
table; ``fold_donor`` performs that fold and the run asserts score parity, so
the folded numbers are a measurement of the deployable model rather than of a
different one.
"""

import argparse
import dataclasses
import gc
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import cast

import equinox as eqx
import fsspec
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from iris.env_resources import TaskResources
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.benchmark_scoring import (
    BASE_URL,
    BENCH_JSON_PREFIX,
    CountingFile,
    list_shards,
    spread,
    tokenize,
)
from experiments.datakit.cluster.quality.fast_transformer.inference import predict
from experiments.datakit.cluster.quality.fast_transformer.joined_labels import EMBED_DIM, embedding_matrix
from experiments.datakit.cluster.quality.fast_transformer.model import COMPUTE_DTYPE, FastTransformer
from experiments.datakit.cluster.quality.fast_transformer.scorer import PooledScorer, load_pooled_scorer

logger = logging.getLogger(__name__)

MODULE = "experiments.datakit.cluster.quality.fast_transformer.benchmark_pretokenized_scoring"
READ_BATCH = 2048  # rows per arrow record batch when streaming the pretokenized shards
PREPARE_READ_BATCH = 512  # rows per arrow record batch when reading source text
# CoreWeave LOTA declines to cache objects at or below this size; they are always
# served from the object-store backend, so shard sizing decides whether a corpus
# is cacheable at all. https://docs.coreweave.com/products/storage/object-storage
LOTA_MIN_CACHED_OBJECT_BYTES = 4 * 1024 * 1024
DEFAULT_BATCHES = (512, 1024, 2048, 4096, 8192)
DEFAULT_CORE_COUNTS = (1, 2, 4, 8, 16, 32, 64)
# Fields carried across when a donor-table model is folded into a plain embedding
# model. `config`, `donor_embed` and `donor_proj` are deliberately absent: the
# folded config has frozen_donor_dim=0 and the folded table replaces the pair.
FOLD_CARRY_FIELDS = (
    "pool_query",
    "proj_w",
    "proj_b",
    "pos_embed",
    "layers",
    "final_query",
    "head_g",
    "head_b",
    "head_w",
    "doc_proj_w",
    "doc_proj_b",
    "doc_ln_g",
    "doc_ln_b",
    "doc_head_w",
    "doc_type_embed",
    "doc_gate",
)


@dataclass(frozen=True)
class Pretokenized:
    """A block of pretokenized rows held in memory."""

    ids: np.ndarray  # [n, max_tokens] int32 compact ids, PAD-padded
    embedding: np.ndarray  # [n, EMBED_DIM] float32, L2-normalized as the forward wants it


def host_facts() -> dict:
    """Which machine the numbers came from. The CPU pool is fungible in Iris, so a
    CPU-only task can land on either a Genoa CPU node or an H100 node's host CPUs,
    and those are different machines with different per-core throughput."""
    model = ""
    with open("/proc/cpuinfo") as fh:
        for line in fh:
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    return {
        "hostname": os.uname().nodename,
        "cpu_model": model,
        "host_cpu_count": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0)),
    }


def physical_cores(cpus: set[int]) -> int:
    """Distinct physical cores behind a set of logical CPUs (SMT siblings collapse)."""
    groups = set()
    for cpu in cpus:
        path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        with open(path) as fh:
            groups.add(fh.read().strip())
    return len(groups)


def thread_count() -> int:
    """Threads in this process — the check that the XLA pool sized to the pinned set."""
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("Threads:"):
                return int(line.split()[1])
    raise ValueError("no Threads: line in /proc/self/status")


def peak_rss_bytes() -> int:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    raise ValueError("no VmHWM: line in /proc/self/status")


def rss_bytes() -> int:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise ValueError("no VmRSS: line in /proc/self/status")


def pin_to_cores(cores: int, offset: int = 0) -> dict:
    """Restrict this process to ``cores`` CPUs and cap the BLAS/OpenMP pools to match.

    Iris's ``--cpu N`` is a Kubernetes *request* with no limit, so an unpinned task
    bursts across the whole node and any per-core number taken from it is fiction.
    ``offset`` lets several processes on one node take disjoint core sets, which is
    what packing many small tasks onto a node actually does.

    Call this before the first JAX computation. The affinity mask is per-thread and
    inherited at thread creation, so it reaches XLA's pool only if the pool does not
    exist yet. The OMP/BLAS caps below are set for the same reason and are honoured
    only by libraries that read them at import.
    """
    allowed = sorted(os.sched_getaffinity(0))[offset : offset + cores]
    if len(allowed) < cores:
        raise ValueError(f"need {cores} cpus at offset {offset}, node offers {len(os.sched_getaffinity(0))}")
    os.sched_setaffinity(0, set(allowed))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(cores)
    return {"pinned_cpus": allowed, "pinned_physical_cores": physical_cores(set(allowed))}


def fold_donor(model: FastTransformer) -> FastTransformer:
    """Fold the frozen donor table and its learned projection into one embedding table.

    The forward computes ``matmul(take(donor, ids), proj)`` in bf16 with f32
    accumulation. Gather and matmul commute row-wise, so folding first and
    gathering after is the identical computation, not an approximation — the
    caller asserts that on real rows. The fold removes a
    ``[batch, max_tokens, 640] @ [640, 256]`` matmul from every forward and drops
    the resident table from ``vocab*640*4`` to ``vocab*256*4`` bytes.
    """
    config = model.config
    if not config.frozen_donor_dim:
        raise ValueError("model has no frozen donor table to fold")
    folded_config = dataclasses.replace(config, frozen_donor_dim=0)
    template = FastTransformer(folded_config, key=jr.PRNGKey(0))
    table = jnp.matmul(
        model.donor_embed.astype(COMPUTE_DTYPE),
        model.donor_proj.astype(COMPUTE_DTYPE),
        preferred_element_type=jnp.float32,
    ).astype(jnp.float32)
    names = ["embed"] + [n for n in FOLD_CARRY_FIELDS if getattr(template, n) is not None]
    values = [table] + [getattr(model, n) for n in names[1:]]
    return cast(FastTransformer, eqx.tree_at(lambda m: [getattr(m, n) for n in names], template, values))


def array_bytes(model: FastTransformer) -> dict:
    """Resident parameter bytes, split into the embedding table and everything else."""
    leaves = jax.tree_util.tree_leaves(cast(FastTransformer, eqx.filter(model, eqx.is_inexact_array)))
    embed = model.embed if model.embed is not None else model.donor_embed
    table = int(embed.size * embed.dtype.itemsize)
    total = sum(int(x.size * x.dtype.itemsize) for x in leaves)
    return {"embedding_table_bytes": table, "params_total_bytes": total, "trunk_bytes": total - table}


# ---------------------------------------------------------------------------
# prepare: build the pretokenized fixture
# ---------------------------------------------------------------------------


def prepare(args) -> dict:
    """Tokenize a fixed text sample once and write the pretokenized shards."""
    scorer = load_pooled_scorer(args.model_dir)
    shards = list_shards(args.base_url, args.sources, args.shards_per_source)
    quota = max(1, args.docs // len(shards))
    fs = fsspec.filesystem("s3")
    out_root = args.out.rstrip("/")
    written: list[dict] = []
    tokenize_seconds = 0.0
    read_seconds = 0.0

    for index, (source, shard) in enumerate(shards):
        ids_rows: list[np.ndarray] = []
        doc_ids: list[str] = []
        embeds: list[np.ndarray] = []
        taken = 0
        t0 = time.monotonic()
        with fs.open(shard, "rb", cache_type="none") as raw:
            parquet = pq.ParquetFile(raw)
            for batch in parquet.iter_batches(batch_size=PREPARE_READ_BATCH, columns=["id", "text", "embedding"]):
                keep = min(batch.num_rows, quota - taken)
                batch = batch.slice(0, keep)
                texts = [t or "" for t in batch.column("text").to_pylist()]
                read_seconds += time.monotonic() - t0
                t1 = time.monotonic()
                ids_rows.append(tokenize("hf", scorer.tokenizer_name, texts, scorer.max_tokens, scorer.remap))
                tokenize_seconds += time.monotonic() - t1
                doc_ids.extend(batch.column("id").to_pylist())
                flat = batch.column("embedding").flatten().to_numpy(zero_copy_only=False)
                embeds.append(flat.reshape(batch.num_rows, EMBED_DIM))
                taken += keep
                t0 = time.monotonic()
                if taken >= quota:
                    break
        ids = np.concatenate(ids_rows)
        embedding = np.concatenate(embeds)
        table = pa.table(
            {
                "id": pa.array(doc_ids, type=pa.string()),
                "ids": pa.FixedSizeListArray.from_arrays(pa.array(ids.reshape(-1), type=pa.int32()), ids.shape[1]),
                "embedding": pa.FixedSizeListArray.from_arrays(
                    pa.array(embedding.reshape(-1), type=pa.int8()), EMBED_DIM
                ),
            }
        )
        path = f"{out_root}/part-{index:05d}.parquet"
        with fs.open(path, "wb") as out:
            pq.write_table(table, out, compression=args.compression)
        size = fs.info(path)["size"]
        written.append({"path": path, "rows": table.num_rows, "bytes": size, "source": source})
        logger.info("prepared %s: %d rows, %.1f MB", path, table.num_rows, size / 1e6)

    rows = sum(w["rows"] for w in written)
    total = sum(w["bytes"] for w in written)
    return {
        "out": out_root,
        "shards": len(written),
        "rows": rows,
        "bytes": total,
        "bytes_per_doc": total / rows,
        "uncompressed_bytes_per_doc": 4 * scorer.max_tokens + EMBED_DIM,
        "max_tokens": scorer.max_tokens,
        "read_seconds": read_seconds,
        "tokenize_seconds": tokenize_seconds,
        "tokenize_docs_per_second": rows / tokenize_seconds,
        "files": written,
    }


# ---------------------------------------------------------------------------
# reading the pretokenized fixture
# ---------------------------------------------------------------------------


def pretokenized_shards(root: str) -> list[str]:
    found = sorted(str(p) for p in StoragePath(f"{root.rstrip('/')}/*.parquet").glob())
    if not found:
        raise ValueError(f"no pretokenized shards under {root}")
    return found


def read_shard(path: str, fs, limit: int | None) -> tuple[list[np.ndarray], list[np.ndarray], int, float, int]:
    """Stream one pretokenized shard.

    Returns (id blocks, embedding blocks, rows, seconds inside ``read``, bytes fetched).
    """
    id_blocks: list[np.ndarray] = []
    embed_blocks: list[np.ndarray] = []
    rows = 0
    with fs.open(path, "rb", cache_type="none") as raw:
        counting = CountingFile(raw)
        parquet = pq.ParquetFile(counting)
        for batch in parquet.iter_batches(batch_size=READ_BATCH, columns=["ids", "embedding"]):
            n = batch.num_rows
            width = len(batch.column("ids")[0])
            id_blocks.append(batch.column("ids").flatten().to_numpy(zero_copy_only=False).reshape(n, width))
            embed_blocks.append(batch.column("embedding").flatten().to_numpy(zero_copy_only=False).reshape(n, EMBED_DIM))
            rows += n
            if limit is not None and rows >= limit:
                break
        io_seconds = counting.read_seconds
        byte_count = counting.bytes_read
    return id_blocks, embed_blocks, rows, io_seconds, byte_count


def load_pretokenized(root: str, docs: int) -> Pretokenized:
    """Load up to ``docs`` pretokenized rows, striping across shards."""
    fs = fsspec.filesystem("s3")
    shards = pretokenized_shards(root)
    ids: list[np.ndarray] = []
    embeds: list[np.ndarray] = []
    have = 0
    for path in shards:
        id_blocks, embed_blocks, rows, _, _ = read_shard(path, fs, docs - have)
        ids.extend(id_blocks)
        embeds.extend(embed_blocks)
        have += rows
        if have >= docs:
            break
    all_ids = np.concatenate(ids)[:docs]
    all_embed = np.concatenate(embeds)[:docs]
    return Pretokenized(ids=np.ascontiguousarray(all_ids), embedding=embedding_matrix(all_embed))


def prestage(paths: list[str], workers: int) -> dict:
    """Warm CoreWeave's LOTA cache for these objects, per the documented recipe.

    A ``HeadObject`` carrying ``Range: bytes=0-0`` makes LOTA pull the *whole*
    object from the backend into the cluster's distributed NVMe cache without
    sending the body to the client, so the subsequent read is served from NVMe
    rather than from the object-store backend. LOTA declines to cache objects
    under ``LOTA_MIN_CACHED_OBJECT_BYTES``, so a shard layout below that size
    cannot be warmed at all — which is why the per-shard rows below carry each
    shard's size.
    """
    fs = fsspec.filesystem("s3")

    def warm(path: str) -> float:
        bucket, key = path.removeprefix("s3://").split("/", 1)
        t0 = time.monotonic()
        fs.call_s3("head_object", Bucket=bucket, Key=key, Range="bytes=0-0")
        return time.monotonic() - t0

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        latencies = list(pool.map(warm, paths))
    return {
        "objects": len(paths),
        "workers": workers,
        "seconds": time.monotonic() - t0,
        "object_latency_seconds": spread(latencies) if latencies else {},
    }


def read_stream(args) -> dict:
    """One sequential reader over its slice of the pretokenized shards.

    ``--repeat`` re-reads the same slice: pass 0 is cold, later passes see
    whatever the node's LOTA cache kept, so the two rates bracket what a worker
    sees on a first sweep of a corpus versus a re-scored one. ``--prestage``
    warms LOTA first, which turns pass 0 into the fully pre-cached case.
    """
    fs = fsspec.filesystem("s3")
    shards = pretokenized_shards(args.pretokenized)[args.reader_index :: args.reader_count]
    if args.max_shards:
        shards = shards[: args.max_shards]
    sizes = {path: fs.info(path)["size"] for path in shards}
    prestaged = prestage(shards, args.prestage_workers) if args.prestage else None
    passes = []
    checksum = 0.0
    for index in range(args.repeat):
        rows = 0
        byte_total = 0
        io_seconds = 0.0
        per_shard = []
        t0 = time.monotonic()
        for path in shards:
            shard_t0 = time.monotonic()
            id_blocks, embed_blocks, shard_rows, shard_io, shard_bytes = read_shard(path, fs, None)
            rows += shard_rows
            io_seconds += shard_io
            byte_total += shard_bytes
            # Touch the decoded arrays so the measurement includes the arrow->numpy
            # materialization the scoring loop actually pays, not just the fetch.
            for block in embed_blocks:
                checksum += float(block[:, 0].sum())
            for block in id_blocks:
                checksum += float(block[0, 0])
            shard_seconds = time.monotonic() - shard_t0
            per_shard.append(
                {
                    "path": path,
                    "size_bytes": sizes[path],
                    "lota_cacheable": sizes[path] > LOTA_MIN_CACHED_OBJECT_BYTES,
                    "seconds": shard_seconds,
                    "megabytes_per_second": shard_bytes / shard_seconds / 1e6,
                }
            )
        elapsed = time.monotonic() - t0
        passes.append(
            {
                "rows": rows,
                "bytes": byte_total,
                "seconds": elapsed,
                "io_seconds": io_seconds,
                "docs_per_second": rows / elapsed,
                "megabytes_per_second": byte_total / elapsed / 1e6,
                "per_shard": per_shard if index == 0 else [],
            }
        )
        logger.info("BENCH read pass %s", json.dumps({k: v for k, v in passes[-1].items() if k != "per_shard"}))
    return {
        "reader_index": args.reader_index,
        "reader_count": args.reader_count,
        "shards": len(shards),
        "prestage": prestaged,
        "rows": passes[0]["rows"],
        "bytes": passes[0]["bytes"],
        "seconds": sum(p["seconds"] for p in passes),
        "cold": passes[0],
        "warm": passes[1:],
        "docs_per_second": passes[0]["docs_per_second"],
        "megabytes_per_second": passes[0]["megabytes_per_second"],
        "checksum": checksum,
        "host": host_facts(),
    }


def run_child(argv: list[str]) -> dict:
    """Run this module as a subprocess and return the JSON it printed."""
    proc = subprocess.run([sys.executable, "-m", MODULE, *argv], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        logger.error("child %s failed (%d):\n%s\n%s", argv, proc.returncode, proc.stdout[-4000:], proc.stderr[-4000:])
        raise RuntimeError(f"child {argv} exited {proc.returncode}")
    for line in proc.stdout.splitlines():
        if line.startswith(BENCH_JSON_PREFIX):
            return json.loads(line[len(BENCH_JSON_PREFIX) :])
    raise RuntimeError(f"child {argv} printed no {BENCH_JSON_PREFIX} line:\n{proc.stdout[-4000:]}")


def read_fanout(args) -> dict:
    """``--readers`` concurrent reader processes over disjoint shard slices, on one node."""
    base = [
        "read",
        "--pretokenized",
        args.pretokenized,
        "--reader-count",
        str(args.readers),
        "--repeat",
        str(args.repeat),
    ]
    if args.max_shards:
        base += ["--max-shards", str(args.max_shards)]
    if args.prestage:
        base += ["--prestage", "--prestage-workers", str(args.prestage_workers)]
    procs = [
        subprocess.Popen(
            [sys.executable, "-m", MODULE, *base, "--reader-index", str(i)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for i in range(args.readers)
    ]
    t0 = time.monotonic()
    results = []
    for proc in procs:
        out, err = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"reader failed ({proc.returncode}):\n{out[-2000:]}\n{err[-2000:]}")
        results.append(
            next(
                json.loads(line[len(BENCH_JSON_PREFIX) :])
                for line in out.splitlines()
                if line.startswith(BENCH_JSON_PREFIX)
            )
        )
    elapsed = time.monotonic() - t0
    # Aggregate over the cold pass, charged to the slowest reader's cold pass: readers
    # start together, so that is the wall time in which all of the cold bytes landed.
    rows = sum(r["cold"]["rows"] for r in results)
    byte_total = sum(r["cold"]["bytes"] for r in results)
    cold_seconds = max(r["cold"]["seconds"] for r in results)
    # Sustained: every pass every reader made, charged to the slowest reader's total.
    # One pass over a small slice is too short to time; the repeated sweep is the
    # number that says how much read bandwidth a node can actually hold open.
    all_rows = sum(r["cold"]["rows"] + sum(w["rows"] for w in r["warm"]) for r in results)
    all_bytes = sum(r["cold"]["bytes"] + sum(w["bytes"] for w in r["warm"]) for r in results)
    all_seconds = max(r["cold"]["seconds"] + sum(w["seconds"] for w in r["warm"]) for r in results)
    return {
        "readers": args.readers,
        "wall_seconds": elapsed,
        "cold_seconds": cold_seconds,
        "rows": rows,
        "bytes": byte_total,
        "aggregate_docs_per_second": rows / cold_seconds,
        "aggregate_megabytes_per_second": byte_total / cold_seconds / 1e6,
        "sustained_docs_per_second": all_rows / all_seconds,
        "sustained_megabytes_per_second": all_bytes / all_seconds / 1e6,
        "sustained_bytes": all_bytes,
        "sustained_seconds": all_seconds,
        "prestage_seconds": max((r["prestage"]["seconds"] for r in results if r["prestage"]), default=0.0),
        "per_reader_cold_docs_per_second": [r["cold"]["docs_per_second"] for r in results],
        "per_reader_warm_docs_per_second": [[w["docs_per_second"] for w in r["warm"]] for r in results],
        "hosts": sorted({r["host"]["hostname"] for r in results}),
        "host": host_facts(),
    }


# ---------------------------------------------------------------------------
# forward: the pinned scaling measurement
# ---------------------------------------------------------------------------


def time_forward(model: FastTransformer, block: Pretokenized, batch: int, min_seconds: float) -> tuple[float, int]:
    """Run whole batches until ``min_seconds`` elapses. Returns (seconds, docs)."""
    rows = block.ids.shape[0]
    docs = 0
    start = 0
    elapsed = 0.0
    t0 = time.monotonic()
    while elapsed < min_seconds:
        if start + batch > rows:
            start = 0
        ids = block.ids[start : start + batch]
        emb = block.embedding[start : start + batch]
        predict(model, ids, batch_size=batch, doc_embed=emb)
        docs += batch
        start += batch
        elapsed = time.monotonic() - t0
    return elapsed, docs


def time_concurrent_forward(
    model: FastTransformer, block: Pretokenized, batch: int, min_seconds: float, shards: int
) -> tuple[float, int]:
    """``shards`` forwards running at once in this process, sharing one model.

    This is the shape a zephyr worker actually runs under ``InlineRunner``: one
    process holds a single copy of the embedding table and executes several
    shards as threads. It is the other way to spend a core budget — inter-request
    parallelism instead of more XLA threads inside one forward — and JAX drops
    the GIL for the duration of a forward, so the threads genuinely overlap.
    """
    if shards == 1:
        return time_forward(model, block, batch, min_seconds)
    with ThreadPoolExecutor(max_workers=shards) as pool:
        t0 = time.monotonic()
        futures = [pool.submit(time_forward, model, block, batch, min_seconds) for _ in range(shards)]
        docs = sum(f.result()[1] for f in futures)
        return time.monotonic() - t0, docs


def forward(args) -> dict:
    """One pinned configuration: sweep batch sizes and report steady-state throughput."""
    result: dict = {
        "cores": args.cores,
        "core_offset": args.core_offset,
        "pin_when": args.pin_when,
        "fold_donor": args.fold_donor,
    }
    if args.pin_when == "before":
        result.update(pin_to_cores(args.cores, args.core_offset))

    load_t0 = time.monotonic()
    scorer: PooledScorer = load_pooled_scorer(args.model_dir)
    model = scorer.model
    result["model_load_seconds"] = time.monotonic() - load_t0
    result["rss_after_load_bytes"] = rss_bytes()

    block = load_pretokenized(args.pretokenized, args.pool_docs)
    result["pool_docs"] = int(block.ids.shape[0])
    result["max_tokens"] = int(block.ids.shape[1])
    result["nonpad_token_fraction"] = float((block.ids != 0).mean())
    result["rss_after_data_bytes"] = rss_bytes()

    if args.fold_donor:
        fold_t0 = time.monotonic()
        folded = fold_donor(model)
        result["fold_seconds"] = time.monotonic() - fold_t0
        probe = min(256, block.ids.shape[0])
        before = predict(model, block.ids[:probe], batch_size=probe, doc_embed=block.embedding[:probe])
        after = predict(folded, block.ids[:probe], batch_size=probe, doc_embed=block.embedding[:probe])
        result["fold_max_abs_score_delta"] = float(np.abs(before - after).max())
        # Drop every reference to the donor-table model, otherwise its 671 MB table
        # stays resident and the folded footprint is not what a deployment would hold.
        del before, after, model, scorer
        gc.collect()
        model = folded
        result["rss_after_fold_bytes"] = rss_bytes()

    result["params"] = array_bytes(model)
    result["flops_per_token"] = model.config.flops_per_token()

    if args.pin_when == "after":
        result.update(pin_to_cores(args.cores, args.core_offset))

    rows = []
    for batch in args.batches:
        if batch > block.ids.shape[0]:
            logger.warning("skipping batch %d: only %d rows loaded", batch, block.ids.shape[0])
            continue
        compile_t0 = time.monotonic()
        predict(model, block.ids[:batch], batch_size=batch, doc_embed=block.embedding[:batch])
        compile_seconds = time.monotonic() - compile_t0
        # Packed runs measure an aggregate, so every process must be timing the same
        # wall-clock window; model load and compile vary enough to smear it otherwise.
        if args.start_at:
            time.sleep(max(0.0, args.start_at - time.time()))
        passes = []
        for _ in range(args.passes):
            seconds, docs = time_concurrent_forward(model, block, batch, args.min_seconds, args.concurrent_shards)
            passes.append(docs / seconds)
        row = {
            "batch": batch,
            "concurrent_shards": args.concurrent_shards,
            "compile_seconds": compile_seconds,
            "docs_per_second": spread(passes),
            "docs_per_second_per_core": spread([p / args.cores for p in passes]),
            "passes": passes,
            "rss_bytes": rss_bytes(),
            "peak_rss_bytes": peak_rss_bytes(),
            "threads": thread_count(),
        }
        rows.append(row)
        logger.info("BENCH forward %s", json.dumps(row))
    result["batches"] = rows
    result["peak_rss_bytes"] = peak_rss_bytes()
    result["threads"] = thread_count()
    result["host"] = host_facts()
    result["task_resources"] = dataclasses.asdict(TaskResources.from_environment())
    return result


def sweep(args) -> dict:
    """Run ``forward`` once per core count, each in its own process on this node."""
    batches = ",".join(str(b) for b in args.batches)
    runs = []
    for cores in args.core_counts:
        argv = [
            "forward",
            "--model-dir",
            args.model_dir,
            "--pretokenized",
            args.pretokenized,
            "--cores",
            str(cores),
            "--batches",
            batches,
            "--pool-docs",
            str(args.pool_docs),
            "--passes",
            str(args.passes),
            "--min-seconds",
            str(args.min_seconds),
            "--pin-when",
            args.pin_when,
            "--concurrent-shards",
            str(args.concurrent_shards),
        ]
        if args.fold_donor:
            argv.append("--fold-donor")
        logger.info("BENCH sweep starting cores=%d", cores)
        runs.append(run_child(argv))
        logger.info("BENCH sweep done cores=%d", cores)
    baseline = next((r for r in runs if r["cores"] == 1), None)
    for run in runs:
        for row in run["batches"]:
            if baseline is None:
                continue
            base = next((b for b in baseline["batches"] if b["batch"] == row["batch"]), None)
            if base:
                row["parallel_efficiency"] = row["docs_per_second"]["mean"] / (
                    base["docs_per_second"]["mean"] * run["cores"]
                )
    return {"host": host_facts(), "fold_donor": args.fold_donor, "pin_when": args.pin_when, "runs": runs}


def pack(args) -> dict:
    """``--tasks`` independent scorer processes on disjoint core sets, timed together.

    The core sweep measures how one forward scales *inside* one process. A labeling
    run instead packs many small tasks onto a node, so what decides the node's
    throughput is how those processes sum — including the memory bandwidth and the
    embedding-table gathers they contend over. Every child is pinned to its own
    cores and told the same start time, so the reported aggregate is one wall-clock
    window rather than a sum of unrelated intervals.
    """
    start_at = time.time() + args.warmup_seconds
    argv = [
        "--model-dir",
        args.model_dir,
        "--pretokenized",
        args.pretokenized,
        "--cores",
        str(args.cores_per_task),
        "--batches",
        str(args.batch),
        "--pool-docs",
        str(args.pool_docs),
        "--passes",
        str(args.passes),
        "--min-seconds",
        str(args.min_seconds),
        "--start-at",
        str(start_at),
    ]
    if args.fold_donor:
        argv.append("--fold-donor")
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                MODULE,
                "forward",
                *argv,
                "--core-offset",
                str(i * args.cores_per_task),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for i in range(args.tasks)
    ]
    results = []
    failures = 0
    for proc in procs:
        out, err = proc.communicate()
        if proc.returncode != 0:
            failures += 1
            logger.error("packed task failed (%d):\n%s\n%s", proc.returncode, out[-2000:], err[-2000:])
            continue
        results.append(
            next(
                json.loads(line[len(BENCH_JSON_PREFIX) :])
                for line in out.splitlines()
                if line.startswith(BENCH_JSON_PREFIX)
            )
        )
    per_task = [r["batches"][0]["docs_per_second"]["mean"] for r in results]
    cores = args.tasks * args.cores_per_task
    return {
        "tasks": args.tasks,
        "cores_per_task": args.cores_per_task,
        "batch": args.batch,
        "total_cores": cores,
        "failures": failures,
        "aggregate_docs_per_second": sum(per_task),
        "docs_per_second_per_core": sum(per_task) / cores,
        "per_task_docs_per_second": per_task,
        "peak_rss_bytes_per_task": [r["peak_rss_bytes"] for r in results],
        "peak_rss_bytes_total": sum(r["peak_rss_bytes"] for r in results),
        "host": host_facts(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)

    prep = sub.add_parser("prepare", help="tokenize a text sample into the pretokenized layout")
    prep.add_argument("--model-dir", required=True)
    prep.add_argument("--out", required=True)
    prep.add_argument("--base-url", default=BASE_URL)
    prep.add_argument("--docs", type=int, default=250_000)
    prep.add_argument("--sources", type=int, default=16)
    prep.add_argument("--shards-per-source", type=int, default=2)
    prep.add_argument("--compression", default="zstd")

    fwd = sub.add_parser("forward", help="one pinned configuration")
    fwd.add_argument("--model-dir", required=True)
    fwd.add_argument("--pretokenized", required=True)
    fwd.add_argument("--cores", type=int, required=True)
    fwd.add_argument("--batches", type=lambda s: [int(x) for x in s.split(",")], default=list(DEFAULT_BATCHES))
    fwd.add_argument("--pool-docs", type=int, default=32_768)
    fwd.add_argument("--passes", type=int, default=3)
    fwd.add_argument("--min-seconds", type=float, default=8.0)
    fwd.add_argument("--pin-when", choices=("before", "after"), default="before")
    fwd.add_argument("--fold-donor", action="store_true")
    fwd.add_argument("--core-offset", type=int, default=0, help="first CPU of this process's pinned set")
    fwd.add_argument("--start-at", type=float, default=0.0, help="unix time to begin timing, after compile")
    fwd.add_argument(
        "--concurrent-shards",
        type=int,
        default=1,
        help="forwards to run at once in this process, as InlineRunner would across shards",
    )

    swp = sub.add_parser("sweep", help="run forward once per core count")
    swp.add_argument("--model-dir", required=True)
    swp.add_argument("--pretokenized", required=True)
    swp.add_argument("--core-counts", type=lambda s: [int(x) for x in s.split(",")], default=list(DEFAULT_CORE_COUNTS))
    swp.add_argument("--batches", type=lambda s: [int(x) for x in s.split(",")], default=list(DEFAULT_BATCHES))
    swp.add_argument("--pool-docs", type=int, default=32_768)
    swp.add_argument("--passes", type=int, default=3)
    swp.add_argument("--min-seconds", type=float, default=8.0)
    swp.add_argument("--pin-when", choices=("before", "after"), default="before")
    swp.add_argument("--fold-donor", action="store_true")
    swp.add_argument("--concurrent-shards", type=int, default=1)

    pk = sub.add_parser("pack", help="many independent scorer processes on disjoint cores")
    pk.add_argument("--model-dir", required=True)
    pk.add_argument("--pretokenized", required=True)
    pk.add_argument("--tasks", type=int, required=True)
    pk.add_argument("--cores-per-task", type=int, required=True)
    pk.add_argument("--batch", type=int, default=512)
    pk.add_argument("--pool-docs", type=int, default=16_384)
    pk.add_argument("--passes", type=int, default=3)
    pk.add_argument("--min-seconds", type=float, default=8.0)
    pk.add_argument("--warmup-seconds", type=float, default=120.0, help="grace for load+compile before timing")
    pk.add_argument("--fold-donor", action="store_true")

    rd = sub.add_parser("read", help="sustained read throughput over the pretokenized shards")
    rd.add_argument("--pretokenized", required=True)
    rd.add_argument("--readers", type=int, default=1, help="concurrent reader processes on this node")
    rd.add_argument("--reader-index", type=int, default=0)
    rd.add_argument("--reader-count", type=int, default=1)
    rd.add_argument("--max-shards", type=int, default=0)
    rd.add_argument("--repeat", type=int, default=2, help="passes over the slice; pass 0 is the cold read")
    rd.add_argument("--prestage", action="store_true", help="warm LOTA over the slice before timing")
    rd.add_argument("--prestage-workers", type=int, default=32)

    args = p.parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()
    if args.mode == "prepare":
        result = prepare(args)
    elif args.mode == "forward":
        result = forward(args)
    elif args.mode == "sweep":
        result = sweep(args)
    elif args.mode == "pack":
        result = pack(args)
    elif args.readers > 1:
        result = read_fanout(args)
    else:
        result = read_stream(args)
    print(BENCH_JSON_PREFIX + json.dumps(result))


if __name__ == "__main__":
    main()
