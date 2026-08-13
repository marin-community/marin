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

Four more modes measure the same corpus on H100s, because a Genoa node delivers
about 7.2 TFLOP/s while one H100 is three orders of magnitude larger on paper and
the model is small enough that dispatch overhead, not arithmetic, may decide:

* ``gpu-forward`` — one process on this task's visible devices: sweep batch size
  with the rows already resident in HBM, so the number is the forward alone, and
  separately drive the same batches from host numpy through ``predict`` so the
  H2D copy and per-call dispatch are priced. Reports docs/s, achieved TFLOP/s and
  MFU against the H100 bf16 peak, per-shape compile time, and peak HBM.
* ``gpu-arms`` — the same device-resident sweep run for several candidate scorer
  shapes at once, one arm per GPU, so a params-versus-throughput frontier comes
  out of one node under one set of conditions. Arms are overrides on the deployed
  folded config (``SCALEUP_ARMS``); the numerics ones keep the trained weights and
  report their score delta, the shape ones run at random init because throughput
  does not depend on the values. Each arm also reports XLA's own flops and
  bytes-accessed for the compiled forward, which is what separates a
  memory-bound trunk from a compute-bound one.
* ``gpu-pack`` — ``--gpus`` independent processes, child *i* pinned to GPU *i* by
  ``CUDA_VISIBLE_DEVICES`` and to its own host cores, timed against a shared
  start. ``iris job run --gpu H100x8`` hands the task one process with all eight
  devices; this measures the other way to spend them, and it is the shape the CPU
  study's "many narrow workers" finding points at.
* ``pipeline`` — the whole loop a labeling run pays: S3 read, arrow decode,
  embedding normalize, H2D, forward. ``--reader-threads`` host threads fill a
  bounded queue and the consumer reports how long it sat waiting on it, which is
  the compute-bound-versus-feed-bound verdict rather than an inference from it.
* ``replicate`` — server-side copy of the fixture into more distinct keys. One
  H100 eats all 16 fixture shards in well under a second, so a feed measurement
  over them measures the node's LOTA cache rather than a first sweep of a corpus.

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
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
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
from rigging.filesystem import StoragePath, open_url
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
from experiments.datakit.cluster.quality.fast_transformer.inference import (
    data_parallel_shardings,
    predict,
    predict_batch,
)
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
# NVIDIA H100 SXM5 dense peaks, structured sparsity excluded (H100 datasheet).
# bf16 is the denominator that matters: ``model.COMPUTE_DTYPE`` is bf16, so every
# matmul in this forward is bf16 with f32 accumulation on CPU and GPU alike, and
# there is no fp32 variant of the model to compare against.
H100_BF16_PEAK_FLOPS = 989.4e12
H100_TF32_PEAK_FLOPS = 494.7e12
H100_FP32_PEAK_FLOPS = 67.0e12
# H100 SXM5 HBM3 peak. The ratio against the bf16 peak is the ~295 FLOP/byte ridge
# an arithmetic intensity has to clear before the tensor cores can be the limit.
H100_HBM_PEAK_BYTES_PER_SECOND = 3.35e12
DEFAULT_GPU_BATCHES = (512, 2048, 8192, 16_384, 32_768, 65_536, 131_072)
# Forwards allowed to run ahead of the device. One in flight leaves the GPU idle
# between kernels; unbounded run-ahead queues activations until HBM fills, which
# turns a throughput measurement into an OOM.
GPU_PIPELINE_DEPTH = 2
# Rows compared between the GPU and CPU backends. bf16 tensor cores and the CPU
# backend's bf16 emulation do not have to round identically, so the delta is
# measured rather than assumed to be zero.
BACKEND_PARITY_ROWS = 512
# Decoded arrow blocks the pipeline readers may run ahead of the forward by.
DEFAULT_FEED_QUEUE_BLOCKS = 64
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
    """Resident parameter counts and bytes, split into the embedding table and the trunk."""
    leaves = jax.tree_util.tree_leaves(cast(FastTransformer, eqx.filter(model, eqx.is_inexact_array)))
    embed = model.embed if model.embed is not None else model.donor_embed
    table = int(embed.size * embed.dtype.itemsize)
    total = sum(int(x.size * x.dtype.itemsize) for x in leaves)
    total_params = sum(int(x.size) for x in leaves)
    return {
        "embedding_table_bytes": table,
        "params_total_bytes": total,
        "trunk_bytes": total - table,
        "table_params": int(embed.size),
        "params_total": total_params,
        "trunk_params": total_params - int(embed.size),
        "embedding_table_dtype": str(embed.dtype),
    }


# ---------------------------------------------------------------------------
# scale-up arms: shape and numerics variants of the deployed trunk
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """A candidate scorer shape, expressed as overrides on the deployed config.

    ``xla_flags`` is appended to the child's ``XLA_FLAGS``; it has to travel in the
    environment because XLA parses it when the GPU client is built, long before any
    flag this module sees. Arms whose overrides leave every parameter shape
    unchanged are measured with the *trained* weights transplanted in, so their
    score delta against the deployed model is a real number; the rest are measured
    with random weights of the same shape, which is a throughput question only.
    """

    name: str
    overrides: dict
    xla_flags: str = ""


CUBLASLT = "--xla_gpu_enable_cublaslt=true"
BF16_STREAM = {"stream_dtype": "bfloat16"}
HEAD64 = {"num_heads": 6}  # hidden_dim 384 / 6 = 64, the width tensor cores want
ARM0 = {**BF16_STREAM, **HEAD64}
D512 = {"hidden_dim": 512, "num_heads": 8}  # head_dim 64 again at the wider trunk
ARM_SEED = 0

SCALEUP_ARMS = {
    a.name: a
    for a in (
        Arm("baseline", {}),
        Arm("a0a_bf16_stream", BF16_STREAM),
        Arm("a0b_cublaslt", {}, CUBLASLT),
        Arm("a0c_head64", HEAD64),
        Arm("a0d_bf16_table", {"embed_dtype": "bfloat16"}),
        Arm("a0_all", ARM0, CUBLASLT),
        Arm("a0e_all_plus_table", {**ARM0, "embed_dtype": "bfloat16"}, CUBLASLT),
        Arm("a1_d512", {**ARM0, **D512}, CUBLASLT),
        Arm("a1_d512_fp32", D512),
        Arm("a2_d512_mlp8", {**ARM0, **D512, "mlp_ratio": 8}, CUBLASLT),
        Arm("a2_d512_mlp8_fp32", {**D512, "mlp_ratio": 8}),
        Arm("a3_embed512", {**ARM0, "embed_dim": 512}, CUBLASLT),
        Arm("a3_embed512_fp32", {"embed_dim": 512}),
    )
}


def leaf_shapes(model: FastTransformer) -> list[tuple[int, ...]]:
    return [x.shape for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))]


def arm_model(trained: FastTransformer, arm: Arm) -> tuple[FastTransformer, bool]:
    """The arm's model, and whether it carries the trained weights.

    Shape-preserving arms (numerics switches, a head-count change) get the trained
    arrays cast into the arm's dtypes, so the arm is the deployed scorer measured
    differently rather than a different scorer. Shape-changing arms cannot, and are
    built at random init: throughput does not depend on the values.
    """
    config = dataclasses.replace(trained.config, **arm.overrides)
    template = FastTransformer(config, key=jr.PRNGKey(ARM_SEED))
    if leaf_shapes(template) != leaf_shapes(trained):
        return template, False
    # Flattened rather than tree_mapped across the pair: the two trees carry
    # different static configs, which makes them different pytree nodes even when
    # every array in them lines up.
    target, static = eqx.partition(template, eqx.is_array)
    target_leaves, treedef = jax.tree_util.tree_flatten(target)
    source_leaves = jax.tree_util.tree_leaves(eqx.filter(trained, eqx.is_array))
    cast_in = jax.tree_util.tree_unflatten(
        treedef, [src.astype(dst.dtype) for dst, src in zip(target_leaves, source_leaves, strict=True)]
    )
    return cast(FastTransformer, eqx.combine(cast_in, static)), True


def arm_score_delta(baseline: FastTransformer, model: FastTransformer, block: Pretokenized) -> dict:
    """Score spread between an arm and the deployed f32 model on the same real rows."""
    rows = min(BACKEND_PARITY_ROWS, block.ids.shape[0])
    ids, emb = block.ids[:rows], block.embedding[:rows]
    before = predict(baseline, ids, batch_size=rows, doc_embed=emb)
    after = predict(model, ids, batch_size=rows, doc_embed=emb)
    delta = np.abs(before - after)
    return {
        "rows": rows,
        "max_abs_score_delta": float(delta.max()),
        "mean_abs_score_delta": float(delta.mean()),
        "quantile_99_abs_score_delta": float(np.quantile(delta, 0.99)),
    }


def compiled_analysis(model: FastTransformer, ids, emb) -> dict:
    """XLA's own view of one forward: flops, HBM bytes, and which gemm calls it emitted.

    ``bytes accessed`` sums each op's operand and output bytes, so it counts a
    tensor once per consumer and ignores cache reuse — an upper bound on HBM
    traffic, and the number whose *ratio* between two arms says whether a change
    moved bytes or only flops. The gemm-call census answers the fusion question
    directly instead of by inference from a timing.
    """
    # A plain ``jax.jit`` closure rather than ``predict_batch.lower``: equinox's
    # compiled wrapper does not forward the analysis methods.
    arrays, static = eqx.partition(model, eqx.is_array)
    lowered = jax.jit(lambda a, i, e: predict_batch(eqx.combine(a, static), i, e)).lower(arrays, ids, emb)
    compiled = lowered.compile()
    cost = compiled.cost_analysis()
    cost = cost[0] if isinstance(cost, list) else cost
    memory = compiled.memory_analysis()
    text = compiled.as_text()
    gemm_lines = [ln.strip()[:300] for ln in text.splitlines() if 'custom_call_target="__cublas' in ln]
    return {
        "xla_flops": float(cost.get("flops", 0.0)),
        "xla_bytes_accessed": float(cost.get("bytes accessed", 0.0)),
        "temp_size_bytes": int(getattr(memory, "temp_size_in_bytes", 0)),
        "argument_size_bytes": int(getattr(memory, "argument_size_in_bytes", 0)),
        "cublas_gemm_calls": sum(1 for ln in gemm_lines if "__cublas$gemm" in ln),
        "cublaslt_matmul_calls": sum(1 for ln in gemm_lines if "__cublas$lt$matmul" in ln),
        "gelu_epilogue_calls": sum(1 for ln in gemm_lines if "GELU" in ln),
        "fusion_ops": sum(1 for ln in text.splitlines() if " fusion(" in ln),
        "gemm_call_sample": gemm_lines[:6],
    }


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


# ---------------------------------------------------------------------------
# gpu: the accelerator measurements
# ---------------------------------------------------------------------------


def device_facts() -> dict:
    """Which accelerators the numbers came from, and how much HBM they hold."""
    devices = jax.devices()
    stats = devices[0].memory_stats() or {}
    return {
        "device_count": len(devices),
        "device_kind": devices[0].device_kind,
        "platform": devices[0].platform,
        "hbm_limit_bytes": int(stats.get("bytes_limit", 0)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def hbm_bytes() -> dict:
    """Current and process-cumulative HBM occupancy across the visible devices.

    ``peak`` never falls, so in a batch sweep it is the high-water mark of every
    shape run so far rather than of the shape in the row that carries it; ``in_use``
    is what the current shape actually holds.
    """
    stats = [d.memory_stats() or {} for d in jax.devices()]
    return {
        "hbm_in_use_bytes": max(int(s.get("bytes_in_use", 0)) for s in stats),
        "hbm_peak_bytes_cumulative": max(int(s.get("peak_bytes_in_use", 0)) for s in stats),
    }


def resident_batches(block: Pretokenized, batch: int) -> list[tuple]:
    """Whole batches placed in device memory, so a timed loop over them is pure compute.

    Placement is paid once and excluded from the measurement on purpose: the H2D
    copy a real pipeline pays is priced separately by the host-fed timing, and
    conflating the two hides which of the forward and the feed is the smaller
    number.
    """
    ndev, _, batch_shard = data_parallel_shardings()
    if batch % ndev:
        raise ValueError(f"batch {batch} does not divide across {ndev} devices")
    whole = (block.ids.shape[0] // batch) * batch
    if not whole:
        raise ValueError(f"pool holds {block.ids.shape[0]} rows, one batch needs {batch}")
    return [
        (
            jax.device_put(jnp.asarray(block.ids[i : i + batch]), batch_shard),
            jax.device_put(jnp.asarray(block.embedding[i : i + batch]), batch_shard),
        )
        for i in range(0, whole, batch)
    ]


def time_resident_forward(model: FastTransformer, batches: list[tuple], min_seconds: float) -> tuple[float, int]:
    """Run device-resident batches until ``min_seconds`` elapses. Returns (seconds, docs)."""
    docs = 0
    index = 0
    pending: list = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < min_seconds:
        ids, emb = batches[index % len(batches)]
        pending.append(predict_batch(model, ids, emb))
        docs += int(ids.shape[0])
        index += 1
        if len(pending) > GPU_PIPELINE_DEPTH:
            jax.block_until_ready(pending.pop(0))
    jax.block_until_ready(pending)
    return time.monotonic() - t0, docs


def on_cpu(model: FastTransformer) -> FastTransformer:
    """The same model with every array moved to the host CPU backend."""
    cpu = jax.devices("cpu")[0]
    arrays, static = eqx.partition(model, eqx.is_array)
    return cast(FastTransformer, eqx.combine(jax.device_put(arrays, cpu), static))


def backend_score_delta(model: FastTransformer, block: Pretokenized) -> dict:
    """Largest score difference between the accelerator and the CPU backend.

    Both run ``COMPUTE_DTYPE`` bf16 matmuls with f32 accumulation, but H100 tensor
    cores and XLA:CPU's bf16 path need not round the same way. A labeling run that
    thresholds on the score cares how big that is.
    """
    rows = min(BACKEND_PARITY_ROWS, block.ids.shape[0])
    ids, emb = block.ids[:rows], block.embedding[:rows]
    accelerator = predict(model, ids, batch_size=rows, doc_embed=emb)
    cpu = jax.devices("cpu")[0]
    host = np.asarray(
        predict_batch(
            on_cpu(model),
            jax.device_put(jnp.asarray(ids), cpu),
            jax.device_put(jnp.asarray(emb), cpu),
        )
    )
    delta = np.abs(accelerator - host)
    return {
        "rows": rows,
        "max_abs_score_delta": float(delta.max()),
        "mean_abs_score_delta": float(delta.mean()),
    }


def load_folded_model(args, result: dict) -> tuple[FastTransformer, Pretokenized]:
    """Load the scorer and the row pool, folding the donor table when asked.

    Shared by every accelerator mode: they all measure the same deployable model
    on the same rows, and the fold's score parity is asserted before any timing.
    """
    load_t0 = time.monotonic()
    scorer: PooledScorer = load_pooled_scorer(args.model_dir)
    model = scorer.model
    result["model_load_seconds"] = time.monotonic() - load_t0

    data_t0 = time.monotonic()
    block = load_pretokenized(args.pretokenized, args.pool_docs)
    result["data_load_seconds"] = time.monotonic() - data_t0
    result["pool_docs"] = int(block.ids.shape[0])
    result["max_tokens"] = int(block.ids.shape[1])

    if args.fold_donor:
        fold_t0 = time.monotonic()
        folded = fold_donor(model)
        result["fold_seconds"] = time.monotonic() - fold_t0
        probe = min(256, block.ids.shape[0])
        before = predict(model, block.ids[:probe], batch_size=probe, doc_embed=block.embedding[:probe])
        after = predict(folded, block.ids[:probe], batch_size=probe, doc_embed=block.embedding[:probe])
        result["fold_max_abs_score_delta"] = float(np.abs(before - after).max())
        del before, after, model, scorer
        gc.collect()
        model = folded

    result["params"] = array_bytes(model)
    result["flops_per_token"] = model.config.flops_per_token()
    result["flops_per_doc"] = model.config.flops_per_token() * int(block.ids.shape[1])
    return model, block


def throughput_row(docs_per_second: dict, flops_per_doc: float) -> dict:
    """Achieved arithmetic rate and MFU for a measured docs/s spread."""
    achieved = docs_per_second["mean"] * flops_per_doc
    return {
        "achieved_tflops": achieved / 1e12,
        "mfu_bf16_peak": achieved / H100_BF16_PEAK_FLOPS,
        "mfu_tf32_peak": achieved / H100_TF32_PEAK_FLOPS,
    }


def gpu_forward(args) -> dict:
    """Sweep batch size on this task's visible devices.

    Each batch is measured twice: device-resident (the forward alone) and host-fed
    through ``predict`` (the forward plus the H2D copy and per-call dispatch a
    pipeline pays).

    The sweep stops at the first batch that exhausts HBM and records where that
    was. On an H100 that ceiling is not reached: the naive footprint of the
    ``[batch, max_tokens, embed_dim]`` embedding activation would be 34 GB at
    batch 65,536, but XLA fuses the table gather into the pooling reduction so it
    never materializes, and resident HBM stays near 1.2 GB from batch 512 to
    131,072. Batch size here is bounded by diminishing returns, not by memory.
    """
    result: dict = {"fold_donor": args.fold_donor}
    if args.host_cores:
        result.update(pin_to_cores(args.host_cores, args.core_offset))
    model, block = load_folded_model(args, result)
    result["device"] = device_facts()

    arm = SCALEUP_ARMS[args.arm]
    result["arm"] = arm.name
    result["arm_overrides"] = arm.overrides
    result["xla_flags"] = os.environ.get("XLA_FLAGS", "")
    if arm.overrides:
        if not args.fold_donor:
            raise ValueError("scale-up arms are defined on the folded deployment config; pass --fold-donor")
        armed, trained_weights = arm_model(model, arm)
        result["arm_carries_trained_weights"] = trained_weights
        if trained_weights:
            result["arm_score_delta"] = arm_score_delta(model, armed, block)
        del model
        gc.collect()
        model = armed
        result["params"] = array_bytes(model)
        result["flops_per_token"] = model.config.flops_per_token()
        result["flops_per_doc"] = model.config.flops_per_token() * int(block.ids.shape[1])
    result["config"] = dataclasses.asdict(model.config)

    # Opt-in: the comparison compiles the forward for XLA:CPU, which on this model
    # costs minutes against the accelerator's ~2 s, and every packed child would
    # pay it. It answers a correctness question, not a throughput one, so run it
    # once rather than on every point of a scaling sweep.
    if args.backend_parity:
        parity_t0 = time.monotonic()
        result["backend_parity"] = backend_score_delta(model, block)
        result["backend_parity"]["seconds"] = time.monotonic() - parity_t0

    rows = []
    for batch in args.batches:
        if batch > block.ids.shape[0]:
            logger.warning("skipping batch %d: only %d rows loaded", batch, block.ids.shape[0])
            continue
        # A batch that exhausts HBM is a data point, not a failure: it marks the
        # ceiling the activation footprint imposes. Nothing past it can fit, so
        # the sweep stops rather than retrying larger shapes.
        try:
            compile_t0 = time.monotonic()
            batches = resident_batches(block, batch)
            jax.block_until_ready(predict_batch(model, *batches[0]))
            compile_seconds = time.monotonic() - compile_t0
            if batch == args.hlo_batch:
                result["hlo"] = compiled_analysis(model, *batches[0])
                logger.info("BENCH gpu-forward-hlo %s", json.dumps(result["hlo"]))
            if args.start_at:
                time.sleep(max(0.0, args.start_at - time.time()))
            resident = spread(
                [d / s for s, d in (time_resident_forward(model, batches, args.min_seconds) for _ in range(args.passes))]
            )
            host_fed = spread(
                [d / s for s, d in (time_forward(model, block, batch, args.min_seconds) for _ in range(args.passes))]
            )
        except jax.errors.JaxRuntimeError as exc:
            logger.warning("batch %d exhausted device memory: %s", batch, exc)
            rows.append({"batch": batch, "out_of_memory": True, "error": str(exc)[:400]})
            break
        row = {
            "batch": batch,
            "out_of_memory": False,
            "compile_seconds": compile_seconds,
            "resident_docs_per_second": resident,
            "host_fed_docs_per_second": host_fed,
            "host_fed_fraction_of_resident": host_fed["mean"] / resident["mean"],
            **hbm_bytes(),
            **throughput_row(resident, result["flops_per_doc"]),
        }
        rows.append(row)
        logger.info("BENCH gpu-forward %s", json.dumps(row))
        del batches
        gc.collect()

    result["batches"] = rows
    result["peak_rss_bytes"] = peak_rss_bytes()
    result["host"] = host_facts()
    result["task_resources"] = dataclasses.asdict(TaskResources.from_environment())
    result["iris_task_id"] = os.environ.get("IRIS_TASK_ID", "")
    return result


def gpu_fanout(
    mode: str, child_argv: list[list[str]], start_at: float, child_env: list[dict] | None = None
) -> list[dict]:
    """Run one child per entry, child *i* holding only device *i*, timed together.

    ``CUDA_VISIBLE_DEVICES`` has to be set in the child's environment: JAX picks
    its devices when the backend is first built, which is long before any flag
    this module parses could take effect. ``child_env`` carries anything else that
    has to be in place that early, notably ``XLA_FLAGS``.
    """
    procs = []
    for index, argv in enumerate(child_argv):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(index), **(child_env[index] if child_env else {})}
        procs.append(
            subprocess.Popen(
                [sys.executable, "-m", MODULE, mode, *argv, "--start-at", str(start_at)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    results = []
    for index, proc in enumerate(procs):
        out, err = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"gpu {index} child failed ({proc.returncode}):\n{out[-3000:]}\n{err[-3000:]}")
        results.append(
            next(
                json.loads(line[len(BENCH_JSON_PREFIX) :])
                for line in out.splitlines()
                if line.startswith(BENCH_JSON_PREFIX)
            )
        )
    return results


def gpu_pack(args) -> dict:
    """One scorer process per GPU on this node, timed against a shared start.

    ``--start-at`` is an absolute unix time so the same value can be handed to
    every replica of a multi-node job; the aggregate is then one wall-clock
    window across the whole gang rather than a sum of unrelated intervals.
    """
    start_at = args.start_at or time.time() + args.warmup_seconds
    argv = [
        "--model-dir",
        args.model_dir,
        "--pretokenized",
        args.pretokenized,
        "--batches",
        str(args.batch),
        "--pool-docs",
        str(args.pool_docs),
        "--passes",
        str(args.passes),
        "--min-seconds",
        str(args.min_seconds),
    ]
    if args.fold_donor:
        argv.append("--fold-donor")
    children = [
        [*argv, "--host-cores", str(args.host_cores_per_gpu), "--core-offset", str(i * args.host_cores_per_gpu)]
        for i in range(args.gpus)
    ]
    results = gpu_fanout("gpu-forward", children, start_at)
    per_gpu = [r["batches"][0]["resident_docs_per_second"]["mean"] for r in results]
    flops_per_doc = results[0]["flops_per_doc"]
    aggregate = sum(per_gpu)
    return {
        "gpus": args.gpus,
        "batch": args.batch,
        "host_cores_per_gpu": args.host_cores_per_gpu,
        "aggregate_docs_per_second": aggregate,
        "docs_per_second_per_gpu": aggregate / args.gpus,
        "per_gpu_docs_per_second": per_gpu,
        "aggregate_tflops": aggregate * flops_per_doc / 1e12,
        "aggregate_mfu_bf16_peak": aggregate * flops_per_doc / (args.gpus * H100_BF16_PEAK_FLOPS),
        "compile_seconds": [r["batches"][0]["compile_seconds"] for r in results],
        "model_load_seconds": [r["model_load_seconds"] for r in results],
        "iris_task_id": os.environ.get("IRIS_TASK_ID", ""),
        "host": host_facts(),
    }


def gemm_shapes(config, batch: int) -> list[tuple[str, int, int, int]]:
    """The ``(name, m, k, n)`` of every matmul one forward runs, at ``batch`` documents."""
    s = config.num_super_tokens + (1 if config.doc_embed_super_token else 0)
    rows = batch * s
    d = config.hidden_dim
    return [
        ("input_proj", batch * config.num_super_tokens, config.pool_out_dim, d),
        ("qkv", rows, d, 3 * d),
        ("attn_out", rows, d, d),
        ("mlp_w1", rows, d, d * config.mlp_ratio),
        ("mlp_w2", rows, d * config.mlp_ratio, d),
    ]


def time_gemm(m: int, k: int, n: int, out_dtype, min_seconds: float) -> dict:
    """Steady-state TFLOP/s of one isolated bf16 gemm with f32 accumulation."""
    key = jr.PRNGKey(0)
    a = jr.normal(key, (m, k), dtype=COMPUTE_DTYPE)
    b = jr.normal(jr.fold_in(key, 1), (k, n), dtype=COMPUTE_DTYPE)
    fn = jax.jit(lambda x, y: jnp.matmul(x, y, preferred_element_type=jnp.float32).astype(out_dtype))
    jax.block_until_ready(fn(a, b))
    calls = 0
    t0 = time.monotonic()
    pending: list = []
    while time.monotonic() - t0 < min_seconds:
        pending.append(fn(a, b))
        calls += 1
        if len(pending) > GPU_PIPELINE_DEPTH:
            jax.block_until_ready(pending.pop(0))
    jax.block_until_ready(pending)
    seconds = time.monotonic() - t0
    flops = 2.0 * m * k * n * calls
    del a, b
    gc.collect()
    return {
        "m": m,
        "k": k,
        "n": n,
        "calls": calls,
        "seconds": seconds,
        "tflops": flops / seconds / 1e12,
        "mfu_bf16_peak": flops / seconds / H100_BF16_PEAK_FLOPS,
    }


def time_streaming_bandwidth(array_bytes_target: int, min_seconds: float) -> dict:
    """Achieved HBM bandwidth of a read-modify-write over one large array."""
    n = array_bytes_target // 4
    x = jnp.full((n,), 1.0, dtype=jnp.float32)
    fn = jax.jit(lambda a: a * 1.0000001)
    jax.block_until_ready(fn(x))
    calls = 0
    t0 = time.monotonic()
    pending: list = []
    while time.monotonic() - t0 < min_seconds:
        pending.append(fn(x))
        calls += 1
        if len(pending) > GPU_PIPELINE_DEPTH:
            jax.block_until_ready(pending.pop(0))
    jax.block_until_ready(pending)
    seconds = time.monotonic() - t0
    moved = 2.0 * n * 4 * calls  # one read and one write per element
    del x
    gc.collect()
    return {"array_bytes": n * 4, "calls": calls, "seconds": seconds, "bytes_per_second": moved / seconds}


def gpu_micro(args) -> dict:
    """The two roofline endpoints this trunk is measured against, on this GPU.

    The model's own matmuls, run in isolation with nothing between them, are the
    throughput a perfectly fused forward of these shapes could reach; the
    streaming bandwidth is what the memory system delivers when nothing else is
    in the way. Together they say whether a low MFU is the shapes' fault or the
    traffic between them.
    """
    scorer = load_pooled_scorer(args.model_dir)
    config = dataclasses.replace(scorer.model.config, frozen_donor_dim=0)
    del scorer
    gc.collect()
    result: dict = {
        "device": device_facts(),
        "batch": args.batch,
        "streaming_bandwidth": time_streaming_bandwidth(args.array_bytes, args.min_seconds),
        "h100_hbm3_peak_bytes_per_second": H100_HBM_PEAK_BYTES_PER_SECOND,
    }
    shapes: dict[str, list] = {}
    for label, overrides in (("baseline", {}), ("a1_d512", D512), ("a2_d512_mlp8", {**D512, "mlp_ratio": 8})):
        armed = dataclasses.replace(config, **overrides)
        rows = []
        for name, m, k, n in gemm_shapes(armed, args.batch):
            for out_dtype in (jnp.float32, COMPUTE_DTYPE):
                rows.append(
                    {
                        "gemm": name,
                        "out_dtype": str(jnp.dtype(out_dtype)),
                        **time_gemm(m, k, n, out_dtype, args.min_seconds),
                    }
                )
            logger.info("BENCH gpu-micro %s %s", label, json.dumps(rows[-2:]))
        shapes[label] = rows
    result["gemms"] = shapes
    result["streaming_bandwidth_fraction_of_peak"] = (
        result["streaming_bandwidth"]["bytes_per_second"] / H100_HBM_PEAK_BYTES_PER_SECOND
    )
    result["host"] = host_facts()
    return result


def arm_frontier_row(arm_name: str, child: dict, baseline_docs_per_second: float | None) -> dict:
    """One line of the params-versus-throughput frontier, at the arm's best batch."""
    ran = [b for b in child["batches"] if not b["out_of_memory"]]
    best = max(ran, key=lambda b: b["resident_docs_per_second"]["mean"]) if ran else {}
    docs = best.get("resident_docs_per_second", {}).get("mean")
    params = child["params"]
    row = {
        "arm": arm_name,
        "best_batch": best.get("batch"),
        "docs_per_second_per_gpu": docs,
        "achieved_tflops": best.get("achieved_tflops"),
        "mfu_bf16_peak": best.get("mfu_bf16_peak"),
        "flops_per_doc": child["flops_per_doc"],
        "trunk_params": params["trunk_params"],
        "table_params": params["table_params"],
        "table_bytes": params["embedding_table_bytes"],
        "params_total_bytes": params["params_total_bytes"],
        "peak_hbm_bytes": max((b["hbm_peak_bytes_cumulative"] for b in ran), default=0),
        "host_fed_fraction_of_resident": best.get("host_fed_fraction_of_resident"),
        "carries_trained_weights": child.get("arm_carries_trained_weights", True),
        "score_delta": child.get("arm_score_delta"),
        "hlo": child.get("hlo"),
    }
    if docs and baseline_docs_per_second:
        row["slowdown_vs_baseline"] = baseline_docs_per_second / docs
        row["holds_baseline_throughput"] = docs >= baseline_docs_per_second
    return row


def gpu_arms(args) -> dict:
    """One arm per GPU on this node, in waves when there are more arms than GPUs.

    Every arm is measured on the same node against the same row pool, so the
    frontier is one comparison rather than a stack of unrelated runs. ``baseline``
    belongs in the arm list: it is the control that says whether this node
    reproduces the number the frontier is quoted against.
    """
    unknown = [a for a in args.arms if a not in SCALEUP_ARMS]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; known: {sorted(SCALEUP_ARMS)}")
    shared = [
        "--model-dir",
        args.model_dir,
        "--pretokenized",
        args.pretokenized,
        "--batches",
        ",".join(str(b) for b in args.batches),
        "--pool-docs",
        str(args.pool_docs),
        "--passes",
        str(args.passes),
        "--min-seconds",
        str(args.min_seconds),
        "--hlo-batch",
        str(args.hlo_batch),
        "--fold-donor",
    ]
    results: dict[str, dict] = {}
    waves = [args.arms[i : i + args.gpus] for i in range(0, len(args.arms), args.gpus)]
    for wave_index, wave in enumerate(waves):
        argv = [
            [
                *shared,
                "--arm",
                name,
                "--host-cores",
                str(args.host_cores_per_gpu),
                "--core-offset",
                str(i * args.host_cores_per_gpu),
            ]
            for i, name in enumerate(wave)
        ]
        env = [{"XLA_FLAGS": f"{os.environ.get('XLA_FLAGS', '')} {SCALEUP_ARMS[n].xla_flags}".strip()} for n in wave]
        logger.info("arm wave %d/%d: %s", wave_index + 1, len(waves), wave)
        start_at = time.time() + args.warmup_seconds
        for name, child in zip(wave, gpu_fanout("gpu-forward", argv, start_at, env), strict=True):
            results[name] = child
            # The frontier row, not the child: a full child result carries every
            # batch's spread and the HLO census, which does not survive a log line.
            logger.info("BENCH gpu-arms-child %s", json.dumps(arm_frontier_row(name, child, None)))

    baseline = results.get("baseline", {})
    ran = [b for b in baseline.get("batches", []) if not b["out_of_memory"]]
    baseline_docs = max((b["resident_docs_per_second"]["mean"] for b in ran), default=None)
    return {
        "arms": args.arms,
        "batches": args.batches,
        "baseline_docs_per_second_per_gpu": baseline_docs,
        "frontier": [arm_frontier_row(n, results[n], baseline_docs) for n in args.arms],
        "per_arm": results,
        "host": host_facts(),
        "iris_task_id": os.environ.get("IRIS_TASK_ID", ""),
    }


# ---------------------------------------------------------------------------
# pipeline: S3 -> decode -> H2D -> forward, end to end
# ---------------------------------------------------------------------------


@dataclass
class FeedCounters:
    """What the readers delivered and what the consumer waited on."""

    docs: int = 0
    bytes_read: int = 0
    queue_wait_seconds: float = 0.0
    forward_seconds: float = 0.0
    reader_seconds: list[float] = field(default_factory=list)


def feed_shards(paths: list[str], blocks: queue.Queue, repeat: int) -> float:
    """Read, decode and normalize this reader's shards onto the queue.

    The int8 -> float32 L2 normalize runs here rather than on the device so the
    host cost the CPU study also paid stays in the host's column, and so the
    per-reader work is the same work a CPU scoring task does.
    """
    fs = fsspec.filesystem("s3")
    t0 = time.monotonic()
    for _ in range(repeat):
        for path in paths:
            with fs.open(path, "rb", cache_type="none") as raw:
                counting = CountingFile(raw)
                parquet = pq.ParquetFile(counting)
                seen_bytes = 0
                for batch in parquet.iter_batches(batch_size=READ_BATCH, columns=["ids", "embedding"]):
                    n = batch.num_rows
                    width = len(batch.column("ids")[0])
                    ids = batch.column("ids").flatten().to_numpy(zero_copy_only=False).reshape(n, width)
                    stored = batch.column("embedding").flatten().to_numpy(zero_copy_only=False).reshape(n, EMBED_DIM)
                    blocks.put((ids, embedding_matrix(stored), counting.bytes_read - seen_bytes))
                    seen_bytes = counting.bytes_read
    blocks.put(None)
    return time.monotonic() - t0


def pipeline(args) -> dict:
    """S3 read -> arrow decode -> embedding normalize -> H2D -> forward, timed as one loop.

    ``queue_wait_seconds`` is the consumer sitting idle because no decoded rows
    were ready; ``forward_seconds`` is it dispatching and copying. Their ratio is
    the compute-bound-versus-feed-bound answer, and ``--reader-threads`` swept
    against it is how many host cores one GPU needs.
    """
    result: dict = {"fold_donor": args.fold_donor}
    if args.host_cores:
        result.update(pin_to_cores(args.host_cores, args.core_offset))
    model, block = load_folded_model(args, result)
    result["device"] = device_facts()

    _, _, batch_shard = data_parallel_shardings()
    compile_t0 = time.monotonic()
    warm = resident_batches(block, args.batch)[0]
    jax.block_until_ready(predict_batch(model, *warm))
    result["compile_seconds"] = time.monotonic() - compile_t0
    del warm, block
    gc.collect()

    # One pipeline process per GPU takes a disjoint stride of the corpus, so eight
    # of them on a node contend for read bandwidth rather than for the same keys.
    shards = pretokenized_shards(args.pretokenized)[args.shard_offset :: args.shard_stride]
    slices = [shards[i :: args.reader_threads] for i in range(args.reader_threads)]
    blocks: queue.Queue = queue.Queue(maxsize=args.queue_blocks)
    counters = FeedCounters()

    if args.start_at:
        time.sleep(max(0.0, args.start_at - time.time()))
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.reader_threads) as pool:
        futures = [pool.submit(feed_shards, s, blocks, args.repeat) for s in slices]
        ids_buffer: list[np.ndarray] = []
        emb_buffer: list[np.ndarray] = []
        buffered = 0
        finished = 0
        pending: list = []
        while finished < args.reader_threads:
            wait_t0 = time.monotonic()
            item = blocks.get()
            counters.queue_wait_seconds += time.monotonic() - wait_t0
            if item is None:
                finished += 1
                continue
            ids, emb, nbytes = item
            ids_buffer.append(ids)
            emb_buffer.append(emb)
            buffered += ids.shape[0]
            counters.bytes_read += nbytes
            while buffered >= args.batch:
                forward_t0 = time.monotonic()
                all_ids = np.concatenate(ids_buffer)
                all_emb = np.concatenate(emb_buffer)
                ids_buffer, emb_buffer = [all_ids[args.batch :]], [all_emb[args.batch :]]
                buffered -= args.batch
                pending.append(
                    predict_batch(
                        model,
                        jax.device_put(jnp.asarray(all_ids[: args.batch]), batch_shard),
                        jax.device_put(jnp.asarray(all_emb[: args.batch]), batch_shard),
                    )
                )
                if len(pending) > GPU_PIPELINE_DEPTH:
                    jax.block_until_ready(pending.pop(0))
                counters.forward_seconds += time.monotonic() - forward_t0
                counters.docs += args.batch
        jax.block_until_ready(pending)
        counters.reader_seconds = [f.result() for f in futures]
    elapsed = time.monotonic() - t0

    achieved = counters.docs / elapsed * result["flops_per_doc"]
    result.update(
        {
            "batch": args.batch,
            "reader_threads": args.reader_threads,
            "queue_blocks": args.queue_blocks,
            "repeat": args.repeat,
            "shards": len(shards),
            "docs": counters.docs,
            "wall_seconds": elapsed,
            "docs_per_second": counters.docs / elapsed,
            "megabytes_per_second": counters.bytes_read / elapsed / 1e6,
            "bytes_per_doc": counters.bytes_read / max(1, counters.docs),
            "queue_wait_seconds": counters.queue_wait_seconds,
            "forward_seconds": counters.forward_seconds,
            "starved_fraction": counters.queue_wait_seconds / elapsed,
            "reader_seconds": counters.reader_seconds,
            "achieved_tflops": achieved / 1e12,
            "mfu_bf16_peak": achieved / H100_BF16_PEAK_FLOPS,
            "peak_rss_bytes": peak_rss_bytes(),
            "host": host_facts(),
            "iris_task_id": os.environ.get("IRIS_TASK_ID", ""),
        }
    )
    logger.info("BENCH pipeline %s", json.dumps({k: v for k, v in result.items() if not isinstance(v, dict)}))
    return result


def pipeline_pack(args) -> dict:
    """One end-to-end pipeline process per GPU, on disjoint host cores and shards."""
    start_at = args.start_at or time.time() + args.warmup_seconds
    argv = [
        "--model-dir",
        args.model_dir,
        "--pretokenized",
        args.pretokenized,
        "--batch",
        str(args.batch),
        "--pool-docs",
        str(args.pool_docs),
        "--reader-threads",
        str(args.reader_threads),
        "--repeat",
        str(args.repeat),
        # Every GPU in the gang takes a disjoint stride of the corpus, so a
        # multi-node run measures the object store under 48 distinct readers
        # rather than 6 nodes re-reading the same keys out of their own caches.
        "--shard-stride",
        str(args.gpus * args.node_count),
    ]
    if args.fold_donor:
        argv.append("--fold-donor")
    children = [
        [
            *argv,
            "--shard-offset",
            str(args.node_index * args.gpus + i),
            "--host-cores",
            str(args.host_cores_per_gpu),
            "--core-offset",
            str(i * args.host_cores_per_gpu),
        ]
        for i in range(args.gpus)
    ]
    results = gpu_fanout("pipeline", children, start_at)
    per_gpu = [r["docs_per_second"] for r in results]
    return {
        "gpus": args.gpus,
        "node_index": args.node_index,
        "node_count": args.node_count,
        "batch": args.batch,
        "reader_threads_per_gpu": args.reader_threads,
        "host_cores_per_gpu": args.host_cores_per_gpu,
        "aggregate_docs_per_second": sum(per_gpu),
        "per_gpu_docs_per_second": per_gpu,
        "aggregate_megabytes_per_second": sum(r["megabytes_per_second"] for r in results),
        "starved_fraction": [r["starved_fraction"] for r in results],
        "iris_task_id": os.environ.get("IRIS_TASK_ID", ""),
        "host": host_facts(),
    }


def replicate(args) -> dict:
    """Server-side copy the fixture shards to ``--copies`` distinct keys.

    The rows repeat, which costs a feed measurement nothing: read bandwidth,
    parquet decode and LOTA admission all depend on object bytes and object
    count, not on whether two objects hold the same documents.
    """
    fs = fsspec.filesystem("s3")
    sources = pretokenized_shards(args.pretokenized)
    out_root = args.out.rstrip("/")
    targets = [
        (src, f"{out_root}/part-{copy * len(sources) + index:05d}.parquet")
        for copy in range(args.copies)
        for index, src in enumerate(sources)
    ]
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda pair: fs.copy(pair[0], pair[1]), targets))
    total = sum(fs.info(dst)["size"] for _, dst in targets)
    return {
        "out": out_root,
        "source_shards": len(sources),
        "objects": len(targets),
        "bytes": total,
        "seconds": time.monotonic() - t0,
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

    gfw = sub.add_parser("gpu-forward", help="batch-size sweep on this task's accelerators")
    gfw.add_argument("--model-dir", required=True)
    gfw.add_argument("--pretokenized", required=True)
    gfw.add_argument("--batches", type=lambda s: [int(x) for x in s.split(",")], default=list(DEFAULT_GPU_BATCHES))
    gfw.add_argument("--pool-docs", type=int, default=131_072)
    gfw.add_argument("--passes", type=int, default=3)
    gfw.add_argument("--min-seconds", type=float, default=8.0)
    gfw.add_argument("--fold-donor", action="store_true")
    gfw.add_argument(
        "--backend-parity",
        action="store_true",
        help="also score the probe rows on XLA:CPU and report the delta (adds a slow CPU compile)",
    )
    gfw.add_argument("--host-cores", type=int, default=0, help="host CPUs to pin to (0 leaves the process unpinned)")
    gfw.add_argument("--core-offset", type=int, default=0)
    gfw.add_argument("--start-at", type=float, default=0.0, help="unix time to begin timing, after compile")
    gfw.add_argument("--arm", default="baseline", choices=sorted(SCALEUP_ARMS), help="scale-up shape to measure")
    gfw.add_argument("--hlo-batch", type=int, default=0, help="batch to run XLA cost/HLO analysis on (0 skips)")

    arms = sub.add_parser("gpu-arms", help="one scale-up arm per GPU, on one node")
    arms.add_argument("--model-dir", required=True)
    arms.add_argument("--pretokenized", required=True)
    arms.add_argument("--arms", type=lambda s: s.split(","), default=sorted(SCALEUP_ARMS))
    arms.add_argument("--gpus", type=int, default=8)
    arms.add_argument("--batches", type=lambda s: [int(x) for x in s.split(",")], default=list(DEFAULT_GPU_BATCHES))
    arms.add_argument("--pool-docs", type=int, default=131_072)
    arms.add_argument("--passes", type=int, default=3)
    arms.add_argument("--min-seconds", type=float, default=8.0)
    arms.add_argument("--hlo-batch", type=int, default=8192)
    arms.add_argument("--host-cores-per-gpu", type=int, default=14)
    arms.add_argument("--warmup-seconds", type=float, default=240.0)

    micro = sub.add_parser("gpu-micro", help="isolated gemm and streaming-bandwidth roofline endpoints")
    micro.add_argument("--model-dir", required=True)
    micro.add_argument("--batch", type=int, default=8192, help="documents the gemm shapes are sized for")
    micro.add_argument("--min-seconds", type=float, default=3.0)
    micro.add_argument("--array-bytes", type=int, default=4 * 1024**3, help="working set of the bandwidth probe")

    gpk = sub.add_parser("gpu-pack", help="one scorer process per GPU on this node")
    gpk.add_argument("--model-dir", required=True)
    gpk.add_argument("--pretokenized", required=True)
    gpk.add_argument("--gpus", type=int, required=True)
    gpk.add_argument("--batch", type=int, required=True)
    gpk.add_argument("--host-cores-per-gpu", type=int, default=16)
    gpk.add_argument("--pool-docs", type=int, default=131_072)
    gpk.add_argument("--passes", type=int, default=3)
    gpk.add_argument("--min-seconds", type=float, default=8.0)
    gpk.add_argument("--warmup-seconds", type=float, default=180.0, help="grace for load+compile before timing")
    gpk.add_argument("--start-at", type=float, default=0.0, help="absolute unix start, shared across a gang")
    gpk.add_argument("--fold-donor", action="store_true")

    pipe = sub.add_parser("pipeline", help="S3 read -> decode -> H2D -> forward, end to end")
    pipe.add_argument("--model-dir", required=True)
    pipe.add_argument("--pretokenized", required=True)
    pipe.add_argument("--batch", type=int, default=8192)
    pipe.add_argument("--reader-threads", type=int, default=16)
    pipe.add_argument("--queue-blocks", type=int, default=DEFAULT_FEED_QUEUE_BLOCKS)
    pipe.add_argument("--repeat", type=int, default=1, help="sweeps over this process's shard slice")
    pipe.add_argument("--shard-offset", type=int, default=0)
    pipe.add_argument("--shard-stride", type=int, default=1)
    pipe.add_argument("--pool-docs", type=int, default=16_384, help="rows loaded up front, only to compile the shape")
    pipe.add_argument("--fold-donor", action="store_true")
    pipe.add_argument("--host-cores", type=int, default=0)
    pipe.add_argument("--core-offset", type=int, default=0)
    pipe.add_argument("--start-at", type=float, default=0.0)

    ppk = sub.add_parser("pipeline-pack", help="one end-to-end pipeline process per GPU")
    ppk.add_argument("--model-dir", required=True)
    ppk.add_argument("--pretokenized", required=True)
    ppk.add_argument("--gpus", type=int, required=True)
    ppk.add_argument("--batch", type=int, default=8192)
    ppk.add_argument("--reader-threads", type=int, default=14)
    ppk.add_argument("--host-cores-per-gpu", type=int, default=16)
    ppk.add_argument("--pool-docs", type=int, default=16_384)
    ppk.add_argument("--repeat", type=int, default=1)
    ppk.add_argument("--node-index", type=int, default=0, help="this replica's rank in a multi-node gang")
    ppk.add_argument("--node-count", type=int, default=1, help="replicas in the gang")
    ppk.add_argument("--warmup-seconds", type=float, default=180.0)
    ppk.add_argument("--start-at", type=float, default=0.0)
    ppk.add_argument("--fold-donor", action="store_true")

    rep = sub.add_parser("replicate", help="server-side copy the fixture into more distinct keys")
    rep.add_argument("--pretokenized", required=True)
    rep.add_argument("--out", required=True)
    rep.add_argument("--copies", type=int, required=True)
    rep.add_argument("--workers", type=int, default=64)

    rd = sub.add_parser("read", help="sustained read throughput over the pretokenized shards")
    rd.add_argument("--pretokenized", required=True)
    rd.add_argument("--readers", type=int, default=1, help="concurrent reader processes on this node")
    rd.add_argument("--reader-index", type=int, default=0)
    rd.add_argument("--reader-count", type=int, default=1)
    rd.add_argument("--max-shards", type=int, default=0)
    rd.add_argument("--repeat", type=int, default=2, help="passes over the slice; pass 0 is the cold read")
    rd.add_argument("--prestage", action="store_true", help="warm LOTA over the slice before timing")
    rd.add_argument("--prestage-workers", type=int, default=32)

    for parser in (arms, micro):
        parser.add_argument("--out", default="", help="path to write the full result JSON to (logs truncate it)")

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
    elif args.mode == "gpu-forward":
        result = gpu_forward(args)
    elif args.mode == "gpu-pack":
        result = gpu_pack(args)
    elif args.mode == "gpu-arms":
        result = gpu_arms(args)
    elif args.mode == "gpu-micro":
        result = gpu_micro(args)
    elif args.mode == "pipeline":
        result = pipeline(args)
    elif args.mode == "pipeline-pack":
        result = pipeline_pack(args)
    elif args.mode == "replicate":
        result = replicate(args)
    elif args.readers > 1:
        result = read_fanout(args)
    else:
        result = read_stream(args)
    if getattr(args, "out", ""):
        with open_url(args.out, "w") as fh:
            fh.write(json.dumps(result))
        logger.info("wrote %s", args.out)
    print(BENCH_JSON_PREFIX + json.dumps(result))


if __name__ == "__main__":
    main()
