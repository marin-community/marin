# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The fast scoring pipeline: read + tokenize on the v6e host threads, forward on the chips.

A v6e-4 VM has 4 chips *and* a 180-vCPU host. The forward is dispatch-bound (~1% MXU) and
scoring a shard's windows on-device costs a few seconds, so throughput is host-bound -- by
the GCS parquet read/decode and by tokenization, not the model. Both parallelize across the
host cores with an ordinary thread pool (the arrow read and the Rust tokenizer both release
the GIL), so this stage keeps one shard on one Zephyr worker and drives its own pool:

  - Each parquet row group is read (``id`` + ``text`` columns only, arrow-native, no per-row
    dict materialization) and windowed by a pool thread, so the ~400 MB text download and
    decode run row-group-parallel instead of as one serial stream.
  - As each row group lands, its docs are cut into ``device_batch``-window blocks and each
    block is submitted as its own tokenize task -- so tokenization fans out across the whole
    pool (not just the ~11 row groups), each thread tokenizing single-threaded with the HF
    rayon pool disabled.
  - A stager thread pulls finished blocks and ships them to the chips (``device_put``); the
    main thread only forwards the staged device arrays and reduces windows -> per-doc scores.
    Reads overlap tokenization; the stager's H2D transfers then overlap the forward and reduce.

Weights/vocab come from the config-faithful scorer dir.
"""

import functools
import json
import logging
import multiprocessing as mp
import os
import posixpath
import queue
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import shared_memory

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq
from fray.cluster import ResourceConfig
from rigging.filesystem import StoragePath, open_url
from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.runners import InlineRunner
from zephyr.writers import ThreadedBatchWriter, write_parquet_file

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from experiments.datakit.cluster.quality.fast_transformer.artifact import BUCKET_EDGES
from experiments.datakit.cluster.quality.fast_transformer.inference import (
    _predict_batch,
    data_parallel_shardings,
)
from experiments.datakit.cluster.quality.fast_transformer.scorer import load_pooled_scorer
from experiments.datakit.cluster.quality.fast_transformer.tpu_bench import tokenize_worker
from experiments.datakit.cluster.quality.fast_transformer.tpu_bench.common import (
    DEFAULT_CORPUS,
    doc_windows,
    load_remap_meta,
    load_shared_tokenizer,
    remap_to_array,
    resolve_dataset_path,
    write_result_json,
)

logger = logging.getLogger(__name__)

READ_COLUMNS = ["id", "text"]
# Default vCPUs requested per worker of the accelerator host. TPU: 128 of a v6e-4's 180 (leave
# headroom so the worker co-schedules with other tenants vs forcing a fresh slice). GPU: a
# smaller share of an 8x-GPU node's host, still plenty for the tokenizer fork pool.
TPU_HOST_CPU = 128
GPU_HOST_CPU = 96
# How many staged (H2D'd) blocks the stager thread may run ahead of the forward. Bounds the
# resident device memory while keeping the chips fed; the forward is never blocked on H2D.
STAGE_QUEUE_DEPTH = 8


@functools.cache
def _load_worker(model_dir: str, device_batch: int, max_tokens: int, calib_file: str):
    """Parent-side warm: scorer + calibration + compiled forward. Tokenization lives in the
    fork children, so the parent never loads the tokenizer or remap."""
    scorer = load_pooled_scorer(model_dir)
    with open_url(str(StoragePath(model_dir) / calib_file), "r") as fh:
        calib = json.loads(fh.read())
    xk = np.asarray(calib["xk"], dtype=np.float64)
    yk = np.asarray(calib["yk"], dtype=np.float64)
    ndev, _, batch_shard = data_parallel_shardings()
    # Warm the compile for the padded launch shape.
    warm = jax.device_put(jnp.zeros((device_batch, max_tokens), dtype=jnp.int32), batch_shard)
    jax.block_until_ready(_predict_batch(scorer.model, warm))
    logger.info("fast worker warm: %d chips, device_batch=%d tok=%d", ndev, device_batch, max_tokens)
    return scorer, xk, yk, batch_shard


@functools.cache
def _get_read_pool(read_threads: int) -> ThreadPoolExecutor:
    """Thread pool for the row-group arrow reads (the read + parquet decode release the GIL)."""
    return ThreadPoolExecutor(max_workers=read_threads, thread_name_prefix="rg-read")


@functools.cache
def _get_fork_pool(model_dir: str, tok_procs: int) -> ProcessPoolExecutor:
    """Fork tokenizer pool -- true multi-core tokenization off the GIL.

    ``fork`` (not ``spawn``): the Zephyr worker's ``__main__`` is the unguarded iris actor
    bootstrap, so spawn's ``_check_not_importing_main`` aborts. Fork is safe only because this
    runs BEFORE the parent initializes JAX (``_load_worker``) -- children inherit a pre-TPU
    image and never touch the chips. The tokenizer + remap are warmed here first so forked
    children inherit them copy-on-write instead of re-staging.
    """
    remap, tokenizer_name, _ = load_remap_meta(model_dir)
    load_shared_tokenizer(tokenizer_name)
    remap_to_array(remap)
    ctx = mp.get_context("fork")
    pool = ProcessPoolExecutor(
        max_workers=tok_procs, mp_context=ctx, initializer=tokenize_worker.child_init, initargs=(model_dir,)
    )
    # Force children to spin up (and bind the tokenizer) now, still before JAX init.
    list(pool.map(tokenize_worker.child_warm, range(tok_procs)))
    logger.info("tokenizer fork pool: %d procs", tok_procs)
    return pool


def _num_row_groups(path: str) -> int:
    with open_url(path, "rb") as f:
        return pq.ParquetFile(f).metadata.num_row_groups


def _read_and_window_rg(path: str, rg_index: int) -> tuple[list, list[list[str]]]:
    """Read one row group's ``id``/``text`` columns and window each doc (runs on a read thread)."""
    with open_url(path, "rb") as f:
        table = pq.ParquetFile(f).read_row_group(rg_index, columns=READ_COLUMNS)
    ids = table.column("id").to_pylist()
    texts = table.column("text").to_pylist()
    return ids, [doc_windows(t or "") for t in texts]


def _iter_blocks(doc_ids: list, doc_win: list[list[str]], device_batch: int):
    """Cut a row group's docs into blocks of <= ``device_batch`` windows, at doc boundaries.

    Yields ``(block_ids, win_texts, win_doc)`` where ``win_doc[i]`` is the block-local doc
    index of window ``i`` -- everything a block needs to reduce its windows back to docs.
    """
    block_ids: list = []
    win_texts: list[str] = []
    win_doc: list[int] = []
    for doc_id, windows in zip(doc_ids, doc_win, strict=True):
        if win_texts and len(win_texts) + len(windows) > device_batch:
            yield block_ids, win_texts, np.asarray(win_doc, dtype=np.int64)
            block_ids, win_texts, win_doc = [], [], []
        local = len(block_ids)
        block_ids.append(doc_id)
        win_texts.extend(windows)
        win_doc.extend([local] * len(windows))
    if win_texts:
        yield block_ids, win_texts, np.asarray(win_doc, dtype=np.int64)


def _read_and_put(shm_name: str, shape: tuple[int, int], batch_shard, device_batch: int):
    """Copy a child's packed block out of shared memory, pad to ``device_batch``, ship H2D.

    The copy lets us ``unlink`` the segment immediately; ``device_put`` then owns the data.
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        packed = np.ndarray(shape, dtype=np.int32, buffer=shm.buf)
        n = shape[0]
        out = np.zeros((device_batch, shape[1]), dtype=np.int32)
        out[:n] = packed
    finally:
        shm.close()
        shm.unlink()
    return jax.device_put(out, batch_shard), n


def _forward_and_reduce(model, block_ids, win_doc, dev, n_real, xk, yk, stats) -> list[dict]:
    """Forward a staged device block and reduce its windows to per-doc score rows."""
    t1 = time.perf_counter()
    win_scores = np.asarray(_predict_batch(model, dev))[:n_real]  # launch forward + D2H
    stats["forward_s"] += time.perf_counter() - t1
    stats["n_windows"] += n_real

    n_docs = len(block_ids)
    sums = np.zeros(n_docs, dtype=np.float64)
    cnts = np.zeros(n_docs, dtype=np.int64)
    np.add.at(sums, win_doc, win_scores)
    np.add.at(cnts, win_doc, 1)
    raw = sums / np.maximum(cnts, 1)
    cal = np.interp(raw, xk, yk)
    buckets = np.digitize(cal, BUCKET_EDGES)
    return [{"id": block_ids[i], "score": float(cal[i]), "quality_bucket": int(buckets[i])} for i in range(n_docs)]


def _score_file(path, model_dir, device_batch, tok_procs, read_threads, calib_file):
    """Score one parquet file's docs; returns ``(rows, stats)``.

    ``rows`` are per-doc ``{id, score, quality_bucket}``; ``stats`` carries the read/forward/
    compute timings and the token/window/doc counts.
    """
    # Fork pool BEFORE JAX init (children must inherit a pre-TPU image).
    fork_pool = _get_fork_pool(model_dir, tok_procs)
    read_pool = _get_read_pool(read_threads)
    scorer, xk, yk, batch_shard = _load_worker(model_dir, device_batch, 512, calib_file)
    stats = {"read_s": 0.0, "forward_s": 0.0, "compute_s": 0.0, "n_tokens": 0, "n_windows": 0}

    n_rg = _num_row_groups(path)
    read_futs = [read_pool.submit(_read_and_window_rg, path, i) for i in range(n_rg)]
    tok_futs: dict = {}
    t_read = time.perf_counter()
    for rf in as_completed(read_futs):
        doc_ids, doc_win = rf.result()
        for block_ids, win_texts, win_doc in _iter_blocks(doc_ids, doc_win, device_batch):
            tok_futs[fork_pool.submit(tokenize_worker.child_pack, win_texts)] = (block_ids, win_doc)
    stats["read_s"] = time.perf_counter() - t_read

    # Stage (shm copy + H2D) on a helper thread so the forward thread never waits on the transfer.
    staged: queue.Queue = queue.Queue(maxsize=STAGE_QUEUE_DEPTH)
    stage_err: list[Exception] = []

    def _stage() -> None:
        try:
            for tf in as_completed(tok_futs):
                block_ids, win_doc = tok_futs[tf]
                shm_name, shape, n_tokens = tf.result()
                dev, n_real = _read_and_put(shm_name, shape, batch_shard, device_batch)
                staged.put((block_ids, win_doc, dev, n_real, n_tokens))
        except Exception as e:  # surface to the main thread instead of wedging its get()
            stage_err.append(e)
        finally:
            staged.put(None)

    rows: list[dict] = []
    t_compute = time.perf_counter()
    stager = threading.Thread(target=_stage, name="stage-h2d")
    stager.start()
    while (item := staged.get()) is not None:
        block_ids, win_doc, dev, n_real, n_tokens = item
        stats["n_tokens"] += n_tokens
        rows.extend(_forward_and_reduce(scorer.model, block_ids, win_doc, dev, n_real, xk, yk, stats))
    stager.join()
    if stage_err:
        raise stage_err[0]
    stats["compute_s"] = time.perf_counter() - t_compute
    stats["n_docs"] = len(rows)
    return rows, stats


def _write_rows(rows: list[dict], out_file: str) -> dict:
    result: dict = {}

    def _sink(items):
        result.update(write_parquet_file(items, output_path=out_file))

    with ThreadedBatchWriter(_sink) as w:
        for row in rows:
            w.submit(row)
    return result


def _writer(output_path, model_dir, device_batch, calib_file, tok_procs, read_threads):
    def writer(paths: Iterator[str], shard: ShardInfo) -> Iterator[dict]:
        for path in paths:
            t0 = time.perf_counter()
            rows, stats = _score_file(path, model_dir, device_batch, tok_procs, read_threads, calib_file)
            out_file = str(StoragePath(output_path) / "outputs" / "main" / posixpath.basename(path))
            t_write = time.perf_counter()
            result = _write_rows(rows, out_file)
            write_s = time.perf_counter() - t_write
            shard_s = time.perf_counter() - t0
            logger.info(
                "shard %s: %.1fs (read %.1f, compute %.1f, write %.1f) -> %d docs, %.0f docs/s",
                posixpath.basename(path),
                shard_s,
                stats["read_s"],
                stats["compute_s"],
                write_s,
                stats["n_docs"],
                stats["n_docs"] / shard_s if shard_s else 0,
            )
            counters.pipeline.update_counter("fast/docs", stats["n_docs"])
            counters.pipeline.update_counter("fast/windows", stats["n_windows"])
            counters.pipeline.update_counter("fast/tokens", stats["n_tokens"])
            counters.pipeline.update_counter("fast/read_ms", int(stats["read_s"] * 1000))
            counters.pipeline.update_counter("fast/forward_ms", int(stats["forward_s"] * 1000))
            counters.pipeline.update_counter("fast/compute_ms", int(stats["compute_s"] * 1000))
            counters.pipeline.update_counter("fast/write_ms", int(write_s * 1000))
            counters.pipeline.update_counter("fast/shard_ms", int(shard_s * 1000))
            yield {"shard_file": posixpath.basename(path), "docs": stats["n_docs"], **result}

    return writer


def _resolve_resources(accelerator: str, worker_cpu: int | None) -> tuple[ResourceConfig, int]:
    """Resolve an accelerator request to ``(ResourceConfig, chips_per_worker)``.

    ``accelerator`` is a TPU type (``"v6e-4"``, ``"v5litepod-16"``) or a GPU ``VARIANTxCOUNT``
    (``"H100x8"``). The forward runs data-parallel across the worker's chips either way.
    """
    if "x" in accelerator:  # GPU, e.g. "H100x8"
        variant, _, count = accelerator.partition("x")
        cpu = worker_cpu if worker_cpu is not None else GPU_HOST_CPU
        return ResourceConfig.with_gpu(variant, count=int(count), cpu=cpu, ram=f"{cpu * 4}g"), int(count)
    cpu = worker_cpu if worker_cpu is not None else TPU_HOST_CPU
    return ResourceConfig.with_tpu(accelerator, cpu=cpu, ram="400g"), int(accelerator.rsplit("-", 1)[1])


def run(
    *,
    corpus_glob,
    model_dir,
    output_path,
    max_files,
    max_workers,
    device_batch,
    tok_procs,
    read_threads,
    calib_file,
    result_json,
    accelerator="v6e-4",
    worker_cpu=None,
):
    corpus_glob = resolve_dataset_path(corpus_glob or DEFAULT_CORPUS)
    model_dir = resolve_dataset_path(model_dir)
    files = sorted(str(m) for m in StoragePath(corpus_glob).glob())[:max_files]
    if not files:
        raise ValueError(f"no files matched {corpus_glob}")
    resources, chips_per_worker = _resolve_resources(accelerator, worker_cpu)
    logger.info(
        "fast: %d files, %d %s workers, device_batch=%d tok_procs=%d read_threads=%d",
        len(files),
        max_workers,
        accelerator,
        device_batch,
        tok_procs,
        read_threads,
    )
    pipeline = Dataset.from_list(files).map_shard(
        _writer(output_path, model_dir, device_batch, calib_file, tok_procs, read_threads)
    )
    ctx = ZephyrContext(
        name="ft-fast",
        resources=resources,
        max_workers=max_workers,
        stage_runner_factory=InlineRunner,
        heartbeat_timeout=1800,
    )
    t0 = time.time()
    agg = dict(ctx.execute(pipeline).counters)
    wall = time.time() - t0
    docs = agg.get("fast/docs", 0)
    tokens = agg.get("fast/tokens", 0)
    n_chips = chips_per_worker * max_workers
    payload = {
        "stage": "fast",
        "accelerator": accelerator,
        "cpu_per_worker": resources.cpu,
        "tok_procs": tok_procs,
        "read_threads": read_threads,
        "files": len(files),
        "workers": max_workers,
        "n_chips": n_chips,
        "device_batch": device_batch,
        "wall_s": round(wall, 1),
        "docs": docs,
        "windows": agg.get("fast/windows", 0),
        "tokens": tokens,
        "docs_per_s": round(docs / wall, 1) if wall else 0,
        "tokens_per_s": round(tokens / wall, 1) if wall else 0,
        "tokens_per_s_per_chip": round(tokens / wall / n_chips, 1) if wall and n_chips else 0,
        "read_worker_s": round(agg.get("fast/read_ms", 0) / 1000, 1),
        "forward_worker_s": round(agg.get("fast/forward_ms", 0) / 1000, 1),
        "compute_worker_s": round(agg.get("fast/compute_ms", 0) / 1000, 1),
        "write_worker_s": round(agg.get("fast/write_ms", 0) / 1000, 1),
        "shard_worker_s": round(agg.get("fast/shard_ms", 0) / 1000, 1),
        "counters": agg,
    }
    print("BENCH " + json.dumps(payload), flush=True)
    if result_json:
        write_result_json(result_json, payload)
    return payload
