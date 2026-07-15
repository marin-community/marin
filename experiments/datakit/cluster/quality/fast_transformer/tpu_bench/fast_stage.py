# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The fast scoring pipeline: parallel-tokenize on the v6e host, forward on the chips.

A v6e-4 VM has 4 chips *and* a 180-vCPU host. The forward is dispatch-bound (~1% MXU) and
scoring a shard's windows on-device costs a few seconds, so throughput is host-bound. This
stage keeps everything on one Zephyr worker (one shard per worker) and parallelizes the
tokenization across the host cores:

  - ``--tok-procs 0``: tokenize on the main thread. The HF fast tokenizer is Rust/rayon
    multi-core, but ONLY when called from the main thread -- calling it from a worker
    thread silently drops to single-core.
  - ``--tok-procs N``: a fork process pool (``tokenize_worker``, jax-free so children never
    touch the TPU) tokenizes window batches across N cores; the main process pulls packed
    batches in order and runs the forward, so tokenization overlaps the forward.

With tokenization parallel, the per-shard wall is dominated by the GCS parquet data path
(read/decode the input, write the output), not by tokenize/forward/reduce. ``--cpu-only``
runs the same path with no TPU (the forward on the host CPUs) for the CPU baseline.
Weights/vocab come from the config-faithful scorer dir.
"""

import functools
import json
import logging
import multiprocessing as mp
import os
import posixpath
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
from fray.cluster import ResourceConfig
from rigging.filesystem import StoragePath, open_url
from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.readers import DEFAULT_FILE_PATH_COLUMN, load_file
from zephyr.runners import InlineRunner
from zephyr.writers import ThreadedBatchWriter, write_parquet_file

os.environ["TOKENIZERS_PARALLELISM"] = "true"

from experiments.datakit.cluster.quality.fast_transformer.artifact import BUCKET_EDGES
from experiments.datakit.cluster.quality.fast_transformer.inference import (
    _predict_batch,
    data_parallel_shardings,
)
from experiments.datakit.cluster.quality.fast_transformer.scorer import load_pooled_scorer
from experiments.datakit.cluster.quality.fast_transformer.tpu_bench import tokenize_worker
from experiments.datakit.cluster.quality.fast_transformer.tpu_bench.common import (
    doc_windows,
    load_remap_meta,
    pack_windows,
    remap_to_array,
)

logger = logging.getLogger(__name__)


@functools.cache
def _load_worker(model_dir: str, device_batch: int, max_tokens: int, calib_file: str):
    scorer = load_pooled_scorer(model_dir)
    remap, tokenizer_name, _ = load_remap_meta(model_dir)
    lut = remap_to_array(remap)
    with open_url(f"{model_dir.rstrip('/')}/{calib_file}", "r") as fh:
        calib = json.loads(fh.read())
    xk = np.asarray(calib["xk"], dtype=np.float64)
    yk = np.asarray(calib["yk"], dtype=np.float64)
    ndev, _, batch_shard = data_parallel_shardings()
    # Warm the compile for the padded launch shape.
    warm = jax.device_put(jnp.zeros((device_batch, max_tokens), dtype=jnp.int32), batch_shard)
    jax.block_until_ready(_predict_batch(scorer.model, warm))
    logger.info(
        "fast worker warm: %d chips, device_batch=%d tok=%d tokenizer=%s", ndev, device_batch, max_tokens, tokenizer_name
    )
    return scorer, lut, tokenizer_name, xk, yk, batch_shard


@functools.cache
def _get_pool(model_dir: str, tok_procs: int) -> ProcessPoolExecutor:
    """One tokenizer pool per worker (children are jax-free, load the tokenizer once).

    Uses ``fork``, not ``spawn``: the Zephyr worker's ``__main__`` is the (unguarded) iris
    actor bootstrap, so spawn's ``_check_not_importing_main`` aborts. Fork is safe here only
    because this is called BEFORE the parent initializes JAX (``_load_worker``) -- the child
    inherits a pre-TPU process image and never touches jax.
    """
    ctx = mp.get_context("fork")
    pool = ProcessPoolExecutor(
        max_workers=tok_procs, mp_context=ctx, initializer=tokenize_worker.child_init, initargs=(model_dir,)
    )
    # Force children to spin up (and load the tokenizer) now, still before JAX init.
    list(pool.map(tokenize_worker.child_warm, range(tok_procs)))
    logger.info("tokenizer pool: %d fork procs", tok_procs)
    return pool


def _stage_to_device(model, ids: np.ndarray, batch_shard, device_batch: int):
    """Pad ``ids`` to ``device_batch`` rows, ship H2D, launch forward -> (jax_out, n_real)."""
    n = ids.shape[0]
    if n < device_batch:
        ids = np.concatenate([ids, np.zeros((device_batch - n, ids.shape[1]), dtype=ids.dtype)], axis=0)
    dev = jax.device_put(jnp.asarray(ids), batch_shard)
    return _predict_batch(model, dev), n


def _score_shard(records, model_dir, device_batch, calib_file, tok_procs):
    """Score one shard's docs; returns (rows, stats). Tokenizes across ``tok_procs`` cores."""
    # Build the fork pool BEFORE JAX inits (children must inherit a pre-TPU process image).
    pool = _get_pool(model_dir, tok_procs) if tok_procs and tok_procs > 1 else None
    scorer, lut, tokenizer_name, xk, yk, batch_shard = _load_worker(model_dir, device_batch, 512, calib_file)
    max_tokens = scorer.max_tokens

    # Materialize the shard's window texts in order (a shard is ~one file). ``doc_windows``
    # always yields >=1 window, so ``ids_out`` (one id per input doc) preserves the exact
    # input row count, and ``doc_pos_of_win`` maps each window back to its doc for the reduce.
    stats = {"tokenize_s": 0.0, "forward_s": 0.0, "window_s": 0.0, "reduce_s": 0.0, "n_tokens": 0}
    t_win = time.perf_counter()
    win_texts: list[str] = []
    doc_pos_of_win: list[int] = []
    ids_out: list[object] = []
    for doc_pos, r in enumerate(records):
        ids_out.append(r["id"])
        for w in doc_windows(r.get("text") or ""):
            win_texts.append(w)
            doc_pos_of_win.append(doc_pos)
    stats["window_s"] = time.perf_counter() - t_win

    n_windows = len(win_texts)
    win_scores = np.empty(n_windows, dtype=np.float32)
    # One tokenize batch == one device launch. A large batch lets the tokenizer's rayon
    # pool fill the host cores on the main thread (its parallelism only engages there).
    starts = list(range(0, n_windows, device_batch))
    batches = [win_texts[s : s + device_batch] for s in starts]

    def forward(start: int, ids: np.ndarray):
        stats["n_tokens"] += int((ids != 0).sum())
        t1 = time.perf_counter()
        out, n = _stage_to_device(scorer.model, ids, batch_shard, device_batch)
        win_scores[start : start + n] = np.asarray(out)[:n]
        stats["forward_s"] += time.perf_counter() - t1

    t0 = time.perf_counter()
    if pool is not None:
        # Children tokenize in parallel; results stream back in submission order and are
        # forwarded on-device by the parent -> tokenization overlaps forward + reduce.
        for start, ids in zip(starts, pool.map(tokenize_worker.child_tokenize, batches), strict=True):
            forward(start, ids)
    else:
        for start, texts in zip(starts, batches, strict=True):
            forward(start, pack_windows(texts, tokenizer_name, lut, max_tokens))
    stats["tokenize_s"] = time.perf_counter() - t0 - stats["forward_s"]

    # Reduce windows -> doc (mean) with a vectorized scatter-add (np.add.at), in doc order.
    t_red = time.perf_counter()
    n_docs = len(ids_out)
    pos = np.asarray(doc_pos_of_win, dtype=np.int64)
    sums = np.zeros(n_docs, dtype=np.float64)
    cnts = np.zeros(n_docs, dtype=np.int64)
    np.add.at(sums, pos, win_scores)
    np.add.at(cnts, pos, 1)
    raw = sums / np.maximum(cnts, 1)
    cal = np.interp(raw, xk, yk)
    buckets = np.digitize(cal, BUCKET_EDGES)
    rows = [{"id": ids_out[i], "score": float(cal[i]), "quality_bucket": int(buckets[i])} for i in range(n_docs)]
    stats["reduce_s"] = time.perf_counter() - t_red
    stats["n_docs"] = n_docs
    stats["n_windows"] = n_windows
    return rows, stats


def _writer(output_path, model_dir, device_batch, calib_file, tok_procs):
    def writer(records: Iterator[dict], shard: ShardInfo) -> Iterator[dict]:
        records = iter(records)
        first = next(records, None)
        if first is None:
            return
        shard_file = posixpath.basename(first[DEFAULT_FILE_PATH_COLUMN])
        rows, stats = _score_shard([first, *records], model_dir, device_batch, calib_file, tok_procs)
        out_file = f"{output_path.rstrip('/')}/outputs/main/{shard_file}"
        result: dict = {}

        def _sink(items):
            result.update(write_parquet_file(items, output_path=out_file))

        with ThreadedBatchWriter(_sink) as w:
            for row in rows:
                w.submit(row)
        counters.pipeline.update_counter("fast/docs", stats["n_docs"])
        counters.pipeline.update_counter("fast/windows", stats["n_windows"])
        counters.pipeline.update_counter("fast/tokens", stats["n_tokens"])
        counters.pipeline.update_counter("fast/tokenize_ms", int(stats["tokenize_s"] * 1000))
        counters.pipeline.update_counter("fast/forward_ms", int(stats["forward_s"] * 1000))
        counters.pipeline.update_counter("fast/window_ms", int(stats["window_s"] * 1000))
        counters.pipeline.update_counter("fast/reduce_ms", int(stats["reduce_s"] * 1000))
        yield {"shard_file": shard_file, "docs": stats["n_docs"], **result}

    return writer


def run(
    *,
    corpus_glob,
    model_dir,
    output_path,
    max_files,
    max_workers,
    device_batch,
    tok_procs,
    calib_file,
    result_json,
    cpu_only=False,
    cpu=180,
):
    files = sorted(str(m) for m in StoragePath(corpus_glob).glob())[:max_files]
    if not files:
        raise ValueError(f"no files matched {corpus_glob}")
    logger.info(
        "fast: %d files, %d workers, device_batch=%d tok_procs=%d cpu_only=%s",
        len(files),
        max_workers,
        device_batch,
        tok_procs,
        cpu_only,
    )
    pipeline = (
        Dataset.from_list(files)
        .flat_map(functools.partial(load_file, include_file_paths=True))
        .map_shard(_writer(output_path, model_dir, device_batch, calib_file, tok_procs))
    )
    # cpu_only measures the "before" baseline the issue describes: the same forward + tokenize
    # path with no TPU, so JAX runs the forward on the host CPUs where it competes with
    # tokenization for cores (on a v6e the forward instead offloads to otherwise-idle chips).
    resources = (
        ResourceConfig(cpu=cpu, ram=f"{cpu * 3}g") if cpu_only else ResourceConfig.with_tpu("v6e-4", cpu=180, ram="600g")
    )
    ctx = ZephyrContext(
        name="ft-fast-cpu" if cpu_only else "ft-fast",
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
    n_chips = 0 if cpu_only else 4 * max_workers
    payload = {
        "stage": "fast_cpu" if cpu_only else "fast",
        "cpu_only": cpu_only,
        "cpu_per_worker": cpu if cpu_only else 180,
        "tok_procs": tok_procs,
        "files": len(files),
        "workers_v6e4": max_workers,
        "n_chips": n_chips,
        "device_batch": device_batch,
        "wall_s": round(wall, 1),
        "docs": docs,
        "windows": agg.get("fast/windows", 0),
        "tokens": tokens,
        "docs_per_s": round(docs / wall, 1) if wall else 0,
        "tokens_per_s": round(tokens / wall, 1) if wall else 0,
        "tokens_per_s_per_chip": round(tokens / wall / n_chips, 1) if wall and n_chips else 0,
        "tokenize_worker_s": round(agg.get("fast/tokenize_ms", 0) / 1000, 1),
        "forward_worker_s": round(agg.get("fast/forward_ms", 0) / 1000, 1),
        "window_worker_s": round(agg.get("fast/window_ms", 0) / 1000, 1),
        "reduce_worker_s": round(agg.get("fast/reduce_ms", 0) / 1000, 1),
        "counters": agg,
    }
    print("BENCH " + json.dumps(payload), flush=True)
    if result_json:
        with open_url(result_json, "w") as fh:
            fh.write(json.dumps(payload, indent=2))
    return payload
