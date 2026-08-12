# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure production-shaped scoring throughput for a pooled fast-transformer.

Runs the deployment loop — read corpus parquet from the object store, cut the
scoring windows, tokenize, forward — over a fixed multi-source sample of the
harrier 50M text sample, timing every stage separately *and* the whole loop end
to end, then normalizing by the CPU allotment the task actually holds
(:class:`iris.env_resources.TaskResources`, not ``os.cpu_count()``, which on a
Kubernetes task reports the whole node).

Two scoring shapes, selected by ``--windows``:

* ``bme`` — the deployed text-only path: mean over begin/middle/end
  ``CHUNK_CHARS`` windows (:func:`scorer.bme_chunks`), so a document costs up to
  three forward rows.
* ``begin`` — the fusion arms' path: one ``max_tokens`` window per document,
  with the stored 1024-d harrier embedding fed alongside. A model whose config
  carries ``doc_embed_dim`` requires the embedding, so the run also measures the
  embedding column read and the int8→float32 normalization separately.

``--tokenizer-backend`` selects the HF tokenizer or the gigatoken Rust backend;
gigatoken runs behind the same exact token-id parity gate the training arms use,
so a measured speedup is never a speedup on different ids.

Every stage is timed over ``--passes`` independent passes so spread is visible,
and the first forward of each shape is timed separately from the rest — XLA
compiles once per (batch, sequence) shape and that cost is warmup, not steady
state.
"""

import argparse
import json
import logging
import os
import statistics
import time
from dataclasses import asdict, dataclass, field

import fsspec
import numpy as np
import pyarrow.parquet as pq
from iris.env_resources import TaskResources
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.data import (
    encode_texts,
    encode_texts_fast,
    load_gigatoken,
    load_tokenizer,
    pack,
)
from experiments.datakit.cluster.quality.fast_transformer.embed_exp import check_gigatoken_parity
from experiments.datakit.cluster.quality.fast_transformer.inference import predict
from experiments.datakit.cluster.quality.fast_transformer.joined_labels import EMBED_DIM, embedding_matrix
from experiments.datakit.cluster.quality.fast_transformer.score import BATCH_SIZE as RECORD_BATCH
from experiments.datakit.cluster.quality.fast_transformer.scorer import PooledScorer, bme_chunks, load_pooled_scorer

logger = logging.getLogger(__name__)

BASE_URL = "s3://marin-us-east-02a/marin/datakit/samples/harrier-oss-v1-0.6b-50m-text-v1"
# PooledScorer.score's own batch: the deployed forward runs this many window rows
# at a time. `predict` then pads each chunk up to its token-budget batch, so the
# padding waste of the deployed shape is measured rather than assumed.
DEPLOYED_SCORE_BATCH = 256
TEXT_COLUMNS = ["id", "text"]
EMBED_COLUMNS = ["id", "text", "embedding"]
MAX_SOURCE_DEPTH = 3
BENCH_JSON_PREFIX = "BENCH_JSON "


class CountingFile:
    """File wrapper that counts bytes returned by ``read`` and time spent in it."""

    def __init__(self, inner):
        self._inner = inner
        self.bytes_read = 0
        self.read_seconds = 0.0

    def read(self, *args):
        t0 = time.monotonic()
        data = self._inner.read(*args)
        self.read_seconds += time.monotonic() - t0
        self.bytes_read += len(data)
        return data

    def __getattr__(self, name):
        return getattr(self._inner, name)


@dataclass(frozen=True)
class Sample:
    """The fixed document sample every configuration is measured on."""

    texts: list[str]
    sources: list[str]
    embedding: np.ndarray | None  # [n, EMBED_DIM] int8, as stored


@dataclass
class ReadStats:
    seconds: float = 0.0
    io_seconds: float = 0.0
    rows: int = 0  # rows actually decoded (a row group is read whole, then sliced)
    bytes_read: int = 0


@dataclass
class StageTimes:
    """Accumulated per-stage seconds and work counts for one scoring pass."""

    window: float = 0.0
    tokenize: float = 0.0
    embed: float = 0.0
    forward: float = 0.0
    docs: int = 0
    rows: int = 0  # forward rows (windows), >= docs under bme
    tokens: int = 0  # non-pad tokens actually carrying content
    padded_tokens: int = 0  # tokens the forward computed on, padding included
    forward_batches: list[float] = field(default_factory=list)


def list_shards(base_url: str, num_sources: int, shards_per_source: int) -> list[tuple[str, str]]:
    """``(source, shard_url)`` pairs, spread deterministically across sources.

    Listing walks single-level globs only: a recursive glob makes s3fs
    ``HeadObject`` the prefix, which the CoreWeave object store answers with 400.
    """
    root = base_url.rstrip("/")
    sources = sorted(str(p).rstrip("/") for p in StoragePath(f"{root}/*").glob())
    if not sources:
        raise ValueError(f"no source directories under {root}")
    stride = max(1, len(sources) // num_sources)
    chosen = sources[::stride][:num_sources]
    out: list[tuple[str, str]] = []
    for source in chosen:
        found: list[str] = []
        level = [source]
        for _ in range(MAX_SOURCE_DEPTH):
            nxt: list[str] = []
            for d in level:
                for entry in sorted(str(m) for m in StoragePath(f"{d}/*").glob()):
                    (found if entry.endswith(".parquet") else nxt).append(entry)
            if found or not nxt:
                break
            level = nxt
        if not found:
            raise ValueError(f"no parquet shards under {source}")
        name = source.rsplit("/", 1)[-1]
        out.extend((name, shard) for shard in found[:shards_per_source])
    return out


def embedding_rows(batch) -> np.ndarray:
    """The stored int8 embedding column of one record batch as ``[n, EMBED_DIM]``.

    Flatten-then-reshape off the Arrow buffer, not ``to_pylist``: the column is a
    fixed-size list, and materializing 1024 Python ints per document costs more
    than every other read stage combined.
    """
    flat = batch.column("embedding").flatten().to_numpy(zero_copy_only=False)
    return flat.reshape(batch.num_rows, EMBED_DIM)


def iter_document_batches(shards: list[tuple[str, str]], quota: int, columns: list[str], stats: ReadStats):
    """Stream ``(source, texts, embedding)`` batches, up to ``quota`` rows per shard.

    ``iter_batches`` rather than whole row groups: the pipeline consumes these
    files as a stream and stops when it has enough, so decoding a 5k-row group to
    keep 500 of it would charge the read stage for rows nothing ever scores.
    Arrow-to-Python conversion is inside the timer because the scoring path pays
    it too; the clock stops across the yield so consumer time never lands here.
    """
    fs = fsspec.filesystem("s3")
    for source, shard in shards:
        taken = 0
        t0 = time.monotonic()
        with fs.open(shard, "rb", cache_type="none") as raw:
            counting = CountingFile(raw)
            parquet = pq.ParquetFile(counting)
            for batch in parquet.iter_batches(batch_size=RECORD_BATCH, columns=columns):
                keep = min(batch.num_rows, quota - taken)
                batch = batch.slice(0, keep)
                texts = [t or "" for t in batch.column("text").to_pylist()]
                embedding = embedding_rows(batch) if "embedding" in columns else None
                taken += keep
                stats.rows += keep
                stats.seconds += time.monotonic() - t0
                yield source, texts, embedding
                t0 = time.monotonic()
                if taken >= quota:
                    break
            stats.bytes_read += counting.bytes_read
            stats.io_seconds += counting.read_seconds
        stats.seconds += time.monotonic() - t0


def read_documents(shards: list[tuple[str, str]], quota: int, columns: list[str]) -> tuple[Sample, ReadStats]:
    """The fixed sample, plus the cost of reading it."""
    stats = ReadStats()
    texts: list[str] = []
    sources: list[str] = []
    embeddings: list[np.ndarray] = []
    for source, batch_texts, embedding in iter_document_batches(shards, quota, columns, stats):
        texts.extend(batch_texts)
        sources.extend([source] * len(batch_texts))
        if embedding is not None:
            embeddings.append(embedding)
    sample = Sample(texts=texts, sources=sources, embedding=np.concatenate(embeddings) if embeddings else None)
    return sample, stats


def tokenize(backend: str, tokenizer_name: str, texts: list[str], max_tokens: int, remap: dict[int, int]) -> np.ndarray:
    """Padded ``[n, max_tokens]`` compact ids, exactly as the scoring path packs them."""
    encode = encode_texts_fast if backend == "gigatoken" else encode_texts
    raw = encode(tokenizer_name, texts, max_tokens)
    return pack(raw, remap, np.zeros(len(raw), dtype=np.float32), max_tokens).ids


def score_documents(
    scorer: PooledScorer,
    texts: list[str],
    embedding: np.ndarray | None,
    *,
    backend: str,
    windows: str,
    score_batch: int,
    times: StageTimes,
) -> np.ndarray:
    """Window, tokenize, and forward one group of documents, accumulating timings."""
    t0 = time.monotonic()
    if windows == "bme":
        flat, spans = bme_chunks(texts)
    else:
        flat, spans = list(texts), [(i, i + 1) for i in range(len(texts))]
    times.window += time.monotonic() - t0

    flat_embed = None
    if embedding is not None:
        t0 = time.monotonic()
        normalized = embedding_matrix(embedding)
        repeats = np.array([b - a for a, b in spans])
        flat_embed = np.repeat(normalized, repeats, axis=0)
        times.embed += time.monotonic() - t0

    scores = np.empty(len(flat), dtype=np.float32)
    for start in range(0, len(flat), score_batch):
        chunk = flat[start : start + score_batch]
        t0 = time.monotonic()
        ids = tokenize(backend, scorer.tokenizer_name, chunk, scorer.max_tokens, scorer.remap)
        times.tokenize += time.monotonic() - t0
        doc_embed = None if flat_embed is None else flat_embed[start : start + score_batch]
        t0 = time.monotonic()
        scores[start : start + len(chunk)] = predict(scorer.model, ids, doc_embed=doc_embed)
        elapsed = time.monotonic() - t0
        times.forward += elapsed
        times.forward_batches.append(elapsed)
        times.tokens += int((ids != 0).sum())
        times.padded_tokens += int(ids.size)
        times.rows += len(chunk)
    times.docs += len(texts)
    return np.array([scores[a:b].mean() for a, b in spans])


def token_stats(ids: np.ndarray) -> dict[str, float]:
    lengths = (ids != 0).sum(axis=1)
    return {
        "rows": len(lengths),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "max": int(lengths.max()),
        "truncated_frac": float((lengths >= ids.shape[1]).mean()),
    }


def spread(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "stdev": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--tokenizer-backend", choices=("hf", "gigatoken"), default="hf")
    p.add_argument("--windows", choices=("bme", "begin"), default="bme")
    p.add_argument("--base-url", default=BASE_URL)
    p.add_argument("--docs", type=int, default=20_000, help="documents in the fixed sample")
    p.add_argument("--sources", type=int, default=8, help="source directories the sample spans")
    p.add_argument("--shards-per-source", type=int, default=1)
    p.add_argument("--passes", type=int, default=3, help="timing passes per stage")
    p.add_argument("--score-batch", type=int, default=DEPLOYED_SCORE_BATCH)
    p.add_argument("--probe-only", action="store_true", help="report the sample and model shape, then stop")
    p.add_argument(
        "--pin-cpu-affinity",
        action="store_true",
        help="restrict the process to as many CPUs as the task was granted. The cgroup cap is a CFS quota, "
        "not a cpuset, so XLA otherwise sizes its thread pool from the whole node and oversubscribes the quota",
    )
    args = p.parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()

    resources = TaskResources.from_environment()
    cores = resources.cpu_cores
    if args.pin_cpu_affinity:
        allowed = sorted(os.sched_getaffinity(0))[: max(1, int(cores))]
        os.sched_setaffinity(0, set(allowed))
    result: dict = {
        "model_dir": args.model_dir,
        "tokenizer_backend": args.tokenizer_backend,
        "windows": args.windows,
        "cpu_cores": cores,
        "host_cpu_count": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0)),
        "pin_cpu_affinity": args.pin_cpu_affinity,
        "memory_bytes": resources.memory_bytes,
        "docs_requested": args.docs,
        "score_batch": args.score_batch,
        "passes": args.passes,
    }

    shards = list_shards(args.base_url, args.sources, args.shards_per_source)
    quota = max(1, args.docs // len(shards))
    logger.info("sample: %d shards x %d rows across %d sources", len(shards), quota, args.sources)

    t0 = time.monotonic()
    scorer = load_pooled_scorer(args.model_dir)
    result["model_load_seconds"] = time.monotonic() - t0
    config = scorer.model.config
    needs_embed = bool(config.doc_embed_dim)
    result["model"] = {
        "tokenizer": scorer.tokenizer_name,
        "max_tokens": scorer.max_tokens,
        "vocab_size": config.vocab_size,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "pool_window": config.pool_window,
        "mlp_ratio": config.mlp_ratio,
        "doc_embed_dim": config.doc_embed_dim,
        "frozen_donor_dim": config.frozen_donor_dim,
        "flops_per_token": config.flops_per_token(),
    }
    logger.info("BENCH model %s", json.dumps(result["model"]))
    if needs_embed and args.windows != "begin":
        logger.warning("model takes a doc embedding; bme windows reuse one document embedding per window")

    columns = EMBED_COLUMNS if needs_embed else TEXT_COLUMNS
    sample, read_stats = read_documents(shards, quota, columns)
    result["sample"] = {
        "docs": len(sample.texts),
        "sources": sorted(set(sample.sources)),
        "chars_mean": float(np.mean([len(t) for t in sample.texts])),
        "chars_median": float(np.median([len(t) for t in sample.texts])),
        "rows_decoded": read_stats.rows,
        "bytes_read": read_stats.bytes_read,
    }
    logger.info("BENCH sample %s", json.dumps(result["sample"]))
    if args.probe_only:
        print(BENCH_JSON_PREFIX + json.dumps(result))
        return

    # Warmup: tokenizer construction, gigatoken parity, and the first forward of
    # each shape (XLA compiles once per shape) are startup costs, not throughput.
    t0 = time.monotonic()
    load_tokenizer(scorer.tokenizer_name)
    result["tokenizer_load_seconds"] = time.monotonic() - t0
    if args.tokenizer_backend == "gigatoken":
        t0 = time.monotonic()
        load_gigatoken(scorer.tokenizer_name)
        result["gigatoken_load_seconds"] = time.monotonic() - t0
        check_gigatoken_parity(scorer.tokenizer_name, sample.texts, np.array(sample.sources))

    warm = StageTimes()
    score_documents(
        scorer,
        sample.texts[: args.score_batch],
        None if sample.embedding is None else sample.embedding[: args.score_batch],
        backend=args.tokenizer_backend,
        windows=args.windows,
        score_batch=args.score_batch,
        times=warm,
    )
    result["warmup"] = {
        "first_forward_seconds": warm.forward_batches[0],
        "later_forward_seconds": warm.forward_batches[1:],
        "tokenize_seconds": warm.tokenize,
    }
    logger.info("BENCH warmup %s", json.dumps(result["warmup"]))

    # Isolated read stage: the same shards re-read per pass (pass 1 is cold, later
    # passes see whatever the node cached), plus a text-only read so the embedding
    # column's share of the read is separated rather than inferred.
    read_passes = []
    for _ in range(args.passes):
        _, stats = read_documents(shards, quota, columns)
        read_passes.append(stats.rows / stats.seconds)
    result["read_docs_per_second"] = spread(read_passes)
    result["read_passes"] = read_passes
    if needs_embed:
        _, text_only = read_documents(shards, quota, TEXT_COLUMNS)
        result["read_text_only"] = {
            "docs_per_second": text_only.rows / text_only.seconds,
            "bytes_read": text_only.bytes_read,
        }

    # Isolated tokenize + forward stages over the in-memory sample.
    stage_passes: list[StageTimes] = []
    ids_for_stats: np.ndarray | None = None
    for index in range(args.passes):
        times = StageTimes()
        for start in range(0, len(sample.texts), RECORD_BATCH):
            score_documents(
                scorer,
                sample.texts[start : start + RECORD_BATCH],
                None if sample.embedding is None else sample.embedding[start : start + RECORD_BATCH],
                backend=args.tokenizer_backend,
                windows=args.windows,
                score_batch=args.score_batch,
                times=times,
            )
        stage_passes.append(times)
        logger.info(
            "BENCH pass %d docs=%d rows=%d window=%.1fs tokenize=%.1fs embed=%.1fs forward=%.1fs",
            index,
            times.docs,
            times.rows,
            times.window,
            times.tokenize,
            times.embed,
            times.forward,
        )
        if ids_for_stats is None:
            flat = bme_chunks(sample.texts[:2048])[0] if args.windows == "bme" else sample.texts[:2048]
            ids_for_stats = tokenize(
                args.tokenizer_backend, scorer.tokenizer_name, flat, scorer.max_tokens, scorer.remap
            )

    docs = stage_passes[0].docs
    rows = stage_passes[0].rows
    result["tokens"] = token_stats(ids_for_stats)
    result["rows_per_doc"] = rows / docs
    result["stages"] = {
        "window_docs_per_second": spread([t.docs / t.window for t in stage_passes]),
        "tokenize_docs_per_second": spread([t.docs / t.tokenize for t in stage_passes]),
        "forward_docs_per_second": spread([t.docs / t.forward for t in stage_passes]),
        "forward_rows_per_second": spread([t.rows / t.forward for t in stage_passes]),
    }
    if needs_embed:
        result["stages"]["embed_docs_per_second"] = spread([t.docs / max(t.embed, 1e-9) for t in stage_passes])

    # End to end: read from the object store and score each batch as it arrives,
    # the way a worker runs it (one zephyr window of records at a time).
    e2e = []
    for _ in range(args.passes):
        times = StageTimes()
        read_stats = ReadStats()
        t0 = time.monotonic()
        for _, batch_texts, batch_embed in iter_document_batches(shards, quota, columns, read_stats):
            score_documents(
                scorer,
                batch_texts,
                batch_embed,
                backend=args.tokenizer_backend,
                windows=args.windows,
                score_batch=args.score_batch,
                times=times,
            )
        elapsed = time.monotonic() - t0
        e2e.append(
            {
                "seconds": elapsed,
                "read_seconds": read_stats.seconds,
                "docs": times.docs,
                "tokens": times.tokens,
                "padded_tokens": times.padded_tokens,
                "docs_per_second": times.docs / elapsed,
                "tokens_per_second": times.tokens / elapsed,
                "padded_tokens_per_second": times.padded_tokens / elapsed,
            }
        )
        logger.info("BENCH e2e %s", json.dumps(e2e[-1]))
    result["end_to_end"] = e2e
    result["end_to_end_docs_per_second"] = spread([r["docs_per_second"] for r in e2e])
    result["end_to_end_docs_per_second_per_core"] = spread([r["docs_per_second"] / cores for r in e2e])
    result["end_to_end_tokens_per_second_per_core"] = spread([r["tokens_per_second"] / cores for r in e2e])
    result["end_to_end_padded_tokens_per_second_per_core"] = spread([r["padded_tokens_per_second"] / cores for r in e2e])
    result["stage_passes"] = [asdict(t) | {"forward_batches": []} for t in stage_passes]
    print(BENCH_JSON_PREFIX + json.dumps(result))


if __name__ == "__main__":
    main()
