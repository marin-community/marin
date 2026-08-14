# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score the Nemotron-tokenized corpus with the fusion quality scorer.

The corpus already carries everything the fusion scorer eats: ``datakit/tokenize``
holds Nemotron ``input_ids`` per document chunk and one complete harrier leaf per
source, named by ``hero_data.harrier``, holds the int8[1024] document embedding.
Both are hash-partitioned from one normalized source and share a shard count and
basename, so a shard pair joins on ``id`` with no shuffle. That co-partitioning is
checked rather than assumed: a source whose two sides came from different normalize
runs has no shard basenames in common, and ``manifest`` refuses to build rather than
letting its documents fall into an unembedded counter that reads as coverage.

Row order within a shard is *not* guaranteed and on at least one source is not
sorted, so the join sorts the embed side itself rather than reading the stored order
as an invariant (see :func:`join_shard`).

Five modes:

* ``fold`` -- collapse the frozen ``[vocab, 2048]`` donor table and its learned
  ``[2048, 256]`` projection into one ``[vocab, 256]`` table and write a new model
  dir. Gather and matmul commute row-wise so this is exact, not an approximation;
  the mode asserts score parity on real rows before writing. Deployment reads the
  folded dir, so 48 workers do not each redo a 1.1 GB read and a fold.
* ``manifest`` -- pair each source's Nemotron tokenize leaf with its harrier embed
  leaf and emit one row per (source, shard_index).
* ``score`` -- one worker, one GPU: take this worker's slice of the manifest and
  stream join -> score -> write for each shard pair it owns.
* ``node`` -- fan out one ``score`` subprocess per visible GPU.
* ``verify`` -- reconcile one source's written score shards against the manifest's
  per-shard embed row counts, and report the score distribution and bucket shares.

Reader *processes*, not threads. A measured 8xH100 study on this cluster found the
scoring loop feed-bound at ~795k docs/s/node with threaded readers: independent
processes sustain 2.2 GB/s per node against a threaded pipeline's 1.06 GB/s,
because the readers contend on the GIL rather than on the network. ``node`` gives
each GPU its own process with its own reader, which is where that bandwidth comes
back. Within a worker a small prefetch pool overlaps the next shard's read with the
current shard's forward; arrow decode releases the GIL, so a shallow pool does not
reintroduce the contention.

The corpus tokenizer (``NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16``) and the donor
tokenizer (``Nemotron-Flash-1B``) are distinct repos that share one 131,072-entry
vocabulary: every token present in both carries the same id, and the two differ
only in the *names* of 8 reserved special slots. Stored ids therefore index the
donor table directly. ``verify_vocab`` re-checks the invariant the remap encodes
rather than trusting it.
"""

import argparse
import dataclasses
import json
import logging
import os
import queue
import random
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import cast

import equinox as eqx
import fsspec
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from iris.cluster.client.job_info import get_job_info
from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import read_artifact
from rigging.filesystem import StoragePath, open_url
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit import hero_data
from experiments.datakit.cluster.quality.fast_transformer.data import NUM_RESERVED, PAD_ID, UNK_ID
from experiments.datakit.cluster.quality.fast_transformer.inference import predict
from experiments.datakit.cluster.quality.fast_transformer.model import COMPUTE_DTYPE, FastTransformer
from experiments.datakit.cluster.quality.fast_transformer.scorer import PooledScorer, artifact_names, load_pooled_scorer
from experiments.datakit.hero_data import NEMOTRON_88K

logger = logging.getLogger(__name__)

MODULE = "experiments.datakit.cluster.quality.fast_transformer.score_corpus"

EMBED_DIM = 1024
SPLIT = "train"
# The embed side is one complete leaf per source, named by `hero_data.harrier`.
# It supersedes the earlier pair of leaves -- one run over what the global fuzzy
# dedup kept, a second over what it dropped -- which had to be unioned to
# reconstruct the corpus. Measured on five sources, the merged leaf's row count
# equals the normalize document count exactly and its shard count equals the token
# side's, while the old pair summed to between 2 and 432 rows *more*: the union
# double-counted a handful of ids. The merged leaf is the corrected set, not a
# repackaging, so nothing unions anything here any more.
DEFAULT_MODEL_DIR = "s3://marin-us-east-02a/marin/user/muchanem/quality_exp/nemotron_donor"
DEFAULT_FOLDED_DIR = "s3://marin-us-east-02a/marin/user/muchanem/quality_scores_run/model/nemotron_88k_folded"
DEFAULT_MANIFEST = "s3://marin-us-east-02a/marin/user/muchanem/quality_scores_run/manifest"
# Cutpoints `calibrate_corpus` fitted through this scoring path; `verify` digitizes
# the written scores under the interior knots to report bucket shares.
DEFAULT_CALIBRATION = f"{DEFAULT_FOLDED_DIR}/calib_bme.json"

# Sources kept out of the manifest, mapped to why. A held source is not scored at
# all, which is safer than scoring it against a leaf chosen by guess, and the
# per-source output layout means adding one back later costs only its own shards.
#
# Empty: the focus crawl's hold is discharged. It was held because its embed leaf
# was the ed4b8bc9 normalize at 4,573 shards against a 333-shard token side with an
# empty basename intersection. Its merged leaf `..._fc8cffa4` now lists 333 shards
# against the token side's 333, with a measured intersection of 333 and nothing
# unmatched on either side, and carries exactly 22,010,234 rows -- the fixed
# normalize's document count, not the 36,327,068 of the older extraction.
HOLD_SOURCES: dict[str, str] = {}

# Rows per arrow record batch when streaming a token shard. The token side is the
# expanded one (a long document occupies several adjacent rows), so this is rows,
# not documents. The token side stays on pyarrow's `iter_batches` because it is the
# side that has to be consumed incrementally: one shard's dense token array reaches
# 16.5 GB, and this polars build exposes no batched parquet reader
# (`read_parquet_batched` and `LazyFrame.collect_iter` are both absent), so a polars
# read of a token shard would have to materialize it whole.
READ_BATCH = 4096
# Transient-fault retry. A shard read is the retry unit: it is idempotent, and the
# token stream cannot be resumed mid-body without double-emitting blocks that were
# already scored, so a reset restarts the shard and the consumer drops its partial
# state. Five attempts over a capped exponential backoff covers the observed
# one-reset-per-minute churn many times over without masking a real fault.
READ_ATTEMPTS = 5
RETRY_BASE_DELAY = 1.0
RETRY_MAX_DELAY = 30.0
# Documents per forward. HBM never exceeded 1.2 GB at any batch size measured --
# XLA fuses the table gather into the pooling -- so this is sized for dispatch
# amortization rather than against a memory ceiling. Passed to `predict`
# explicitly: a batch that changes shape between calls recompiles, and a
# 256-vs-512 shape was measured at 2.12x in the forward.
DEFAULT_BATCH = 32_768
# Shard reads run ahead of the forward by this many shards. Two, because the
# reader is CPU-bound in parquet decode rather than latency-bound: measured on
# one H100 over the same four shards, depth 2 gave 24,553 docs/s while depth 6
# gave 13,176, with the summed read time rising 40s -> 157s. Deeper prefetch
# convoys on the GIL instead of overlapping, which is the same knee the CPU
# study found. The way to more node throughput is more processes.
DEFAULT_PREFETCH_SHARDS = 2
# Documents per joined block. Shard sizes are extremely skewed -- mean 86,698
# documents but a maximum of 2,682,446, whose dense token and embedding arrays
# come to 16.5 GB -- so a worker that materialized whole shards would size its
# memory to the largest one. Blocking caps a worker's resident join at roughly
# `(prefetch + 1) * block_docs * 6 KB` whatever the shard holds.
DEFAULT_BLOCK_DOCS = 32_768
# The fold is exact in principle; assert it on real rows rather than trust it.
FOLD_PARITY_ROWS = 256
FOLD_PARITY_TOLERANCE = 1e-5
# Carried across when the donor-table model is folded into a plain embedding model.
# `config`, `donor_embed` and `donor_proj` are deliberately absent: the folded
# config has frozen_donor_dim=0 and the folded table replaces the pair.
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

# ---------------------------------------------------------------------------
# model: fold the donor table
# ---------------------------------------------------------------------------


def fold_donor(model: FastTransformer) -> FastTransformer:
    """Fold the frozen donor table and its learned projection into one table.

    The forward computes ``matmul(take(donor, ids), proj)``. Gather and matmul
    commute row-wise, so folding first and gathering after is the identical
    computation. With the Nemotron donor this drops the resident table from
    ``vocab*2048*4`` to ``vocab*256*4`` bytes and removes a
    ``[batch, max_tokens, 2048] @ [2048, 256]`` matmul from every forward.
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


def rss_bytes() -> int:
    """This process's resident set, for sizing the worker pool against the cap."""
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise ValueError("no VmRSS: line in /proc/self/status")


def verify_vocab(remap: dict[int, int]) -> int:
    """Assert the remap is the full-vocab identity offset and return the vocab size.

    ``train_exp.full_vocab_remap`` maps every raw tokenizer id to ``id +
    NUM_RESERVED``. Scoring exploits that: the remap becomes an add rather than a
    per-token dict lookup, which over ~6e12 tokens is the difference between a
    vectorized op and a Python loop. If a checkpoint ever ships a *compacted*
    remap this assert fires instead of silently scoring scrambled ids.
    """
    size = len(remap)
    expected = {t: t + NUM_RESERVED for t in range(size)}
    if remap != expected:
        wrong = [t for t in range(size) if remap.get(t) != t + NUM_RESERVED]
        raise ValueError(
            f"remap is not the full-vocab identity offset ({len(wrong)} of {size} entries differ, "
            f"e.g. {wrong[:5]}); score_corpus assumes raw_id + {NUM_RESERVED}"
        )
    return size


def fold_mode(args) -> dict:
    """Fold the trained checkpoint and write a deployable model dir."""
    scorer = load_pooled_scorer(args.model_dir)
    vocab = verify_vocab(scorer.remap)
    logger.info(
        "loaded %s: tokenizer=%s max_tokens=%d vocab=%d frozen_donor_dim=%d",
        args.model_dir,
        scorer.tokenizer_name,
        scorer.max_tokens,
        vocab,
        scorer.model.config.frozen_donor_dim,
    )

    rng = np.random.default_rng(0)
    probe_ids = rng.integers(NUM_RESERVED, vocab, size=(FOLD_PARITY_ROWS, scorer.max_tokens), dtype=np.int32)
    probe_embed = rng.normal(size=(FOLD_PARITY_ROWS, EMBED_DIM)).astype(np.float32)
    probe_embed /= np.maximum(np.linalg.norm(probe_embed, axis=1, keepdims=True), 1e-6)

    before = predict(scorer.model, probe_ids, batch_size=FOLD_PARITY_ROWS, doc_embed=probe_embed)
    t0 = time.monotonic()
    folded = fold_donor(scorer.model)
    fold_seconds = time.monotonic() - t0
    after = predict(folded, probe_ids, batch_size=FOLD_PARITY_ROWS, doc_embed=probe_embed)
    delta = float(np.abs(before - after).max())
    logger.info("fold took %.1fs; max abs score delta over %d rows = %.3e", fold_seconds, FOLD_PARITY_ROWS, delta)
    if delta > FOLD_PARITY_TOLERANCE:
        raise ValueError(f"fold changed scores by {delta:.3e} (> {FOLD_PARITY_TOLERANCE:.0e}); refusing to write")

    out_dir = args.out_dir.rstrip("/")
    eqx_name, remap_name, meta_name = artifact_names(args.stem)
    local = f"/tmp/{eqx_name}"
    eqx.tree_serialise_leaves(local, folded)
    size = os.path.getsize(local)
    with open(local, "rb") as fh, open_url(f"{out_dir}/{eqx_name}", "wb") as out:
        out.write(fh.read())
    with open_url(f"{out_dir}/{remap_name}", "w") as fh:
        fh.write(json.dumps({str(k): v for k, v in scorer.remap.items()}))
    meta = {
        "tokenizer": scorer.tokenizer_name,
        "max_tokens": scorer.max_tokens,
        "config": dataclasses.asdict(folded.config),
        "folded_from": args.model_dir,
        "fold_max_abs_score_delta": delta,
    }
    with open_url(f"{out_dir}/{meta_name}", "w") as fh:
        fh.write(json.dumps(meta, indent=2))
    logger.info("wrote folded model to %s (%.1f MB)", out_dir, size / 1e6)
    return {"out_dir": out_dir, "eqx_bytes": size, "fold_max_abs_score_delta": delta, "vocab": vocab}


# ---------------------------------------------------------------------------
# manifest: pair the tokenize and embed leaves
# ---------------------------------------------------------------------------


def manifest_mode(args) -> dict:
    """Pair every source's Nemotron token shards with its harrier embed shards.

    Every path comes from :mod:`hero_data`: it names the sources, resolves the
    tokenize leaf and the score output path from step identity, and maps each
    source to its one complete harrier leaf. Nothing is discovered by listing a
    stage root, so a leaf whose hash no longer matches current code, or whose
    artifact records no result at all, is still found.
    """
    fs = fsspec.filesystem("s3")
    registered = hero_data.source_names()
    sources = [s for s in registered if s not in HOLD_SOURCES]
    missing_embed = sorted(s for s in sources if s not in hero_data.harrier_paths())

    # Held sources are named and counted, never merely absent: a source that
    # silently drops out of the manifest reads downstream as a corpus that never
    # had it, which is the difference between "excluded" and "missing".
    held = {}
    for source, reason in HOLD_SOURCES.items():
        if source not in registered:
            raise ValueError(f"held source {source!r} is not registered; the hold is stale")
        counters = read_artifact(hero_data.normalized(source).output_path, NormalizedData).counters
        held[source] = {"reason": reason, "counters": counters}
        logger.warning("HELD OUT of the score manifest: %s -- %s; normalize counters %s", source, reason, counters)
    logger.info(
        "sources=%d of %d registered (%d held) no-embed=%s", len(sources), len(registered), len(held), missing_embed
    )
    if missing_embed:
        raise ValueError(f"no harrier leaf mapped for {len(missing_embed)} source(s): {missing_embed}")

    def leaf_rows(source: str) -> list[dict]:
        tok_dir = f"{hero_data.tokenized(source, NEMOTRON_88K.tokenizer).output_path.rstrip('/')}/{SPLIT}"
        out_dir = hero_data.quality(source, NEMOTRON_88K).output_path.rstrip("/")
        tok = sorted(fs.ls(tok_dir.removeprefix("s3://"), detail=True), key=lambda e: e["name"])
        tok = [e for e in tok if e["name"].endswith(".parquet")]
        embed_dir = hero_data.harrier(source).rstrip("/")
        present = {
            e["name"].rsplit("/", 1)[-1]: (f"s3://{e['name']}", e["size"])
            for e in fs.ls(embed_dir.removeprefix("s3://"), detail=True)
            if e["name"].endswith(".parquet")
        }
        rows = []
        for index, entry in enumerate(tok):
            base = entry["name"].rsplit("/", 1)[-1]
            found = present.get(base)
            rows.append(
                {
                    "source": source,
                    "shard_index": index,
                    "num_shards": len(tok),
                    "tokens_path": f"s3://{entry['name']}",
                    "embed_path": found[0] if found else "",
                    "output_path": f"{out_dir}/{base}",
                    "embed_leaves": 1 if found else 0,
                    "tokens_bytes": entry["size"],
                    "embed_bytes": found[1] if found else 0,
                }
            )
        return rows

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.discovery_threads) as pool:
        for got in pool.map(leaf_rows, sources):
            rows.extend(got)
    if not rows:
        raise ValueError("no shard tasks discovered; the stage layout likely moved")

    # A source whose token and embed sides were built from different normalize runs
    # has disjoint shard basenames, so every one of its shards pairs with nothing.
    # That is a geometry break, not an empty source, and it is invisible to row
    # counts and artifact metadata -- left alone it lands in `unembedded_shards` and
    # reads as coverage. Refuse the manifest instead, and name the source.
    matched_by_source: dict[str, int] = {}
    for row in rows:
        matched_by_source[row["source"]] = matched_by_source.get(row["source"], 0) + row["embed_leaves"]
    unpaired = sorted(s for s, matched in matched_by_source.items() if not matched)
    if unpaired:
        raise ValueError(
            f"{len(unpaired)} source(s) have an embed leaf but no shard basename in common with their "
            f"token side, so every document would resolve to no embedding: {unpaired}. "
            f"Hold them explicitly in HOLD_SOURCES or repartition one side."
        )

    # Footer reads, once, here: `embed_rows` is what `verify` compares the written
    # score count against, and computing it at verify time would reread the embed
    # side of a leaf that has already been scored. Two range GETs per shard.
    def embed_rows(row: dict) -> int:
        if not row["embed_path"]:
            return 0
        with fs.open(row["embed_path"], "rb", cache_type="none") as raw:
            return pq.ParquetFile(raw).metadata.num_rows

    with ThreadPoolExecutor(max_workers=args.footer_threads) as pool:
        for row, count in zip(rows, pool.map(embed_rows, rows), strict=True):
            row["embed_rows"] = count
    path = f"{args.manifest.rstrip('/')}/manifest.parquet"
    with open_url(path, "wb") as fh:
        pl.DataFrame(rows).write_parquet(fh)
    result = {
        "manifest": path,
        "tasks": len(rows),
        "sources": len(sources),
        "registered_sources": len(registered),
        "held_sources": held,
        "sources_without_embed": missing_embed,
        "embedded_shards": sum(1 for r in rows if r["embed_leaves"]),
        "unembedded_shards": sum(1 for r in rows if not r["embed_leaves"]),
        "embed_rows": sum(r["embed_rows"] for r in rows),
    }
    logger.info("wrote %s: %s", path, json.dumps(result, default=str))
    return result


def read_manifest(manifest: str) -> pl.DataFrame:
    root = manifest.rstrip("/")
    shards = sorted(str(p) for p in StoragePath(f"{root}/*.parquet").glob())
    if not shards:
        raise ValueError(f"no manifest parquet under {root}")
    frames = []
    for shard in shards:
        with StoragePath(shard).open("rb") as fh:
            frames.append(pl.read_parquet(fh))
    return pl.concat(frames, how="vertical_relaxed")


# ---------------------------------------------------------------------------
# the join
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardTask:
    source: str
    shard_index: int
    tokens_path: str
    embed_path: str
    output_path: str
    total_bytes: int


@dataclass(frozen=True)
class Block:
    """A bounded run of matched documents, ready for one or more forwards."""

    doc_ids: np.ndarray  # [n] object, the 32-char hex ids
    ids: np.ndarray  # [n, max_tokens] int32 compact ids, PAD-padded
    embedding: np.ndarray  # [n, EMBED_DIM] float32, L2-normalized


@dataclass(frozen=True)
class ShardRestart:
    """A shard is being re-read from the start; drop what it already produced."""


@dataclass(frozen=True)
class ShardStats:
    """What one shard pair cost and whether its join was complete."""

    token_rows: int  # rows read on the token side, before the chunk filter
    embed_rows: int  # rows on the embed side
    matched: int  # documents that joined
    read_seconds: float

    @property
    def unmatched_embed(self) -> int:
        """Embed rows the token side did not carry.

        Containment (embed shard k is a subset of token shard k) was validated on
        8 of 166,275 shard pairs, so the run measures it on all of them rather
        than assuming it: every embedded document has a chunk-0 token row, so
        this is zero unless the co-partitioning assumption fails somewhere
        unsampled. Surfacing it as a number is what keeps a violation from
        looking like a silently smaller output.
        """
        return self.embed_rows - self.matched


def _ragged_to_padded(column: pa.ChunkedArray | pa.Array, max_tokens: int, vocab_size: int) -> np.ndarray:
    """A parquet ``list<int>`` column as a dense ``[n, max_tokens]`` compact-id array.

    The stored ids are raw Nemotron ids and the model's remap is
    ``raw + NUM_RESERVED`` (asserted by :func:`verify_vocab`), so the remap is an
    add. Everything here is vectorized: a per-row Python loop over ~6e12 tokens is
    not affordable.

    An id at or above ``vocab_size`` becomes ``UNK_ID``. The corpus and donor
    tokenizers share a 131,072-entry vocabulary so this should never fire, but a
    jax gather clamps out-of-range indices rather than raising, which would score
    the row against an unrelated embedding with no visible error.
    """
    array = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if isinstance(array, pa.ListArray | pa.LargeListArray):
        offsets = array.offsets.to_numpy()
        values = array.values.to_numpy(zero_copy_only=False)
    else:  # FixedSizeListArray
        width = array.type.list_size
        values = array.values.to_numpy(zero_copy_only=False)
        offsets = np.arange(len(array) + 1, dtype=np.int64) * width
    n = len(offsets) - 1
    starts = offsets[:-1]
    lengths = np.minimum(np.diff(offsets), max_tokens)
    positions = np.arange(max_tokens, dtype=np.int64)
    mask = positions[None, :] < lengths[:, None]
    # Clamp the gather so out-of-range lanes read a valid element; `mask` discards
    # them. An empty values buffer would make the clamp negative, so guard it.
    gather = np.minimum(starts[:, None] + positions[None, :], max(len(values) - 1, 0))
    raw = (values[gather] if len(values) else np.zeros((n, max_tokens), dtype=np.int64)).astype(np.int64)
    compact = np.where(raw < vocab_size - NUM_RESERVED, raw + NUM_RESERVED, UNK_ID)
    return np.where(mask, compact, PAD_ID).astype(np.int32)


def with_retry(operation, what: str, attempts: int = READ_ATTEMPTS):
    """Run ``operation``, retrying transient object-store faults with backoff.

    CoreWeave severs connections mid-body at roughly one per minute against a
    single leaf, surfacing as ``ClientPayloadError`` and friends, at request sizes
    from 65 KB to 8.7 MB alike -- connection churn, with no size selectivity. Every
    observed instance recovered on the first retry. Over tens of thousands of
    shards and hours of wall clock, leaving that unretried turns a routine reset
    into a task failure that looks exactly like a real data fault.

    Deliberately broad in what it retries and strict in how often: anything raised
    by a read is treated as possibly transient, but only a handful of times, so a
    genuine fault still fails the task rather than spinning.
    """
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt == attempts:
                logger.error("%s failed after %d attempts: %r", what, attempts, exc)
                raise
            delay = min(RETRY_BASE_DELAY * 2 ** (attempt - 1), RETRY_MAX_DELAY)
            delay *= 1.0 + random.random()  # jitter, so parallel readers do not resynchronize
            logger.warning("%s failed (attempt %d/%d): %r; retrying in %.1fs", what, attempt, attempts, exc, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def read_one_embed_shard(path: str, fs) -> tuple[np.ndarray, np.ndarray]:
    """One shard's ``(ids, int8[n, 1024] embeddings)``.

    Read through polars, which types the column as a fixed-width ``Array`` rather
    than a variable-length list. That has no offsets buffer, so the int32 offset
    ceiling that made a whole-column pyarrow read of the largest shards fail with
    "List index overflow" -- 2,682,446 documents is 2.75e9 values against 2^31-1 --
    does not apply, and ``to_numpy`` hands back a contiguous ``[n, 1024]`` block
    with no reshape and no second copy.
    """
    with fs.open(path, "rb", cache_type="none") as raw:
        frame = pl.read_parquet(raw, columns=["id", "embedding"])
    return frame.get_column("id").to_numpy(), frame.get_column("embedding").to_numpy()


def read_embed_side(path: str, fs) -> tuple[np.ndarray, np.ndarray]:
    """A shard's ``(ids, embeddings)``, retried through transient resets."""
    return with_retry(lambda: read_one_embed_shard(path, fs), f"embed read {path}")


def normalize_embeddings(rows: np.ndarray) -> np.ndarray:
    """int8[1024] rows -> float32, L2-normalized, exactly as training fed them.

    ``joined_labels.embedding_matrix`` normalizes the raw int8 without applying the
    0.3/127 quantization scale, and it is right to: a uniform scale cancels under
    L2 normalization, so dequantizing first is a no-op. This reproduces the
    training path rather than a variant of it.
    """
    x = rows.astype(np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)


def join_shard(task: ShardTask, max_tokens: int, vocab_size: int, block_docs: int, fs) -> Iterator[Block | ShardStats]:
    """Stream a shard pair's inner join on ``id`` as bounded blocks.

    Yields :class:`Block` values of about ``block_docs`` documents and a final
    :class:`ShardStats`. Blocks rather than one array per shard because shard
    sizes are extremely skewed: the largest holds 2.68M documents, whose dense
    token and embedding arrays come to 16.5 GB, and a worker pool holding even a
    few of those at once exceeds the container's memory cap. Peak memory here is
    set by ``block_docs``, not by the shard.

    The embed side is one row per document and the token side a chunk-expanded
    superset, so the two are never positionally alignable and the match is on the
    key. It is also on the key *without* assuming either side is ordered. The shard
    writers do not guarantee ordering, and ``common-crawl-focus-2026-22`` is a
    source where they did not deliver it -- its shard 0 carries 3,081 ``id``
    inversions in 6,096 rows. A match that read the stored order as sorted
    degenerated silently there, emitting 3 rows for a 6,048-document shard rather
    than failing, and lost 36.3M documents across the leaf. Sorting the embed ids
    here costs one ``argsort`` per shard and makes the join independent of how
    either side happens to be laid out.

    Only ``chunk_index == 0`` is scored (the model reads the first 512 tokens);
    later chunks are dropped as soon as they are decoded.
    """
    t0 = time.monotonic()
    embed_ids, embeddings = read_embed_side(task.embed_path, fs)
    embed_rows = len(embed_ids)
    # `order` maps a rank in `lookup` back to its row in `embeddings`; the gather
    # into the embedding array therefore stays on original row positions.
    order = np.argsort(embed_ids, kind="stable")
    lookup = embed_ids[order]

    id_blocks: list[np.ndarray] = []
    token_blocks: list[np.ndarray] = []
    embed_blocks: list[np.ndarray] = []
    pending = 0
    matched_total = 0
    token_rows = 0

    def flush() -> Block:
        nonlocal id_blocks, token_blocks, embed_blocks, pending
        block = Block(
            doc_ids=np.concatenate(id_blocks),
            ids=np.concatenate(token_blocks),
            embedding=normalize_embeddings(embeddings[np.concatenate(embed_blocks)]),
        )
        id_blocks, token_blocks, embed_blocks, pending = [], [], [], 0
        return block

    with fs.open(task.tokens_path, "rb", cache_type="none") as raw:
        parquet = pq.ParquetFile(raw)
        for batch in parquet.iter_batches(batch_size=READ_BATCH, columns=["id", "chunk_index", "input_ids"]):
            token_rows += batch.num_rows
            chunk_index = batch.column("chunk_index").to_numpy(zero_copy_only=False)
            first = np.flatnonzero(chunk_index == 0)
            if not len(first):
                continue
            doc_ids = batch.column("id").take(pa.array(first)).to_numpy(zero_copy_only=False)
            # Inner join: keep only ids the (dedup-filtered) embed side carries.
            rank = np.searchsorted(lookup, doc_ids)
            rank = np.minimum(rank, max(embed_rows - 1, 0))
            keep = np.flatnonzero(embed_rows and (lookup[rank] == doc_ids))
            if not len(keep):
                continue
            # Narrow to the matched rows before decoding `input_ids`. Dedup drops a
            # large minority of chunk-0 documents (41.6% on the shards measured),
            # and the ragged-to-dense conversion is the heaviest CPU work in the
            # reader thread, so decoding rows the join discards is pure waste.
            id_blocks.append(doc_ids[keep])
            matched = batch.column("input_ids").take(pa.array(first[keep]))
            token_blocks.append(_ragged_to_padded(matched, max_tokens, vocab_size))
            # Positions, not rows: the gather happens once per block.
            embed_blocks.append(order[rank[keep]])
            pending += len(keep)
            matched_total += len(keep)
            if pending >= block_docs:
                yield flush()
    if pending:
        yield flush()
    yield ShardStats(
        token_rows=token_rows, embed_rows=embed_rows, matched=matched_total, read_seconds=time.monotonic() - t0
    )


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def load_folded_scorer(model_dir: str) -> PooledScorer:
    """Load the deployable scorer, folding on the fly only if the dir is unfolded."""
    scorer = load_pooled_scorer(model_dir)
    verify_vocab(scorer.remap)
    if scorer.model.config.frozen_donor_dim:
        logger.warning("model dir %s is unfolded; folding in-process (prefer a pre-folded dir)", model_dir)
        scorer = dataclasses.replace(scorer, model=fold_donor(scorer.model))
    return scorer


def write_scores(path: str, doc_ids: np.ndarray, scores: np.ndarray, model_tag: str, fs) -> int:
    """Write the narrow score shard: id + score + a model tag, nothing else.

    Deliberately not joined back to the input. Keeping the output to the join key
    and the score keeps the write cheap and leaves it trivially re-joinable by
    ``id`` against any other attribute over the same corpus.
    """
    frame = pl.DataFrame(
        {
            "id": pl.Series(doc_ids, dtype=pl.String),
            "score": pl.Series(scores, dtype=pl.Float32),
            "model": pl.Series([model_tag] * len(doc_ids), dtype=pl.String),
        }
    )
    with fs.open(path, "wb") as out:
        frame.write_parquet(out, compression="zstd")
    return frame.height


def score_mode(args) -> dict:
    """Score this worker's slice of the manifest."""
    configure_coreweave_s3()
    fs = fsspec.filesystem("s3")
    scorer = load_folded_scorer(args.model_dir)
    vocab_size = scorer.model.config.vocab_size
    logger.info("worker %d rss after model load: %.2f GB", args.worker, rss_bytes() / 1e9)
    logger.info(
        "worker %d/%d: model max_tokens=%d devices=%s batch=%d",
        args.worker,
        args.num_workers,
        scorer.max_tokens,
        jax.devices(),
        args.batch_size,
    )

    manifest = read_manifest(args.manifest)
    tasks = [
        ShardTask(
            source=row["source"],
            shard_index=row["shard_index"],
            tokens_path=row["tokens_path"],
            embed_path=row["embed_path"],
            output_path=row["output_path"],
            total_bytes=(row.get("tokens_bytes") or 0) + (row.get("embed_bytes") or 0),
        )
        for row in manifest.to_dicts()
        # A shard with no embedded documents has nothing to score: the embedding is
        # a required forward input, so an inner join empties it anyway. A shard
        # whose documents were all dedup-dropped still writes a zero-row parquet on
        # the surviving side, so this is a row count check rather than an existence
        # check, and it is taken over the union of the leaves.
        if row["embed_rows"]
        # Rescoring one source: the output is overwritten shard by shard, so a
        # partially-written leaf is repaired rather than skipped.
        and (not args.source or row["source"] == args.source)
    ]
    if args.source and not tasks:
        raise ValueError(f"no manifest rows for source {args.source!r}")
    # Deal largest-first round-robin over manifest *rows*, not source_keys. Leaves
    # range from 1 to 25,962 shards, so a per-leaf fan-out would hand one worker
    # an entire large source; and shard sizes vary enough within a leaf that
    # striding alone leaves the tail uneven. Sorting by bytes first makes every
    # worker's total within a shard of every other's.
    tasks.sort(key=lambda t: t.total_bytes, reverse=True)
    mine = tasks[args.worker :: args.num_workers]
    if args.limit:
        mine = mine[: args.limit]
    logger.info(
        "worker %d owns %d of %d shard tasks (%.2f TB of %.2f TB), rss %.2f GB",
        args.worker,
        len(mine),
        len(tasks),
        sum(t.total_bytes for t in mine) / 1e12,
        sum(t.total_bytes for t in tasks) / 1e12,
        rss_bytes() / 1e9,
    )

    done = 0
    docs = 0
    unmatched_embed = 0
    token_rows = 0
    read_seconds = 0.0
    score_seconds = 0.0
    write_seconds = 0.0
    started = time.monotonic()

    # One reader thread over the whole slice, feeding a bounded queue. The reader
    # runs ahead of the forward so S3 and the GPU overlap, but the queue bound
    # plus the block bound cap resident memory at a few hundred MB per worker
    # regardless of how large a shard is.
    work: queue.Queue = queue.Queue(maxsize=args.prefetch)

    def read_one(task: ShardTask) -> None:
        """Stream one shard, restarting it whole on a transient fault.

        The token stream cannot resume mid-body: blocks already handed to the
        consumer have been scored, so replaying from an offset would double-count
        and replaying from the start without warning would too. A restart therefore
        announces itself, and the consumer drops the shard's partial state before
        the first block of the new attempt arrives.
        """
        for attempt in range(1, READ_ATTEMPTS + 1):
            try:
                for item in join_shard(task, scorer.max_tokens, vocab_size, args.block_docs, fs):
                    work.put((task, item))
                return
            except Exception as exc:
                if attempt == READ_ATTEMPTS:
                    raise
                delay = min(RETRY_BASE_DELAY * 2 ** (attempt - 1), RETRY_MAX_DELAY) * (1.0 + random.random())
                logger.warning(
                    "%s shard %d failed mid-read (attempt %d/%d): %r; restarting shard in %.1fs",
                    task.source,
                    task.shard_index,
                    attempt,
                    READ_ATTEMPTS,
                    exc,
                    delay,
                )
                time.sleep(delay)
                work.put((task, ShardRestart()))

    def produce() -> None:
        try:
            for task in mine:
                read_one(task)
        except Exception as exc:  # surfaced on the consumer thread
            work.put((None, exc))
        finally:
            work.put((None, None))

    reader = threading.Thread(target=produce, name="reader", daemon=True)
    reader.start()

    shard_ids: list[np.ndarray] = []
    shard_scores: list[np.ndarray] = []
    while True:
        task, item = work.get()
        if task is None:
            if isinstance(item, BaseException):
                raise item
            break
        if isinstance(item, ShardRestart):
            # The shard is being re-read from the start; everything scored from the
            # abandoned attempt must go, or its documents are written twice.
            logger.warning("%s shard %d: discarding %d partial blocks", task.source, task.shard_index, len(shard_ids))
            shard_ids, shard_scores = [], []
            continue
        if isinstance(item, Block):
            t0 = time.monotonic()
            shard_scores.append(predict(scorer.model, item.ids, batch_size=args.batch_size, doc_embed=item.embedding))
            score_seconds += time.monotonic() - t0
            shard_ids.append(item.doc_ids)
            continue

        # ShardStats: the shard is fully read and scored, so write it out. Only
        # ids and scores were retained across its blocks, which is ~36 bytes a
        # document even for the largest shard.
        read_seconds += item.read_seconds
        token_rows += item.token_rows
        if item.unmatched_embed:
            unmatched_embed += item.unmatched_embed
            logger.warning(
                "%s shard %d: %d embed rows absent from the token side; containment does not hold here",
                task.source,
                task.shard_index,
                item.unmatched_embed,
            )
        if shard_ids:
            t1 = time.monotonic()
            rows = write_scores(
                task.output_path,
                np.concatenate(shard_ids),
                np.concatenate(shard_scores),
                args.model_tag,
                fs,
            )
            write_seconds += time.monotonic() - t1
            docs += rows
        else:
            logger.warning("%s shard %d: no joined rows", task.source, task.shard_index)
        shard_ids, shard_scores = [], []
        done += 1
        if done % 25 == 0 or done == len(mine):
            elapsed = time.monotonic() - started
            logger.info(
                "worker %d: %d/%d shards, %d docs, %.0f docs/s "
                "(read %.0fs score %.0fs write %.0fs of %.0fs, rss %.1f GB)",
                args.worker,
                done,
                len(mine),
                docs,
                docs / max(elapsed, 1e-9),
                read_seconds,
                score_seconds,
                write_seconds,
                elapsed,
                rss_bytes() / 1e9,
            )

    elapsed = time.monotonic() - started
    result = {
        "worker": args.worker,
        "shards": done,
        "docs": docs,
        "unmatched_embed": unmatched_embed,
        "token_rows": token_rows,
        "bytes": sum(t.total_bytes for t in mine),
        "seconds": elapsed,
        "docs_per_second": docs / max(elapsed, 1e-9),
        "read_seconds": read_seconds,
        "score_seconds": score_seconds,
        "write_seconds": write_seconds,
    }
    logger.info("worker %d done: %s", args.worker, json.dumps(result))
    return result


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def read_score_shard(path: str, fs) -> tuple[np.ndarray, set[str]] | None:
    """A written score shard's scores and its model tags, or None if absent."""
    if not fs.exists(path):
        return None
    with fs.open(path, "rb", cache_type="none") as raw:
        frame = pl.read_parquet(raw, columns=["score", "model"])
    return frame.get_column("score").to_numpy(), set(frame.get_column("model").unique().to_list())


def verify_mode(args) -> dict:
    """Reconcile one source's written scores against its manifest and calibration.

    Reads every score shard the manifest names for the source, so the row count is
    the written one rather than a footer sum over whatever happens to be in the
    output directory. The per-shard comparison is against ``embed_rows``: the join
    is an inner join and every embedded document has a chunk-0 token row, so a
    shard whose score count falls short of its embed count is the signature of a
    join that did not complete.
    """
    fs = fsspec.filesystem("s3")
    rows = [r for r in read_manifest(args.manifest).to_dicts() if r["source"] == args.source]
    if not rows:
        raise ValueError(f"no manifest rows for source {args.source!r}")
    rows.sort(key=lambda r: r["shard_index"])

    with ThreadPoolExecutor(max_workers=args.verify_threads) as pool:
        found = list(pool.map(lambda r: read_score_shard(r["output_path"], fs), rows))

    missing = [r["shard_index"] for r, got in zip(rows, found, strict=True) if got is None]
    empty = [r["shard_index"] for r, got in zip(rows, found, strict=True) if got is not None and not len(got[0])]
    short = [
        {"shard_index": r["shard_index"], "score_rows": len(got[0]), "embed_rows": r["embed_rows"]}
        for r, got in zip(rows, found, strict=True)
        if got is not None and len(got[0]) != r["embed_rows"]
    ]
    tags = sorted({tag for got in found if got for tag in got[1]})
    scores = np.concatenate([got[0] for got in found if got]) if any(found) else np.zeros(0, dtype=np.float32)
    embed_rows = sum(r["embed_rows"] for r in rows)

    with open_url(args.calibration, "rb") as fh:
        knots = json.loads(fh.read())["default"]["xk"]
    edges = np.asarray(knots[1:-1], dtype=np.float64)
    counts = np.bincount(np.digitize(scores, edges), minlength=len(edges) + 1)

    result = {
        "source": args.source,
        "manifest_shards": len(rows),
        "missing_score_files": len(missing),
        "empty_score_files": len(empty),
        "shards_below_embed_rows": len(short),
        "score_rows": len(scores),
        "embed_rows": embed_rows,
        "shortfall": embed_rows - len(scores),
        "model_tags": tags,
        "score_mean": float(scores.mean()) if len(scores) else None,
        "score_std": float(scores.std()) if len(scores) else None,
        "score_percentiles": {str(q): float(np.percentile(scores, q)) for q in (1, 25, 50, 75, 99) if len(scores)},
        "bucket_edges": edges.tolist(),
        "bucket_counts": counts.tolist(),
        "bucket_shares": (counts / max(len(scores), 1)).tolist(),
        "worst_shards": sorted(short, key=lambda s: s["score_rows"] - s["embed_rows"])[:10],
        "missing_shard_indices": missing[:20],
    }
    logger.info("verify %s", json.dumps(result, indent=2))
    return result


def node_placement(args) -> tuple[int, int]:
    """This replica's (index, count), from Iris when it is running one.

    ``--replicas N`` gang-schedules N identical commands, so a replica learns
    which slice of the manifest is its own from the job rather than from argv.
    The explicit flags stay as an override for a local or single-node run.
    """
    if args.node_index is not None and args.num_nodes is not None:
        return args.node_index, args.num_nodes
    info = get_job_info()
    if info is None:
        raise ValueError("no Iris job context; pass --node-index and --num-nodes explicitly")
    return info.task_index, info.num_tasks


def node_mode(args) -> dict:
    """One ``score`` subprocess per visible GPU, each on its own device.

    Independent processes are the point: the pipeline is feed-bound and threaded
    readers contend on the GIL, so the per-node bandwidth only appears when the
    readers are in separate address spaces.
    """
    gpus = args.gpus_per_node
    per_node = args.procs_per_node or gpus
    node_index, num_nodes = node_placement(args)
    logger.info("node %d of %d, %d workers over %d GPUs", node_index, num_nodes, per_node, gpus)
    base = node_index * per_node
    procs = []
    for local in range(per_node):
        env = dict(os.environ)
        # More workers than GPUs on purpose. The forward is ~12% of a worker's
        # wall time and the reader is CPU-bound, so a node's 128 vCPU are the
        # scarce resource, not its 8 devices; workers share a device round-robin.
        # HBM per worker is ~1.2 GB, so cap the preallocator rather than let the
        # first worker on a device claim 75% of it and starve its co-tenants.
        env["CUDA_VISIBLE_DEVICES"] = str(local % gpus)
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{max(0.05, 0.7 / max(1, per_node // gpus)):.2f}"
        env["TOKENIZERS_PARALLELISM"] = "false"
        argv = [
            sys.executable,
            "-m",
            MODULE,
            "score",
            "--manifest",
            args.manifest,
            "--model-dir",
            args.model_dir,
            "--worker",
            str(base + local),
            "--num-workers",
            str(num_nodes * per_node),
            "--batch-size",
            str(args.batch_size),
            "--prefetch",
            str(args.prefetch),
            "--block-docs",
            str(args.block_docs),
            "--model-tag",
            args.model_tag,
            "--source",
            args.source,
        ]
        if args.limit:
            argv += ["--limit", str(args.limit)]
        logger.info("launching worker %d on local GPU %d", base + local, local)
        procs.append(subprocess.Popen(argv, env=env))
    codes = [p.wait() for p in procs]
    failed = [i for i, c in enumerate(codes) if c != 0]
    if failed:
        raise RuntimeError(f"workers {failed} exited non-zero: {codes}")
    return {"node_index": node_index, "workers": per_node, "exit_codes": codes}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)

    f = sub.add_parser("fold", help="fold the donor table and write a deployable model dir")
    f.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    f.add_argument("--out-dir", default=DEFAULT_FOLDED_DIR)
    f.add_argument("--stem", default="nemotron_88k_folded")

    m = sub.add_parser("manifest", help="build the minimum shard-task manifest from listings")
    m.add_argument("--manifest", default=DEFAULT_MANIFEST)
    m.add_argument("--footer-threads", type=int, default=256)
    m.add_argument("--discovery-threads", type=int, default=48)

    s = sub.add_parser("score", help="score this worker's slice of the manifest")
    s.add_argument("--manifest", default=DEFAULT_MANIFEST)
    s.add_argument("--model-dir", default=DEFAULT_FOLDED_DIR)
    s.add_argument("--worker", type=int, required=True)
    s.add_argument("--num-workers", type=int, required=True)
    s.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    s.add_argument("--prefetch", type=int, default=DEFAULT_PREFETCH_SHARDS, help="blocks queued ahead of the forward")
    s.add_argument("--block-docs", type=int, default=DEFAULT_BLOCK_DOCS)
    s.add_argument("--model-tag", default="nemotron88k_v1")
    s.add_argument("--source", default="", help="score only this source (default: all)")
    s.add_argument("--limit", type=int, default=0, help="cap shards per worker (smoke runs)")

    n = sub.add_parser("node", help="fan out one score subprocess per GPU on this node")
    n.add_argument("--manifest", default=DEFAULT_MANIFEST)
    n.add_argument("--model-dir", default=DEFAULT_FOLDED_DIR)
    n.add_argument("--node-index", type=int, default=None, help="default: this Iris replica's index")
    n.add_argument("--num-nodes", type=int, default=None, help="default: the Iris job's replica count")
    n.add_argument("--gpus-per-node", type=int, default=8)
    n.add_argument("--procs-per-node", type=int, default=0, help="worker processes per node (default: one per GPU)")
    n.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    n.add_argument("--prefetch", type=int, default=DEFAULT_PREFETCH_SHARDS)
    n.add_argument("--block-docs", type=int, default=DEFAULT_BLOCK_DOCS)
    n.add_argument("--model-tag", default="nemotron88k_v1")
    n.add_argument("--source", default="", help="score only this source (default: all)")
    n.add_argument("--limit", type=int, default=0)

    v = sub.add_parser("verify", help="reconcile one source's written scores against the manifest")
    v.add_argument("--manifest", default=DEFAULT_MANIFEST)
    v.add_argument("--source", required=True)
    v.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    v.add_argument("--verify-threads", type=int, default=64)

    args = p.parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()
    modes = {
        "fold": fold_mode,
        "manifest": manifest_mode,
        "score": score_mode,
        "node": node_mode,
        "verify": verify_mode,
    }
    result = modes[args.mode](args)
    logger.info("%s result: %s", args.mode, json.dumps(result, default=str))


if __name__ == "__main__":
    main()
