# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score the Nemotron-tokenized corpus with the fusion quality scorer.

The corpus already carries everything the fusion scorer eats: ``datakit/tokenize``
holds Nemotron ``input_ids`` per document chunk and ``datakit/embed/harrier`` holds
the int8[1024] document embedding. Both are hash-partitioned from one normalized
source, share a shard count and basename, and are sorted by ``id`` within a shard,
so a shard pair joins with a streaming merge on ``id`` and no shuffle.

Four modes:

* ``fold`` -- collapse the frozen ``[vocab, 2048]`` donor table and its learned
  ``[2048, 256]`` projection into one ``[vocab, 256]`` table and write a new model
  dir. Gather and matmul commute row-wise so this is exact, not an approximation;
  the mode asserts score parity on real rows before writing. Deployment reads the
  folded dir, so 48 workers do not each redo a 1.1 GB read and a fold.
* ``manifest`` -- pair the Nemotron tokenize leaves with the harrier embed leaves
  on ``result.source_key`` and emit one row per (source_key, shard_index).
* ``score`` -- one worker, one GPU: take this worker's slice of the manifest and
  stream join -> score -> write for each shard pair it owns.
* ``node`` -- fan out one ``score`` subprocess per visible GPU.

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
import subprocess
import sys
import time
from collections import deque
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
from iris.cluster.client.job_info import get_job_info
from rigging.filesystem import StoragePath, open_url
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.data import NUM_RESERVED, PAD_ID, UNK_ID
from experiments.datakit.cluster.quality.fast_transformer.inference import predict
from experiments.datakit.cluster.quality.fast_transformer.model import COMPUTE_DTYPE, FastTransformer
from experiments.datakit.cluster.quality.fast_transformer.scorer import PooledScorer, artifact_names, load_pooled_scorer

logger = logging.getLogger(__name__)

MODULE = "experiments.datakit.cluster.quality.fast_transformer.score_corpus"

EMBED_DIM = 1024
NEMOTRON_CORPUS_TOKENIZER = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
TOKENIZE_ROOT = "s3://marin-us-east-02a/marin/datakit/tokenize"
EMBED_ROOT = "s3://marin-us-east-02a/marin/datakit/embed/harrier"
DEFAULT_OUT_ROOT = "s3://marin-us-east-02a/marin/datakit/quality-scores"
DEFAULT_MODEL_DIR = "s3://marin-us-east-02a/marin/user/muchanem/quality_exp/nemotron_donor"
DEFAULT_FOLDED_DIR = "s3://marin-us-east-02a/marin/user/muchanem/quality_scores_run/model/nemotron_88k_folded"
DEFAULT_MANIFEST = "s3://marin-us-east-02a/marin/user/muchanem/quality_scores_run/manifest"

# Rows per arrow record batch when streaming a token shard. The token side is the
# expanded one (a long document occupies several adjacent rows), so this is rows,
# not documents.
READ_BATCH = 4096
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


def _leaf_artifacts(fs, root: str, threads: int) -> list[dict]:
    """Every ``<source>/<subset>_<hash>`` leaf's ``.artifact.json`` under a stage root.

    Delimiter listings one level at a time rather than a recursive glob. The
    tokenize tree holds on the order of a million parquet shards, and a
    ``*/*/.artifact.json`` pattern makes s3fs walk all of them to match three
    path components -- measured at over 20 minutes without returning.
    """
    base = root.removeprefix("s3://").rstrip("/")
    with ThreadPoolExecutor(max_workers=threads) as pool:
        sources = fs.ls(base, detail=False)
        leaves = [leaf for got in pool.map(lambda s: fs.ls(s, detail=False), sources) for leaf in got]
        logger.info("%s: %d sources, %d leaves", root, len(sources), len(leaves))

        def read(leaf: str) -> dict | None:
            try:
                return json.loads(fs.cat(f"{leaf.rstrip('/')}/.artifact.json"))
            except Exception as exc:
                logger.warning("unreadable artifact under %s: %s", leaf, exc)
                return None

        return [a for a in pool.map(read, leaves) if a]


def _leaf_rel(output_dir: str, marker: str) -> str:
    """``<source>/<subset>_<hash>`` from a stage output dir.

    Split on the stage marker rather than by counting path components: one
    source_key carries an anomalous ``data/datakit/normalized/...`` prefix, so
    positional parsing is wrong on it.
    """
    index = output_dir.find(marker)
    if index < 0:
        raise ValueError(f"{output_dir!r} does not contain {marker!r}")
    return output_dir[index + len(marker) :].strip("/").removesuffix("/train")


def manifest_mode(args) -> dict:
    """Build the minimum manifest the scorer needs: paths and an existence flag.

    STEP 1's ``build_quality_scores_manifest`` writes a richer artifact carrying
    shard sizes and row counts for planning. This is the fallback that needs only
    directory listings, so it costs two listings per leaf rather than a footer
    read per shard, and is consumed by exactly the same reader.
    """
    fs = fsspec.filesystem("s3")
    tokenize, embed = {}, {}
    # Not every leaf artifact carries a result: a stage that failed or is still
    # running writes one without `source_key`, and reading it as a pair would
    # either crash or silently pair the wrong directories.
    for art in _leaf_artifacts(fs, TOKENIZE_ROOT, args.discovery_threads):
        cfg, res = art.get("config") or {}, art.get("result") or {}
        train = (res.get("output_dirs") or {}).get("train")
        if cfg.get("tokenizer") == NEMOTRON_CORPUS_TOKENIZER and res.get("source_key") and train:
            tokenize[res["source_key"]] = train
    for art in _leaf_artifacts(fs, EMBED_ROOT, args.discovery_threads):
        res = art.get("result") or {}
        if res.get("source_key") and res.get("output_dir"):
            embed[res["source_key"]] = res["output_dir"]
    shared = sorted(set(tokenize) & set(embed))
    logger.info("nemotron leaves=%d harrier leaves=%d paired=%d", len(tokenize), len(embed), len(shared))

    def leaf_rows(source_key: str) -> list[dict]:
        tok_dir, emb_dir = tokenize[source_key].rstrip("/"), embed[source_key].rstrip("/")
        tok = sorted(f"s3://{p}" for p in fs.glob(f"{tok_dir.removeprefix('s3://')}/*.parquet"))
        emb = {p.rsplit("/", 1)[-1] for p in fs.glob(f"{emb_dir.removeprefix('s3://')}/*.parquet")}
        leaf = _leaf_rel(tok_dir, "/datakit/tokenize/")
        rows = []
        for index, path in enumerate(tok):
            base = path.rsplit("/", 1)[-1]
            rows.append(
                {
                    "source_key": source_key,
                    "shard_index": index,
                    "num_shards": len(tok),
                    "tokens_path": path,
                    "embed_path": f"{emb_dir}/{base}",
                    "output_path": f"{args.out_root.rstrip('/')}/{leaf}/{base}",
                    "embed_exists": base in emb,
                }
            )
        return rows

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.discovery_threads) as pool:
        for got in pool.map(leaf_rows, shared):
            rows.extend(got)
    missing = sum(1 for r in rows if not r["embed_exists"])
    table = pa.table({c: [r[c] for r in rows] for c in rows[0]})
    path = f"{args.manifest.rstrip('/')}/manifest.parquet"
    with open_url(path, "wb") as fh:
        pq.write_table(table, fh)
    logger.info("wrote %s: %d tasks over %d sources (%d without embeds)", path, len(rows), len(shared), missing)
    return {"manifest": path, "tasks": len(rows), "sources": len(shared), "missing_embed": missing}


def read_manifest(manifest: str) -> pa.Table:
    root = manifest.rstrip("/")
    shards = sorted(str(p) for p in StoragePath(f"{root}/*.parquet").glob())
    if not shards:
        raise ValueError(f"no manifest parquet under {root}")
    tables = []
    for shard in shards:
        with StoragePath(shard).open("rb") as fh:
            tables.append(pq.ParquetFile(fh).read())
    return pa.concat_tables(tables, promote_options="default")


# ---------------------------------------------------------------------------
# the join
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardTask:
    source_key: str
    shard_index: int
    tokens_path: str
    embed_path: str
    output_path: str


@dataclass(frozen=True)
class Joined:
    """One shard pair's matched documents, ready for the forward."""

    doc_ids: np.ndarray  # [n] object, the 32-char hex ids
    ids: np.ndarray  # [n, max_tokens] int32 compact ids, PAD-padded
    embedding: np.ndarray  # [n, EMBED_DIM] float32, L2-normalized
    token_rows: int  # rows read on the token side, before the chunk filter
    embed_rows: int  # rows on the embed side
    read_seconds: float
    bytes_read: int


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


def _normalize_embeddings(column: pa.ChunkedArray | pa.Array, rows: int) -> np.ndarray:
    """int8[1024] rows -> float32, L2-normalized, exactly as training fed them.

    ``joined_labels.embedding_matrix`` normalizes the raw int8 without applying the
    0.3/127 quantization scale, and it is right to: a uniform scale cancels under
    L2 normalization, so dequantizing first is a no-op. This reproduces the
    training path rather than a variant of it.
    """
    array = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    flat = array.flatten().to_numpy(zero_copy_only=False)
    x = flat.reshape(rows, EMBED_DIM).astype(np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)


def join_shard(task: ShardTask, max_tokens: int, vocab_size: int, fs) -> Joined:
    """Stream a shard pair's inner join on ``id``.

    Both sides are ascending by ``id`` within a shard: the embed side is a
    dedup-filtered subset, the token side a chunk-expanded superset. Neither is
    positionally alignable with the other, so this joins on the key -- matching by
    ``searchsorted`` against the sorted embed ids rather than by row position.
    Only ``chunk_index == 0`` is scored (the model reads the first 512 tokens);
    later chunks are dropped as soon as they are decoded.
    """
    t0 = time.monotonic()
    with fs.open(task.embed_path, "rb", cache_type="none") as raw:
        embed_table = pq.ParquetFile(raw).read(columns=["id", "embedding"])
    embed_rows = embed_table.num_rows
    embed_ids = embed_table.column("id").to_numpy(zero_copy_only=False)
    embeddings = _normalize_embeddings(embed_table.column("embedding"), embed_rows)
    del embed_table

    id_blocks: list[np.ndarray] = []
    token_blocks: list[np.ndarray] = []
    embed_blocks: list[np.ndarray] = []
    token_rows = 0
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
            position = np.searchsorted(embed_ids, doc_ids)
            position = np.minimum(position, max(embed_rows - 1, 0))
            keep = np.flatnonzero(embed_rows and (embed_ids[position] == doc_ids))
            if not len(keep):
                continue
            # Narrow to the matched rows before decoding `input_ids`. Dedup drops a
            # large minority of chunk-0 documents (41.6% on the shards measured),
            # and the ragged-to-dense conversion is the heaviest CPU work in the
            # reader thread, so decoding rows the join discards is pure waste.
            id_blocks.append(doc_ids[keep])
            matched = batch.column("input_ids").take(pa.array(first[keep]))
            token_blocks.append(_ragged_to_padded(matched, max_tokens, vocab_size))
            embed_blocks.append(embeddings[position[keep]])

    read_seconds = time.monotonic() - t0
    if not id_blocks:
        empty_ids = np.empty((0, max_tokens), dtype=np.int32)
        empty_emb = np.empty((0, EMBED_DIM), dtype=np.float32)
        return Joined(np.empty(0, dtype=object), empty_ids, empty_emb, token_rows, embed_rows, read_seconds, 0)
    return Joined(
        doc_ids=np.concatenate(id_blocks),
        ids=np.concatenate(token_blocks),
        embedding=np.concatenate(embed_blocks),
        token_rows=token_rows,
        embed_rows=embed_rows,
        read_seconds=read_seconds,
        bytes_read=0,
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
    table = pa.table(
        {
            "id": pa.array(doc_ids, type=pa.string()),
            "score": pa.array(scores, type=pa.float32()),
            "model": pa.array([model_tag] * len(doc_ids), type=pa.string()),
        }
    )
    with fs.open(path, "wb") as out:
        pq.write_table(table, out, compression="zstd")
    return table.num_rows


def score_mode(args) -> dict:
    """Score this worker's slice of the manifest."""
    configure_coreweave_s3()
    fs = fsspec.filesystem("s3")
    scorer = load_folded_scorer(args.model_dir)
    vocab_size = scorer.model.config.vocab_size
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
            source_key=row["source_key"],
            shard_index=row["shard_index"],
            tokens_path=row["tokens_path"],
            embed_path=row["embed_path"],
            output_path=row["output_path"],
        )
        for row in manifest.to_pylist()
        # A token shard whose embed side is missing has no scorable documents: the
        # embedding is a required forward input, so an inner join drops it anyway.
        if row.get("embed_exists", True)
    ]
    # Interleave rather than block-partition: consecutive manifest rows are shards
    # of one source and so share a size profile, and a block slice would hand one
    # worker an entire large source.
    mine = tasks[args.worker :: args.num_workers]
    if args.limit:
        mine = mine[: args.limit]
    logger.info("worker %d owns %d of %d shard tasks", args.worker, len(mine), len(tasks))

    done = 0
    docs = 0
    token_rows = 0
    read_seconds = 0.0
    score_seconds = 0.0
    write_seconds = 0.0
    started = time.monotonic()

    def read(task: ShardTask) -> tuple[ShardTask, Joined]:
        return task, join_shard(task, scorer.max_tokens, vocab_size, fs)

    with ThreadPoolExecutor(max_workers=args.prefetch) as pool:
        # Bounded prefetch. `pool.map` would submit every shard at once and hold
        # the whole slice's joined arrays in memory; this keeps at most
        # `prefetch` reads in flight ahead of the forward.
        pending: deque = deque()
        upcoming = iter(mine)
        for task in upcoming:
            pending.append(pool.submit(read, task))
            if len(pending) >= args.prefetch:
                break
        while pending:
            task, joined = pending.popleft().result()
            nxt = next(upcoming, None)
            if nxt is not None:
                pending.append(pool.submit(read, nxt))
            read_seconds += joined.read_seconds
            token_rows += joined.token_rows
            if not len(joined.doc_ids):
                logger.warning("%s shard %d: no joined rows", task.source_key, task.shard_index)
                continue
            t0 = time.monotonic()
            scores = predict(scorer.model, joined.ids, batch_size=args.batch_size, doc_embed=joined.embedding)
            score_seconds += time.monotonic() - t0
            t1 = time.monotonic()
            rows = write_scores(task.output_path, joined.doc_ids, scores, args.model_tag, fs)
            write_seconds += time.monotonic() - t1
            docs += rows
            done += 1
            if done % 25 == 0 or done == len(mine):
                elapsed = time.monotonic() - started
                logger.info(
                    "worker %d: %d/%d shards, %d docs, %.0f docs/s "
                    "(read %.0fs score %.0fs write %.0fs of %.0fs)",
                    args.worker,
                    done,
                    len(mine),
                    docs,
                    docs / max(elapsed, 1e-9),
                    read_seconds,
                    score_seconds,
                    write_seconds,
                    elapsed,
                )

    elapsed = time.monotonic() - started
    result = {
        "worker": args.worker,
        "shards": done,
        "docs": docs,
        "token_rows": token_rows,
        "seconds": elapsed,
        "docs_per_second": docs / max(elapsed, 1e-9),
        "read_seconds": read_seconds,
        "score_seconds": score_seconds,
        "write_seconds": write_seconds,
    }
    logger.info("worker %d done: %s", args.worker, json.dumps(result))
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
            "--model-tag",
            args.model_tag,
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
    m.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    m.add_argument("--discovery-threads", type=int, default=48)

    s = sub.add_parser("score", help="score this worker's slice of the manifest")
    s.add_argument("--manifest", default=DEFAULT_MANIFEST)
    s.add_argument("--model-dir", default=DEFAULT_FOLDED_DIR)
    s.add_argument("--worker", type=int, required=True)
    s.add_argument("--num-workers", type=int, required=True)
    s.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    s.add_argument("--prefetch", type=int, default=DEFAULT_PREFETCH_SHARDS)
    s.add_argument("--model-tag", default="nemotron88k_v1")
    s.add_argument("--limit", type=int, default=0, help="cap shards per worker (smoke runs)")

    n = sub.add_parser("node", help="fan out one score subprocess per GPU on this node")
    n.add_argument("--manifest", default=DEFAULT_MANIFEST)
    n.add_argument("--model-dir", default=DEFAULT_FOLDED_DIR)
    n.add_argument("--node-index", type=int, default=None, help="default: this Iris replica's index")
    n.add_argument("--num-nodes", type=int, default=None, help="default: the Iris job's replica count")
    n.add_argument("--gpus-per-node", type=int, default=8)
    n.add_argument(
        "--procs-per-node", type=int, default=0, help="worker processes per node (default: one per GPU)"
    )
    n.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    n.add_argument("--prefetch", type=int, default=DEFAULT_PREFETCH_SHARDS)
    n.add_argument("--model-tag", default="nemotron88k_v1")
    n.add_argument("--limit", type=int, default=0)

    args = p.parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()
    modes = {"fold": fold_mode, "manifest": manifest_mode, "score": score_mode, "node": node_mode}
    result = modes[args.mode](args)
    logger.info("%s result: %s", args.mode, json.dumps(result, default=str))


if __name__ == "__main__":
    main()
