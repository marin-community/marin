# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-document head-to-head table for the deployed v3 scorer and the fusion candidate.

:mod:`compare_fusion_vs_deployed` reports what the two models do in aggregate. A
reader who wants to check them against the documents themselves needs the rows:
what each model scored, which window drove it, and which way the bucket moved.

``score`` runs both models over the 80,000-document domain evaluation set and
writes one row per document — calibrated score, bucket, content type and the
per-window raw scores behind each aggregate — carrying no text. ``payload``
joins the text back on, aggregates the population by source family and bucket,
draws a deterministic stratified sample and writes the single JSON blob a viewer
page reads.

The stages are split because scoring is the expensive half: resampling or
reshaping the viewer payload does not re-run the models.
"""

import argparse
import hashlib
import json
import logging

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath, open_url
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer import content_type, domain_mlp
from experiments.datakit.cluster.quality.fast_transformer.artifact import BUCKET_EDGES
from experiments.datakit.cluster.quality.fast_transformer.calibrate import apply_calibration
from experiments.datakit.cluster.quality.fast_transformer.joined_labels import embedding_matrix
from experiments.datakit.cluster.quality.fast_transformer.scorer import bme_chunks, load_pooled_scorer
from experiments.datakit.scripts.compare_fusion_vs_deployed import (
    DEPLOYED_DIR,
    DEPLOYED_SCORED,
    DOMAIN_MLP,
    EVAL_DOCS,
    EVAL_EMBEDDINGS,
    FUSION_CALIB_NAME,
    FUSION_DIR,
    REPORT_ROOT,
    SCORE_BATCH,
    _compare_against_stored,
    _load_embedding_index,
)

logger = logging.getLogger(__name__)

DOC_TABLE = f"{REPORT_ROOT}/doc_table.parquet"
VIEWER_PAYLOAD = f"{REPORT_ROOT}/head_to_head_payload.json"

BUCKET_LABELS = ("junk", "poor", "usable", "good", "excellent")
BUCKET_COLORS = ("#a23e2a", "#8f6b38", "#6e6456", "#385c8f", "#224a82")

# Sampling. Every non-empty (family, v3 bucket) cell gets MIN_PER_CELL documents
# so no family is represented only where it is dense; the remaining budget is
# spread proportional to cell population, so the sample tracks the corpus.
DEFAULT_SAMPLE_DOCS = 3_000
MIN_PER_CELL = 3
MAX_PER_CELL = 60
# Characters kept per displayed window, matching the published v3 overview's hard
# 900-character cut. That page shows one window per example; a head-to-head shows
# every window the deployed model read.
DEFAULT_WINDOW_CHARS = 900
SCORE_DECIMALS = 4


def _buckets(calibrated: np.ndarray) -> np.ndarray:
    return np.digitize(calibrated, BUCKET_EDGES)


def _family(source: str) -> str:
    """The source family a source belongs to: its first path segment.

    ``nemotron_cc_v2/diverse_qa`` and ``nemotron_cc_v2/high_quality`` are two
    renderings of one corpus and belong under one heading, which is how the
    published v3 overview groups them.
    """
    return source.split("/", 1)[0]


def _score_windows(scorer, texts: list[str]) -> list[np.ndarray]:
    """Raw per-window scores for each document, in begin/middle/end order.

    ``score_bme`` mean-pools these away. The head-to-head keeps them because the
    mean is the number under dispute: a document whose begin window scores well
    and whose end window scores badly is exactly where the two models' reading
    windows disagree.
    """
    flat, spans = bme_chunks(texts)
    raw = scorer.score(flat, batch_size=SCORE_BATCH)
    return [raw[a:b] for a, b in spans]


def run_score(args) -> None:
    """Both models over the evaluation set, one row per document, no text."""
    index, embed_int8 = _load_embedding_index(args.eval_embeddings)
    typer, typer_labels = domain_mlp.load(args.domain_mlp)
    text_typer = content_type.load(f"{args.deployed_dir.rstrip('/')}/content_type.npz")
    with open_url(f"{args.fusion_dir.rstrip('/')}/{FUSION_CALIB_NAME}", "r") as fh:
        fusion_calib = json.loads(fh.read())
    with open_url(f"{args.deployed_dir.rstrip('/')}/calib_bme.json", "r") as fh:
        deployed_calib = json.loads(fh.read())

    deployed_scorer = load_pooled_scorer(args.deployed_dir)
    fusion_scorer = load_pooled_scorer(args.fusion_dir)
    if fusion_scorer.model.config.doc_embed_dim != embed_int8.shape[1]:
        raise ValueError(
            f"{args.fusion_dir} expects a {fusion_scorer.model.config.doc_embed_dim}-d embedding, "
            f"got {embed_int8.shape[1]}"
        )

    shards = sorted(str(m) for m in StoragePath(f"{args.eval_docs.rstrip('/')}/*.parquet").glob())
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    logger.info("doc table: %d shards", len(shards))

    ids: list[str] = []
    sources: list[str] = []
    domains: list[str] = []
    chars: list[int] = []
    windows: list[list[float]] = []
    deployed_raw: list[float] = []
    fusion_raw: list[float] = []
    deployed_types: list[str] = []
    fusion_types: list[str] = []
    missing_embedding = 0
    seen: set[str] = set()

    for n, shard in enumerate(shards, 1):
        with StoragePath(shard).open("rb") as fh:
            table = pq.ParquetFile(fh).read(columns=["id", "source", "domain_id", "text"])
        data = {c: table.column(c).to_pylist() for c in ("id", "source", "domain_id", "text")}
        rows = []
        for i, doc_id in enumerate(data["id"]):
            if doc_id in seen:
                continue
            if doc_id not in index:
                missing_embedding += 1
                continue
            seen.add(doc_id)
            rows.append(i)
        texts = [data["text"][i] or "" for i in rows]
        raw_embed = [embed_int8[index[data["id"][i]]] for i in rows]

        per_window = _score_windows(deployed_scorer, texts)
        fusion = fusion_scorer.score(texts, batch_size=SCORE_BATCH, doc_embed=embedding_matrix(raw_embed))

        ids.extend(data["id"][i] for i in rows)
        sources.extend(data["source"][i] for i in rows)
        domains.extend(str(data["domain_id"][i]) for i in rows)
        chars.extend(len(t) for t in texts)
        windows.extend([float(v) for v in w] for w in per_window)
        deployed_raw.extend(float(w.mean()) for w in per_window)
        fusion_raw.extend(float(v) for v in fusion)
        deployed_types.extend(content_type.predict(text_typer, texts))
        fusion_types.extend(str(t) for t in domain_mlp.predict(typer, typer_labels, raw_embed))
        logger.info("doc table: shard %d/%d, %d rows scored", n, len(shards), len(rows))

    deployed = np.array(deployed_raw, dtype=np.float64)
    fusion = np.array(fusion_raw, dtype=np.float64)
    deployed_cal = apply_calibration(deployed, np.array(deployed_types), deployed_calib)
    fusion_cal = apply_calibration(fusion, np.array(fusion_types), fusion_calib)

    table = pa.table(
        {
            "id": pa.array(ids, pa.string()),
            "source": pa.array(sources, pa.string()),
            "domain_id": pa.array(domains, pa.string()),
            "chars": pa.array(chars, pa.int32()),
            "v3_raw": pa.array(deployed, pa.float32()),
            "v3_score": pa.array(deployed_cal, pa.float32()),
            "v3_bucket": pa.array(_buckets(deployed_cal), pa.int8()),
            "v3_windows": pa.array(windows, pa.list_(pa.float32())),
            "v3_type": pa.array(deployed_types, pa.string()),
            "fusion_raw": pa.array(fusion, pa.float32()),
            "fusion_score": pa.array(fusion_cal, pa.float32()),
            "fusion_bucket": pa.array(_buckets(fusion_cal), pa.int8()),
            "fusion_type": pa.array(fusion_types, pa.string()),
        }
    )
    with StoragePath(args.doc_table).open("wb") as fh:
        pq.write_table(table, fh, compression="zstd")
    logger.info(
        "doc table: wrote %s — %d rows, %d without an embedding, %d sources",
        args.doc_table,
        table.num_rows,
        missing_embedding,
        len(set(sources)),
    )


def _cell_quota(cell_sizes: dict, budget: int) -> dict:
    """Documents to draw per ``(family, bucket)`` cell.

    A floor for every non-empty cell, then the remaining budget spread
    proportional to cell population. The spread water-fills: a cell that hits
    ``MAX_PER_CELL`` hands its excess back to the cells still under the cap
    instead of forfeiting it, so a skewed population still spends the budget.
    Deterministic given the same populations.
    """
    ceiling = {key: min(MAX_PER_CELL, n) for key, n in cell_sizes.items()}
    quota = {key: min(MIN_PER_CELL, n) for key, n in cell_sizes.items()}
    while True:
        remaining = budget - sum(quota.values())
        open_cells = [key for key in cell_sizes if quota[key] < ceiling[key]]
        if remaining <= 0 or not open_cells:
            return quota
        total = sum(cell_sizes[key] for key in open_cells)
        added = 0
        for key in sorted(open_cells):
            take = min(ceiling[key] - quota[key], int(remaining * cell_sizes[key] / total))
            quota[key] += take
            added += take
        if added:
            continue
        # Every proportional share has rounded below one; hand the remainder to
        # the largest cells so the budget is spent rather than left on the table.
        for key in sorted(open_cells, key=lambda k: (-cell_sizes[k], k))[:remaining]:
            quota[key] += 1
        return quota


def _rank_key(doc_id: str) -> int:
    """A stable pseudo-random order over document ids, independent of shard order."""
    return int.from_bytes(hashlib.blake2b(doc_id.encode(), digest_size=8).digest(), "big")


def _windows_for(text: str, cap: int) -> list[str]:
    """The begin/middle/end windows the deployed model read, each truncated to ``cap``."""
    flat, _ = bme_chunks([text])
    return [w[:cap] for w in flat]


def run_payload(args) -> None:
    """Aggregate the population, draw the stratified sample, attach text."""
    with StoragePath(args.doc_table).open("rb") as fh:
        scored = pq.read_table(fh).to_pydict()

    ids = scored["id"]
    sources = scored["source"]
    families = [_family(s) for s in sources]
    v3_bucket = np.array(scored["v3_bucket"], dtype=int)
    fusion_bucket = np.array(scored["fusion_bucket"], dtype=int)
    total = len(ids)
    logger.info("payload: %d scored rows, %d sources, %d families", total, len(set(sources)), len(set(families)))

    # Aggregates over the whole population, so the shares on the page describe
    # the evaluation set and not the sample drawn from it.
    corpus = {
        "n": total,
        "v3": [int((v3_bucket == b).sum()) for b in range(5)],
        "fusion": [int((fusion_bucket == b).sum()) for b in range(5)],
        "agree": int((v3_bucket == fusion_bucket).sum()),
    }

    by_family: dict[str, list[int]] = {}
    for i, family in enumerate(families):
        by_family.setdefault(family, []).append(i)

    cell_sizes: dict[tuple[str, int], int] = {}
    for family, rows in by_family.items():
        for b in range(5):
            n = sum(1 for i in rows if v3_bucket[i] == b)
            if n:
                cell_sizes[(family, b)] = n
    quota = _cell_quota(cell_sizes, args.sample_docs)

    picked: dict[tuple[str, int], list[int]] = {}
    for (family, b), k in quota.items():
        rows = [i for i in by_family[family] if v3_bucket[i] == b]
        rows.sort(key=lambda i: _rank_key(ids[i]))
        picked[(family, b)] = rows[:k]
    wanted = {i for rows in picked.values() for i in rows}
    logger.info("payload: sampled %d documents across %d cells", len(wanted), len(picked))

    # Text for the sampled rows only, read back from the evaluation shards.
    wanted_ids = {ids[i]: i for i in wanted}
    text_by_row: dict[int, str] = {}
    shards = sorted(str(m) for m in StoragePath(f"{args.eval_docs.rstrip('/')}/*.parquet").glob())
    for n, shard in enumerate(shards, 1):
        with StoragePath(shard).open("rb") as fh:
            table = pq.ParquetFile(fh).read(columns=["id", "text"])
        for doc_id, text in zip(table.column("id").to_pylist(), table.column("text").to_pylist(), strict=True):
            row = wanted_ids.get(doc_id)
            if row is not None and row not in text_by_row:
                text_by_row[row] = text or ""
        logger.info("payload: text shard %d/%d, %d/%d recovered", n, len(shards), len(text_by_row), len(wanted))
    if len(text_by_row) != len(wanted):
        raise ValueError(f"recovered text for {len(text_by_row)} of {len(wanted)} sampled documents")

    source_index = {name: i for i, name in enumerate(sorted(set(sources)))}
    order = sorted(wanted, key=lambda i: (families[i], v3_bucket[i], _rank_key(ids[i])))
    doc_slot = {row: slot for slot, row in enumerate(order)}
    docs = []
    for row in order:
        docs.append(
            [
                ids[row],
                source_index[sources[row]],
                int(scored["chars"][row]),
                round(float(scored["v3_score"][row]), SCORE_DECIMALS),
                int(v3_bucket[row]),
                [round(float(v), SCORE_DECIMALS) for v in scored["v3_windows"][row]],
                scored["v3_type"][row],
                round(float(scored["fusion_score"][row]), SCORE_DECIMALS),
                int(fusion_bucket[row]),
                round(float(scored["fusion_raw"][row]), SCORE_DECIMALS),
                scored["fusion_type"][row],
                _windows_for(text_by_row[row], args.window_chars),
            ]
        )

    family_blocks = []
    for family in sorted(by_family, key=lambda f: -len(by_family[f])):
        rows = by_family[family]
        v3_counts = [int(sum(1 for i in rows if v3_bucket[i] == b)) for b in range(5)]
        fusion_counts = [int(sum(1 for i in rows if fusion_bucket[i] == b)) for b in range(5)]
        family_blocks.append(
            {
                "name": family,
                "sources": sorted({source_index[sources[i]] for i in rows}),
                "n": len(rows),
                "v3": v3_counts,
                "fusion": fusion_counts,
                "agree": int(sum(1 for i in rows if v3_bucket[i] == fusion_bucket[i])),
                "cells": {
                    str(b): [doc_slot[i] for i in picked.get((family, b), [])] for b in range(5) if (family, b) in picked
                },
            }
        )

    payload = {
        "meta": {
            "population": total,
            "sampled": len(docs),
            "sources": len(source_index),
            "families": len(family_blocks),
            "window_chars": args.window_chars,
            "min_per_cell": MIN_PER_CELL,
            "max_per_cell": MAX_PER_CELL,
            "eval_docs": args.eval_docs,
            "v3_model": args.deployed_dir,
            "fusion_model": args.fusion_dir,
        },
        "bucket_labels": list(BUCKET_LABELS),
        "bucket_colors": list(BUCKET_COLORS),
        "doc_fields": [
            "id",
            "source",
            "chars",
            "v3_score",
            "v3_bucket",
            "v3_windows",
            "v3_type",
            "fusion_score",
            "fusion_bucket",
            "fusion_raw",
            "fusion_type",
            "window_text",
        ],
        "corpus": corpus,
        "source_names": sorted(source_index),
        "families": family_blocks,
        "docs": docs,
    }
    blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    with StoragePath(args.payload).open("w") as fh:
        fh.write(blob)
    logger.info("payload: wrote %s — %d documents, %.1f MB", args.payload, len(docs), len(blob.encode()) / 1e6)


def run_verify(args) -> None:
    """Check the table's v3 column against the scores the deployed model shipped.

    The page presents the v3 column as what production scored, so it has to be
    what production scored. ``score`` recomputes it rather than reading the
    stored run, which is only safe if the two agree.
    """
    with StoragePath(args.doc_table).open("rb") as fh:
        scored = pq.read_table(fh).to_pydict()
    stored = _compare_against_stored(
        np.array(scored["id"]),
        np.array(scored["v3_bucket"], dtype=int),
        np.array(scored["v3_score"], dtype=float),
        args.deployed_scored,
    )
    logger.info("verify: %s", json.dumps(stored))
    if stored["status"] != "compared":
        raise ValueError(f"could not compare against {args.deployed_scored}: {stored}")
    if stored["bucket_agreement"] < 1.0:
        raise ValueError(f"v3 buckets disagree with the shipped scores: {stored}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=("score", "payload", "verify"))
    p.add_argument("--fusion-dir", default=FUSION_DIR)
    p.add_argument("--deployed-dir", default=DEPLOYED_DIR)
    p.add_argument("--domain-mlp", default=DOMAIN_MLP)
    p.add_argument("--eval-docs", default=EVAL_DOCS)
    p.add_argument("--eval-embeddings", default=EVAL_EMBEDDINGS)
    p.add_argument("--doc-table", default=DOC_TABLE)
    p.add_argument("--deployed-scored", default=DEPLOYED_SCORED)
    p.add_argument("--payload", default=VIEWER_PAYLOAD)
    p.add_argument("--sample-docs", type=int, default=DEFAULT_SAMPLE_DOCS)
    p.add_argument("--window-chars", type=int, default=DEFAULT_WINDOW_CHARS)
    p.add_argument("--limit-shards", type=int, default=0, help="score stage smoke run: only N shards")
    args = p.parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()

    if args.stage == "score":
        run_score(args)
    elif args.stage == "payload":
        run_payload(args)
    else:
        run_verify(args)


if __name__ == "__main__":
    main()
