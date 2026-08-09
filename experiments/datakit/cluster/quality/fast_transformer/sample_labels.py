# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Choose which documents the oracle labels, from a scored sample tree.

The deployed label set is what caps the scorer: 5,578 rows over 113 sources, of
which 75 (1.3%) are quality-5. A model trained on that cannot rank the top of the
distribution, and it shows — the deployed scorer puts 2.4% of documents in its top
bucket and 82% in the middle two. Drawing more labels uniformly would reproduce the
same shape, only larger, because a uniform draw from web-scale data is mostly
average documents.

So the draw is stratified by the *current* scorer's opinion, per source:

* **Every source is represented.** ``per_source_floor`` documents each, so all 292
  sources reach the oracle instead of the 113 the deployed set covers. Source
  families that arrived since the last labeling round (agent trajectories, the
  ``finepdfs`` language splits, safety data) are otherwise invisible to it.
* **The top of each source is over-drawn.** Within a source, quota is spread over
  v0-score quintiles with ``QUINTILE_WEIGHTS``, weighted toward the top. Each
  source's own best documents get labeled, so the oracle's 5s can come from every
  content type rather than only from the types v0 already scores highly (math lands
  88.5% in the top two buckets; multilingual 23.5%).
* **The bottom is kept.** A scale needs both ends: the bottom quintile still draws,
  so the oracle sees junk and the 1s stay populated.

Stratifying on v0 imports v0's bias into *what gets looked at* — but not into the
labels, which the oracle assigns independently. The floor is what bounds that risk:
a source v0 misjudges wholesale still reaches the oracle through its floor.

Documents come from the scoring run's ``outputs/main`` joined by id to the corpus
text, not from its ``outputs/samples`` side output. The side output is the obvious
input — it carries ``text`` beside ``score`` and needs no join — but it keeps a
fixed 2% of each shard, which rounds to zero for a source holding a handful of
documents. Roughly half the registry is agent-trajectory sources that small, so
reading it drops precisely the sources the per-source floor exists to guarantee:
drawing from the side output covered 292 sources on paper and returned 7,271 rows
against a 20,000 target, with the smallest sources contributing nothing.
"""

import argparse
import logging
import posixpath
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from marin.datakit.sources import all_sources
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

# Relative draw weight per v0-score quintile, lowest first. The top two quintiles
# carry half the quota: that is where the deployed scorer cannot discriminate and
# where selection actually reads. The bottom quintile keeps the 1s populated.
QUINTILE_WEIGHTS = (0.15, 0.10, 0.15, 0.25, 0.35)
N_QUINTILES = len(QUINTILE_WEIGHTS)
DEFAULT_TARGET = 20_000
DEFAULT_FLOOR = 24
# Read a few times the quota before sampling, so quintiles are cut over a
# representative slice of a large source rather than one shard.
SHARD_OVERDRAW = 4
# Rows per output shard, so a large draw streams instead of being held in memory.
SHARD_ROWS = 5_000
# Concurrent per-source draws. The work is object-store round trips, not CPU.
DEFAULT_WORKERS = 16
# Oracle prompts carry the document inline, so cap what a single row contributes.
#
# The cap must not look like corpus damage. The rubric marks a document
# "truncated mid-token" invalid, and invalid forces quality 1 — so a hard slice at
# a fixed offset teaches the grader that long documents are garbage. It did exactly
# that: at a 12k hard cut, 34.7% of rows hit the cap, those rows were called invalid
# 28.6% of the time against 2.4% for complete documents, 85.5% of every quality-1
# label sat at the cap, and length-to-quality correlation reached -0.25. A model
# trained on that target learns "long document = low quality".
#
# So cut on a paragraph boundary and mark the result as an excerpt
# (:data:`EXCERPT_NOTICE`), which leaves no mid-token edge for the grader to read as
# damage.
MAX_TEXT_CHARS = 12_000
# How far back to look for a clean break before giving up and cutting at the cap.
BOUNDARY_SEARCH_CHARS = 2_000
# Tried in order, coarsest first. The closing-brace forms matter for structured
# sources: a JSON-per-line corpus has no paragraph or sentence breaks at all, and
# every over-cap document in massive_function_calling is a single line.
BOUNDARY_MARKERS = ("\n\n", ". ", "\n", "}, ", "},", "} ", "; ", ", ")
EXCERPT_NOTICE = "\n\n[Excerpt ends here — the document continues beyond this point.]"


def excerpt(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    """``text`` capped to ``limit``, cut on a boundary and marked as an excerpt.

    A document under the cap is returned unchanged, so most rows carry no notice at
    all and the grader has nothing to react to.

    ``limit`` is a parameter because the labeler applies a second, smaller cap: this
    one keeps the stored label set readable, while the labeler needs the prompt to
    fit a token budget that characters only approximate.
    """
    if len(text) <= limit:
        return text
    window = text[:limit]
    floor = limit - BOUNDARY_SEARCH_CHARS
    for marker in BOUNDARY_MARKERS:
        cut = window.rfind(marker, floor)
        if cut > 0:
            return window[: cut + len(marker)].rstrip() + EXCERPT_NOTICE
    # No structural boundary: break at the last whitespace instead. Structured
    # sources exist with none at all — every document over the cap in
    # massive_function_calling is one line — and for those a fixed slice lands
    # mid-token, which is precisely the damage the rubric punishes.
    space = max(window.rfind(c, floor) for c in (" ", "\t"))
    if space > 0:
        return window[:space].rstrip() + EXCERPT_NOTICE
    return window + EXCERPT_NOTICE


def _read_columns(path: str, columns: tuple[str, ...]) -> pa.Table | None:
    """``columns`` from one parquet shard, or ``None`` when the shard does not carry them.

    An empty shard is written with no columns at all, so asking for names raises
    rather than returning an empty table. Reading the schema first turns that into
    a skip instead of aborting a 292-source draw.
    """
    with StoragePath(path).open("rb") as handle:
        parquet = pq.ParquetFile(handle)
        if not set(columns) <= set(parquet.schema_arrow.names):
            return None
        return parquet.read(columns=list(columns))


def _source_frame(
    source: str, scored_prefix: str, corpus_prefix: str, rng: np.random.Generator, quota: int
) -> list[dict]:
    """Draw ``quota`` documents for one source, spread over its v0-score quintiles.

    Scores are read first and text only for the documents actually drawn. Reading a
    shard's text to pick a few dozen rows from it does not fit in memory: a shard of
    a large source holds ~200k documents, so materializing its text costs about a
    gigabyte to keep 68 rows, and doing that per source is an OOM rather than a slow
    path.

    Documents come from the scored ``outputs/main``, not the scorer's
    ``outputs/samples`` side output. The side output carries text beside score and
    needs no join, but keeps a fixed 2% of each shard, which rounds to *zero* for a
    source holding a handful of documents — silently dropping exactly the small
    agent-trajectory sources the per-source floor exists to cover.
    """
    scored_dir = f"{scored_prefix.rstrip('/')}/{source}/outputs/main"
    corpus_dir = f"{corpus_prefix.rstrip('/')}/{source}/outputs/main"
    shards = sorted(str(m) for m in StoragePath(f"{scored_dir}/*.parquet").glob())
    if not shards:
        return []

    # Shuffled so a large source is not always drawn from its first shard, and
    # truncated so a 40-shard source does not read every shard to fill 68 rows.
    rng.shuffle(shards)
    scored_rows: list[tuple[str, str, float]] = []  # (shard basename, id, score)
    for shard in shards:
        table = _read_columns(shard, ("id", "score"))
        if table is None:
            logger.warning("sample_labels: %s: shard %s carries no columns", source, posixpath.basename(shard))
            continue
        name = posixpath.basename(shard)
        scored_rows.extend(
            (name, doc_id, float(score))
            for doc_id, score in zip(table.column("id").to_pylist(), table.column("score").to_pylist(), strict=True)
        )
        if len(scored_rows) >= quota * SHARD_OVERDRAW:
            break
    if not scored_rows:
        return []

    scores = np.array([r[2] for r in scored_rows], dtype=np.float64)
    order = np.argsort(scores, kind="stable")
    bounds = np.linspace(0, len(order), N_QUINTILES + 1).astype(int)
    picked: list[int] = []
    for q, weight in enumerate(QUINTILE_WEIGHTS):
        band = order[bounds[q] : bounds[q + 1]]
        if len(band) == 0:
            continue
        want = min(len(band), max(1, round(quota * weight)))
        picked.extend(rng.choice(band, size=want, replace=False).tolist())

    # One corpus read per shard, keeping only the drawn ids.
    wanted: dict[str, dict[str, float]] = {}
    for i in picked:
        shard_name, doc_id, score = scored_rows[i]
        wanted.setdefault(shard_name, {})[doc_id] = score

    rows: list[dict] = []
    for shard_name, id_scores in wanted.items():
        corpus = _read_columns(f"{corpus_dir}/{shard_name}", ("id", "text"))
        if corpus is None:
            continue
        for doc_id, text in zip(corpus.column("id").to_pylist(), corpus.column("text").to_pylist(), strict=True):
            if doc_id in id_scores:
                rows.append(
                    {
                        "source": source,
                        "id": doc_id,
                        "v0_score": id_scores[doc_id],
                        "text": excerpt(text or ""),
                    }
                )
    return rows


def build_label_set(
    *,
    scored_prefix: str,
    corpus_prefix: str,
    out_dir: str,
    sources: list[str],
    target: int = DEFAULT_TARGET,
    per_source_floor: int = DEFAULT_FLOOR,
    seed: int = 42,
    workers: int = DEFAULT_WORKERS,
) -> int:
    """Draw the oracle's labeling set across ``sources``; returns rows written.

    Shards are written as sources are drawn rather than accumulated. At a 100k
    target the set is over a gigabyte of text, and holding it to build one table
    doubles that at the moment of writing.

    Quota is the per-source floor plus an equal share of whatever ``target`` leaves,
    so small and large sources both clear the floor and the remainder spreads evenly
    rather than by corpus size — the oracle needs each source's own range, not a
    corpus-proportional sample.
    """
    extra = max(0, target - per_source_floor * len(sources))
    quota = per_source_floor + extra // max(1, len(sources))
    logger.info(
        "sample_labels: %d sources x %d docs (floor %d) -> target ~%d",
        len(sources),
        quota,
        per_source_floor,
        quota * len(sources),
    )

    out_root = out_dir.rstrip("/")
    buffer: list[dict] = []
    written = 0
    shard_index = 0
    # Ids already drawn. A corpus shard can carry the same document more than once —
    # one document appeared 19 times in a single common-crawl source — and paying the
    # oracle to grade it repeatedly also over-weights it in training.
    seen_ids: set[str] = set()
    duplicates = 0

    def flush() -> None:
        nonlocal buffer, written, shard_index
        if not buffer:
            return
        with StoragePath(f"{out_root}/part-{shard_index:05d}.parquet").open("wb") as handle:
            pq.write_table(pa.Table.from_pylist(buffer), handle)
        written += len(buffer)
        shard_index += 1
        buffer = []

    # Sources are drawn concurrently: the work is object-store reads, and serially a
    # 100k draw is hours of waiting on round trips. Each source gets its own seeded
    # generator so a draw stays reproducible regardless of completion order.
    def draw(indexed: tuple[int, str]) -> tuple[str, list[dict]]:
        index, name = indexed
        return name, _source_frame(name, scored_prefix, corpus_prefix, np.random.default_rng([seed, index]), quota)

    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (name, drawn) in enumerate(pool.map(draw, enumerate(sources)), 1):
            if not drawn:
                logger.warning("sample_labels: %s contributed no rows", name)
            with lock:
                fresh = [row for row in drawn if row["id"] not in seen_ids]
                duplicates += len(drawn) - len(fresh)
                seen_ids.update(row["id"] for row in fresh)
                buffer.extend(fresh)
                if len(buffer) >= SHARD_ROWS:
                    flush()
                total = written + len(buffer)
            if i % 25 == 0:
                logger.info("sample_labels: %d/%d sources, %d rows", i, len(sources), total)
    flush()
    logger.info(
        "sample_labels: drew %d rows from %d sources into %d shards (%d duplicate ids dropped)",
        written,
        len(sources),
        shard_index,
        duplicates,
    )
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-prefix", required=True, help="root of a scoring run (per-source dirs)")
    parser.add_argument("--corpus-prefix", required=True, help="sample tree holding the documents' text")
    parser.add_argument(
        "--sources-file",
        default=None,
        help="newline-separated source names; omit to draw from every registered source",
    )
    parser.add_argument("--out", required=True, help="directory the labeling-set shards are written to")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET, help="approximate total documents to draw")
    parser.add_argument("--per-source-floor", type=int, default=DEFAULT_FLOOR, help="minimum documents per source")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="concurrent per-source draws")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()

    if args.sources_file:
        with open(args.sources_file) as handle:
            sources = [line.strip() for line in handle if line.strip()]
    else:
        sources = sorted(all_sources())
        logger.info("sample_labels: drawing from every registered source (%d)", len(sources))
    written = build_label_set(
        scored_prefix=args.scored_prefix,
        corpus_prefix=args.corpus_prefix,
        out_dir=args.out,
        sources=sources,
        target=args.target,
        per_source_floor=args.per_source_floor,
        seed=args.seed,
        workers=args.workers,
    )
    logger.info("sample_labels: wrote %d rows to %s", written, args.out)


if __name__ == "__main__":
    main()
