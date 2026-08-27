# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score the incumbent router and fit a candidate against the routing study table.

The question is not "which model has better AUC". It is whether the routing decision can be made
better *at a fixed cost*, where the two errors are not symmetric:

* Sending a document to the VLM that Docling would have read correctly costs GPU time. It is the
  expensive mistake, and it is recoverable -- the text is fine.
* Sending a document to Docling that only the VLM can read costs corpus quality, silently. The
  document lands in training data as truncated, garbled or misordered text, and nothing downstream
  will flag it.

So the model is scored on the quality-loss/cost frontier rather than on accuracy, and the operating
point is chosen on that curve. :func:`route_frontier` reports, for every threshold, what fraction
of the corpus goes to the VLM and how much Docling-side quality loss is accepted for it.

**The label.** ``docling_ok`` is derived from the agreement columns, thresholded at
:data:`RECALL_FLOOR` on bigram recall with a page-level floor as well, and forced false wherever the
Docling route produced nothing at all. Documents whose VLM extraction is itself untrustworthy --
truncated pages, loop repair, pages rendered below the legibility floor -- are dropped rather than
labeled, because on those rows disagreement measures the VLM's failure rather than Docling's, and
training on them teaches the router the wrong lesson.
"""

import logging
from dataclasses import dataclass

import fsspec
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# Bigram recall at or above this counts as "Docling read the same document". Bigram rather than
# unigram because reading order is one of the properties being protected, and the page floor
# because a mean hides one destroyed page in a long report.
RECALL_FLOOR = 0.80
MAX_DESTROYED_PAGE_FRACTION = 0.10

# The incumbent's operating point, for reference on the frontier.
INCUMBENT_THRESHOLD = 0.20


def load(paths: list[str], storage_options: dict[str, str] | None = None) -> pl.DataFrame:
    return pl.scan_parquet(paths, storage_options=storage_options).collect(engine="streaming")


def read_table(prefix: str, fs: fsspec.AbstractFileSystem) -> pl.DataFrame:
    """Read every part file under a prefix into one frame.

    Shard by shard rather than one ``scan_parquet``: a shard in which no document failed types its
    error columns as null, and a strict concatenation of that against a shard that did have a
    failure errors out.
    """
    paths = sorted(fs.glob(f"{prefix}/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no shards under {prefix}")
    frames = []
    for path in paths:
        with fs.open(path, "rb") as stream:
            frames.append(pl.read_parquet(stream))
    logger.info("read %d shards from %s", len(paths), prefix)
    return pl.concat(frames, how="diagonal_relaxed")


def route_ok(prefix: str, missing: pl.Expr, metric: str = "bigram") -> pl.Expr:
    """``docling_ok`` generalized to any cheap route, from that route's agreement columns.

    The identical construction the published label uses: recall at or above the floor, no more than
    :data:`MAX_DESTROYED_PAGE_FRACTION` of pages destroyed outright, and false wherever the route
    produced nothing at all. The metric is a parameter only so the unigram variant can be reported
    alongside; the decision is made on bigrams, because reading-order damage is invisible to
    unigrams and this repository has already retired a backend that scored 0.935 unigram F1 while
    splicing multi-column reading order.
    """
    return (
        missing.not_()
        & (pl.col(f"{prefix}_{metric}_recall_mean") >= RECALL_FLOOR)
        & (pl.col(f"{prefix}_frac_pages_{metric}_below_50") <= MAX_DESTROYED_PAGE_FRACTION)
    )


def label(frame: pl.DataFrame) -> pl.DataFrame:
    """Attach ``docling_ok`` and ``trustworthy``, the two columns the rest of this module needs.

    ``trustworthy`` marks rows where the VLM side is clean enough to be treated as the reference.
    It is deliberately strict: the whole method rests on the VLM extraction being the better one,
    and a truncated or loop-repaired page is neither better nor comparable.
    """
    return frame.with_columns(
        trustworthy=(
            (pl.col("pages_truncated") == 0)
            & (pl.col("pages_failed") == 0)
            & (pl.col("pages_unrendered") == 0)
            & (pl.col("loop_chars_dropped") == 0)
            & (pl.col("pages_below_legibility_floor") == 0)
        ),
        docling_ok=(
            pl.col("docling_missing").not_()
            & (pl.col("bigram_recall_mean") >= RECALL_FLOOR)
            & (pl.col("frac_pages_bigram_below_50") <= MAX_DESTROYED_PAGE_FRACTION)
        ),
    )


@dataclass(frozen=True)
class RoutePoint:
    """One operating point: what it sends to the VLM, and what it lets through broken."""

    threshold: float
    vlm_fraction: float
    """Share of documents routed to the VLM -- directly proportional to GPU cost."""
    quality_loss: float
    """Share of *all* documents sent to Docling that Docling reads badly. The silent failure."""
    wasted_vlm_fraction: float
    """Share of all documents sent to the VLM that Docling would have read correctly."""
    recall_of_bad: float
    """Share of Docling-unreadable documents correctly sent to the VLM."""


def route_frontier(scores: np.ndarray, docling_ok: np.ndarray, thresholds: np.ndarray) -> list[RoutePoint]:
    """Evaluate a score's routing behaviour across thresholds.

    A document goes to the VLM when its score is at or above the threshold, so a lower threshold
    buys quality with GPU time.
    """
    total = len(scores)
    points = []
    for threshold in thresholds:
        to_vlm = scores >= threshold
        points.append(
            RoutePoint(
                threshold=float(threshold),
                vlm_fraction=float(to_vlm.mean()),
                quality_loss=float((~to_vlm & ~docling_ok).sum() / total),
                wasted_vlm_fraction=float((to_vlm & docling_ok).sum() / total),
                recall_of_bad=float((to_vlm & ~docling_ok).sum() / max((~docling_ok).sum(), 1)),
            )
        )
    return points


def point_at_budget(points: list[RoutePoint], vlm_fraction: float) -> RoutePoint:
    """The operating point closest to a given VLM budget, for cost-matched comparison.

    Comparing two routers at their own preferred thresholds compares two different budgets. The
    honest comparison holds the GPU spend fixed and asks which router loses less quality for it.
    """
    return min(points, key=lambda point: abs(point.vlm_fraction - vlm_fraction))


def describe_incumbent(frame: pl.DataFrame) -> dict:
    """What the FinePDFs router actually did on this corpus, by its own shipped rule.

    Its rule is not the probability alone: a document also goes to OCR when any sampled page
    produced a replacement character, whatever the model thought.
    """
    routed_to_vlm = (pl.col("ocr_prob") >= INCUMBENT_THRESHOLD) | (pl.col("garbled_text_ratio") > 0.0)
    scored = frame.filter(pl.col("ocr_prob").is_not_null())
    decided = scored.with_columns(to_vlm=routed_to_vlm)
    total = decided.height
    return {
        "documents": total,
        "vlm_fraction": decided["to_vlm"].mean(),
        "quality_loss": decided.filter(pl.col("to_vlm").not_() & pl.col("docling_ok").not_()).height / total,
        "wasted_vlm_fraction": decided.filter(pl.col("to_vlm") & pl.col("docling_ok")).height / total,
        "garbled_override_share": (
            decided.filter((pl.col("ocr_prob") < INCUMBENT_THRESHOLD) & (pl.col("garbled_text_ratio") > 0.0)).height
            / total
        ),
    }
