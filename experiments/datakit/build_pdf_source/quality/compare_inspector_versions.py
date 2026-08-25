# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""What changed between two builds of pdf-inspector, measured on the same documents.

Two Stage 1 tables built over the same 100,000 documents by two builds of the library
(:mod:`~experiments.datakit.build_pdf_source.quality.build_inspector_study`), joined on
``source_id``. Every quantity here is a **paired** difference: the same document, read twice.

Pairing is not a refinement, it is the whole design. The published pass measured ~0.012 of
split-draw noise on the routing frontier -- the spread across domain-disjoint splits of one dataset
-- and several of the effects at issue are smaller than that. An unpaired before-and-after would put
the effect and the noise in the same number and could not separate them. Differencing per document
removes everything the two runs share, which is the corpus, the sample, the VLM reference, Docling,
the normalizer and the split.

What the paired difference cannot remove is a real change that happens to be small. So the
per-stratum tables carry a **paired bootstrap interval over domains, not documents**: the crawl
holds ~9.8% exact-duplicate PDFs and many more near-duplicates from one publisher's template, so
documents are not independent and an interval computed over them would be too narrow by whatever
the duplication rate is. Resampling registered domains keeps a publisher's documents together.

Three questions, in the order they can invalidate each other:

*Did anything break?* Outcome classes per build. A new panic class or a document the new build
refuses is a correctness result and comes before any quality number computed on the survivors --
and it is also a bias: if a build fails the documents it reads worst, its mean improves for the
wrong reason. Reported as a transition matrix, so a document that changed outcome is visible rather
than netted out against one that changed the other way.

*Did it get slower?* The library's own per-page timings, paired. The report's cost model rests on a
single number (extract ms/page against Docling's ~1000), and the table and layout work of 1.15-1.17
adds passes. A tail that lengthens while the median holds is the shape to look for and the reason
this reports quantiles of the per-document *ratio* rather than a ratio of means.

*Did it read the documents better?* Bigram recall against the VLM, per document, per stratum. The
strata are the adjudication set's own
(:mod:`~experiments.datakit.build_pdf_source.quality.build_adjudication_set`), assigned from the
baseline table so the two builds are sliced identically -- a stratum defined from the new build's
own signals would move underneath the comparison, since ``table_heavy`` is defined on a column the
extraction produces.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name pdf-inspector-compare --extra pdf \\
        --cpu 16 --memory 48GB --disk 16GB --enable-extra-resources \\
        -- python -m experiments.datakit.build_pdf_source.quality.compare_inspector_versions
"""

import json
import logging

import numpy as np
import polars as pl
from rigging.log_setup import configure_logging

from experiments.datakit.build_pdf_source.quality.analyze_route_study import label, read_table
from experiments.datakit.build_pdf_source.quality.build_adjudication_set import (
    ROUTE_STUDY_PREFIX,
    STRATA,
    stratum_of,
)
from experiments.datakit.build_pdf_source.quality.build_inspector_study import BASELINE_PREFIX, OUTPUT_PREFIX
from experiments.datakit.build_pdf_source.quality.build_route_study import storage

logger = logging.getLogger(__name__)

RESULT_PATH = "s3://marin-us-east-02a/marin/data/pdf_quality/cc_focus_2026_22_inspector_version_compare.json"

BASELINE_VERSION = "1.14.1"
CANDIDATE_VERSION = "1.17.0"

# The published split-draw noise floor: the spread the routing frontier shows across domain-disjoint
# splits of one dataset, with nothing else changed. A paired difference is not subject to it -- that
# is the point of pairing -- but the report has to say whether a moved number cleared it, because
# the published numbers it is being compared against are.
SPLIT_DRAW_NOISE = 0.012

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260825

# The agreement columns worth differencing. Bigram leads: reading-order damage is invisible to
# unigrams, and this repository has already retired a backend that scored 0.935 unigram F1 while
# splicing multi-column reading order.
METRICS = (
    "inspector_vlm_bigram_recall_mean",
    "inspector_vlm_bigram_precision_mean",
    "inspector_vlm_unigram_recall_mean",
    "inspector_vlm_frac_pages_bigram_below_50",
    "inspector_docling_bigram_recall_mean",
)
HEADLINE_METRIC = "inspector_vlm_bigram_recall_mean"

# Timing columns, paired per document.
TIMINGS = ("inspector_detect_ms_per_page", "inspector_extract_ms_per_page")

CARRIED = (
    "source_id",
    "domain",
    "num_pages",
    "inspector_outcome",
    "inspector_error",
    "inspector_extracted_pages",
    "inspector_markdown_chars",
    "inspector_page_count",
    "inspector_extract_pages_with_tables",
    "inspector_extract_pages_with_columns",
    "inspector_extract_is_complex_layout",
    *METRICS,
    *TIMINGS,
)

# Columns the strata predicates read out of the route study, which is version-independent.
_OLD = "_old"


def joined(fs) -> pl.DataFrame:
    """Both builds' tables and the route study, on one row per document.

    The route study supplies ``trustworthy`` and the script/layout features the strata are defined
    on; both inspector tables supply their own reading of the same document. Suffixed rather than
    renamed column by column so a column added to Stage 1 later arrives here without ceremony.
    """
    route = label(read_table(ROUTE_STUDY_PREFIX, fs))
    baseline = read_table(BASELINE_PREFIX, fs).select(CARRIED)
    candidate = read_table(OUTPUT_PREFIX, fs).select(CARRIED)
    logger.info(
        "route %d, %s %d, %s %d", route.height, BASELINE_VERSION, baseline.height, CANDIDATE_VERSION, candidate.height
    )

    frame = candidate.join(baseline, on="source_id", how="inner", suffix=_OLD)
    frame = frame.join(route.drop("url", "num_pages", "pdf_bytes", "docling_missing"), on="source_id", how="inner")
    logger.info("joined %d documents", frame.height)
    return frame.with_columns(stratum=stratum_of())


def evaluable(frame: pl.DataFrame) -> pl.DataFrame:
    """Documents where a paired quality difference means something.

    Trustworthy on the VLM side, for the reason the published report drops 10.7% of the sample: where
    the VLM extraction is itself truncated or loop-repaired, a disagreement measures the VLM's
    failure. Both builds must also have produced an agreement number -- a document one build lost is
    a *survival* result, counted in the outcome matrix, and averaging quality over the survivors of
    two different failure sets would compare two different corpora.
    """
    return frame.filter(
        pl.col("trustworthy")
        & pl.col("feature_error").is_null()
        & (pl.col("domain") != "")
        & pl.col(HEADLINE_METRIC).is_not_null()
        & pl.col(f"{HEADLINE_METRIC}{_OLD}").is_not_null()
    )


def outcome_matrix(frame: pl.DataFrame) -> dict:
    """How each document's outcome class moved between the builds."""
    counts = (
        frame.group_by(f"inspector_outcome{_OLD}", "inspector_outcome")
        .len()
        .sort("len", descending=True)
        .iter_rows(named=True)
    )
    transitions = {f"{row[f'inspector_outcome{_OLD}']} -> {row['inspector_outcome']}": row["len"] for row in counts}
    errors = (
        frame.filter(pl.col("inspector_outcome") != "ok")["inspector_error"]
        .drop_nulls()
        .value_counts()
        .sort("count", descending=True)
        .head(10)
    )
    return {
        "transitions": transitions,
        f"failures_{BASELINE_VERSION}": int((frame[f"inspector_outcome{_OLD}"] != "ok").sum()),
        f"failures_{CANDIDATE_VERSION}": int((frame["inspector_outcome"] != "ok").sum()),
        "new_failures": int(((frame[f"inspector_outcome{_OLD}"] == "ok") & (frame["inspector_outcome"] != "ok")).sum()),
        "recovered": int(((frame[f"inspector_outcome{_OLD}"] != "ok") & (frame["inspector_outcome"] == "ok")).sum()),
        f"errors_{CANDIDATE_VERSION}": dict(zip(errors["inspector_error"], errors["count"], strict=True)),
    }


def timing_shift(frame: pl.DataFrame) -> dict:
    """Paired per-page timings, as a distribution of ratios rather than a ratio of aggregates.

    An aggregate ratio is a page-weighted mean and hides which documents moved. The table and layout
    passes added since 1.14.1 do not run on every document, so the expected signature is a median
    close to unchanged with a heavy upper tail -- which a single number cannot show.
    """
    shifts = {}
    for column in TIMINGS:
        rows = frame.filter(
            pl.col(column).is_not_null() & pl.col(f"{column}{_OLD}").is_not_null() & (pl.col(f"{column}{_OLD}") > 0)
        )
        if rows.height == 0:
            continue
        ratio = (rows[column] / rows[f"{column}{_OLD}"]).to_numpy()
        pages = rows["inspector_page_count"].fill_null(0).to_numpy().astype(np.float64)
        weighted_old = float((rows[f"{column}{_OLD}"].to_numpy() * pages).sum() / max(pages.sum(), 1))
        weighted_new = float((rows[column].to_numpy() * pages).sum() / max(pages.sum(), 1))
        shifts[column] = {
            "documents": rows.height,
            f"page_weighted_{BASELINE_VERSION}": weighted_old,
            f"page_weighted_{CANDIDATE_VERSION}": weighted_new,
            "page_weighted_ratio": weighted_new / weighted_old if weighted_old else float("nan"),
            "ratio_quantiles": {str(q): float(np.quantile(ratio, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)},
            "share_slower_5pct": float((ratio > 1.05).mean()),
            "share_faster_5pct": float((ratio < 0.95).mean()),
        }
    return shifts


def _bootstrap_domain_ci(values: np.ndarray, domains: np.ndarray) -> tuple[float, float]:
    """A 95% interval for a paired mean, resampling domains rather than documents.

    Near-duplicates cluster by publisher, so a document-level interval would count the same document
    several times as independent evidence. Resampling whole domains keeps them together.
    """
    unique, inverse = np.unique(domains, return_inverse=True)
    if unique.size < 2:
        return (float("nan"), float("nan"))
    by_domain = [values[inverse == index] for index in range(unique.size)]
    sums = np.array([group.sum() for group in by_domain])
    counts = np.array([group.size for group in by_domain])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    picks = rng.integers(0, unique.size, size=(BOOTSTRAP_RESAMPLES, unique.size))
    means = sums[picks].sum(axis=1) / np.maximum(counts[picks].sum(axis=1), 1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def paired_delta(frame: pl.DataFrame, column: str) -> dict:
    """The mean paired change in one metric, with a domain-clustered interval."""
    rows = frame.filter(pl.col(column).is_not_null() & pl.col(f"{column}{_OLD}").is_not_null())
    if rows.height == 0:
        return {"documents": 0}
    delta = (rows[column] - rows[f"{column}{_OLD}"]).to_numpy().astype(np.float64)
    low, high = _bootstrap_domain_ci(delta, rows["domain"].to_numpy())
    return {
        "documents": rows.height,
        "domains": int(rows["domain"].n_unique()),
        BASELINE_VERSION: float(rows[f"{column}{_OLD}"].mean()),
        CANDIDATE_VERSION: float(rows[column].mean()),
        "delta": float(delta.mean()),
        "ci95": [low, high],
        # Whether the interval excludes zero at all, and whether the change is bigger than the noise
        # the published unpaired numbers carry. Two different questions and both are asked.
        "significant": bool(low > 0 or high < 0),
        "exceeds_split_draw_noise": bool(abs(delta.mean()) > SPLIT_DRAW_NOISE),
        "share_improved": float((delta > 0).mean()),
        "share_worsened": float((delta < 0).mean()),
        "share_unchanged": float((delta == 0).mean()),
    }


def by_stratum(frame: pl.DataFrame, column: str) -> dict:
    """The paired change in one metric, sliced by the adjudication set's strata."""
    output = {}
    for stratum in STRATA:
        rows = frame.filter(pl.col("stratum") == stratum.name)
        if rows.height:
            output[stratum.name] = paired_delta(rows, column)
    output["ALL"] = paired_delta(frame, column)
    return output


def report(results: dict) -> str:
    lines = [
        f"pdf-inspector {BASELINE_VERSION} -> {CANDIDATE_VERSION}, paired on {results['evaluable']} documents "
        f"of {results['joined']} joined ({results['domains']} domains)",
        f"split-draw noise floor from the published pass: {SPLIT_DRAW_NOISE}",
        "",
        "== survival ==",
        json.dumps(results["outcomes"], indent=2),
        "",
        "== speed (library's own per-page timings, paired) ==",
    ]
    for column, shift in results["timings"].items():
        lines.append(
            f"{column}: {shift[f'page_weighted_{BASELINE_VERSION}']:.3f} -> "
            f"{shift[f'page_weighted_{CANDIDATE_VERSION}']:.3f} ms/page "
            f"({shift['page_weighted_ratio']:.2f}x), per-document ratio "
            f"p50 {shift['ratio_quantiles']['0.5']:.2f} p90 {shift['ratio_quantiles']['0.9']:.2f} "
            f"p99 {shift['ratio_quantiles']['0.99']:.2f}, {shift['share_slower_5pct']:.1%} slower"
        )

    for metric, table in results["metrics"].items():
        lines.append(f"\n== {metric} ==")
        lines.append(
            f"{'stratum':<32} {'n':>6} {'dom':>5} {BASELINE_VERSION:>9} {CANDIDATE_VERSION:>9} "
            f"{'delta':>9} {'ci95':>20} {'sig':>4} {'>noise':>7}"
        )
        for name, entry in table.items():
            if not entry.get("documents"):
                continue
            low, high = entry["ci95"]
            lines.append(
                f"{name:<32} {entry['documents']:>6} {entry['domains']:>5} {entry[BASELINE_VERSION]:>9.4f} "
                f"{entry[CANDIDATE_VERSION]:>9.4f} {entry['delta']:>+9.4f} "
                f"[{low:>+8.4f},{high:>+8.4f}] {'yes' if entry['significant'] else 'no':>4} "
                f"{'yes' if entry['exceeds_split_draw_noise'] else 'no':>7}"
            )
    return "\n".join(lines)


def main() -> None:
    configure_logging(logging.INFO)
    fs = storage()
    frame = joined(fs)
    rows = evaluable(frame)
    logger.info("evaluable %d documents, %d domains", rows.height, rows["domain"].n_unique())

    results = {
        "baseline_version": BASELINE_VERSION,
        "candidate_version": CANDIDATE_VERSION,
        "baseline_prefix": BASELINE_PREFIX,
        "candidate_prefix": OUTPUT_PREFIX,
        "split_draw_noise": SPLIT_DRAW_NOISE,
        "joined": frame.height,
        "evaluable": rows.height,
        "domains": int(rows["domain"].n_unique()),
        # Survival is read over everything joined, not over the evaluable subset: a document one
        # build lost is exactly the row the evaluable filter removes.
        "outcomes": outcome_matrix(frame),
        "timings": timing_shift(rows),
        "metrics": {metric: by_stratum(rows, metric) for metric in METRICS},
        "output_size": paired_delta(rows, "inspector_markdown_chars"),
        "tables_detected": paired_delta(rows, "inspector_extract_pages_with_tables"),
        "columns_detected": paired_delta(rows, "inspector_extract_pages_with_columns"),
    }
    with fs.open(RESULT_PATH, "w") as stream:
        json.dump(results, stream, indent=2, default=float)
    print(report(results))
    logger.info("wrote %s", RESULT_PATH)


if __name__ == "__main__":
    main()
