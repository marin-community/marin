# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY -- page-count and bytes-per-page statistics from the OCR routing table.

DELETE once the numbers are recorded. Nothing in the pipeline imports this.

Reads the classify step's routing table (~30 MB, no PDF bytes) and reports, for all classified PDFs
and for the OCR-routed subset separately: the page-count distribution, total pages, and mean MB per
page. It then extrapolates each total to the whole focus crawl.

The extrapolation factor is ``1 / SAMPLE_FRACTION``, and that is a design-based estimator rather than
a guess: :func:`~experiments.build_pdf_source.plan.sample_ranges` selects coalesced ranges uniformly,
and every PDF belongs to exactly one range, so every PDF in the crawl had the same inclusion
probability regardless of how many PDFs shared its range. Scaling any sample total by the reciprocal
of that probability is therefore unbiased. Range clustering inflates the variance, not the mean --
:data:`CRAWL_PDF_RECORDS` gives an independent check on how far off the realised draw was.

Run on the cluster, where the routing table lives::

    uv run iris --cluster=marin job run --target-cluster cw-us-east-08a \\
        --job-name analyze-page-counts \\
        -- python -m experiments.build_pdf_source._analyze_page_counts
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from pydantic import BaseModel
from rigging.filesystem import url_to_fs
from rigging.log_setup import configure_logging

from experiments.build_pdf_source.classify import classify_step, model_step
from experiments.build_pdf_source.common import PdfClassificationData
from experiments.build_pdf_source.fetch import fetch_step
from experiments.build_pdf_source.plan import SAMPLE_FRACTION, plan_step

logger = logging.getLogger(__name__)

# Untruncated 200-response PDF records the whole cc-index holds for this crawl, measured by scanning
# all ten index parts. Used only to sanity-check the extrapolation against a known total.
CRAWL_PDF_RECORDS = 3_136_363

_COLUMNS = ["needs_ocr", "num_pages", "pdf_bytes", "classification_error"]
_PAGE_BUCKETS = (1, 2, 3, 6, 11, 26, 51, 101, 501)
_PERCENTILES = (10, 25, 50, 75, 90, 99)
_READ_THREADS = 32
_MEGABYTE = 1 << 20

_RESOURCES = ResourceConfig(cpu=4, ram="16g", disk="8g")


class CohortStats(BaseModel):
    """Page and size statistics for one set of PDFs."""

    label: str
    documents: int
    total_pages: int
    mean_pages: float
    percentile_pages: dict[str, float]
    max_pages: int
    total_megabytes: float
    mean_megabytes_per_page: float
    mean_megabytes_per_document: float
    page_histogram: dict[str, int]
    estimated_crawl_pages: int
    estimated_crawl_megabytes: float


class PageCountReport(BaseModel):
    version: str = "v1"
    sample_fraction: float
    scale_factor: float
    documents_read: int
    documents_without_page_count: int
    cohorts: list[CohortStats]
    estimated_crawl_pdf_records: int
    known_crawl_pdf_records: int


def _read_shard(path: str, filesystem) -> pa.Table:
    with filesystem.open(path, "rb") as stream:
        return pq.read_table(stream, columns=_COLUMNS)


def _histogram(pages: np.ndarray) -> dict[str, int]:
    """Bucket page counts into human-readable ranges."""
    edges = list(_PAGE_BUCKETS)
    labels = []
    for index, low in enumerate(edges):
        high = edges[index + 1] - 1 if index + 1 < len(edges) else None
        labels.append(f"{low}" if high == low else (f"{low}-{high}" if high else f"{low}+"))
    counts = np.digitize(pages, edges, right=False) - 1
    return {label: int((counts == index).sum()) for index, label in enumerate(labels)}


def _cohort_stats(label: str, pages: np.ndarray, byte_sizes: np.ndarray, scale: float) -> CohortStats:
    total_pages = int(pages.sum())
    total_bytes = int(byte_sizes.sum())
    return CohortStats(
        label=label,
        documents=int(pages.size),
        total_pages=total_pages,
        mean_pages=float(pages.mean()),
        percentile_pages={f"p{percentile}": float(np.percentile(pages, percentile)) for percentile in _PERCENTILES},
        max_pages=int(pages.max()),
        total_megabytes=total_bytes / _MEGABYTE,
        # Aggregate ratio, not the mean of per-document ratios: the question is how many bytes a page
        # costs on average across the corpus, which small documents must not dominate.
        mean_megabytes_per_page=(total_bytes / _MEGABYTE / total_pages) if total_pages else 0.0,
        mean_megabytes_per_document=total_bytes / _MEGABYTE / pages.size,
        page_histogram=_histogram(pages),
        estimated_crawl_pages=round(total_pages * scale),
        estimated_crawl_megabytes=total_bytes / _MEGABYTE * scale,
    )


def analyze(output_path: str, classification_output_path: str) -> PageCountReport:
    """Compute page-count and MB/page statistics and extrapolate to the whole crawl."""
    classification = read_artifact(classification_output_path, PdfClassificationData)
    filesystem, path = url_to_fs(classification.main_output_dir)
    shards = sorted(filesystem.glob(f"{path}/*.parquet"))
    if not shards:
        raise RuntimeError(f"No routing table under {classification.main_output_dir}")
    logger.info("Reading %d routing-table shards", len(shards))

    with ThreadPoolExecutor(max_workers=_READ_THREADS) as pool:
        table = pa.concat_tables(pool.map(partial(_read_shard, filesystem=filesystem), shards))
    logger.info("Routing table holds %d rows", table.num_rows)

    # Drop unreadable documents in Arrow, before any NumPy conversion. Their num_pages is null, and
    # a null in an integer column converts to a float NaN, which `astype(int64)` silently turns into
    # INT64_MIN -- poisoning every sum rather than raising.
    readable_table = table.filter(pc.is_valid(table.column("num_pages")))
    unreadable = table.num_rows - readable_table.num_rows

    pages = readable_table.column("num_pages").to_numpy(zero_copy_only=False).astype(np.int64)
    sizes = readable_table.column("pdf_bytes").to_numpy(zero_copy_only=False).astype(np.int64)
    ocr = pc.fill_null(readable_table.column("needs_ocr"), False).to_numpy(zero_copy_only=False).astype(bool)

    if pages.min() < 0:
        raise ValueError(f"Negative page count after filtering nulls: min={pages.min()}")

    scale = 1.0 / SAMPLE_FRACTION
    cohorts = [
        _cohort_stats("all_classified", pages, sizes, scale),
        _cohort_stats("needs_ocr", pages[ocr], sizes[ocr], scale),
        _cohort_stats("text_extractable", pages[~ocr], sizes[~ocr], scale),
    ]

    report = PageCountReport(
        sample_fraction=SAMPLE_FRACTION,
        scale_factor=scale,
        documents_read=int(table.num_rows),
        documents_without_page_count=unreadable,
        cohorts=cohorts,
        estimated_crawl_pdf_records=round(pages.size * scale),
        known_crawl_pdf_records=CRAWL_PDF_RECORDS,
    )

    logger.info("=== PDF PAGE COUNT REPORT ===")
    logger.info("  sample_fraction=%.3f scale=%.2fx", report.sample_fraction, report.scale_factor)
    logger.info(
        "  rows=%d without_page_count=%d estimated_crawl_records=%d (index holds %d)",
        report.documents_read,
        report.documents_without_page_count,
        report.estimated_crawl_pdf_records,
        report.known_crawl_pdf_records,
    )
    for cohort in cohorts:
        logger.info("  --- %s ---", cohort.label)
        logger.info(
            "    documents=%d total_pages=%d mean_pages=%.2f max_pages=%d",
            cohort.documents,
            cohort.total_pages,
            cohort.mean_pages,
            cohort.max_pages,
        )
        logger.info("    percentiles=%s", {k: round(v, 1) for k, v in cohort.percentile_pages.items()})
        logger.info(
            "    total_MB=%.1f MB_per_page=%.4f MB_per_document=%.4f",
            cohort.total_megabytes,
            cohort.mean_megabytes_per_page,
            cohort.mean_megabytes_per_document,
        )
        logger.info("    page_histogram=%s", cohort.page_histogram)
        logger.info(
            "    EXTRAPOLATED crawl_pages=%d crawl_TiB=%.3f",
            cohort.estimated_crawl_pages,
            cohort.estimated_crawl_megabytes / (1 << 20),
        )
    return report


def analysis_step(classification_output_path: str) -> StepSpec:
    """Build the analysis step. Classify is named by path, not depended on, so it is never re-run."""
    return StepSpec(
        name="data/datakit/analyze/focus_crawl_pdf_page_counts",
        hash_attrs={"classification_output_path": classification_output_path, "attempt": 2},
        fn=remote(
            partial(analyze, classification_output_path=classification_output_path),
            resources=_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )


def main() -> None:
    configure_logging(logging.INFO)
    plan = plan_step()
    classify = classify_step(fetch_step(plan), model_step())
    logger.info("Reading routing table from %s", classify.output_path)
    StepRunner().run([analysis_step(classify.output_path)])


if __name__ == "__main__":
    main()
