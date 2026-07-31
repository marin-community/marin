# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared constants, types, and HTTP plumbing for the focus-crawl PDF source build.

The science focus crawl (``CC-SUPPLEMENTAL-2026-22``) holds 3,169,664 PDF responses over
4.92 TB -- 73.5% of the crawl's bytes, against 1.02 TB of HTML. ``experiments/datakit/focus_crawl.py``
(archived at ``20b7003fe``) extracted only the HTML; this package fetches the PDF bytes so the
extraction work in #7618 has real input, and does it on a sample so a first run costs a fraction
of the full crawl.

Two steps, so the expensive fetch can be re-planned without re-reading the index and re-run without
re-planning:

* :mod:`plan` reads the crawl-wide columnar index, coalesces each WARC's PDF byte ranges into
  range GETs, samples those ranges, and packs them into byte-budgeted fetch tasks.
* :mod:`fetch` maps over those tasks and writes the raw PDF bytes with WARC provenance.

Sampling deliberately happens *after* coalescing. Every fetched range is then byte-identical to
one a full-crawl run would issue, and packing to a byte budget makes per-task cost independent of
the sample size -- so a 10% run measures the same per-task behavior the full run will have.
Sampling records first and coalescing after would destroy both: a WARC averages 1.5 GB holding
~686 kept PDF records, so at 10% the mean spacing between kept records is ~22 MB, well past any
sane coalesce gap, and every range would degenerate to a singleton GET.
"""

from dataclasses import dataclass
from functools import cache

import requests
from marin.datakit.download.http_session import build_retrying_session
from pydantic import BaseModel

FOCUS_CRAWL = "CC-SUPPLEMENTAL-2026-22"
COMMON_CRAWL_BASE_URL = "https://data.commoncrawl.org"

# The crawl-wide columnar (cc-index) table: ten Spark part files sharing one job UUID, ~3.0 GB
# total. Sorted by ``url_surtkey``, so every part spans nearly every WARC -- a single part is not
# a usable subset of the crawl, only a subset of its rows.
FOCUS_INDEX_DIR = (
    "https://data.commoncrawl.org/projects/cc-open-athena-test/CC-SUPPLEMENTAL-2026-22"
    "/index/table/cc-supplemental/warc/crawl=CC-SUPPLEMENTAL-2026-22/subset=warc"
)
FOCUS_INDEX_JOB_UUID = "8637f21e-a055-46d1-8233-990f59974248"
FOCUS_INDEX_PART_COUNT = 10
FOCUS_WARC_FILE_COUNT = 4_573

# Tika-detected type (the index's ``content_mime_detected``), not the server-declared
# ``content_mime_type``. The declared-type buckets are unreliable here: 71.4 GB of the crawl
# declares ``unk``/octet-stream and is almost entirely genomics and CERN research binaries, only
# 34 records of which Tika identifies as PDFs.
PDF_MIME_TYPE = "application/pdf"
FETCH_SUCCESS_STATUS = 200

USER_AGENT = "marin-focus-crawl-pdf-ingress/1.0"
REQUEST_TIMEOUT = (30, 300)
DOWNLOAD_CHUNK_BYTES = 1 << 20
_RETRY_STATUS = (403, 429, 500, 502, 503, 504)
_RETRY_TOTAL = 10
_RETRY_BACKOFF_FACTOR = 2.0


@cache
def session() -> requests.Session:
    """A process-wide retrying session for ``data.commoncrawl.org``.

    Common Crawl answers sustained parallel range GETs with 403s and 429s as often as with 5xx, so
    both are retried rather than surfaced.
    """
    return build_retrying_session(
        total=_RETRY_TOTAL,
        backoff_factor=_RETRY_BACKOFF_FACTOR,
        status_forcelist=_RETRY_STATUS,
    )


@dataclass(frozen=True)
class RangeFetch:
    """One coalesced byte range within one WARC, and the PDF records it was built from.

    ``record_offsets`` are absolute offsets into the WARC, so they are the selection key on the
    fetch side: a coalesced range also spans the gap records between the PDFs it covers, including
    PDFs the plan deliberately excluded (truncated, or non-200).
    """

    warc_filename: str
    start: int
    stop: int
    record_offsets: tuple[int, ...]

    @property
    def size(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class FetchTask:
    """A byte-budgeted group of coalesced ranges -- one Zephyr shard, one output Parquet file."""

    task_id: int
    ranges: tuple[RangeFetch, ...]

    @property
    def size(self) -> int:
        return sum(selected.size for selected in self.ranges)


class PdfFetchPlan(BaseModel):
    """Outcome of the plan step: a Parquet fetch plan plus the totals it commits to.

    ``fetch_bytes`` is what the fetch step will pull from ``data.commoncrawl.org``; it exceeds the
    PDF payload total because a coalesced range also transfers the gap records inside it.
    """

    version: str = "v1"
    plan_path: str
    num_warcs: int
    num_ranges: int
    num_pdfs: int
    fetch_bytes: int
    num_tasks: int


class PdfSourceData(BaseModel):
    """Outcome of the fetch step: raw PDF bytes with WARC provenance, as Parquet."""

    version: str = "v1"
    main_output_dir: str
    counters: dict[str, int | float]


class OcrModelData(BaseModel):
    """Outcome of the model step: the staged FinePDFs OCR router, pinned by content hash."""

    version: str = "v1"
    model_path: str
    revision: str
    sha256: str


class PdfClassificationData(BaseModel):
    """Outcome of the classify step: one row per PDF, routing it to OCR or to text extraction."""

    version: str = "v1"
    main_output_dir: str
    counters: dict[str, int | float]


class LayoutModelData(BaseModel):
    """Outcome of the layout model step: an INT8 OpenVINO build of docling's layout model.

    ``label_map`` travels with the graph because the compiled IR carries class indices and no
    names; reading it from the source repository at inference time would let the two drift.
    """

    version: str = "v1"
    model_path: str
    source_repo: str
    source_revision: str
    label_map: dict[int, str]
    calibration_images: int
    fp32_megabytes: float
    int8_megabytes: float
