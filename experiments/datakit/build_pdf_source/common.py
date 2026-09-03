# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared constants, types, and HTTP plumbing for the focus-crawl PDF source build.

Fetching is two steps, so the expensive fetch can be re-planned without re-reading the index and
re-run without re-planning:

* :mod:`plan` reads the crawl-wide columnar index, coalesces each WARC's PDF byte ranges into
  range GETs, samples those ranges, and packs them into byte-budgeted fetch tasks.
* :mod:`fetch` maps over those tasks and writes the raw PDF bytes with WARC provenance.

Sampling happens after coalescing, so every fetched range is byte-identical to one a full-crawl run
would issue.
"""

from dataclasses import dataclass
from functools import cache

import requests
from marin.datakit.download.http_session import build_retrying_session
from pydantic import BaseModel

FOCUS_CRAWL = "CC-SUPPLEMENTAL-2026-22"
COMMON_CRAWL_BASE_URL = "https://data.commoncrawl.org"

# One Parquet shard per input shard, numbered by the input's position in the sorted listing the
# step mapped over. The fetch, the pdf-inspector extraction and the routing table all write this
# pattern 1:1 over the previous step's listing, so a shard's basename is the join between them.
SHARD_PATTERN = "part-{shard:05d}-of-{total:05d}.parquet"
# The column the reader injects so a batch knows which shard it came from. Never written to an output.
SOURCE_FILE_COLUMN = "_source_file"

# The crawl-wide columnar (cc-index) table: ten Spark part files sharing one job UUID, sorted by
# ``url_surtkey`` so every part spans nearly every WARC.
FOCUS_INDEX_DIR = (
    "https://data.commoncrawl.org/projects/cc-open-athena-test/CC-SUPPLEMENTAL-2026-22"
    "/index/table/cc-supplemental/warc/crawl=CC-SUPPLEMENTAL-2026-22/subset=warc"
)
FOCUS_INDEX_JOB_UUID = "8637f21e-a055-46d1-8233-990f59974248"
FOCUS_INDEX_PART_COUNT = 10
FOCUS_WARC_FILE_COUNT = 4_573

# Matched against the Tika-detected type (the index's ``content_mime_detected``), not the
# server-declared ``content_mime_type``.
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
    """A process-wide retrying session for ``data.commoncrawl.org``; 403 and 429 are retried like 5xx."""
    return build_retrying_session(
        total=_RETRY_TOTAL,
        backoff_factor=_RETRY_BACKOFF_FACTOR,
        status_forcelist=_RETRY_STATUS,
    )


@dataclass(frozen=True)
class RangeFetch:
    """One coalesced byte range within one WARC, and the PDF records it was built from.

    ``record_offsets`` are absolute offsets into the WARC and select which records the fetch keeps:
    a coalesced range also spans the gap records between the PDFs it covers.
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

    ``fetch_bytes`` includes the gap records inside each coalesced range, so it exceeds the PDF
    payload total.
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


class StagedModelData(BaseModel):
    """Outcome of a model-staging step: one model artifact copied into the marin prefix.

    ``model_path`` is a single file or a directory, whichever the consumer loads.
    """

    version: str = "v1"
    model_path: str
    revision: str
    sha256: str


class PdfDocumentsData(BaseModel):
    """Outcome of a step that writes document records: one Parquet shard per input shard.

    Each shard carries the shared record in :mod:`document_record` plus the producer's own
    columns, unsorted and undeduplicated; the normalize step downstream sorts and dedups.
    """

    version: str = "v1"
    main_output_dir: str
    counters: dict[str, int | float]


class PdfClassificationData(BaseModel):
    """Outcome of the routing step: one routing shard per extraction shard, named after it.

    Each row says whether the VLM re-reads the document and at what render budget; a consumer reads
    the decisions for the shard it is processing by name (:func:`classify.shard_routing`).
    """

    version: str = "v1"
    main_output_dir: str
    counters: dict[str, int | float]
