# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Downloader for the UWF Zeek sample used by issue #5057.

Issue: https://github.com/marin-community/marin/issues/5057 (parent #5005).

This module materializes a *small* text-PPL slice from the listable UWF Zeek CSV export at
``https://datasets.uwf.edu/data/UWF-ZeekDataSum25-2/csv/``. We intentionally fetch only one CSV
per chosen category and cap rows per category so the eval bundle stays cheap and bounded.

Each emitted record preserves Zeek field names plus CSV delimiter structure by rendering:

    header1,header2,...
    value1,value2,...

as the ``text`` field inside a gzipped JSONL document::

    {"id": str, "text": str, "source": str, "category": str, "row_index": int}
"""

from __future__ import annotations

import csv
import json
import logging
import posixpath
from dataclasses import dataclass
from html.parser import HTMLParser
from io import StringIO
from typing import Any
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from rigging.filesystem import open_url
from urllib3.util import Retry
from zephyr.writers import atomic_rename

from marin.execution.executor import THIS_OUTPUT_PATH
from marin.execution.step_spec import StepSpec
from marin.utils import fsspec_mkdirs

logger = logging.getLogger(__name__)

DEFAULT_UWF_ZEEK_BASE_URL = "https://datasets.uwf.edu/data/UWF-ZeekDataSum25-2/csv/"
DEFAULT_UWF_ZEEK_CATEGORIES: tuple[str, ...] = ("Benign", "Discovery", "Reconnaissance")
DEFAULT_HTTP_TIMEOUT_SECONDS = 120
DEFAULT_MAX_ROWS_PER_CATEGORY = 128
DEFAULT_OUTPUT_FILENAME = "data.jsonl.gz"


@dataclass(frozen=True)
class UwfZeekSampleSource:
    """Describes the small UWF Zeek slice we materialize for evals."""

    slice_key: str
    base_url: str = DEFAULT_UWF_ZEEK_BASE_URL
    categories: tuple[str, ...] = DEFAULT_UWF_ZEEK_CATEGORIES
    max_rows_per_category: int = DEFAULT_MAX_ROWS_PER_CATEGORY
    source_label: str = ""

    def resolved_source_label(self) -> str:
        return self.source_label or self.slice_key

    def validate(self) -> None:
        if not self.base_url.endswith("/"):
            raise ValueError("base_url must end with '/'")
        if not self.categories:
            raise ValueError("categories must not be empty")
        if self.max_rows_per_category <= 0:
            raise ValueError("max_rows_per_category must be positive")


@dataclass
class DownloadUwfZeekSampleConfig:
    """Runtime config for :func:`download_uwf_zeek_sample`."""

    source: UwfZeekSampleSource
    output_path: str = THIS_OUTPUT_PATH
    output_filename: str = DEFAULT_OUTPUT_FILENAME
    http_timeout_seconds: int = DEFAULT_HTTP_TIMEOUT_SECONDS


class _HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)


def _build_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _list_hrefs(session: requests.Session, url: str, *, timeout_seconds: int) -> list[str]:
    response = session.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    parser = _HrefParser()
    parser.feed(response.text)
    return parser.hrefs


def _category_csv_url(
    session: requests.Session, source: UwfZeekSampleSource, *, category: str, timeout_seconds: int
) -> str:
    category_url = urljoin(source.base_url, f"{category}/")
    hrefs = _list_hrefs(session, category_url, timeout_seconds=timeout_seconds)
    candidates = sorted(
        {urljoin(category_url, href) for href in hrefs if href.lower().endswith(".csv") or ".csv?" in href.lower()}
    )
    if not candidates:
        raise ValueError(f"no CSV files found for category {category!r} at {category_url}")
    return candidates[0]


def _render_csv_text(fieldnames: list[str], row: dict[str, str]) -> str:
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerow({field: row.get(field, "") for field in fieldnames})
    return buf.getvalue().rstrip("\n")


def _iter_csv_records(
    session: requests.Session,
    *,
    csv_url: str,
    source: UwfZeekSampleSource,
    category: str,
    timeout_seconds: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    response = session.get(csv_url, timeout=timeout_seconds, stream=True)
    response.raise_for_status()
    rows: list[dict[str, Any]] = []
    try:
        reader = csv.DictReader(response.iter_lines(decode_unicode=True))
        if reader.fieldnames is None:
            raise ValueError(f"CSV at {csv_url} has no header row")
        fieldnames = list(reader.fieldnames)
        for row_index, row in enumerate(reader):
            rows.append(
                {
                    "id": f"{source.slice_key}#{category.lower()}:{row_index}",
                    "text": _render_csv_text(fieldnames, row),
                    "source": source.resolved_source_label(),
                    "category": category,
                    "row_index": row_index,
                    "csv_url": csv_url,
                }
            )
            if len(rows) >= source.max_rows_per_category:
                break
    finally:
        response.close()
    manifest = {
        "category": category,
        "csv_url": csv_url,
        "fieldnames": fieldnames,
        "rows_written": len(rows),
    }
    return rows, manifest


def download_uwf_zeek_sample(config: DownloadUwfZeekSampleConfig) -> dict[str, Any]:
    """Download the bounded UWF Zeek CSV sample and write a gzipped JSONL output."""

    source = config.source
    source.validate()
    output_path = str(config.output_path)
    fsspec_mkdirs(output_path, exist_ok=True)

    session = _build_session()
    output_file = posixpath.join(output_path, config.output_filename)
    manifest_entries: list[dict[str, Any]] = []
    total_rows = 0
    try:
        with atomic_rename(output_file) as temp_path:
            with open_url(temp_path, "wt", encoding="utf-8", compression="gzip") as handle:
                for category in source.categories:
                    csv_url = _category_csv_url(
                        session, source, category=category, timeout_seconds=config.http_timeout_seconds
                    )
                    records, manifest = _iter_csv_records(
                        session,
                        csv_url=csv_url,
                        source=source,
                        category=category,
                        timeout_seconds=config.http_timeout_seconds,
                    )
                    manifest_entries.append(manifest)
                    total_rows += len(records)
                    logger.info(
                        "Materialized %d UWF Zeek rows for %s from %s",
                        len(records),
                        category,
                        csv_url,
                    )
                    for record in records:
                        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                        handle.write("\n")
    finally:
        session.close()

    manifest = {
        "slice_key": source.slice_key,
        "base_url": source.base_url,
        "categories": list(source.categories),
        "max_rows_per_category": source.max_rows_per_category,
        "total_rows": total_rows,
        "files": manifest_entries,
    }
    manifest_path = posixpath.join(output_path, "manifest.json")
    with atomic_rename(manifest_path) as temp_path:
        with open_url(temp_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def uwf_zeek_sample_step(
    source: UwfZeekSampleSource,
    *,
    name: str | None = None,
    http_timeout_seconds: int = DEFAULT_HTTP_TIMEOUT_SECONDS,
) -> StepSpec:
    """Create the StepSpec that materializes the bounded UWF Zeek eval sample."""

    source.validate()
    step_name = name or f"raw/{source.slice_key}"
    return StepSpec(
        name=step_name,
        fn=lambda output_path: download_uwf_zeek_sample(
            DownloadUwfZeekSampleConfig(
                source=source,
                output_path=output_path,
                http_timeout_seconds=http_timeout_seconds,
            )
        ),
        hash_attrs={
            "slice_key": source.slice_key,
            "base_url": source.base_url,
            "categories": list(source.categories),
            "max_rows_per_category": source.max_rows_per_category,
            "http_timeout_seconds": http_timeout_seconds,
        },
    )
