# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract the Common Crawl science focus crawl to normalized Datakit Parquet.

The crawl is read from WARC response records so Michael Ryan's jusText fork can
remove boilerplate from the original HTML. Invoke through the canonical module
path so remote callables remain importable::

    python -c 'from experiments.datakit.focus_crawl import main; main()'
"""

import codecs
import logging
import re
from collections.abc import Iterator
from email.message import Message
from functools import cache, partial
from typing import BinaryIO

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from fray.types import ResourceConfig
from marin.datakit.download.http_session import build_retrying_session
from marin.datakit.normalize import DEFAULT_MAX_WHITESPACE_RUN_CHARS, NormalizedData, generate_id
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from resiliparse.parse.encoding import bytes_to_str, detect_encoding, map_encoding_to_html5
from rigging.filesystem import prefix_join
from rigging.log_setup import configure_logging
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.runners import InlineRunner

logger = logging.getLogger(__name__)

FOCUS_CRAWL = "CC-SUPPLEMENTAL-2026-22"
FOCUS_INDEX_PARQUET = (
    "https://data.commoncrawl.org/projects/cc-open-athena-test/CC-SUPPLEMENTAL-2026-22"
    "/index/table/cc-supplemental/warc/crawl=CC-SUPPLEMENTAL-2026-22/subset=warc"
    "/part-00000-8637f21e-a055-46d1-8233-990f59974248.c000.gz.parquet"
)
FOCUS_WARC_FILE_COUNT = 4_573
COMMON_CRAWL_BASE_URL = "https://data.commoncrawl.org"
JUSTEXT_REPOSITORY = "https://github.com/XenonMolecule/jusText"
JUSTEXT_COMMIT = "1652a1497b36c4b9941c609ffa1714eeefedc70b"
JUSTEXT_MODEL = "sklearn"
JUSTEXT_STOPLIST = "English"

_USER_AGENT = "marin-focus-crawl-ingress/1.0"
_RETRY_STATUS = (429, 500, 502, 503, 504)
_REQUEST_TIMEOUT = (30, 300)
_HTML_MIME_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_LONG_WHITESPACE = re.compile(r"\s{" + str(DEFAULT_MAX_WHITESPACE_RUN_CHARS + 1) + r",}")
_OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("warc_filename", pa.string(), nullable=False),
        pa.field("warc_date", pa.string(), nullable=False),
        pa.field("content_type", pa.string(), nullable=False),
        pa.field("content_encoding", pa.string(), nullable=False),
    ]
)
_BOM_ENCODINGS = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)
_DRIVER_RESOURCES = ResourceConfig(cpu=1, ram="4g")
_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="8g", disk="10g")
_MAX_WORKERS = 256


@cache
def _session() -> requests.Session:
    return build_retrying_session(status_forcelist=_RETRY_STATUS)


def focus_crawl_warc_paths(index_parquet: str, expected_files: int) -> list[str]:
    """Return the focus crawl's WARC file keys from its columnar URL index."""
    warc_files: set[str] = set()
    with fsspec.open(index_parquet, "rb") as stream:
        parquet = pq.ParquetFile(stream)
        for batch in parquet.iter_batches(columns=["warc_filename"]):
            warc_files.update(path for path in batch.column(0).to_pylist() if path)

    paths = sorted(warc_files)
    if len(paths) != expected_files:
        raise ValueError(f"Expected {expected_files} WARC files for {FOCUS_CRAWL}, found {len(paths)}")
    return paths


def _header_encoding(content_type: str) -> str | None:
    message = Message()
    message["content-type"] = content_type
    charset = message.get_param("charset")
    if not isinstance(charset, str):
        return None
    return map_encoding_to_html5(charset, fallback_utf8=False)


def _html_text(payload: bytes, content_type: str) -> tuple[str, str]:
    encoding = next((name for bom, name in _BOM_ENCODINGS if payload.startswith(bom)), None)
    if encoding is None:
        encoding = _header_encoding(content_type)
    if encoding is None:
        detected = detect_encoding(payload, html5_compatible=False, from_html_meta=True)
        encoding = map_encoding_to_html5(detected, fallback_utf8=False) if detected else None
    if encoding is None:
        encoding = "cp1252"
    return bytes_to_str(payload, encoding=encoding, errors="replace", fallback_encodings=("utf-8", "cp1252")), encoding


@cache
def _stoplist() -> frozenset[str]:
    import justext  # noqa: PLC0415

    return justext.get_stoplist(JUSTEXT_STOPLIST)


@cache
def _model() -> object:
    import justext  # noqa: PLC0415

    model = justext.get_model()
    if model is None:
        raise RuntimeError("The pinned jusText sklearn model could not be loaded")
    return model


def _clean_text(html: str) -> str:
    import justext  # noqa: PLC0415

    paragraphs = justext.justext(html, _stoplist(), model=_model())
    text = "\n\n".join(paragraph.text for paragraph in paragraphs if not paragraph.is_boilerplate)
    compacted = _LONG_WHITESPACE.sub(lambda match: match.group(0)[:DEFAULT_MAX_WHITESPACE_RUN_CHARS], text)
    if len(compacted) != len(text):
        counters.pipeline.update_counter("focus_crawl/compacted_whitespace", 1)
    return compacted.strip()


def iter_warc_text_records(stream: BinaryIO, warc_filename: str) -> Iterator[dict[str, str]]:
    """Yield normalized text records from HTML responses in one WARC stream."""
    from warcio.archiveiterator import ArchiveIterator  # noqa: PLC0415

    for record in ArchiveIterator(stream):
        if record.rec_type != "response" or record.http_headers is None:
            continue

        status = record.http_headers.get_statuscode()
        if status is None or not status.startswith("2"):
            counters.pipeline.update_counter("focus_crawl/non_success_response", 1)
            continue

        http_content_type = record.http_headers.get_header("Content-Type") or ""
        identified_type = record.rec_headers.get_header("WARC-Identified-Payload-Type") or ""
        mime_types = {
            content_type.partition(";")[0].strip().lower()
            for content_type in (http_content_type, identified_type)
            if content_type
        }
        if not mime_types & _HTML_MIME_TYPES:
            counters.pipeline.update_counter("focus_crawl/non_html_response", 1)
            continue

        payload = record.content_stream().read()
        html, encoding = _html_text(payload, http_content_type)
        source_id = record.rec_headers.get_header("WARC-Record-ID") or ""
        url = record.rec_headers.get_header("WARC-Target-URI") or ""
        if "\ufffd" in html:
            logger.warning("Skipping WARC record with replacement characters: %s %s", source_id, url)
            counters.pipeline.update_counter("focus_crawl/replacement_character", 1)
            continue

        text = _clean_text(html)
        if not text:
            counters.pipeline.update_counter("focus_crawl/empty_extraction", 1)
            continue

        counters.pipeline.update_counter("focus_crawl/documents", 1)
        counters.pipeline.update_counter("focus_crawl/text_bytes", len(text.encode("utf-8")))
        yield {
            "id": generate_id(text),
            "text": text,
            "source_id": source_id,
            "source": FOCUS_CRAWL,
            "url": url,
            "warc_filename": warc_filename,
            "warc_date": record.rec_headers.get_header("WARC-Date") or "",
            "content_type": http_content_type or identified_type,
            "content_encoding": encoding,
        }


def download_warc_text_records(warc_filename: str, base_url: str) -> Iterator[dict[str, str]]:
    """Stream one Common Crawl WARC file and yield its extracted text records."""
    url = f"{base_url.rstrip('/')}/{warc_filename.lstrip('/')}"
    logger.info("Extracting %s", url)
    with _session().get(url, headers={"user-agent": _USER_AGENT}, stream=True, timeout=_REQUEST_TIMEOUT) as response:
        response.raise_for_status()
        yield from iter_warc_text_records(response.raw, warc_filename)


def extract_warc_paths(
    output_path: str,
    warc_paths: list[str],
    *,
    base_url: str,
    worker_resources: ResourceConfig,
    max_workers: int,
) -> NormalizedData:
    """Extract WARC paths into the normalized Parquet layout used by Datakit."""
    if not warc_paths:
        raise ValueError("No WARC paths to extract")

    output_dir = prefix_join(output_path, "outputs/main")
    output_pattern = prefix_join(output_dir, "part-{shard:05d}-of-{total:05d}.parquet")
    pipeline = (
        Dataset.from_list(warc_paths)
        .flat_map(partial(download_warc_text_records, base_url=base_url))
        .write_parquet(output_pattern, schema=_OUTPUT_SCHEMA, skip_existing=True)
    )
    outcome = ZephyrContext(
        name="focus-crawl-justext",
        resources=worker_resources,
        max_workers=min(max_workers, len(warc_paths)),
        stage_runner_factory=InlineRunner,
    ).execute(pipeline)
    return NormalizedData(main_output_dir=output_dir, dup_output_dir="", counters=dict(outcome.counters))


def extract_focus_crawl(output_path: str) -> NormalizedData:
    paths = focus_crawl_warc_paths(FOCUS_INDEX_PARQUET, FOCUS_WARC_FILE_COUNT)
    return extract_warc_paths(
        output_path,
        paths,
        base_url=COMMON_CRAWL_BASE_URL,
        worker_resources=_WORKER_RESOURCES,
        max_workers=_MAX_WORKERS,
    )


def focus_crawl_step() -> StepSpec:
    """Build the canonical focus-crawl WARC extraction step."""
    return StepSpec(
        name="data/datakit/normalized/common_crawl_focus_2026_22",
        hash_attrs={
            "crawl": FOCUS_CRAWL,
            "index_parquet": FOCUS_INDEX_PARQUET,
            "warc_file_count": FOCUS_WARC_FILE_COUNT,
            "justext_repository": JUSTEXT_REPOSITORY,
            "justext_commit": JUSTEXT_COMMIT,
            "justext_model": JUSTEXT_MODEL,
            "justext_stoplist": JUSTEXT_STOPLIST,
            "schema_version": 1,
        },
        fn=remote(
            extract_focus_crawl,
            resources=_DRIVER_RESOURCES,
            env_vars={"JUSTEXT_MODEL": JUSTEXT_MODEL, "JUSTEXT_NO_DOWNLOAD": "1"},
            pip_dependency_groups=["datakit"],
        ),
    )


def main() -> None:
    configure_logging(logging.INFO)
    StepRunner().run([focus_crawl_step()])


if __name__ == "__main__":
    main()
