# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract the Common Crawl science focus crawl to normalized Datakit Parquet.

The crawl is read from WARC response records so Michael Ryan's jusText fork can
remove boilerplate from the original HTML. Invoke through the canonical module
path so remote callables remain importable::

    python -c 'from experiments.datakit.focus_crawl import main; main()'
"""

import codecs
import gzip
import http.client
import json
import logging
import re
import tempfile
import unicodedata
import warnings
from collections.abc import Iterator, Mapping
from email.message import Message
from functools import cache, partial
from typing import BinaryIO, NamedTuple

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
from zephyr.runners import SubprocessRunner

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
JUSTEXT_COMMIT = "20d27c00ebfbe927f86281933da687d3e636cba3"
JUSTEXT_MODEL = "sklearn"
JUSTEXT_STOPLIST = "English"

_USER_AGENT = "marin-focus-crawl-ingress/1.0"
_RETRY_STATUS = (403, 429, 500, 502, 503, 504)
_RETRY_TOTAL = 10
_RETRY_BACKOFF_FACTOR = 2.0
_RETRY_BACKOFF_JITTER = 10.0
_REQUEST_TIMEOUT = (30, 300)
_DOWNLOAD_CHUNK_BYTES = 1 << 20
_MAX_DOWNLOAD_STALLS = 8
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")
_RANGE_COALESCE_GAP_BYTES = 1 << 20
_HTML_MIME_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_SKIPPED_RECORD_IDS = frozenset({"<urn:uuid:88562b9a-a0da-40be-a939-330f53017c9d>"})
_LONG_WHITESPACE = re.compile(r"\s{" + str(DEFAULT_MAX_WHITESPACE_RUN_CHARS + 1) + r",}")
_INVALID_XML_CHARACTERS = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")
_NUMERIC_CHARACTER_REFERENCE = re.compile(r"&#(x[0-9a-f]+|[0-9]+);?", re.IGNORECASE)
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
_WORKER_POOL = "cpu-genoa"
_DRIVER_RESOURCES = ResourceConfig(cpu=1, ram="4g", pool=_WORKER_POOL)
_WORKER_RESOURCES = ResourceConfig(cpu=8, ram="64g", disk="10g", pool=_WORKER_POOL)
_MAP_TASK_RESOURCES = ResourceConfig(cpu=1, ram="8g", disk="1g")
_MAX_WORKERS = 66
_HEARTBEAT_TIMEOUT = 15 * 60


class _WarcRange(NamedTuple):
    start: int
    stop: int


@cache
def _session() -> requests.Session:
    return build_retrying_session(
        total=_RETRY_TOTAL,
        backoff_factor=_RETRY_BACKOFF_FACTOR,
        backoff_jitter=_RETRY_BACKOFF_JITTER,
        status_forcelist=_RETRY_STATUS,
    )


def focus_crawl_warc_paths(index_parquet: str, expected_files: int) -> list[str]:
    """Return the focus crawl's WARC file keys from its columnar URL index."""
    warc_files: set[str] = set()
    with _session().get(index_parquet, stream=True, timeout=_REQUEST_TIMEOUT) as response:
        response.raise_for_status()
        with tempfile.TemporaryFile(prefix="focus-crawl-index-", suffix=".parquet") as stream:
            for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                stream.write(chunk)
            stream.seek(0)
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

    warnings.filterwarnings(
        "ignore",
        message=r"`sklearn\.utils\.parallel\.delayed` should be used.*",
        category=UserWarning,
        module=r"sklearn\.utils\.parallel",
    )
    model = justext.get_model()
    if model is None:
        raise RuntimeError("The pinned jusText sklearn model could not be loaded")
    model.model.set_params(n_jobs=1)
    return model


def _clean_text(html: str) -> str:
    import justext  # noqa: PLC0415

    try:
        paragraphs = justext.justext(html, _stoplist(), model=_model())
    except ValueError as error:
        if not str(error).startswith("invalid literal for int() with base 10"):
            raise
        normalized_html, replacements = _normalize_compatibility_digits(html)
        if not replacements:
            raise
        counters.pipeline.update_counter("focus_crawl/normalized_compatibility_digit", replacements)
        paragraphs = justext.justext(normalized_html, _stoplist(), model=_model())
    text = "\n\n".join(paragraph.text for paragraph in paragraphs if not paragraph.is_boilerplate)
    compacted = _LONG_WHITESPACE.sub(lambda match: match.group(0)[:DEFAULT_MAX_WHITESPACE_RUN_CHARS], text)
    if len(compacted) != len(text):
        counters.pipeline.update_counter("focus_crawl/compacted_whitespace", 1)
    return compacted.strip()


def _normalize_compatibility_digits(text: str) -> tuple[str, int]:
    replacements = 0
    normalized: list[str] = []
    for character in text:
        if character.isdigit() and not character.isdecimal():
            normalized.append(str(unicodedata.digit(character)))
            replacements += 1
        else:
            normalized.append(character)
    return "".join(normalized), replacements


def _sanitize_html(html: str) -> tuple[str, int, int]:
    sanitized, xml_replacements = _INVALID_XML_CHARACTERS.subn("", html)
    digit_replacements = 0

    def strip_invalid_reference(match: re.Match[str]) -> str:
        nonlocal digit_replacements, xml_replacements
        value = match.group(1)
        codepoint = int(value[1:], 16) if value.lower().startswith("x") else int(value)
        if codepoint > 0x10FFFF:
            return match.group(0)
        character = chr(codepoint)
        if _INVALID_XML_CHARACTERS.fullmatch(character):
            xml_replacements += 1
            return ""
        if character.isdigit() and not character.isdecimal():
            digit_replacements += 1
            return str(unicodedata.digit(character))
        return match.group(0)

    return _NUMERIC_CHARACTER_REFERENCE.sub(strip_invalid_reference, sanitized), xml_replacements, digit_replacements


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

        source_id = record.rec_headers.get_header("WARC-Record-ID") or ""
        url = record.rec_headers.get_header("WARC-Target-URI") or ""
        if source_id in _SKIPPED_RECORD_IDS:
            logger.warning("Skipping excluded WARC record: %s %s", source_id, url)
            counters.pipeline.update_counter("focus_crawl/skipped_record", 1)
            continue

        payload = record.content_stream().read()
        html, encoding = _html_text(payload, http_content_type)
        if "\ufffd" in html:
            logger.warning("Skipping WARC record with replacement characters: %s %s", source_id, url)
            counters.pipeline.update_counter("focus_crawl/replacement_character", 1)
            continue

        sanitized_html, xml_replacements, digit_replacements = _sanitize_html(html)
        if xml_replacements:
            counters.pipeline.update_counter("focus_crawl/xml_invalid_character", xml_replacements)
        if digit_replacements:
            counters.pipeline.update_counter("focus_crawl/normalized_compatibility_digit", digit_replacements)

        try:
            text = _clean_text(sanitized_html)
        except AssertionError:
            logger.warning("Skipping WARC record after jusText assertion: %s %s", source_id, url)
            counters.pipeline.update_counter("focus_crawl/justext_assertion", 1)
            continue
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


def _cdx_path(warc_filename: str) -> str:
    prefix, separator, filename = warc_filename.rpartition("/warc/")
    if not separator or not filename.endswith(".warc.gz"):
        raise ValueError(f"Unexpected Common Crawl WARC path: {warc_filename}")
    return f"{prefix}/cdx/warc/{filename.removesuffix('.warc.gz')}.cdx.gz"


def _is_html_cdx_record(record: Mapping[str, object]) -> bool:
    status = str(record.get("status", ""))
    mime_types = {str(record.get(field, "")).partition(";")[0].strip().lower() for field in ("mime", "mime-detected")}
    return status.startswith("2") and bool(mime_types & _HTML_MIME_TYPES)


def _coalesced_html_ranges(stream: BinaryIO) -> list[_WarcRange]:
    ranges: list[_WarcRange] = []
    with gzip.open(stream, "rt", encoding="utf-8") as cdx:
        for line in cdx:
            fields = line.split(" ", 2)
            if len(fields) != 3:
                raise ValueError(f"Malformed CDX line: {line[:200]}")
            record = json.loads(fields[2])
            if not _is_html_cdx_record(record):
                continue
            start = int(record["offset"])
            selected = _WarcRange(start, start + int(record["length"]))
            if ranges and selected.start - ranges[-1].stop <= _RANGE_COALESCE_GAP_BYTES:
                ranges[-1] = _WarcRange(ranges[-1].start, max(ranges[-1].stop, selected.stop))
            else:
                ranges.append(selected)
    return ranges


def _warc_html_ranges(warc_filename: str, base_url: str) -> list[_WarcRange]:
    url = f"{base_url.rstrip('/')}/{_cdx_path(warc_filename).lstrip('/')}"
    with _session().get(url, headers={"user-agent": _USER_AGENT}, stream=True, timeout=_REQUEST_TIMEOUT) as response:
        response.raise_for_status()
        return _coalesced_html_ranges(response.raw)


def _download_range(url: str, selected: _WarcRange, destination: BinaryIO) -> None:
    expected_bytes = selected.stop - selected.start
    stalls = 0
    while destination.tell() < expected_bytes:
        written = destination.tell()
        request_start = selected.start + written
        headers = {
            "Range": f"bytes={request_start}-{selected.stop - 1}",
            "user-agent": _USER_AGENT,
        }

        error: Exception | None = None
        try:
            with _session().get(url, headers=headers, stream=True, timeout=_REQUEST_TIMEOUT) as response:
                response.raise_for_status()
                content_range = response.headers.get("Content-Range", "")
                match = _CONTENT_RANGE.fullmatch(content_range)
                if response.status_code != http.client.PARTIAL_CONTENT or match is None:
                    raise RuntimeError(
                        f"WARC range {request_start}-{selected.stop - 1} did not return a valid partial response"
                    )
                response_start, response_stop, _ = (int(value) for value in match.groups())
                if response_start != request_start or response_stop != selected.stop - 1:
                    raise RuntimeError(
                        f"WARC range requested {request_start}-{selected.stop - 1}, "
                        f"received {response_start}-{response_stop}"
                    )

                for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                    if chunk:
                        destination.write(chunk)
        except (requests.exceptions.RequestException, http.client.IncompleteRead) as exc:
            error = exc

        if destination.tell() > written:
            stalls = 0
        else:
            stalls += 1
        if stalls > _MAX_DOWNLOAD_STALLS:
            raise RuntimeError(
                f"WARC range download stalled at {destination.tell()}/{expected_bytes} bytes after {stalls} attempts"
            ) from error
        if destination.tell() > expected_bytes:
            raise RuntimeError(f"WARC range download exceeded expected size {expected_bytes}")
        if error is not None:
            logger.warning(
                "WARC range download interrupted at %d/%d bytes; resuming: %s",
                destination.tell(),
                expected_bytes,
                error,
            )

    destination.seek(0)


def download_warc_text_records(warc_filename: str, base_url: str) -> Iterator[dict[str, str]]:
    """Download one Common Crawl WARC's HTML ranges and yield extracted text records."""
    url = f"{base_url.rstrip('/')}/{warc_filename.lstrip('/')}"
    ranges = _warc_html_ranges(warc_filename, base_url)
    transferred_bytes = sum(selected.stop - selected.start for selected in ranges)
    counters.pipeline.update_counter("focus_crawl/range_requests", len(ranges))
    counters.pipeline.update_counter("focus_crawl/range_bytes", transferred_bytes)
    logger.info("Extracting %s via %d ranges (%d bytes)", url, len(ranges), transferred_bytes)
    for selected in ranges:
        with tempfile.TemporaryFile(prefix="focus-crawl-", suffix=".warc.gz", dir=".") as warc_file:
            _download_range(url, selected, warc_file)
            yield from iter_warc_text_records(warc_file, warc_filename)


def extract_warc_paths(
    output_path: str,
    warc_paths: list[str],
    *,
    base_url: str,
    worker_resources: ResourceConfig,
    task_resources: ResourceConfig,
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
        stage_runner_factory=SubprocessRunner,
        map_task_resources=task_resources,
        heartbeat_timeout=_HEARTBEAT_TIMEOUT,
    ).execute(pipeline)
    return NormalizedData(main_output_dir=output_dir, dup_output_dir="", counters=dict(outcome.counters))


def extract_focus_crawl(output_path: str) -> NormalizedData:
    paths = focus_crawl_warc_paths(FOCUS_INDEX_PARQUET, FOCUS_WARC_FILE_COUNT)
    return extract_warc_paths(
        output_path,
        paths,
        base_url=COMMON_CRAWL_BASE_URL,
        worker_resources=_WORKER_RESOURCES,
        task_resources=_MAP_TASK_RESOURCES,
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
            "skipped_record_ids": sorted(_SKIPPED_RECORD_IDS),
            "range_coalesce_gap_bytes": _RANGE_COALESCE_GAP_BYTES,
            "worker_pool": _WORKER_POOL,
            "schema_version": 3,
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
