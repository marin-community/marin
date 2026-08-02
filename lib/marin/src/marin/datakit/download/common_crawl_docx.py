# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Discover, extract, and normalize DOCX documents from Common Crawl indexes."""

import io
import zipfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cache, partial
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from rigging.filesystem import prefix_join
from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext

from marin.datakit.download.common_crawl_warc import (
    COMMON_CRAWL_DATA_URL,
    COMMON_CRAWL_USER_AGENT,
    CommonCrawlClient,
    CommonCrawlRequestRejectedError,
    CommonCrawlTransientError,
    CommonCrawlWarcRecord,
    MainIndexedRecord,
    OriginResponseStatusError,
    SupplementalIndexedRecord,
    WarcParsingError,
    WarcPayloadTooLargeError,
    WarcRecordRange,
    WarcRecordTooLargeError,
    WarcRevisitError,
    common_crawl_index_partitions,
    main_record_from_index_row,
    supplemental_record_from_index_row,
    verify_supplemental_record,
    verify_url_index_record,
)
from marin.datakit.normalize import DedupMode, normalize_step
from marin.execution.remote import remote
from marin.execution.step_spec import StepSpec

DOCX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
DOCX_MIME_TYPES = frozenset({DOCX_MIME_TYPE})
DEFAULT_MAXIMUM_WARC_RECORD_BYTES = 64 << 20
DEFAULT_MAXIMUM_PAYLOAD_BYTES = 64 << 20
DEFAULT_MAXIMUM_ZIP_ENTRIES = 10_000
DEFAULT_MAXIMUM_UNCOMPRESSED_BYTES = 512 << 20
DEFAULT_INDEX_BATCH_ROWS = 16_384
DEFAULT_EXTRACTION_SHARDS = 1_024
DEFAULT_MAX_WORKERS = 128
DEFAULT_LANGUAGE_MINIMUM_CHARS = 50
DEFAULT_LANGUAGE_SAMPLE_CHARS = 2_000
EXTRACTOR_VERSION = "docling-2.99.0-smart-markdown-v2"
LANGUAGE_DETECTOR_VERSION = "lingua-2.2.0-all-languages-v1"

_INDEX_COLUMNS = (
    "url",
    "fetch_status",
    "content_mime_type",
    "content_mime_detected",
    "content_digest",
    "content_truncated",
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
)
_REQUIRED_DOCX_MEMBERS = frozenset({"[Content_Types].xml", "word/document.xml"})


class CommonCrawlIndexKind(StrEnum):
    """Common Crawl index schema used to discover and verify records."""

    MAIN = "main"
    SUPPLEMENTAL = "supplemental"


class DocxSelectionReason(StrEnum):
    """Highest-confidence signal that selected an index row."""

    DECLARED_MIME = "declared_mime"
    DETECTED_MIME = "detected_mime"
    URL_SUFFIX = "url_suffix"


CANDIDATE_SCHEMA = pa.schema(
    [
        pa.field("crawl_id", pa.string(), nullable=False),
        pa.field("index_kind", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("fetch_status", pa.int32(), nullable=False),
        pa.field("content_mime_type", pa.string()),
        pa.field("content_mime_detected", pa.string()),
        pa.field("content_digest", pa.string(), nullable=False),
        pa.field("warc_filename", pa.string(), nullable=False),
        pa.field("warc_record_offset", pa.int64(), nullable=False),
        pa.field("warc_record_length", pa.int64(), nullable=False),
        pa.field("warc_record_id", pa.string()),
        pa.field("selection_reason", pa.string(), nullable=False),
        pa.field("selected_by_declared_mime", pa.bool_(), nullable=False),
        pa.field("selected_by_detected_mime", pa.bool_(), nullable=False),
        pa.field("selected_by_url_suffix", pa.bool_(), nullable=False),
    ]
)

COMMON_CRAWL_DOCX_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("crawl_id", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("warc_filename", pa.string(), nullable=False),
        pa.field("warc_record_offset", pa.int64(), nullable=False),
        pa.field("warc_record_length", pa.int64(), nullable=False),
        pa.field("warc_date", pa.string()),
        pa.field("http_status", pa.int32(), nullable=False),
        pa.field("http_content_type", pa.string()),
        pa.field("identified_payload_type", pa.string()),
        pa.field("content_digest", pa.string(), nullable=False),
        pa.field("index_status", pa.int32(), nullable=False),
        pa.field("index_content_type", pa.string()),
        pa.field("index_detected_type", pa.string()),
        pa.field("selection_reason", pa.string(), nullable=False),
        pa.field("language", pa.string(), nullable=False),
        pa.field("language_confidence", pa.float64(), nullable=False),
        pa.field("word_count", pa.int64(), nullable=False),
        pa.field("table_count", pa.int64(), nullable=False),
        pa.field("image_count", pa.int64(), nullable=False),
        pa.field("extractor", pa.string(), nullable=False),
        pa.field("language_detector", pa.string(), nullable=False),
    ]
)
RAW_COMMON_CRAWL_DOCX_SCHEMA = pa.schema(list(COMMON_CRAWL_DOCX_SCHEMA)[1:])


class InvalidDocxError(ValueError):
    """Raised when a payload is not a safe, structurally valid DOCX container."""


class DocxExtractionError(RuntimeError):
    """Raised when a valid DOCX container cannot be converted to text."""


class EmptyDocxTextError(DocxExtractionError):
    """Raised when DOCX extraction produces no usable text."""


@dataclass(frozen=True)
class ExtractedDocument:
    """Extracted text and inexpensive document-level metrics."""

    text: str
    word_count: int
    table_count: int
    image_count: int


@dataclass(frozen=True)
class DocxSelection:
    """Signals that selected one Common Crawl index row."""

    reason: DocxSelectionReason
    declared_mime: bool
    detected_mime: bool
    url_suffix: bool


@dataclass(frozen=True)
class LanguageDetection:
    """ISO 639-1 language prediction and detector confidence."""

    language: str
    confidence: float


class DocxTextExtractor(Protocol):
    """Text extraction boundary for a validated DOCX payload."""

    @property
    def version(self) -> str: ...

    def extract(self, payload: bytes) -> ExtractedDocument: ...


class LanguageDetector(Protocol):
    """Language identification boundary for extracted text."""

    @property
    def version(self) -> str: ...

    def detect(self, text: str) -> LanguageDetection: ...


class WarcRecordFetcher(Protocol):
    """Transport boundary used by candidate extraction workers."""

    def fetch_record(self, location: WarcRecordRange) -> CommonCrawlWarcRecord: ...


@dataclass(frozen=True)
class DoclingDocxExtractor:
    """Extract DOCX text with Docling while avoiding padded Markdown tables."""

    version: str = EXTRACTOR_VERSION

    def extract(self, payload: bytes) -> ExtractedDocument:
        # Docling is an optional datakit dependency, so importing this module
        # must not require it until extraction is actually requested.
        from docling.datamodel.base_models import DocumentStream  # noqa: PLC0415
        from docling.exceptions import ConversionError, SecurityError  # noqa: PLC0415

        try:
            result = _docling_converter().convert(DocumentStream(name="document.docx", stream=io.BytesIO(payload)))
        except (ConversionError, SecurityError) as error:
            raise DocxExtractionError("Docling failed to extract the DOCX payload") from error
        return _extracted_document(result.document)


@dataclass(frozen=True)
class LinguaLanguageDetector:
    """Detect language with Lingua's complete language model set."""

    minimum_chars: int = DEFAULT_LANGUAGE_MINIMUM_CHARS
    sample_chars: int = DEFAULT_LANGUAGE_SAMPLE_CHARS
    version: str = LANGUAGE_DETECTOR_VERSION

    def detect(self, text: str) -> LanguageDetection:
        if len(text) < self.minimum_chars:
            return LanguageDetection(language="unknown", confidence=0.0)
        confidence_values = _lingua_detector().compute_language_confidence_values(text[: self.sample_chars])
        if not confidence_values:
            return LanguageDetection(language="unknown", confidence=0.0)
        prediction = confidence_values[0]
        return LanguageDetection(
            language=prediction.language.iso_code_639_1.name.lower(),
            confidence=prediction.value,
        )


@dataclass(frozen=True)
class CommonCrawlDocxCandidate:
    """One DOCX candidate selected from a Common Crawl index row."""

    indexed_record: MainIndexedRecord | SupplementalIndexedRecord
    index_kind: CommonCrawlIndexKind
    index_status: int
    index_content_type: str | None
    index_detected_type: str | None
    selection_reason: DocxSelectionReason
    selected_by_declared_mime: bool
    selected_by_detected_mime: bool
    selected_by_url_suffix: bool


@dataclass(frozen=True)
class CommonCrawlDocxSource:
    """Common Crawl source and resource limits for the three-stage pipeline."""

    crawl_id: str
    index_kind: CommonCrawlIndexKind
    paths_manifest_url: str
    base_url: str = COMMON_CRAWL_DATA_URL
    maximum_warc_record_bytes: int = DEFAULT_MAXIMUM_WARC_RECORD_BYTES
    maximum_payload_bytes: int = DEFAULT_MAXIMUM_PAYLOAD_BYTES
    maximum_zip_entries: int = DEFAULT_MAXIMUM_ZIP_ENTRIES
    maximum_uncompressed_bytes: int = DEFAULT_MAXIMUM_UNCOMPRESSED_BYTES
    index_batch_rows: int = DEFAULT_INDEX_BATCH_ROWS
    extraction_shards: int = DEFAULT_EXTRACTION_SHARDS
    max_workers: int = DEFAULT_MAX_WORKERS

    def __post_init__(self) -> None:
        if not self.crawl_id:
            raise ValueError("crawl_id must be non-empty")
        if not self.paths_manifest_url:
            raise ValueError("paths_manifest_url must be non-empty")
        for field_name in (
            "maximum_warc_record_bytes",
            "maximum_payload_bytes",
            "maximum_zip_entries",
            "maximum_uncompressed_bytes",
            "index_batch_rows",
            "extraction_shards",
            "max_workers",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")


@cache
def _docling_converter() -> Any:
    from docling.datamodel.base_models import InputFormat  # noqa: PLC0415
    from docling.document_converter import DocumentConverter  # noqa: PLC0415

    return DocumentConverter(allowed_formats=[InputFormat.DOCX])


@cache
def _lingua_detector() -> Any:
    from lingua import LanguageDetectorBuilder  # noqa: PLC0415

    return LanguageDetectorBuilder.from_all_languages().build()


def _extracted_document(document: Any) -> ExtractedDocument:
    from docling_core.types.doc.labels import DocItemLabel  # noqa: PLC0415

    non_table_content = document.export_to_markdown(labels=set(DocItemLabel) - {DocItemLabel.TABLE})
    tables: list[str] = []
    for table in document.tables:
        rows = [" | ".join(cell.text for cell in row if cell.text) for row in table.data.grid]
        if table_text := "\n".join(row for row in rows if row):
            tables.append(table_text)
    text = "\n\n".join((non_table_content, *tables)).strip()
    return ExtractedDocument(
        text=text,
        word_count=len(text.split()),
        table_count=len(document.tables),
        image_count=len(document.pictures),
    )


def _docx_selection(row: Mapping[str, object]) -> DocxSelection | None:
    status = row.get("fetch_status")
    if isinstance(status, bool) or not isinstance(status, int) or not 200 <= status < 300:
        return None
    declared = _mime_type(row.get("content_mime_type")) in DOCX_MIME_TYPES
    detected = _mime_type(row.get("content_mime_detected")) in DOCX_MIME_TYPES
    url = row.get("url")
    suffix = isinstance(url, str) and unquote(urlsplit(url).path).lower().endswith(".docx")
    if declared:
        reason = DocxSelectionReason.DECLARED_MIME
    elif detected:
        reason = DocxSelectionReason.DETECTED_MIME
    elif suffix:
        reason = DocxSelectionReason.URL_SUFFIX
    else:
        return None
    return DocxSelection(reason, declared, detected, suffix)


def is_docx_index_row(row: Mapping[str, object]) -> bool:
    """Return whether an index row is a successful, non-truncated DOCX candidate."""
    return row.get("content_truncated") is None and _docx_selection(row) is not None


def docx_candidates(
    index_partition: str,
    *,
    crawl_id: str,
    index_kind: CommonCrawlIndexKind,
    base_url: str,
    batch_rows: int,
) -> Iterator[CommonCrawlDocxCandidate]:
    """Yield DOCX candidates from one URL Index Parquet partition."""
    partition_url = f"{base_url.rstrip('/')}/{index_partition.lstrip('/')}"
    open_options = {"headers": {"User-Agent": COMMON_CRAWL_USER_AGENT}} if partition_url.startswith("http") else {}
    with fsspec.open(partition_url, "rb", **open_options) as stream:
        parquet = pq.ParquetFile(stream)
        available_columns = set(parquet.schema_arrow.names)
        required_columns = set(_INDEX_COLUMNS)
        if index_kind is CommonCrawlIndexKind.MAIN:
            required_columns.add("warc_record_id")
        if missing_columns := required_columns - available_columns:
            raise ValueError(f"Index partition is missing required columns: {sorted(missing_columns)}")
        columns = list(_INDEX_COLUMNS)
        if "warc_record_id" in available_columns:
            columns.append("warc_record_id")

        for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns):
            for row in batch.to_pylist():
                counters.pipeline.update_counter("common_crawl_docx/index_rows", 1)
                if row.get("content_truncated") is not None:
                    counters.pipeline.update_counter("common_crawl_docx/truncated_index_rows", 1)
                    continue
                if not is_docx_index_row(row):
                    continue
                selection = _docx_selection(row)
                assert selection is not None
                try:
                    indexed_record = (
                        main_record_from_index_row(row, crawl_id=crawl_id)
                        if index_kind is CommonCrawlIndexKind.MAIN
                        else supplemental_record_from_index_row(row, crawl_id=crawl_id)
                    )
                except ValueError:
                    counters.pipeline.update_counter("common_crawl_docx/invalid_index_rows", 1)
                    continue
                counters.pipeline.update_counter("common_crawl_docx/candidates", 1)
                for selected, counter in (
                    (selection.declared_mime, "selected_by_declared_mime"),
                    (selection.detected_mime, "selected_by_detected_mime"),
                    (selection.url_suffix, "selected_by_url_suffix"),
                ):
                    if selected:
                        counters.pipeline.update_counter(f"common_crawl_docx/{counter}", 1)
                yield CommonCrawlDocxCandidate(
                    indexed_record=indexed_record,
                    index_kind=index_kind,
                    index_status=row["fetch_status"],
                    index_content_type=_optional_string(row.get("content_mime_type")),
                    index_detected_type=_optional_string(row.get("content_mime_detected")),
                    selection_reason=selection.reason,
                    selected_by_declared_mime=selection.declared_mime,
                    selected_by_detected_mime=selection.detected_mime,
                    selected_by_url_suffix=selection.url_suffix,
                )


def candidate_record(candidate: CommonCrawlDocxCandidate) -> dict[str, object]:
    """Serialize a candidate into the stable discovery-manifest schema."""
    indexed = candidate.indexed_record
    expectation = indexed.expectation
    record_range = indexed.record_range
    return {
        "crawl_id": record_range.crawl_id,
        "index_kind": candidate.index_kind.value,
        "url": expectation.url,
        "fetch_status": candidate.index_status,
        "content_mime_type": candidate.index_content_type,
        "content_mime_detected": candidate.index_detected_type,
        "content_digest": expectation.content_digest,
        "warc_filename": record_range.warc_filename,
        "warc_record_offset": record_range.offset,
        "warc_record_length": record_range.length,
        "warc_record_id": expectation.warc_record_id if isinstance(indexed, MainIndexedRecord) else None,
        "selection_reason": candidate.selection_reason.value,
        "selected_by_declared_mime": candidate.selected_by_declared_mime,
        "selected_by_detected_mime": candidate.selected_by_detected_mime,
        "selected_by_url_suffix": candidate.selected_by_url_suffix,
    }


def candidate_from_record(row: Mapping[str, object]) -> CommonCrawlDocxCandidate:
    """Deserialize one discovery-manifest row."""
    index_kind = CommonCrawlIndexKind(row["index_kind"])
    indexed_record = (
        main_record_from_index_row(row, crawl_id=str(row["crawl_id"]))
        if index_kind is CommonCrawlIndexKind.MAIN
        else supplemental_record_from_index_row(row, crawl_id=str(row["crawl_id"]))
    )
    return CommonCrawlDocxCandidate(
        indexed_record=indexed_record,
        index_kind=index_kind,
        index_status=int(row["fetch_status"]),
        index_content_type=_optional_string(row.get("content_mime_type")),
        index_detected_type=_optional_string(row.get("content_mime_detected")),
        selection_reason=DocxSelectionReason(row["selection_reason"]),
        selected_by_declared_mime=bool(row["selected_by_declared_mime"]),
        selected_by_detected_mime=bool(row["selected_by_detected_mime"]),
        selected_by_url_suffix=bool(row["selected_by_url_suffix"]),
    )


def discover_partition_records(index_partition: str, *, source: CommonCrawlDocxSource) -> Iterator[dict[str, object]]:
    """Produce manifest rows for one index partition."""
    for candidate in docx_candidates(
        index_partition,
        crawl_id=source.crawl_id,
        index_kind=source.index_kind,
        base_url=source.base_url,
        batch_rows=source.index_batch_rows,
    ):
        yield candidate_record(candidate)


def discover_common_crawl_docx(output_path: str, source: CommonCrawlDocxSource) -> None:
    """Write a reusable candidate manifest for one Common Crawl index."""
    partitions = common_crawl_index_partitions(source.paths_manifest_url, crawl_id=source.crawl_id)
    pipeline = (
        Dataset.from_list(list(partitions))
        .flat_map(partial(discover_partition_records, source=source))
        .write_parquet(
            prefix_join(output_path, "candidates/part-{shard:05d}-of-{total:05d}.parquet"),
            schema=CANDIDATE_SCHEMA,
            skip_existing=True,
        )
    )
    ZephyrContext(
        name=f"common-crawl-docx-discovery-{source.crawl_id.lower()}",
        resources=ResourceConfig(cpu=1, ram="8g"),
        max_workers=min(source.max_workers, len(partitions)),
    ).execute(pipeline)


def validate_docx(payload: bytes, *, maximum_entries: int, maximum_uncompressed_bytes: int) -> None:
    """Validate required DOCX members and reject oversized ZIP containers."""
    if not payload.startswith(b"PK\x03\x04"):
        raise InvalidDocxError("DOCX payload does not start with a ZIP local-file header")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            if len(entries) > maximum_entries:
                raise InvalidDocxError(f"DOCX contains more than {maximum_entries} ZIP entries")
            if sum(entry.file_size for entry in entries) > maximum_uncompressed_bytes:
                raise InvalidDocxError(f"DOCX expands beyond {maximum_uncompressed_bytes} uncompressed bytes")
            if missing := _REQUIRED_DOCX_MEMBERS - {entry.filename for entry in entries}:
                raise InvalidDocxError(f"DOCX is missing required members: {sorted(missing)}")
    except (zipfile.BadZipFile, OSError) as error:
        raise InvalidDocxError("DOCX payload is not a readable ZIP archive") from error


def extracted_docx_record(
    candidate: CommonCrawlDocxCandidate,
    record: CommonCrawlWarcRecord,
    *,
    extractor: DocxTextExtractor,
    language_detector: LanguageDetector,
    maximum_zip_entries: int,
    maximum_uncompressed_bytes: int,
) -> dict[str, object]:
    """Validate and extract one verified WARC payload into the raw schema."""
    validate_docx(
        record.payload,
        maximum_entries=maximum_zip_entries,
        maximum_uncompressed_bytes=maximum_uncompressed_bytes,
    )
    extracted = extractor.extract(record.payload)
    text = extracted.text.strip()
    if not text:
        raise EmptyDocxTextError("DOCX extraction produced empty text")
    language = language_detector.detect(text)
    location = candidate.indexed_record.record_range
    return {
        "text": text,
        "source_id": record.warc_record_id,
        "source": location.crawl_id,
        "crawl_id": location.crawl_id,
        "url": record.target_url,
        "warc_filename": location.warc_filename,
        "warc_record_offset": location.offset,
        "warc_record_length": location.length,
        "warc_date": record.warc_date,
        "http_status": record.http_status,
        "http_content_type": record.http_content_type,
        "identified_payload_type": record.identified_payload_type,
        "content_digest": record.payload_digest,
        "index_status": candidate.index_status,
        "index_content_type": candidate.index_content_type,
        "index_detected_type": candidate.index_detected_type,
        "selection_reason": candidate.selection_reason.value,
        "language": language.language,
        "language_confidence": language.confidence,
        "word_count": extracted.word_count,
        "table_count": extracted.table_count,
        "image_count": extracted.image_count,
        "extractor": extractor.version,
        "language_detector": language_detector.version,
    }


def _verify_candidate(candidate: CommonCrawlDocxCandidate, record: CommonCrawlWarcRecord) -> None:
    indexed_record = candidate.indexed_record
    if isinstance(indexed_record, MainIndexedRecord):
        verify_url_index_record(record, indexed_record.expectation)
    else:
        verify_supplemental_record(record, indexed_record.expectation)


def process_docx_candidate(
    client: WarcRecordFetcher,
    candidate: CommonCrawlDocxCandidate,
    *,
    extractor: DocxTextExtractor,
    language_detector: LanguageDetector,
    maximum_zip_entries: int,
    maximum_uncompressed_bytes: int,
) -> dict[str, object] | None:
    """Fetch one candidate, propagating transient and provenance failures."""
    try:
        record = client.fetch_record(candidate.indexed_record.record_range)
        counters.pipeline.update_counter("common_crawl_docx/fetched", 1)
        _verify_candidate(candidate, record)
    except CommonCrawlTransientError:
        counters.pipeline.update_counter("common_crawl_docx/transient_warc_failures", 1)
        raise
    except WarcRevisitError:
        counters.pipeline.update_counter("common_crawl_docx/revisits", 1)
        return None
    except OriginResponseStatusError:
        counters.pipeline.update_counter("common_crawl_docx/non_success_origin", 1)
        return None
    except WarcRecordTooLargeError:
        counters.pipeline.update_counter("common_crawl_docx/warc_records_too_large", 1)
        return None
    except WarcPayloadTooLargeError:
        counters.pipeline.update_counter("common_crawl_docx/payloads_too_large", 1)
        return None
    except CommonCrawlRequestRejectedError:
        counters.pipeline.update_counter("common_crawl_docx/rejected_requests", 1)
        return None
    except WarcParsingError:
        counters.pipeline.update_counter("common_crawl_docx/warc_parsing_errors", 1)
        return None

    try:
        output = extracted_docx_record(
            candidate,
            record,
            extractor=extractor,
            language_detector=language_detector,
            maximum_zip_entries=maximum_zip_entries,
            maximum_uncompressed_bytes=maximum_uncompressed_bytes,
        )
    except InvalidDocxError:
        counters.pipeline.update_counter("common_crawl_docx/invalid_files", 1)
        return None
    except EmptyDocxTextError:
        counters.pipeline.update_counter("common_crawl_docx/empty_text", 1)
        return None
    except DocxExtractionError:
        counters.pipeline.update_counter("common_crawl_docx/docling_errors", 1)
        return None

    counters.pipeline.update_counter("common_crawl_docx/valid_files", 1)
    counters.pipeline.update_counter("common_crawl_docx/text_bytes", len(output["text"].encode("utf-8")))
    counters.pipeline.update_counter("common_crawl_docx/words", output["word_count"])
    counters.pipeline.update_counter("common_crawl_docx/tables", output["table_count"])
    counters.pipeline.update_counter("common_crawl_docx/images", output["image_count"])
    if output["table_count"]:
        counters.pipeline.update_counter("common_crawl_docx/documents_with_tables", 1)
    if output["language"] == "unknown":
        counters.pipeline.update_counter("common_crawl_docx/unknown_language", 1)
    return output


def process_docx_shard(
    rows: Iterable[Mapping[str, object]],
    _shard: ShardInfo,
    *,
    source: CommonCrawlDocxSource,
    extractor: DocxTextExtractor,
    language_detector: LanguageDetector,
) -> Iterator[dict[str, object]]:
    """Fetch and extract one independently retryable candidate shard."""
    with CommonCrawlClient(
        maximum_warc_record_bytes=source.maximum_warc_record_bytes,
        maximum_payload_bytes=source.maximum_payload_bytes,
        base_url=source.base_url,
    ) as client:
        for row in rows:
            output = process_docx_candidate(
                client,
                candidate_from_record(row),
                extractor=extractor,
                language_detector=language_detector,
                maximum_zip_entries=source.maximum_zip_entries,
                maximum_uncompressed_bytes=source.maximum_uncompressed_bytes,
            )
            if output is not None:
                yield output


def extract_common_crawl_docx(
    output_path: str,
    candidate_path: str,
    source: CommonCrawlDocxSource,
    *,
    extractor: DocxTextExtractor,
    language_detector: LanguageDetector,
) -> None:
    """Extract a previously materialized Common Crawl DOCX candidate manifest."""
    pipeline = (
        Dataset.from_files(prefix_join(candidate_path, "candidates/*.parquet"))
        .load_parquet()
        .reshard(source.extraction_shards)
        .map_shard(partial(process_docx_shard, source=source, extractor=extractor, language_detector=language_detector))
        .write_parquet(
            prefix_join(output_path, "data/part-{shard:05d}-of-{total:05d}.parquet"),
            schema=RAW_COMMON_CRAWL_DOCX_SCHEMA,
            skip_existing=True,
        )
    )
    ZephyrContext(
        name=f"common-crawl-docx-extraction-{source.crawl_id.lower()}",
        resources=ResourceConfig(cpu=2, ram="16g"),
        max_workers=source.max_workers,
    ).execute(pipeline)


def common_crawl_docx_steps(
    source: CommonCrawlDocxSource,
    *,
    extractor: DocxTextExtractor = DoclingDocxExtractor(),
    language_detector: LanguageDetector = LinguaLanguageDetector(),
) -> tuple[StepSpec, StepSpec, StepSpec]:
    """Build discovery, extraction, and exact-deduplicating normalization steps."""
    slug = source.crawl_id.lower()
    discovery = StepSpec(
        name=f"raw/common-crawl-docx-candidates/{slug}",
        fn=remote(
            partial(discover_common_crawl_docx, source=source),
            resources=ResourceConfig(cpu=1, ram="4g"),
            pip_dependency_groups=["datakit"],
        ),
        hash_attrs={
            "crawl_id": source.crawl_id,
            "index_kind": source.index_kind,
            "paths_manifest_url": source.paths_manifest_url,
            "base_url": source.base_url,
            "docx_mime_types": sorted(DOCX_MIME_TYPES),
            "schema_version": 2,
        },
    )
    extraction = StepSpec(
        name=f"raw/common-crawl-docx-extracted/{slug}",
        fn=remote(
            partial(
                extract_common_crawl_docx,
                candidate_path=discovery.output_path,
                source=source,
                extractor=extractor,
                language_detector=language_detector,
            ),
            resources=ResourceConfig(cpu=1, ram="4g"),
            pip_dependency_groups=["datakit"],
        ),
        deps=[discovery],
        hash_attrs={
            "maximum_warc_record_bytes": source.maximum_warc_record_bytes,
            "maximum_payload_bytes": source.maximum_payload_bytes,
            "maximum_zip_entries": source.maximum_zip_entries,
            "maximum_uncompressed_bytes": source.maximum_uncompressed_bytes,
            "extractor": extractor.version,
            "language_detector": language_detector.version,
            "schema_version": 2,
        },
    )
    normalized = normalize_step(
        name=f"normalized/common-crawl-docx/{slug}",
        download=extraction,
        relative_input_path="data",
        file_extensions=(".parquet",),
        id_field="source_id",
        dedup_mode=DedupMode.EXACT,
        output_schema=COMMON_CRAWL_DOCX_SCHEMA,
    )
    return discovery, extraction, normalized


def _mime_type(value: object) -> str | None:
    return value.partition(";")[0].strip().lower() if isinstance(value, str) else None


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
