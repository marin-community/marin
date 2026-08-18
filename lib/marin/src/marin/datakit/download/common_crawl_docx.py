# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Discover, extract, and normalize DOCX documents from Common Crawl indexes."""

import io
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cache, partial
from typing import Any, Protocol

import pyarrow as pa
from fray.types import ResourceConfig
from pydantic import BaseModel
from rigging.filesystem import prefix_join
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from marin.datakit.download.common_crawl_plan import (
    CommonCrawlDiscoveryOptions,
    CommonCrawlFetchTask,
    CommonCrawlPlanOptions,
    CommonCrawlPlanSummary,
    CommonCrawlSelection,
    CommonCrawlSource,
    FetchedCommonCrawlRecord,
    common_crawl_discovery_step,
    common_crawl_plan_step,
    fetch_common_crawl_task,
    read_common_crawl_tasks,
)
from marin.datakit.normalize import DedupMode, normalize_step
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_spec import StepSpec

DOCX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
DOCX_MIME_TYPES = frozenset({DOCX_MIME_TYPE})
DEFAULT_MAXIMUM_WARC_RECORD_BYTES = 64 << 20
DEFAULT_MAXIMUM_PAYLOAD_BYTES = 64 << 20
DEFAULT_MAXIMUM_ZIP_ENTRIES = 10_000
DEFAULT_MAXIMUM_UNCOMPRESSED_BYTES = 512 << 20
DEFAULT_INDEX_BATCH_ROWS = 16_384
DEFAULT_MAX_WORKERS = 128
DEFAULT_LANGUAGE_MINIMUM_CHARS = 50
DEFAULT_LANGUAGE_SAMPLE_CHARS = 2_000
EXTRACTOR_VERSION = "docling-2.99.0-smart-markdown-v2"
LANGUAGE_DETECTOR_VERSION = "lingua-2.2.0-all-languages-v1"

_REQUIRED_DOCX_MEMBERS = frozenset({"[Content_Types].xml", "word/document.xml"})


class DocxSelectionReason(StrEnum):
    """Highest-confidence signal that selected an index row."""

    DECLARED_MIME = "declared_mime"
    DETECTED_MIME = "detected_mime"
    URL_SUFFIX = "url_suffix"


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
class DocxRecordSelector:
    """Select successful, untruncated DOCX records and preserve index signals."""

    @property
    def identity(self) -> Mapping[str, object]:
        return {
            "mime_types": sorted(DOCX_MIME_TYPES),
            "url_suffixes": [".docx"],
            "successful_responses": True,
            "exclude_truncated": True,
        }

    def select(self, row: Mapping[str, object]) -> CommonCrawlSelection | None:
        status = row.get("fetch_status")
        if isinstance(status, bool) or not isinstance(status, int) or not 200 <= status < 300:
            return None
        if row.get("content_truncated") is not None:
            return None
        declared_type = _mime_type(row.get("content_mime_type"))
        detected_type = _mime_type(row.get("content_mime_detected"))
        url = row.get("url")
        declared = declared_type in DOCX_MIME_TYPES
        detected = detected_type in DOCX_MIME_TYPES
        suffix = isinstance(url, str) and url.partition("?")[0].lower().endswith(".docx")
        if not (declared or detected or suffix):
            return None
        reason = (
            DocxSelectionReason.DECLARED_MIME
            if declared
            else DocxSelectionReason.DETECTED_MIME if detected else DocxSelectionReason.URL_SUFFIX
        )
        return CommonCrawlSelection(
            {
                "index_status": status,
                "index_content_type": declared_type,
                "index_detected_type": detected_type,
                "selection_reason": reason.value,
                "selected_by_declared_mime": declared,
                "selected_by_detected_mime": detected,
                "selected_by_url_suffix": suffix,
            }
        )


@dataclass(frozen=True)
class CommonCrawlDocxConfig:
    """Common Crawl sources, plan policy, and DOCX extraction limits."""

    name: str
    sources: tuple[CommonCrawlSource, ...]
    plan: CommonCrawlPlanOptions = field(default_factory=CommonCrawlPlanOptions)
    maximum_warc_record_bytes: int = DEFAULT_MAXIMUM_WARC_RECORD_BYTES
    maximum_payload_bytes: int = DEFAULT_MAXIMUM_PAYLOAD_BYTES
    maximum_zip_entries: int = DEFAULT_MAXIMUM_ZIP_ENTRIES
    maximum_uncompressed_bytes: int = DEFAULT_MAXIMUM_UNCOMPRESSED_BYTES
    index_batch_rows: int = DEFAULT_INDEX_BATCH_ROWS
    max_workers: int = DEFAULT_MAX_WORKERS

    def __post_init__(self) -> None:
        if not self.name or not self.sources:
            raise ValueError("name and sources must not be empty")
        for field_name in (
            "maximum_warc_record_bytes",
            "maximum_payload_bytes",
            "maximum_zip_entries",
            "maximum_uncompressed_bytes",
            "index_batch_rows",
            "max_workers",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")


class CommonCrawlDocxStageResult(BaseModel):
    """Output location and aggregate Zephyr counters for one pipeline stage."""

    data_dir: str
    counters: dict[str, int | float]


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
    fetched: FetchedCommonCrawlRecord,
    *,
    extractor: DocxTextExtractor,
    language_detector: LanguageDetector,
    maximum_zip_entries: int,
    maximum_uncompressed_bytes: int,
) -> dict[str, object]:
    """Validate and extract one verified WARC payload into the raw schema."""
    record = fetched.observed_record
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
    location = fetched.indexed_record.record_range
    metadata = fetched.selection.metadata
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
        "index_status": int(metadata["index_status"]),
        "index_content_type": _optional_string(metadata.get("index_content_type")),
        "index_detected_type": _optional_string(metadata.get("index_detected_type")),
        "selection_reason": str(metadata["selection_reason"]),
        "language": language.language,
        "language_confidence": language.confidence,
        "word_count": extracted.word_count,
        "table_count": extracted.table_count,
        "image_count": extracted.image_count,
        "extractor": extractor.version,
        "language_detector": language_detector.version,
    }


def process_fetched_docx(
    fetched: FetchedCommonCrawlRecord,
    *,
    extractor: DocxTextExtractor,
    language_detector: LanguageDetector,
    maximum_zip_entries: int,
    maximum_uncompressed_bytes: int,
) -> dict[str, object] | None:
    """Extract one fetched DOCX, skipping document-local conversion failures."""
    try:
        output = extracted_docx_record(
            fetched,
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


def process_docx_task(
    task: CommonCrawlFetchTask,
    *,
    config: CommonCrawlDocxConfig,
    extractor: DocxTextExtractor,
    language_detector: LanguageDetector,
) -> Iterator[dict[str, object]]:
    """Fetch one source-local task and extract each verified DOCX payload."""
    for fetched in fetch_common_crawl_task(
        task,
        maximum_warc_record_bytes=config.maximum_warc_record_bytes,
        maximum_payload_bytes=config.maximum_payload_bytes,
    ):
        counters.pipeline.update_counter("common_crawl_docx/fetched", 1)
        output = process_fetched_docx(
            fetched,
            extractor=extractor,
            language_detector=language_detector,
            maximum_zip_entries=config.maximum_zip_entries,
            maximum_uncompressed_bytes=config.maximum_uncompressed_bytes,
        )
        if output is not None:
            yield output


def extract_common_crawl_docx(
    output_path: str,
    plan_output_path: str,
    config: CommonCrawlDocxConfig,
    *,
    extractor: DocxTextExtractor,
    language_detector: LanguageDetector,
) -> CommonCrawlDocxStageResult:
    """Fetch and extract a shared Common Crawl plan."""
    plan = read_artifact(plan_output_path, CommonCrawlPlanSummary)
    tasks = read_common_crawl_tasks(plan.manifest_path)
    pipeline = (
        Dataset.from_list(tasks)
        .flat_map(partial(process_docx_task, config=config, extractor=extractor, language_detector=language_detector))
        .write_parquet(
            prefix_join(output_path, "data/part-{shard:05d}-of-{total:05d}.parquet"),
            schema=RAW_COMMON_CRAWL_DOCX_SCHEMA,
            skip_existing=True,
        )
    )
    outcome = ZephyrContext(
        name=f"common-crawl-docx-extraction-{config.name}",
        resources=ResourceConfig(cpu=2, ram="16g"),
        max_workers=max(1, min(config.max_workers, len(tasks))),
    ).execute(pipeline)
    return CommonCrawlDocxStageResult(
        data_dir=prefix_join(output_path, "data"),
        counters=dict(outcome.counters),
    )


def common_crawl_docx_steps(
    config: CommonCrawlDocxConfig,
    *,
    extractor: DocxTextExtractor = DoclingDocxExtractor(),
    language_detector: LanguageDetector = LinguaLanguageDetector(),
) -> tuple[StepSpec, StepSpec, StepSpec, StepSpec]:
    """Build shared discovery and planning, DOCX extraction, and normalization steps."""
    slug = config.name.lower()
    discovery = common_crawl_discovery_step(
        name=f"raw/common-crawl-docx-discovery/{slug}",
        sources=config.sources,
        selector=DocxRecordSelector(),
        options=CommonCrawlDiscoveryOptions(
            batch_rows=config.index_batch_rows,
            max_workers=config.max_workers,
        ),
    )
    plan = common_crawl_plan_step(
        name=f"raw/common-crawl-docx-plan/{slug}",
        discovery=discovery,
        options=config.plan,
    )
    extraction = StepSpec(
        name=f"raw/common-crawl-docx-extracted/{slug}",
        fn=remote(
            partial(
                extract_common_crawl_docx,
                plan_output_path=plan.output_path,
                config=config,
                extractor=extractor,
                language_detector=language_detector,
            ),
            resources=ResourceConfig(cpu=1, ram="4g"),
            pip_dependency_groups=["datakit"],
        ),
        deps=[plan],
        hash_attrs={
            "maximum_warc_record_bytes": config.maximum_warc_record_bytes,
            "maximum_payload_bytes": config.maximum_payload_bytes,
            "maximum_zip_entries": config.maximum_zip_entries,
            "maximum_uncompressed_bytes": config.maximum_uncompressed_bytes,
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
    return discovery, plan, extraction, normalized


def _mime_type(value: object) -> str | None:
    return value.partition(";")[0].strip().lower() if isinstance(value, str) else None


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
