# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Discover, fetch, extract, identify, and normalize DOCX documents from Common Crawl indexes."""

import io
import re
import zipfile
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cache, partial
from typing import Any, Protocol

import pyarrow as pa
from fray.types import ResourceConfig
from pydantic import BaseModel, ValidationError
from rigging.filesystem import prefix_join
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from marin.datakit.download.common_crawl.plan import (
    CommonCrawlDiscoveryOptions,
    CommonCrawlFetchTask,
    CommonCrawlPlanOptions,
    CommonCrawlPlanSummary,
    CommonCrawlSelection,
    CommonCrawlSource,
    FetchedCommonCrawlRecord,
    common_crawl_discovery_step,
    common_crawl_fetch_failure_counts,
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
DEFAULT_LANGUAGE_CHUNK_CHARS = 2_000
DEFAULT_LANGUAGE_SAMPLE_CHUNKS = 12
DEFAULT_LANGUAGE_MINIMUM_ALPHA_BYTES = 50
DEFAULT_LANGUAGE_MINIMUM_ALPHA_RATIO = 0.2
DEFAULT_LANGUAGE_MAXIMUM_TABLE_ALPHA_BYTES = 1_000
DEFAULT_LANGUAGE_DISTRIBUTION_TOP_K = 5
DEFAULT_LANGUAGE_MINIMUM_SCORE = 0.5
EXTRACTOR_VERSION = "docling-2.99.0-smart-markdown-v2"
LANGUAGE_DETECTOR_VERSION = "lingua-2.2.0-chunk-weighted-v2"

_REQUIRED_DOCX_MEMBERS = frozenset({"[Content_Types].xml", "word/document.xml"})


class DocxSelectionReason(StrEnum):
    """Highest-confidence signal that selected an index row."""

    DECLARED_MIME = "declared_mime"
    DETECTED_MIME = "detected_mime"
    URL_SUFFIX = "url_suffix"


_PROVENANCE_FIELDS = [
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
]
FETCHED_COMMON_CRAWL_DOCX_SCHEMA = pa.schema([pa.field("payload", pa.binary(), nullable=False), *_PROVENANCE_FIELDS])
_LANGUAGE_BLOCK_TYPE = pa.struct(
    [
        pa.field("start", pa.int64(), nullable=False),
        pa.field("stop", pa.int64(), nullable=False),
        pa.field("is_table", pa.bool_(), nullable=False),
    ]
)
EXTRACTED_COMMON_CRAWL_DOCX_SCHEMA = pa.schema(
    [
        pa.field("text", pa.string(), nullable=False),
        pa.field("language_blocks", pa.list_(_LANGUAGE_BLOCK_TYPE), nullable=False),
        *_PROVENANCE_FIELDS,
        pa.field("word_count", pa.int64(), nullable=False),
        pa.field("table_count", pa.int64(), nullable=False),
        pa.field("image_count", pa.int64(), nullable=False),
        pa.field("extractor", pa.string(), nullable=False),
    ]
)
_LANGUAGE_FIELDS = [
    pa.field("language", pa.string(), nullable=False),
    pa.field("language_score", pa.float64(), nullable=False),
    pa.field("language_distribution", pa.map_(pa.string(), pa.float64()), nullable=False),
    pa.field("language_chunks_total", pa.int32(), nullable=False),
    pa.field("language_chunks_scored", pa.int32(), nullable=False),
    pa.field("language_detector", pa.string(), nullable=False),
]
LANGUAGE_COMMON_CRAWL_DOCX_SCHEMA = pa.schema(
    [field for field in EXTRACTED_COMMON_CRAWL_DOCX_SCHEMA if field.name != "language_blocks"] + _LANGUAGE_FIELDS
)
COMMON_CRAWL_DOCX_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        *LANGUAGE_COMMON_CRAWL_DOCX_SCHEMA,
    ]
)


class InvalidDocxError(ValueError):
    """Raised when a payload is not a safe, structurally valid DOCX container."""


class DocxExtractionError(RuntimeError):
    """Raised when a valid DOCX container cannot be converted to text."""


class EmptyDocxTextError(DocxExtractionError):
    """Raised when DOCX extraction produces no usable text."""


@dataclass(frozen=True)
class DocumentBlock:
    """Extracted document content with structural information needed by LID."""

    text: str
    is_table: bool


@dataclass(frozen=True)
class ExtractedDocument:
    """Extracted text and inexpensive document-level metrics."""

    text: str
    word_count: int
    table_count: int
    image_count: int
    language_blocks: tuple[DocumentBlock, ...]


@dataclass(frozen=True)
class LanguageDetection:
    """Document language prediction and weighted score distribution."""

    language: str
    score: float
    distribution: Mapping[str, float]
    chunks_total: int
    chunks_scored: int


class DocxTextExtractor(Protocol):
    """Text extraction boundary for a validated DOCX payload."""

    @property
    def version(self) -> str: ...

    def extract(self, payload: bytes) -> ExtractedDocument: ...


class LanguageDetector(Protocol):
    """Language identification boundary for extracted text."""

    @property
    def version(self) -> str: ...

    def score(self, text: str) -> Mapping[str, float]: ...


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
            return _extracted_document(result.document)
        except (ConversionError, SecurityError, RuntimeError, ValidationError) as error:
            raise DocxExtractionError("Docling failed to extract the DOCX payload") from error


@dataclass(frozen=True)
class LinguaLanguageDetector:
    """Detect language with Lingua's complete language model set."""

    version: str = LANGUAGE_DETECTOR_VERSION

    def score(self, text: str) -> Mapping[str, float]:
        confidence_values = _lingua_detector().compute_language_confidence_values(text)
        return {prediction.language.iso_code_639_1.name.lower(): prediction.value for prediction in confidence_values}


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
    language_chunk_chars: int = DEFAULT_LANGUAGE_CHUNK_CHARS
    language_sample_chunks: int = DEFAULT_LANGUAGE_SAMPLE_CHUNKS
    language_minimum_alpha_bytes: int = DEFAULT_LANGUAGE_MINIMUM_ALPHA_BYTES
    language_minimum_alpha_ratio: float = DEFAULT_LANGUAGE_MINIMUM_ALPHA_RATIO
    language_maximum_table_alpha_bytes: int = DEFAULT_LANGUAGE_MAXIMUM_TABLE_ALPHA_BYTES
    language_distribution_top_k: int = DEFAULT_LANGUAGE_DISTRIBUTION_TOP_K
    language_minimum_score: float = DEFAULT_LANGUAGE_MINIMUM_SCORE
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
            "language_chunk_chars",
            "language_sample_chunks",
            "language_minimum_alpha_bytes",
            "language_maximum_table_alpha_bytes",
            "language_distribution_top_k",
            "index_batch_rows",
            "max_workers",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if not 0.0 < self.language_minimum_alpha_ratio <= 1.0:
            raise ValueError("language_minimum_alpha_ratio must be in (0, 1]")
        if not 0.0 <= self.language_minimum_score <= 1.0:
            raise ValueError("language_minimum_score must be in [0, 1]")


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

    non_table_content = document.export_to_markdown(labels=set(DocItemLabel) - {DocItemLabel.TABLE}).strip()
    tables: list[str] = []
    for table in document.tables:
        rows = [" | ".join(cell.text for cell in row if cell.text) for row in table.data.grid]
        if table_text := "\n".join(row for row in rows if row):
            tables.append(table_text)
    body_blocks = (DocumentBlock(non_table_content, is_table=False),) if non_table_content else ()
    language_blocks = (*body_blocks, *(DocumentBlock(table, is_table=True) for table in tables))
    text = "\n\n".join(block.text for block in language_blocks)
    return ExtractedDocument(
        text=text,
        word_count=len(text.split()),
        table_count=len(document.tables),
        image_count=len(document.pictures),
        language_blocks=language_blocks,
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


def fetched_docx_record(fetched: FetchedCommonCrawlRecord) -> dict[str, object]:
    """Serialize one verified Common Crawl response for reusable downstream processing."""
    record = fetched.observed_record
    location = fetched.indexed_record.record_range
    metadata = fetched.selection.metadata
    return {
        "payload": record.payload,
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
    }


def extracted_docx_record(
    fetched: Mapping[str, object],
    *,
    extractor: DocxTextExtractor,
    maximum_zip_entries: int,
    maximum_uncompressed_bytes: int,
) -> dict[str, object]:
    """Validate and extract one persisted DOCX payload without running LID."""
    payload = fetched["payload"]
    if not isinstance(payload, bytes):
        raise TypeError("Fetched DOCX payload must be bytes")
    validate_docx(
        payload,
        maximum_entries=maximum_zip_entries,
        maximum_uncompressed_bytes=maximum_uncompressed_bytes,
    )
    extracted = extractor.extract(payload)
    if not extracted.language_blocks:
        raise DocxExtractionError("DOCX extractor returned no language blocks")
    text = extracted.text.strip()
    if not text:
        raise EmptyDocxTextError("DOCX extraction produced empty text")
    return {
        "text": text,
        "language_blocks": _language_block_spans(text, extracted.language_blocks),
        **{field.name: fetched[field.name] for field in _PROVENANCE_FIELDS},
        "word_count": extracted.word_count,
        "table_count": extracted.table_count,
        "image_count": extracted.image_count,
        "extractor": extractor.version,
    }


def process_fetched_docx(
    fetched: Mapping[str, object],
    *,
    extractor: DocxTextExtractor,
    maximum_zip_entries: int,
    maximum_uncompressed_bytes: int,
) -> dict[str, object] | None:
    """Extract one fetched DOCX, skipping document-local conversion failures."""
    try:
        output = extracted_docx_record(
            fetched,
            extractor=extractor,
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
    return output


def fetch_docx_task(
    task: CommonCrawlFetchTask,
    *,
    config: CommonCrawlDocxConfig,
    failure_output_path: str,
) -> Iterator[dict[str, object]]:
    """Fetch one source-local task and persist each verified DOCX payload."""
    for fetched in fetch_common_crawl_task(
        task,
        maximum_warc_record_bytes=config.maximum_warc_record_bytes,
        maximum_payload_bytes=config.maximum_payload_bytes,
        failure_output_path=failure_output_path,
    ):
        counters.pipeline.update_counter("common_crawl_docx/fetched", 1)
        counters.pipeline.update_counter("common_crawl_docx/fetched_payload_bytes", len(fetched.observed_record.payload))
        yield fetched_docx_record(fetched)


def fetch_common_crawl_docx(
    output_path: str,
    plan_output_path: str,
    config: CommonCrawlDocxConfig,
) -> CommonCrawlDocxStageResult:
    """Fetch a shared Common Crawl plan into reusable payload shards.

    Failed task invocations are persisted under ``.metrics/fetch-failures`` so
    retries do not erase their counts.
    """
    plan = read_artifact(plan_output_path, CommonCrawlPlanSummary)
    tasks = read_common_crawl_tasks(plan.manifest_path)
    failure_output_path = prefix_join(output_path, ".metrics/fetch-failures")
    pipeline = (
        Dataset.from_list(tasks)
        .flat_map(partial(fetch_docx_task, config=config, failure_output_path=failure_output_path))
        .write_parquet(
            prefix_join(output_path, "data/part-{shard:05d}-of-{total:05d}.parquet"),
            schema=FETCHED_COMMON_CRAWL_DOCX_SCHEMA,
            skip_existing=True,
        )
    )
    outcome = ZephyrContext(
        name=f"common-crawl-docx-fetch-{config.name}",
        resources=ResourceConfig(cpu=1, ram="8g"),
        max_workers=max(1, min(config.max_workers, len(tasks))),
    ).execute(pipeline)
    return CommonCrawlDocxStageResult(
        data_dir=prefix_join(output_path, "data"),
        counters={**outcome.counters, **common_crawl_fetch_failure_counts(failure_output_path)},
    )


def extract_common_crawl_docx(
    output_path: str,
    fetched_input_path: str,
    config: CommonCrawlDocxConfig,
    *,
    extractor: DocxTextExtractor,
) -> CommonCrawlDocxStageResult:
    """Read persisted payload shards and extract DOCX text."""
    input_glob = prefix_join(fetched_input_path, "data/**/*.parquet")
    pipeline = (
        Dataset.from_files(input_glob)
        .load_parquet()
        .map(
            partial(
                process_fetched_docx,
                extractor=extractor,
                maximum_zip_entries=config.maximum_zip_entries,
                maximum_uncompressed_bytes=config.maximum_uncompressed_bytes,
            )
        )
        .filter(_is_present)
        .write_parquet(
            prefix_join(output_path, "data/part-{shard:05d}-of-{total:05d}.parquet"),
            schema=EXTRACTED_COMMON_CRAWL_DOCX_SCHEMA,
            skip_existing=True,
        )
    )
    outcome = ZephyrContext(
        name=f"common-crawl-docx-extraction-{config.name}",
        resources=ResourceConfig(cpu=2, ram="16g"),
        max_workers=config.max_workers,
    ).execute(pipeline)
    return CommonCrawlDocxStageResult(data_dir=prefix_join(output_path, "data"), counters=dict(outcome.counters))


@dataclass(frozen=True)
class LanguageChunk:
    """Cleaned LID input and its uncapped and effective aggregation weights."""

    text: str
    alpha_bytes: int
    weight: float
    is_table: bool


_MARKDOWN_DELIMITERS = re.compile(r"(?:^|\s)(?:#{1,6}|[-*_]{3,}|```+)(?=\s|$)|[|`*_~]+")


def language_chunks(
    blocks: Sequence[DocumentBlock],
    *,
    chunk_chars: int = DEFAULT_LANGUAGE_CHUNK_CHARS,
    minimum_alpha_bytes: int = DEFAULT_LANGUAGE_MINIMUM_ALPHA_BYTES,
    minimum_alpha_ratio: float = DEFAULT_LANGUAGE_MINIMUM_ALPHA_RATIO,
) -> list[LanguageChunk]:
    """Clean and gate structural LID chunks without changing stored document text."""
    chunks: list[LanguageChunk] = []
    for block in blocks:
        for paragraph in re.split(r"\n\s*\n", block.text):
            cleaned = _clean_lid_text(paragraph)
            if not cleaned:
                continue
            for start in range(0, len(cleaned), chunk_chars):
                candidate = cleaned[start : start + chunk_chars]
                alpha_bytes = _alpha_bytes(candidate)
                if alpha_bytes < minimum_alpha_bytes:
                    continue
                if alpha_bytes / max(1, len(candidate.encode("utf-8"))) < minimum_alpha_ratio:
                    continue
                chunks.append(
                    LanguageChunk(
                        text=candidate,
                        alpha_bytes=alpha_bytes,
                        weight=float(alpha_bytes),
                        is_table=block.is_table,
                    )
                )
    return chunks


def _clean_lid_text(text: str) -> str:
    return " ".join(_MARKDOWN_DELIMITERS.sub(" ", text).split())


def _alpha_bytes(text: str) -> int:
    return sum(len(character.encode("utf-8")) for character in text if character.isalpha())


def evenly_spaced_sample[T](values: Sequence[T], maximum_values: int) -> tuple[T, ...]:
    """Select at most ``maximum_values`` values while retaining both ends."""
    if maximum_values <= 0:
        raise ValueError("maximum_values must be positive")
    if len(values) <= maximum_values:
        return tuple(values)
    if maximum_values == 1:
        return (values[len(values) // 2],)
    indices = [round(index * (len(values) - 1) / (maximum_values - 1)) for index in range(maximum_values)]
    return tuple(values[index] for index in indices)


def detect_document_language(
    blocks: Sequence[DocumentBlock],
    *,
    detector: LanguageDetector,
    chunk_chars: int = DEFAULT_LANGUAGE_CHUNK_CHARS,
    sample_chunks: int = DEFAULT_LANGUAGE_SAMPLE_CHUNKS,
    minimum_alpha_bytes: int = DEFAULT_LANGUAGE_MINIMUM_ALPHA_BYTES,
    minimum_alpha_ratio: float = DEFAULT_LANGUAGE_MINIMUM_ALPHA_RATIO,
    maximum_table_alpha_bytes: int = DEFAULT_LANGUAGE_MAXIMUM_TABLE_ALPHA_BYTES,
    distribution_top_k: int = DEFAULT_LANGUAGE_DISTRIBUTION_TOP_K,
    minimum_score: float = DEFAULT_LANGUAGE_MINIMUM_SCORE,
) -> LanguageDetection:
    """Aggregate per-chunk detector scores using alphabetic-byte weights."""
    all_chunks = language_chunks(
        blocks,
        chunk_chars=chunk_chars,
        minimum_alpha_bytes=minimum_alpha_bytes,
        minimum_alpha_ratio=minimum_alpha_ratio,
    )
    selected = _cap_table_weight(evenly_spaced_sample(all_chunks, sample_chunks), maximum_table_alpha_bytes)
    weighted_scores: defaultdict[str, float] = defaultdict(float)
    total_weight = 0.0
    scored_chunks = 0
    for chunk in selected:
        scores = detector.score(chunk.text)
        if not scores:
            continue
        scored_chunks += 1
        total_weight += chunk.weight
        for language, score in scores.items():
            weighted_scores[language] += score * chunk.weight
    if total_weight == 0:
        return LanguageDetection("unknown", 0.0, {}, len(all_chunks), 0)
    full_distribution = {language: score / total_weight for language, score in weighted_scores.items()}
    ranked = sorted(full_distribution.items(), key=lambda item: item[1], reverse=True)
    distribution = dict(ranked[:distribution_top_k])
    winning_language, winning_score = ranked[0]
    language = winning_language if winning_score >= minimum_score else "unknown"
    return LanguageDetection(language, winning_score, distribution, len(all_chunks), scored_chunks)


def _cap_table_weight(chunks: Sequence[LanguageChunk], maximum_table_alpha_bytes: int) -> tuple[LanguageChunk, ...]:
    table_alpha_bytes = sum(chunk.alpha_bytes for chunk in chunks if chunk.is_table)
    if table_alpha_bytes <= maximum_table_alpha_bytes:
        return tuple(chunks)
    table_scale = maximum_table_alpha_bytes / table_alpha_bytes
    return tuple(
        LanguageChunk(
            text=chunk.text,
            alpha_bytes=chunk.alpha_bytes,
            weight=chunk.alpha_bytes * table_scale if chunk.is_table else chunk.weight,
            is_table=chunk.is_table,
        )
        for chunk in chunks
    )


def language_identified_docx_record(
    extracted: Mapping[str, object], *, detector: LanguageDetector, config: CommonCrawlDocxConfig
) -> dict[str, object]:
    detection = detect_document_language(
        _document_blocks(extracted["language_blocks"], str(extracted["text"])),
        detector=detector,
        chunk_chars=config.language_chunk_chars,
        sample_chunks=config.language_sample_chunks,
        minimum_alpha_bytes=config.language_minimum_alpha_bytes,
        minimum_alpha_ratio=config.language_minimum_alpha_ratio,
        maximum_table_alpha_bytes=config.language_maximum_table_alpha_bytes,
        distribution_top_k=config.language_distribution_top_k,
        minimum_score=config.language_minimum_score,
    )
    output = {key: value for key, value in extracted.items() if key != "language_blocks"}
    output.update(
        language=detection.language,
        language_score=detection.score,
        language_distribution=dict(detection.distribution),
        language_chunks_total=detection.chunks_total,
        language_chunks_scored=detection.chunks_scored,
        language_detector=detector.version,
    )
    if detection.language == "unknown":
        counters.pipeline.update_counter("common_crawl_docx/unknown_language", 1)
    return output


def identify_common_crawl_docx_language(
    output_path: str,
    extracted_input_path: str,
    config: CommonCrawlDocxConfig,
    *,
    detector: LanguageDetector,
) -> CommonCrawlDocxStageResult:
    """Read extracted DOCX shards and add chunk-aggregated language metadata."""
    input_glob = prefix_join(extracted_input_path, "data/**/*.parquet")
    pipeline = (
        Dataset.from_files(input_glob)
        .load_parquet()
        .map(partial(language_identified_docx_record, detector=detector, config=config))
        .write_parquet(
            prefix_join(output_path, "data/part-{shard:05d}-of-{total:05d}.parquet"),
            schema=LANGUAGE_COMMON_CRAWL_DOCX_SCHEMA,
            skip_existing=True,
        )
    )
    outcome = ZephyrContext(
        name=f"common-crawl-docx-language-identification-{config.name}",
        resources=ResourceConfig(cpu=2, ram="8g"),
        max_workers=config.max_workers,
    ).execute(pipeline)
    return CommonCrawlDocxStageResult(data_dir=prefix_join(output_path, "data"), counters=dict(outcome.counters))


def common_crawl_docx_steps(
    config: CommonCrawlDocxConfig,
    *,
    extractor: DocxTextExtractor = DoclingDocxExtractor(),
    language_detector: LanguageDetector = LinguaLanguageDetector(),
) -> tuple[StepSpec, StepSpec, StepSpec, StepSpec, StepSpec, StepSpec]:
    """Build discovery, planning, fetch, extraction, LID, and normalization steps."""
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
    fetch = StepSpec(
        name=f"raw/common-crawl-docx-fetched/{slug}",
        fn=remote(
            partial(fetch_common_crawl_docx, plan_output_path=plan.output_path, config=config),
            resources=ResourceConfig(cpu=1, ram="4g"),
            pip_dependency_groups=["datakit"],
        ),
        deps=[plan],
        hash_attrs={
            "maximum_warc_record_bytes": config.maximum_warc_record_bytes,
            "maximum_payload_bytes": config.maximum_payload_bytes,
            "schema_version": 1,
        },
    )
    extraction = StepSpec(
        name=f"raw/common-crawl-docx-extracted/{slug}",
        fn=remote(
            partial(
                extract_common_crawl_docx,
                fetched_input_path=fetch.output_path,
                config=config,
                extractor=extractor,
            ),
            resources=ResourceConfig(cpu=1, ram="4g"),
            pip_dependency_groups=["datakit"],
        ),
        deps=[fetch],
        hash_attrs={
            "maximum_zip_entries": config.maximum_zip_entries,
            "maximum_uncompressed_bytes": config.maximum_uncompressed_bytes,
            "extractor": extractor.version,
            "schema_version": 5,
        },
    )
    language_identification = StepSpec(
        name=f"raw/common-crawl-docx-language/{slug}",
        fn=remote(
            partial(
                identify_common_crawl_docx_language,
                extracted_input_path=extraction.output_path,
                config=config,
                detector=language_detector,
            ),
            resources=ResourceConfig(cpu=1, ram="4g"),
            pip_dependency_groups=["datakit"],
        ),
        deps=[extraction],
        hash_attrs={
            "language_detector": language_detector.version,
            "chunk_chars": config.language_chunk_chars,
            "sample_chunks": config.language_sample_chunks,
            "minimum_alpha_bytes": config.language_minimum_alpha_bytes,
            "minimum_alpha_ratio": config.language_minimum_alpha_ratio,
            "maximum_table_alpha_bytes": config.language_maximum_table_alpha_bytes,
            "distribution_top_k": config.language_distribution_top_k,
            "minimum_score": config.language_minimum_score,
            "schema_version": 3,
        },
    )
    normalized = normalize_step(
        name=f"normalized/common-crawl-docx/{slug}",
        download=language_identification,
        relative_input_path="data",
        file_extensions=(".parquet",),
        id_field="source_id",
        dedup_mode=DedupMode.EXACT,
        output_schema=COMMON_CRAWL_DOCX_SCHEMA,
    )
    return discovery, plan, fetch, extraction, language_identification, normalized


def _mime_type(value: object) -> str | None:
    return value.partition(";")[0].strip().lower() if isinstance(value, str) else None


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _is_present(record: object | None) -> bool:
    return record is not None


def _language_block_spans(text: str, blocks: Sequence[DocumentBlock]) -> list[dict[str, object]]:
    spans: list[dict[str, object]] = []
    cursor = 0
    for block in blocks:
        stop = cursor + len(block.text)
        if text[cursor:stop] != block.text:
            raise DocxExtractionError("Extracted text does not align with language blocks")
        spans.append({"start": cursor, "stop": stop, "is_table": block.is_table})
        cursor = stop + 2
    if cursor - 2 != len(text):
        raise DocxExtractionError("Language blocks do not cover extracted text")
    return spans


def _document_blocks(value: object, text: str) -> tuple[DocumentBlock, ...]:
    if not isinstance(value, list):
        raise TypeError("Expected a list of language blocks")
    blocks: list[DocumentBlock] = []
    for item in value:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("start"), int)
            or not isinstance(item.get("stop"), int)
            or not isinstance(item.get("is_table"), bool)
        ):
            raise TypeError("Expected language block offsets and is_table fields")
        start = item["start"]
        stop = item["stop"]
        if start < 0 or stop <= start or stop > len(text):
            raise ValueError("Language block offsets are outside extracted text")
        blocks.append(DocumentBlock(text=text[start:stop], is_table=item["is_table"]))
    return tuple(blocks)
