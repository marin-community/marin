# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from docx import Document
from marin.datakit.download.common_crawl_docx import (
    DOCX_MIME_TYPE,
    CommonCrawlDocxCandidate,
    CommonCrawlDocxSource,
    CommonCrawlIndexKind,
    DoclingDocxExtractor,
    DocxExtractionError,
    DocxSelectionReason,
    ExtractedDocument,
    InvalidDocxError,
    LanguageDetection,
    LinguaLanguageDetector,
    candidate_from_record,
    candidate_record,
    common_crawl_docx_steps,
    docx_candidates,
    extracted_docx_record,
    is_docx_index_row,
    process_docx_candidate,
    process_docx_shard,
    validate_docx,
)
from marin.datakit.download.common_crawl_warc import (
    CommonCrawlDownloadError,
    CommonCrawlWarcRecord,
    RecordVerificationError,
    SupplementalIndexedRecord,
    WarcRevisitError,
    content_digest,
    main_record_from_index_row,
)
from marin.execution.remote import RemoteCallable
from zephyr.dataset import ShardInfo

from experiments.datakit.common_crawl_docx_sample import sample_report_markdown, stratified_partition_slice

CRAWL_ID = "CC-MAIN-2026-30"
RECORD_ID = "<urn:uuid:019f8700-d21d-78d8-8eb1-99eaa22579da>"
URL = "https://example.com/report.docx"


def _docx_payload(*, document: bytes = b"<w:document/>") -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", b"<Types/>")
        archive.writestr("_rels/.rels", b"<Relationships/>")
        archive.writestr("word/document.xml", document)
    return output.getvalue()


def _real_docx_payload(text: str) -> bytes:
    output = io.BytesIO()
    document = Document()
    document.add_heading("Common Crawl DOCX", level=1)
    document.add_paragraph(text)
    document.save(output)
    return output.getvalue()


def _real_docx_with_table() -> bytes:
    output = io.BytesIO()
    document = Document()
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Quarter"
    table.cell(0, 1).text = "Revenue"
    table.cell(1, 0).text = "Q1"
    table.cell(1, 1).text = "$42"
    document.save(output)
    return output.getvalue()


def _index_row(payload: bytes) -> dict[str, object]:
    return {
        "url": URL,
        "fetch_status": 200,
        "content_mime_type": DOCX_MIME_TYPE,
        "content_mime_detected": "application/zip",
        "content_digest": content_digest(payload),
        "content_truncated": None,
        "warc_filename": "crawl-data/test.warc.gz",
        "warc_record_offset": 42,
        "warc_record_length": 100,
        "warc_record_id": RECORD_ID,
    }


def _candidate(payload: bytes) -> CommonCrawlDocxCandidate:
    row = _index_row(payload)
    return CommonCrawlDocxCandidate(
        indexed_record=main_record_from_index_row(row, crawl_id=CRAWL_ID),
        index_kind=CommonCrawlIndexKind.MAIN,
        index_status=200,
        index_content_type=DOCX_MIME_TYPE,
        index_detected_type="application/zip",
        selection_reason=DocxSelectionReason.DECLARED_MIME,
        selected_by_declared_mime=True,
        selected_by_detected_mime=False,
        selected_by_url_suffix=True,
    )


def _warc_record(payload: bytes) -> CommonCrawlWarcRecord:
    return CommonCrawlWarcRecord(
        payload=payload,
        payload_digest=content_digest(payload),
        warc_record_id=RECORD_ID,
        target_url=URL,
        http_status=200,
        http_content_type=DOCX_MIME_TYPE,
        warc_date="2026-07-21T21:48:44Z",
        identified_payload_type="application/zip",
    )


@dataclass(frozen=True)
class _Extractor:
    text: str
    version: str = "test-extractor-v1"

    def extract(self, payload: bytes) -> ExtractedDocument:
        assert payload.startswith(b"PK")
        return ExtractedDocument(text=self.text, word_count=len(self.text.split()), table_count=2, image_count=1)


@dataclass(frozen=True)
class _LanguageDetector:
    detection: LanguageDetection
    version: str = "test-detector-v1"

    def detect(self, text: str) -> LanguageDetection:
        assert text
        return self.detection


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"fetch_status": 200, "content_mime_type": DOCX_MIME_TYPE, "url": "https://example.com/a"}, True),
        ({"fetch_status": 200, "content_mime_type": "application/octet-stream", "url": URL}, True),
        ({"fetch_status": 200, "content_mime_detected": DOCX_MIME_TYPE, "url": "https://example.com/a"}, True),
        ({"fetch_status": 404, "content_mime_type": DOCX_MIME_TYPE, "url": URL}, False),
        ({"fetch_status": 200, "content_mime_type": "text/html", "url": "https://example.com/a"}, False),
    ],
)
def test_is_docx_index_row_uses_status_mime_and_url(row: dict[str, object], expected: bool) -> None:
    assert is_docx_index_row(row) is expected


def test_docx_candidates_reads_main_index_partition(tmp_path: Path) -> None:
    payload = _docx_payload()
    parquet_path = tmp_path / "index.parquet"
    pq.write_table(pa.Table.from_pylist([_index_row(payload)]), parquet_path)

    candidates = list(
        docx_candidates(
            parquet_path.name,
            crawl_id=CRAWL_ID,
            index_kind=CommonCrawlIndexKind.MAIN,
            base_url=str(tmp_path),
            batch_rows=16,
        )
    )

    assert len(candidates) == 1
    assert candidates[0].indexed_record.record_range.offset == 42
    assert candidates[0].indexed_record.expectation.url == URL


def test_docx_candidates_reads_supplemental_index_without_record_id(tmp_path: Path) -> None:
    payload = _docx_payload()
    row = _index_row(payload)
    del row["warc_record_id"]
    parquet_path = tmp_path / "supplemental.parquet"
    pq.write_table(pa.Table.from_pylist([row]), parquet_path)

    candidates = list(
        docx_candidates(
            parquet_path.name,
            crawl_id="CC-SUPPLEMENTAL-2026-22",
            index_kind=CommonCrawlIndexKind.SUPPLEMENTAL,
            base_url=str(tmp_path),
            batch_rows=16,
        )
    )

    assert len(candidates) == 1
    assert isinstance(candidates[0].indexed_record, SupplementalIndexedRecord)


def test_validate_docx_accepts_required_office_members() -> None:
    validate_docx(_docx_payload(), maximum_entries=10, maximum_uncompressed_bytes=1024)


def test_docling_extractor_reads_real_docx_fixture() -> None:
    extracted = DoclingDocxExtractor().extract(_real_docx_payload("A small document extracted by Docling."))

    assert "Common Crawl DOCX" in extracted.text
    assert "A small document extracted by Docling." in extracted.text
    assert extracted.word_count > 0


def test_docling_extractor_preserves_table_text_without_markdown_padding() -> None:
    extracted = DoclingDocxExtractor().extract(_real_docx_with_table())

    assert "Quarter | Revenue" in extracted.text
    assert "Q1 | $42" in extracted.text
    assert "---" not in extracted.text
    assert extracted.table_count == 1


def test_lingua_detector_identifies_multilingual_fixture() -> None:
    french = "Ceci est un document français avec suffisamment de mots pour identifier correctement sa langue. " * 3

    detection = LinguaLanguageDetector().detect(french)

    assert detection.language == "fr"
    assert detection.confidence > 0.5


@pytest.mark.parametrize(
    "payload",
    [b"not a zip", _docx_payload(document=b"x")[:-8]],
)
def test_validate_docx_rejects_malformed_payload(payload: bytes) -> None:
    with pytest.raises(InvalidDocxError):
        validate_docx(payload, maximum_entries=10, maximum_uncompressed_bytes=1024)


def test_validate_docx_rejects_missing_required_member() -> None:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("[Content_Types].xml", b"<Types/>")

    with pytest.raises(InvalidDocxError):
        validate_docx(output.getvalue(), maximum_entries=10, maximum_uncompressed_bytes=1024)


def test_extracted_docx_record_preserves_provenance_language_and_metrics() -> None:
    payload = _docx_payload()
    output = extracted_docx_record(
        _candidate(payload),
        _warc_record(payload),
        extractor=_Extractor("Bonjour tout le monde. Ceci est un document français."),
        language_detector=_LanguageDetector(LanguageDetection("fr", 0.98)),
        maximum_zip_entries=10,
        maximum_uncompressed_bytes=1024,
    )

    assert output["text"] == "Bonjour tout le monde. Ceci est un document français."
    assert output["language"] == "fr"
    assert output["language_confidence"] == 0.98
    assert output["source_id"] == RECORD_ID
    assert output["crawl_id"] == CRAWL_ID
    assert output["warc_filename"] == "crawl-data/test.warc.gz"
    assert output["warc_record_offset"] == 42
    assert output["warc_record_length"] == 100
    assert output["warc_date"] == "2026-07-21T21:48:44Z"
    assert output["index_status"] == 200
    assert output["index_content_type"] == DOCX_MIME_TYPE
    assert output["index_detected_type"] == "application/zip"
    assert "id" not in output
    assert output["selection_reason"] == "declared_mime"
    assert output["table_count"] == 2
    assert output["image_count"] == 1


def test_extracted_docx_record_rejects_empty_extraction() -> None:
    payload = _docx_payload()

    with pytest.raises(DocxExtractionError):
        extracted_docx_record(
            _candidate(payload),
            _warc_record(payload),
            extractor=_Extractor("  \n"),
            language_detector=_LanguageDetector(LanguageDetection("unknown", 0.0)),
            maximum_zip_entries=10,
            maximum_uncompressed_bytes=1024,
        )


def test_docx_candidates_rejects_truncated_rows(tmp_path: Path) -> None:
    row = _index_row(_docx_payload())
    row["content_truncated"] = "length"
    parquet_path = tmp_path / "index.parquet"
    pq.write_table(pa.Table.from_pylist([row]), parquet_path)

    assert not list(
        docx_candidates(
            parquet_path.name,
            crawl_id=CRAWL_ID,
            index_kind=CommonCrawlIndexKind.MAIN,
            base_url=str(tmp_path),
            batch_rows=16,
        )
    )


def test_main_index_requires_record_id_column(tmp_path: Path) -> None:
    row = _index_row(_docx_payload())
    del row["warc_record_id"]
    parquet_path = tmp_path / "index.parquet"
    pq.write_table(pa.Table.from_pylist([row]), parquet_path)

    with pytest.raises(ValueError, match="warc_record_id"):
        list(
            docx_candidates(
                parquet_path.name,
                crawl_id=CRAWL_ID,
                index_kind=CommonCrawlIndexKind.MAIN,
                base_url=str(tmp_path),
                batch_rows=16,
            )
        )


def test_candidate_manifest_round_trip() -> None:
    candidate = _candidate(_docx_payload())

    restored = candidate_from_record(candidate_record(candidate))

    assert restored == candidate


@dataclass
class _Client:
    result: CommonCrawlWarcRecord | Exception

    def fetch_record(self, location: object) -> CommonCrawlWarcRecord:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@dataclass(frozen=True)
class _Outcome:
    counters: dict[str, int]


def _process(client: _Client, payload: bytes) -> dict[str, object] | None:
    return process_docx_candidate(
        client,
        _candidate(payload),
        extractor=_Extractor("some extracted words"),
        language_detector=_LanguageDetector(LanguageDetection("en", 0.9)),
        maximum_zip_entries=10,
        maximum_uncompressed_bytes=1024,
    )


def test_process_docx_candidate_propagates_provenance_mismatch() -> None:
    payload = _docx_payload()
    mismatched = _warc_record(payload)
    mismatched = CommonCrawlWarcRecord(**{**mismatched.__dict__, "target_url": "https://wrong.example/file.docx"})

    with pytest.raises(RecordVerificationError):
        _process(_Client(mismatched), payload)


def test_process_docx_candidate_propagates_transient_download() -> None:
    payload = _docx_payload()

    with pytest.raises(CommonCrawlDownloadError):
        _process(_Client(CommonCrawlDownloadError("temporary")), payload)


def test_process_docx_candidate_skips_revisit() -> None:
    payload = _docx_payload()

    assert _process(_Client(WarcRevisitError("revisit")), payload) is None


def test_validate_docx_enforces_entry_and_size_limits() -> None:
    payload = _docx_payload(document=b"x" * 100)

    with pytest.raises(InvalidDocxError, match="entries"):
        validate_docx(payload, maximum_entries=1, maximum_uncompressed_bytes=1024)
    with pytest.raises(InvalidDocxError, match="expands"):
        validate_docx(payload, maximum_entries=10, maximum_uncompressed_bytes=10)


def test_validate_docx_does_not_require_package_relationships() -> None:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("[Content_Types].xml", b"<Types/>")
        archive.writestr("word/document.xml", b"<w:document/>")

    validate_docx(output.getvalue(), maximum_entries=10, maximum_uncompressed_bytes=1024)


def test_lingua_returns_unknown_below_minimum_length() -> None:
    assert LinguaLanguageDetector(minimum_chars=10).detect("short") == LanguageDetection("unknown", 0.0)


def test_common_crawl_docx_steps_separates_operational_and_output_identity() -> None:
    source = CommonCrawlDocxSource(
        crawl_id=CRAWL_ID,
        index_kind=CommonCrawlIndexKind.MAIN,
        paths_manifest_url="https://example.com/index.paths.gz",
        index_batch_rows=7,
        extraction_shards=9,
        max_workers=11,
    )

    discovery, extraction, normalized = common_crawl_docx_steps(
        source,
        extractor=_Extractor("text", version="extractor-v7"),
        language_detector=_LanguageDetector(LanguageDetection("en", 1.0), version="detector-v3"),
    )

    assert discovery in extraction.deps
    assert extraction in normalized.deps
    assert "index_batch_rows" not in discovery.hash_attrs
    assert "extraction_shards" not in extraction.hash_attrs
    assert "max_workers" not in extraction.hash_attrs
    assert extraction.hash_attrs["extractor"] == "extractor-v7"
    assert extraction.hash_attrs["language_detector"] == "detector-v3"


def test_process_docx_shard_accepts_zephyr_shard_info() -> None:
    source = CommonCrawlDocxSource(
        crawl_id=CRAWL_ID,
        index_kind=CommonCrawlIndexKind.MAIN,
        paths_manifest_url="https://example.com/index.paths.gz",
    )

    assert (
        list(
            process_docx_shard(
                [],
                ShardInfo(shard_idx=0, total_shards=1),
                source=source,
                extractor=_Extractor("text"),
                language_detector=_LanguageDetector(LanguageDetection("en", 1.0)),
            )
        )
        == []
    )


def test_extraction_step_binds_step_runner_output_path_first(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = CommonCrawlDocxSource(
        crawl_id=CRAWL_ID,
        index_kind=CommonCrawlIndexKind.MAIN,
        paths_manifest_url="https://example.com/index.paths.gz",
    )
    _, extraction, _ = common_crawl_docx_steps(
        source,
        extractor=_Extractor("text"),
        language_detector=_LanguageDetector(LanguageDetection("en", 1.0)),
    )
    assert isinstance(extraction.fn, RemoteCallable)
    executed = False

    def record_execution(self: object, pipeline: object) -> _Outcome:
        nonlocal executed
        executed = True
        return _Outcome(counters={})

    monkeypatch.setattr("marin.datakit.download.common_crawl_docx.ZephyrContext.execute", record_execution)

    extraction.fn.fn(str(tmp_path))

    assert executed


def test_stratified_partition_slice_spans_manifest() -> None:
    partitions = tuple(f"part-{index}" for index in range(10))

    assert stratified_partition_slice(partitions, 4) == ("part-0", "part-3", "part-6", "part-9")


def test_sample_report_shows_per_reason_yield_and_bounded_examples() -> None:
    source = CommonCrawlDocxSource(
        crawl_id=CRAWL_ID,
        index_kind=CommonCrawlIndexKind.MAIN,
        paths_manifest_url="https://example.com/index.paths.gz",
    )
    candidates = [
        {"selection_reason": DocxSelectionReason.DECLARED_MIME.value},
        {"selection_reason": DocxSelectionReason.DECLARED_MIME.value},
        {"selection_reason": DocxSelectionReason.URL_SUFFIX.value},
    ]
    extracted = [
        {
            "selection_reason": DocxSelectionReason.DECLARED_MIME.value,
            "url": URL,
            "language": "en",
            "word_count": 4,
            "table_count": 1,
            "text": "A short extracted document.",
        }
    ]

    markdown, examples = sample_report_markdown(
        source=source,
        candidate_rows=candidates,
        extracted_rows=extracted,
        normalized_rows=extracted,
        extraction_counters={"common_crawl_docx/invalid_files": 2},
        examples_per_reason=1,
    )

    assert "| `declared_mime` | 2 | 1 | 50.0% |" in markdown
    assert "| `url_suffix` | 1 | 0 | 0.0% |" in markdown
    assert "`invalid_files`: 2" in markdown
    assert examples[0]["excerpt"] == "A short extracted document."
