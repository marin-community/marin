# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download and normalize the public English text from DocxCorpus."""

import hashlib
import logging
import re
from dataclasses import dataclass

import pyarrow as pa
from fray.types import ResourceConfig
from rigging.filesystem import prefix_join
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.writers import write_parquet_file

from marin.datakit.download.http_session import build_retrying_session
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec
from marin.utilities.validation_utils import write_provenance_json

logger = logging.getLogger(__name__)

DATASET_ID = "superdoc-dev/docx-corpus"
LANGUAGE = "en"
MANIFEST_URL = f"https://api.docxcorp.us/manifest?lang={LANGUAGE}"
MANIFEST_SHA256 = "34c7a3beae63d942a8d02dd3b49d4421acc1f351be5106b42262c192fc58fea9"
RAW_URL_PATTERN = re.compile(r"https://docxcorp\.us/documents/([0-9a-f]{64})\.docx")
EXTRACTED_URL_TEMPLATE = "https://docxcorp.us/extracted/{id}.txt"
SOURCE_VERSION = "2026-07-28"
USER_AGENT = "marin-docx-corpus-ingress/1.0 (https://github.com/marin-community/marin)"
HTTP_TIMEOUT = (10, 120)
HTTP_RETRY_TOTAL = 8
HTTP_BACKOFF_FACTOR = 2.0
DOWNLOAD_BATCH_SIZE = 128
DOWNLOAD_MAX_WORKERS = 4

DOCX_CORPUS_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("raw_url", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("language", pa.string(), nullable=False),
    ]
)


@dataclass(frozen=True)
class DocxCorpusTask:
    id: str
    raw_url: str
    extracted_url: str


@dataclass(frozen=True)
class DocxCorpusBatch:
    index: int
    total: int
    tasks: tuple[DocxCorpusTask, ...]
    manifest_rows: int = 0
    duplicate_ids: int = 0


def manifest_tasks(manifest: str, *, language: str = LANGUAGE) -> list[DocxCorpusTask]:
    """Validate and deduplicate a DocxCorpus manifest."""
    if language != LANGUAGE:
        raise ValueError(f"DocxCorpus source only supports language {LANGUAGE!r}, got {language!r}")

    tasks: list[DocxCorpusTask] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(manifest.splitlines(), start=1):
        raw_url = line.strip()
        if not raw_url:
            continue
        match = RAW_URL_PATTERN.fullmatch(raw_url)
        if match is None:
            raise ValueError(f"Invalid DocxCorpus manifest URL on line {line_number}: {raw_url!r}")
        document_id = match.group(1)
        if document_id in seen_ids:
            continue
        seen_ids.add(document_id)
        tasks.append(
            DocxCorpusTask(
                id=document_id,
                raw_url=raw_url,
                extracted_url=EXTRACTED_URL_TEMPLATE.format(id=document_id),
            )
        )
    return tasks


def _download_record(session, task: DocxCorpusTask) -> dict[str, str]:
    response = session.get(
        task.extracted_url,
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT,
    )
    try:
        response.raise_for_status()
        text = response.content.decode("utf-8")
    finally:
        response.close()
    if not text.strip():
        raise ValueError(f"Empty extracted text for DocxCorpus document {task.id}")
    counters.pipeline.update_counter("docx_corpus/successful_downloads", 1)
    return {
        "id": task.id,
        "text": text,
        "url": task.extracted_url,
        "raw_url": task.raw_url,
        "source": DATASET_ID,
        "language": LANGUAGE,
    }


def download_batch(batch: DocxCorpusBatch, output_path: str) -> dict:
    """Download one batch and write its records to a Parquet shard."""
    counters.pipeline.update_counter("docx_corpus/manifest_rows", batch.manifest_rows)
    counters.pipeline.update_counter("docx_corpus/unique_ids", len(batch.tasks))
    counters.pipeline.update_counter("docx_corpus/duplicate_ids", batch.duplicate_ids)

    session = build_retrying_session(total=HTTP_RETRY_TOTAL, backoff_factor=HTTP_BACKOFF_FACTOR)
    try:
        records = (_download_record(session, task) for task in batch.tasks)
        output_file = prefix_join(output_path, f"data-{batch.index:05d}-of-{batch.total:05d}.parquet")
        result = write_parquet_file(records, output_file, schema=DOCX_CORPUS_SCHEMA)
    finally:
        session.close()
    return {"batch": batch.index, **result}


def _fetch_manifest() -> str:
    session = build_retrying_session(total=HTTP_RETRY_TOTAL, backoff_factor=HTTP_BACKOFF_FACTOR)
    response = session.get(
        MANIFEST_URL,
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT,
    )
    try:
        response.raise_for_status()
        manifest = response.content
        digest = hashlib.sha256(manifest).hexdigest()
        if digest != MANIFEST_SHA256:
            raise ValueError(f"DocxCorpus manifest digest changed: expected {MANIFEST_SHA256}, got {digest}")
        return manifest.decode("utf-8")
    finally:
        response.close()
        session.close()


def _download_batches(tasks: list[DocxCorpusTask], manifest_rows: int) -> list[DocxCorpusBatch]:
    total = (len(tasks) + DOWNLOAD_BATCH_SIZE - 1) // DOWNLOAD_BATCH_SIZE
    duplicate_ids = manifest_rows - len(tasks)
    return [
        DocxCorpusBatch(
            index=index,
            total=total,
            tasks=tuple(tasks[offset : offset + DOWNLOAD_BATCH_SIZE]),
            manifest_rows=manifest_rows if index == 0 else 0,
            duplicate_ids=duplicate_ids if index == 0 else 0,
        )
        for index, offset in enumerate(range(0, len(tasks), DOWNLOAD_BATCH_SIZE))
    ]


def download_docx_corpus(output_path: str) -> None:
    """Download the live English DocxCorpus manifest and extracted text."""
    manifest = _fetch_manifest()
    manifest_rows = sum(bool(line.strip()) for line in manifest.splitlines())
    tasks = manifest_tasks(manifest)
    if not tasks:
        raise ValueError(f"No English documents found in {MANIFEST_URL}")

    batches = _download_batches(tasks, manifest_rows)
    logger.info(
        "Downloading %d unique DocxCorpus documents from %d manifest rows in %d batches",
        len(tasks),
        manifest_rows,
        len(batches),
    )
    pipeline = (
        Dataset.from_list(batches)
        .map(lambda batch: download_batch(batch, output_path))
        .write_jsonl(
            prefix_join(output_path, ".metrics/download-{shard:05d}-of-{total:05d}.jsonl"),
            skip_existing=True,
        )
    )
    context = ZephyrContext(
        name="download-docx-corpus-en",
        max_workers=DOWNLOAD_MAX_WORKERS,
        resources=ResourceConfig(cpu=1, ram="4g", disk="5g"),
    )
    context.execute(pipeline)

    write_provenance_json(
        output_path,
        metadata={
            "dataset": DATASET_ID,
            "version": SOURCE_VERSION,
            "language": LANGUAGE,
            "manifest_sha256": MANIFEST_SHA256,
            "manifest_rows": manifest_rows,
            "unique_ids": len(tasks),
            "links": [MANIFEST_URL, "https://docxcorp.us/download"],
        },
    )


def docx_corpus_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the download and normalize chain for English DocxCorpus."""
    download = StepSpec(
        name="raw/docx-corpus/en",
        fn=download_docx_corpus,
        hash_attrs={
            "dataset": DATASET_ID,
            "language": LANGUAGE,
            "manifest_url": MANIFEST_URL,
            "manifest_sha256": MANIFEST_SHA256,
            "source_version": SOURCE_VERSION,
            "download_batch_size": DOWNLOAD_BATCH_SIZE,
            "download_max_workers": DOWNLOAD_MAX_WORKERS,
        },
    )
    return (
        download,
        normalize_step(
            name="normalized/docx-corpus/en",
            download=download,
            id_field="id",
            file_extensions=(".parquet",),
        ),
    )
