# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""yoonholee/olympiad-books-open-source download + cleanup helpers.

The upstream dataset is a small chunked collection of math textbooks. This
source keeps the CC BY / CC BY-SA reviewed book subset, repairs the observed
row-local duplicated-half artifact, removes exact duplicate text after cleanup,
and emits Dolma-shaped document rows for Datakit normalization.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem import url_to_fs
from zephyr import counters

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "yoonholee/olympiad-books-open-source"
HF_REVISION = "016c098"
TRAIN_PARQUET_GLOB = "data/train-00000-of-00001.parquet"
OLYMPIAD_BOOKS_OPEN_SOURCE_CC_BY_SA_REVIEWED_ROUGH_TOKENS_B = 0.002

PROCESSED_NAME = "processed/olympiad-books-open-source/cc-by-sa-reviewed"
NORMALIZED_NAME = "normalized/olympiad-books-open-source/cc-by-sa-reviewed"

BOOK_LICENSES = {
    "napkin": "CC BY-SA 4.0 text; GPL v3 source files",
    "mathematical-reasoning": "CC BY-NC-SA 3.0",
    "exploring-combinatorial-math": "GFDL",
    "discrete-mathematics": "CC BY-SA 4.0",
    "aata": "GFDL",
    "applied-combinatorics": "CC BY-SA 4.0",
    "bogart": "GFDL",
    "fcla": "GFDL",
    "ent": "Free under Springer agreement",
    "ra": "CC BY-NC-SA 4.0 + CC BY-SA 4.0",
    "ibl-intro-proof": "CC BY-SA 4.0",
    "openlogic": "CC BY 4.0",
}

CC_BY_SA_REVIEWED_BOOK_KEYS = frozenset(
    {
        "napkin",
        "discrete-mathematics",
        "applied-combinatorics",
        "ibl-intro-proof",
        "openlogic",
    }
)

REPEATED_HALF_SEARCH_RADIUS = 8


def _clean_required_text(row: dict, key: str) -> str | None:
    value = row.get(key)
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    return text


def _optional_text(row: dict, key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        return ""

    return value.strip()


def _optional_int(row: dict, key: str) -> int | None:
    value = row.get(key)
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalized_text_for_hash(text: str) -> str:
    return " ".join(text.split())


def _text_hash(text: str) -> str:
    return hashlib.sha256(_normalized_text_for_hash(text).encode("utf-8")).hexdigest()


def _collapse_repeated_half(text: str) -> tuple[str, bool]:
    stripped = text.strip()
    midpoint = len(stripped) // 2
    for delta in range(-REPEATED_HALF_SEARCH_RADIUS, REPEATED_HALF_SEARCH_RADIUS + 1):
        split = midpoint + delta
        if split <= 0 or split >= len(stripped):
            continue

        left = stripped[:split].strip()
        right = stripped[split:].strip()
        if not left:
            continue
        if _normalized_text_for_hash(left) == _normalized_text_for_hash(right):
            return left, True

    return stripped, False


def row_to_doc(row: dict[str, Any]) -> list[dict[str, Any]]:
    text = _clean_required_text(row, "text")
    if text is None:
        counters.increment("olympiad_books_open_source/dropped_empty")
        return []

    book_key = _optional_text(row, "book_key")
    if book_key not in BOOK_LICENSES:
        counters.increment("olympiad_books_open_source/dropped_unknown_book")
        return []

    if book_key not in CC_BY_SA_REVIEWED_BOOK_KEYS:
        counters.increment("olympiad_books_open_source/dropped_license")
        return []

    text, collapsed = _collapse_repeated_half(text)
    if collapsed:
        counters.increment("olympiad_books_open_source/collapsed_repeated_half")

    text_hash = _text_hash(text)
    source_file = _optional_text(row, "source_file")
    chunk_id = _optional_int(row, "chunk_id")
    doc_id = hashlib.sha256(f"{book_key}:{source_file}:{chunk_id}:{text_hash}".encode()).hexdigest()

    return [
        {
            "id": doc_id,
            "text": text,
            "source": HF_DATASET_ID,
            "source_revision": HF_REVISION,
            "book": _optional_text(row, "book"),
            "book_key": book_key,
            "book_license": BOOK_LICENSES[book_key],
            "license_view": "cc-by-sa-reviewed",
            "subject": _optional_text(row, "subject"),
            "level": _optional_text(row, "level"),
            "part": _optional_text(row, "part"),
            "chapter": _optional_text(row, "chapter"),
            "section": _optional_text(row, "section"),
            "source_file": source_file,
            "chunk_id": chunk_id,
            "tokens_est": _optional_int(row, "tokens_est"),
            "text_hash": text_hash,
        }
    ]


def transform(input_path: str, output_path: str) -> None:
    input_fs, input_base = url_to_fs(input_path)
    parquet_paths = sorted(input_fs.glob(f"{input_base}/data/*.parquet"))

    seen_text_hashes: set[str] = set()
    docs: list[dict[str, Any]] = []
    for parquet_path in parquet_paths:
        with input_fs.open(parquet_path, "rb") as input_file:
            for row in pq.read_table(input_file).to_pylist():
                for doc in row_to_doc(row):
                    text_hash = doc["text_hash"]
                    if text_hash in seen_text_hashes:
                        counters.increment("olympiad_books_open_source/dropped_exact_duplicate")
                        continue
                    seen_text_hashes.add(text_hash)
                    counters.increment("olympiad_books_open_source/kept")
                    docs.append(doc)

    output_fs, output_base = url_to_fs(output_path)
    output_fs.makedirs(output_base, exist_ok=True)
    with output_fs.open(f"{output_base}/data-00000-of-00001.parquet", "wb") as output_file:
        pq.write_table(pa.Table.from_pylist(docs), output_file)


def download_olympiad_books_open_source_step() -> StepSpec:
    """Download and transform the reviewed Olympiad Books subset into documents."""
    dl = download_hf_step(
        "raw/olympiad-books-open-source",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[TRAIN_PARQUET_GLOB],
        override_output_path=f"raw/olympiad-books-open-source-{HF_REVISION}",
    )

    return StepSpec(
        name=PROCESSED_NAME,
        deps=[dl],
        fn=lambda output_path: transform(input_path=dl.output_path, output_path=output_path),
        hash_attrs={
            "version": "v1",
            "license_view": "cc-by-sa-reviewed",
            "book_keys": sorted(CC_BY_SA_REVIEWED_BOOK_KEYS),
            "drop_exact_duplicates": True,
            "collapse_repeated_half_rows": True,
        },
    )


def olympiad_books_open_source_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the ``(download+transform, normalize)`` chain for Olympiad Books."""
    processed = download_olympiad_books_open_source_step()
    return (
        processed,
        normalize_step(name=NORMALIZED_NAME, download=processed),
    )
