# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lightweight audit tooling for Longmino and FinePDF long-context corpora."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import fsspec
import pyarrow.parquet as pq
from levanter.data.sharded_datasource import JsonlDataSource, ParquetDataSource
from levanter.tokenizers import load_tokenizer

from experiments.long_context_datasets.finepdfs import (
    finepdfs_by_language,
    finepdfs_edu_by_language,
)
from experiments.long_context_datasets.longmino import longmino_by_bucket
from experiments.marin_models import marin_tokenizer
from marin.execution.executor import InputName
from rigging.filesystem import marin_prefix, open_url

TextFormat = Literal["jsonl", "parquet"]

_LINE_END_RE = re.compile(r"[.!?:;)\]\"']$")
_PAGE_LINE_RE = re.compile(r"^(page\s+\d+|\d+)$", re.IGNORECASE)
_HEADER_FOOTER_RE = re.compile(r"^[A-Z0-9][A-Z0-9 .,:;_/-]{10,}$")


@dataclass(frozen=True)
class AuditSource:
    """Configuration for one audited corpus or bucket."""

    key: str
    raw_path: str
    format: TextFormat
    text_paths: tuple[tuple[str, ...], ...]
    id_paths: tuple[tuple[str, ...], ...] = (("id",), ("doc_id",), ("metadata", "id"))
    language_paths: tuple[tuple[str, ...], ...] = (
        ("language",),
        ("metadata", "language"),
        ("metadata", "lang"),
    )
    token_count_paths: tuple[tuple[str, ...], ...] = (
        ("token_count",),
        ("metadata", "token_count"),
    )
    tokenized_cache_glob: str | None = None

    def absolute_raw_path(self, prefix: str) -> str:
        return _join_path(prefix, self.raw_path)

    def absolute_cache_glob(self, prefix: str) -> str | None:
        if self.tokenized_cache_glob is None:
            return None
        return _join_path(prefix, self.tokenized_cache_glob)


@dataclass(frozen=True)
class ExtractedDocument:
    """Minimal normalized view of one raw record."""

    doc_id: str | None
    text: str
    token_count: int | None
    language: str | None


@dataclass(frozen=True)
class TextQuality:
    """Cheap quality signals computed from document text."""

    char_count: int
    line_count: int
    paragraph_count: int
    whitespace_ratio: float
    non_alnum_ratio: float
    repeat_line_ratio: float
    repeat_paragraph_ratio: float
    broken_line_ratio: float
    boilerplate_line_ratio: float
    flags: tuple[str, ...]


@dataclass(frozen=True)
class ExactSourceStats:
    """Exact or near-exact source totals discovered outside the sample."""

    total_documents: int | None
    total_tokens: int | None
    stats_source: str


@dataclass(frozen=True)
class SourceSummary:
    """Summary emitted for one source."""

    source_key: str
    sample_documents: int
    total_documents: int | None
    total_tokens: int | None
    total_documents_source: str | None
    total_tokens_source: str | None
    char_count_p50: int | None
    char_count_p90: int | None
    char_count_p99: int | None
    token_count_p50: int | None
    token_count_p90: int | None
    token_count_p99: int | None
    empty_rate: float
    repetition_rate: float
    ocr_noise_rate: float
    formatting_noise_rate: float
    language_counts: dict[str, int]


@dataclass(frozen=True)
class ReviewExample:
    """One record for human review."""

    source_key: str
    doc_id: str | None
    token_count: int | None
    char_count: int
    language: str | None
    flags: tuple[str, ...]
    first_text: str
    last_text: str


def build_default_sources() -> list[AuditSource]:
    """Build audit specs from the existing long-context dataset definitions."""
    sources = [
        AuditSource(
            key=f"longmino/{bucket}",
            raw_path=_relative_path(longmino_by_bucket[bucket]),
            format="jsonl",
            text_paths=(("text",),),
            tokenized_cache_glob=f"tokenized/longmino_{bucket}_llama3*",
        )
        for bucket in longmino_by_bucket
    ]
    sources.extend(
        [
            AuditSource(
                key="finepdfs/eng_Latn",
                raw_path=_relative_path(finepdfs_by_language["eng_Latn"]),
                format="parquet",
                text_paths=(("text",), ("content",)),
                tokenized_cache_glob="tokenized/finepdfs_eng_Latn_llama3*",
            ),
            AuditSource(
                key="finepdfs-edu/eng_Latn",
                raw_path=_relative_path(finepdfs_edu_by_language["eng_Latn"]),
                format="parquet",
                text_paths=(("text",), ("content",)),
                tokenized_cache_glob="tokenized/finepdfs_edu_eng_Latn_llama3*",
            ),
        ]
    )
    return sources


def extract_document_fields(record: dict[str, Any], source: AuditSource) -> ExtractedDocument:
    """Normalize one raw record into the audit schema."""
    text = _pick_first_string(record, source.text_paths) or ""
    doc_id = _pick_first_string(record, source.id_paths)
    language = _pick_first_string(record, source.language_paths)
    token_count = _pick_first_int(record, source.token_count_paths)
    return ExtractedDocument(doc_id=doc_id, text=text, token_count=token_count, language=language)


def compute_text_quality(text: str) -> TextQuality:
    """Compute lightweight repetition and extraction-noise signals."""
    stripped = text.strip()
    if not stripped:
        return TextQuality(
            char_count=0,
            line_count=0,
            paragraph_count=0,
            whitespace_ratio=0.0,
            non_alnum_ratio=0.0,
            repeat_line_ratio=0.0,
            repeat_paragraph_ratio=0.0,
            broken_line_ratio=0.0,
            boilerplate_line_ratio=0.0,
            flags=("empty",),
        )

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", stripped) if paragraph.strip()]
    char_count = len(stripped)
    whitespace_ratio = sum(char.isspace() for char in stripped) / char_count
    non_alnum_ratio = sum((not char.isalnum()) and (not char.isspace()) for char in stripped) / char_count
    repeat_line_ratio, boilerplate_line_ratio = _repeat_ratio(lines)
    repeat_paragraph_ratio, _ = _repeat_ratio(paragraphs)
    broken_line_ratio = _broken_line_ratio(lines)

    flags: list[str] = []
    if repeat_line_ratio >= 0.2 or repeat_paragraph_ratio >= 0.15 or boilerplate_line_ratio >= 0.15:
        flags.append("repetition")
    if broken_line_ratio >= 0.2 and (non_alnum_ratio >= 0.03 or boilerplate_line_ratio >= 0.1):
        flags.append("ocr_noise")
    if whitespace_ratio >= 0.35:
        flags.append("formatting_noise")

    return TextQuality(
        char_count=char_count,
        line_count=len(lines),
        paragraph_count=len(paragraphs),
        whitespace_ratio=whitespace_ratio,
        non_alnum_ratio=non_alnum_ratio,
        repeat_line_ratio=repeat_line_ratio,
        repeat_paragraph_ratio=repeat_paragraph_ratio,
        broken_line_ratio=broken_line_ratio,
        boilerplate_line_ratio=boilerplate_line_ratio,
        flags=tuple(flags),
    )


def summarize_source(
    source: AuditSource,
    documents: Sequence[ExtractedDocument],
    exact_stats: ExactSourceStats | None = None,
) -> SourceSummary:
    """Aggregate sampled documents plus any cache-derived totals."""
    qualities = [compute_text_quality(document.text) for document in documents]
    char_counts = [quality.char_count for quality in qualities if quality.char_count > 0]
    token_counts = [document.token_count for document in documents if document.token_count is not None]
    language_counts = Counter(document.language for document in documents if document.language)
    sample_count = len(documents)

    return SourceSummary(
        source_key=source.key,
        sample_documents=sample_count,
        total_documents=exact_stats.total_documents if exact_stats else None,
        total_tokens=exact_stats.total_tokens if exact_stats else None,
        total_documents_source=(
            exact_stats.stats_source if exact_stats and exact_stats.total_documents is not None else None
        ),
        total_tokens_source=(exact_stats.stats_source if exact_stats and exact_stats.total_tokens is not None else None),
        char_count_p50=_quantile_int(char_counts, 0.5),
        char_count_p90=_quantile_int(char_counts, 0.9),
        char_count_p99=_quantile_int(char_counts, 0.99),
        token_count_p50=_quantile_int(token_counts, 0.5),
        token_count_p90=_quantile_int(token_counts, 0.9),
        token_count_p99=_quantile_int(token_counts, 0.99),
        empty_rate=_flag_rate(qualities, "empty"),
        repetition_rate=_flag_rate(qualities, "repetition"),
        ocr_noise_rate=_flag_rate(qualities, "ocr_noise"),
        formatting_noise_rate=_flag_rate(qualities, "formatting_noise"),
        language_counts=dict(language_counts),
    )


def run_audit(
    *,
    output_dir: str,
    sample_size: int,
    review_size: int,
    max_shards: int,
    seed: int,
    prefix: str | None = None,
    source_keys: Sequence[str] | None = None,
) -> list[SourceSummary]:
    """Run the audit and write JSON, Markdown, and review-sample outputs."""
    resolved_prefix = (prefix or marin_prefix()).rstrip("/")
    sources = build_default_sources()
    if source_keys:
        key_set = set(source_keys)
        sources = [source for source in sources if source.key in key_set]

    _ensure_dir(output_dir)
    tokenizer = _TokenCounter()

    summaries: list[SourceSummary] = []
    review_examples: list[ReviewExample] = []
    for source in sources:
        exact_stats = discover_exact_stats(source, prefix=resolved_prefix)
        sampled_documents = sample_documents(
            source,
            prefix=resolved_prefix,
            sample_size=sample_size,
            max_shards=max_shards,
            seed=seed,
        )
        sampled_documents = [
            ExtractedDocument(
                doc_id=document.doc_id,
                text=document.text,
                token_count=document.token_count if document.token_count is not None else tokenizer.count(document.text),
                language=document.language,
            )
            for document in sampled_documents
        ]
        summaries.append(summarize_source(source, sampled_documents, exact_stats=exact_stats))
        review_examples.extend(select_review_examples(source.key, sampled_documents, review_size))

    _write_json(_join_path(output_dir, "summary.json"), [asdict(summary) for summary in summaries])
    _write_text(_join_path(output_dir, "summary.md"), render_markdown_summary(summaries))
    _write_jsonl(_join_path(output_dir, "review_sample.jsonl"), review_examples)
    return summaries


def discover_exact_stats(source: AuditSource, *, prefix: str) -> ExactSourceStats | None:
    """Discover exact source totals from cache stats or cheap parquet metadata."""
    cache_stats = _discover_cache_stats(source, prefix=prefix)
    parquet_docs = _count_parquet_rows(source, prefix=prefix) if source.format == "parquet" else None

    if cache_stats is None and parquet_docs is None:
        return None

    if cache_stats is None:
        return ExactSourceStats(total_documents=parquet_docs, total_tokens=None, stats_source="parquet_metadata")

    total_documents = cache_stats.total_documents
    stats_source = cache_stats.stats_source
    if total_documents is None and parquet_docs is not None:
        total_documents = parquet_docs
        stats_source = "cache_and_parquet_metadata"

    return ExactSourceStats(
        total_documents=total_documents,
        total_tokens=cache_stats.total_tokens,
        stats_source=stats_source,
    )


def sample_documents(
    source: AuditSource,
    *,
    prefix: str,
    sample_size: int,
    max_shards: int,
    seed: int,
) -> list[ExtractedDocument]:
    """Read a bounded, deterministic sample across shard files."""
    if sample_size <= 0:
        return []

    data_source = _build_data_source(source, prefix=prefix)
    shard_names = list(data_source.shard_names)
    if not shard_names:
        return []

    shard_count = min(max_shards, len(shard_names))
    rng = random.Random(f"{seed}:{source.key}")
    rng.shuffle(shard_names)
    chosen_shards = shard_names[:shard_count]
    docs_per_shard = max(1, math.ceil(sample_size / shard_count))

    documents: list[ExtractedDocument] = []
    for shard_name in chosen_shards:
        per_shard = 0
        for record in data_source.open_shard_at_row(shard_name, 0):
            documents.append(extract_document_fields(record, source))
            per_shard += 1
            if per_shard >= docs_per_shard or len(documents) >= sample_size:
                break
        if len(documents) >= sample_size:
            break

    return documents


def select_review_examples(
    source_key: str,
    documents: Sequence[ExtractedDocument],
    review_size: int,
) -> list[ReviewExample]:
    """Select evenly spaced documents by token length for human review."""
    if review_size <= 0 or not documents:
        return []

    indexed_documents = sorted(
        enumerate(documents),
        key=lambda item: (item[1].token_count or len(item[1].text), item[0]),
    )
    if review_size >= len(indexed_documents):
        chosen = [document for _, document in indexed_documents]
    else:
        chosen = []
        last_index = len(indexed_documents) - 1
        for offset in range(review_size):
            position = round(offset * last_index / max(review_size - 1, 1))
            chosen.append(indexed_documents[position][1])

    return [
        ReviewExample(
            source_key=source_key,
            doc_id=document.doc_id,
            token_count=document.token_count,
            char_count=len(document.text.strip()),
            language=document.language,
            flags=compute_text_quality(document.text).flags,
            first_text=document.text[:2000],
            last_text=document.text[-1000:],
        )
        for document in chosen
    ]


def render_markdown_summary(summaries: Sequence[SourceSummary]) -> str:
    """Render a terse Markdown table for the issue follow-up."""
    lines = [
        "| source | sample_docs | total_docs | total_tokens | token_p50 | token_p90 | repetition | ocr_noise |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        total_docs = _format_optional_int(summary.total_documents)
        total_tokens = _format_optional_int(summary.total_tokens)
        token_p50 = _format_optional_int(summary.token_count_p50)
        token_p90 = _format_optional_int(summary.token_count_p90)
        lines.append(
            "| "
            f"{summary.source_key} | {summary.sample_documents} | {total_docs} | {total_tokens} | "
            f"{token_p50} | {token_p90} | {summary.repetition_rate:.3f} | {summary.ocr_noise_rate:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", required=True, help="Directory for summary.json, summary.md, review_sample.jsonl"
    )
    parser.add_argument("--sample-size", type=int, default=2048, help="Number of raw documents to sample per source")
    parser.add_argument("--review-size", type=int, default=100, help="Number of review examples per source")
    parser.add_argument("--max-shards", type=int, default=64, help="Maximum shard files to sample per source")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed")
    parser.add_argument("--prefix", default=None, help="Optional Marin storage prefix override")
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Optional subset of source keys, for example longmino/8k-16k finepdfs-edu/eng_Latn",
    )
    args = parser.parse_args()
    run_audit(
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        review_size=args.review_size,
        max_shards=args.max_shards,
        seed=args.seed,
        prefix=args.prefix,
        source_keys=args.sources,
    )


def _relative_path(input_name: InputName) -> str:
    step = input_name.step
    if step is None or step.override_output_path is None or input_name.name is None:
        raise ValueError(f"Expected a relative dataset path, got {input_name!r}")
    return os.path.join(step.override_output_path, input_name.name)


def _pick_first_string(record: dict[str, Any], paths: Sequence[tuple[str, ...]]) -> str | None:
    for path in paths:
        value = _nested_value(record, path)
        if isinstance(value, str) and value:
            return value
    return None


def _pick_first_int(record: dict[str, Any], paths: Sequence[tuple[str, ...]]) -> int | None:
    for path in paths:
        value = _nested_value(record, path)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _nested_value(record: dict[str, Any], path: Sequence[str]) -> Any:
    current: Any = record
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _repeat_ratio(chunks: Sequence[str]) -> tuple[float, float]:
    if not chunks:
        return 0.0, 0.0

    normalized = [re.sub(r"\s+", " ", chunk.strip()) for chunk in chunks if chunk.strip()]
    if not normalized:
        return 0.0, 0.0

    counts = Counter(normalized)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    boilerplate = 0
    for chunk, count in counts.items():
        if count < 2:
            continue
        if _PAGE_LINE_RE.match(chunk) or _HEADER_FOOTER_RE.match(chunk):
            boilerplate += count
    total = len(normalized)
    return repeated / total, boilerplate / total


def _broken_line_ratio(lines: Sequence[str]) -> float:
    if not lines:
        return 0.0

    broken = 0
    candidates = 0
    for line in lines:
        if len(line) < 12 or _PAGE_LINE_RE.match(line):
            continue
        candidates += 1
        if not _LINE_END_RE.search(line):
            broken += 1
    if candidates == 0:
        return 0.0
    return broken / candidates


def _quantile_int(values: Sequence[int], quantile: float) -> int | None:
    if not values:
        return None
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return int(sorted_values[lower])
    weight = position - lower
    interpolated = sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
    return round(interpolated)


def _flag_rate(qualities: Sequence[TextQuality], flag: str) -> float:
    if not qualities:
        return 0.0
    return sum(flag in quality.flags for quality in qualities) / len(qualities)


def _build_data_source(source: AuditSource, *, prefix: str):
    absolute_path = source.absolute_raw_path(prefix)
    if source.format == "jsonl":
        return JsonlDataSource([absolute_path])
    if source.format == "parquet":
        return ParquetDataSource([absolute_path])
    raise ValueError(f"Unsupported source format {source.format!r}")


def _discover_cache_stats(source: AuditSource, *, prefix: str) -> ExactSourceStats | None:
    cache_glob = source.absolute_cache_glob(prefix)
    if cache_glob is None:
        return None

    matches = _glob(cache_glob)
    best_total_tokens = -1
    best_documents: int | None = None
    for match in matches:
        stats_path = _join_path(match, "train/.stats.json")
        try:
            with open_url(stats_path, "r") as f:
                stats = json.load(f)
        except FileNotFoundError:
            continue
        total_tokens = int(stats.get("total_tokens", 0))
        total_documents = stats.get("total_elements")
        if total_tokens > best_total_tokens:
            best_total_tokens = total_tokens
            best_documents = int(total_documents) if total_documents is not None else None

    if best_total_tokens < 0:
        return None

    return ExactSourceStats(
        total_documents=best_documents,
        total_tokens=best_total_tokens,
        stats_source="tokenized_cache",
    )


def _count_parquet_rows(source: AuditSource, *, prefix: str) -> int | None:
    if source.format != "parquet":
        return None
    total_rows = 0
    for parquet_path in _glob(source.absolute_raw_path(prefix)):
        with open_url(parquet_path, "rb") as f:
            total_rows += pq.ParquetFile(f).metadata.num_rows
    return total_rows


def _glob(path_pattern: str) -> list[str]:
    fs, fs_path = fsspec.core.url_to_fs(path_pattern)
    matches = sorted(fs.glob(fs_path))
    if "://" not in path_pattern:
        return matches
    scheme = path_pattern.split("://", 1)[0]
    return [f"{scheme}://{match}" for match in matches]


def _ensure_dir(path: str) -> None:
    fs, fs_path = fsspec.core.url_to_fs(path)
    fs.makedirs(fs_path, exist_ok=True)


def _write_json(path: str, payload: Any) -> None:
    with open_url(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_jsonl(path: str, rows: Iterable[ReviewExample]) -> None:
    with open_url(path, "w") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), sort_keys=True))
            f.write("\n")


def _write_text(path: str, text: str) -> None:
    with open_url(path, "w") as f:
        f.write(text)


def _join_path(prefix: str, suffix: str) -> str:
    return f"{prefix.rstrip('/')}/{suffix.lstrip('/')}"


def _format_optional_int(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,}"


class _TokenCounter:
    """Lazy tokenizer loader for sample-only token counting."""

    def __init__(self):
        self._tokenizer = None

    def count(self, text: str) -> int | None:
        stripped = text.strip()
        if not stripped:
            return None
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(marin_tokenizer)
        return len(self._tokenizer.encode(stripped))


if __name__ == "__main__":
    main()
