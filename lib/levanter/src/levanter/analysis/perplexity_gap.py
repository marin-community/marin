# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import heapq
import itertools
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import fsspec
import numpy as np
import regex
from rigging.filesystem import open_url

from levanter.data.sharded_datasource import ShardedDataSource
from levanter.data.text import DatasetComponent, TextLmDatasetFormat
from levanter.tokenizers import MarinTokenizer, _safe_split_for_tokenizer


LOG2E = float(np.log2(np.e))
VISIBLE_WHITESPACE = str.maketrans(
    {
        " ": "\u2420",
        "\n": "\u23ce",
        "\t": "\u21e5",
        "\r": "\u21b5",
    }
)

_URL_RE = regex.compile(r"(?:https?://|www\.)\S+")
_NUMBER_RE = regex.compile(r"\p{N}+(?:[.,:/-]\p{N}+)*")
_WORD_RE = regex.compile(r"[\p{L}\p{M}_]+(?:['\u2019-][\p{L}\p{M}_]+)*")
_SEGMENT_RE = regex.compile(
    r"(?:https?://|www\.)\S+|\s+|\p{N}+(?:[.,:/-]\p{N}+)*|"
    r"[\p{L}\p{M}_]+(?:['\u2019-][\p{L}\p{M}_]+)*|[^\p{L}\p{M}\p{N}\s]+"
)


@dataclass(frozen=True)
class RawTextDocument:
    dataset_name: str
    tags: tuple[str, ...]
    shard_name: str
    row_index: int
    text: str


@dataclass(frozen=True)
class TokenizedDocument:
    token_ids: np.ndarray
    byte_starts: np.ndarray
    byte_ends: np.ndarray
    num_bytes: int


@dataclass(frozen=True)
class TokenizedChunk:
    doc_index: int
    token_ids: np.ndarray
    byte_starts: np.ndarray
    byte_ends: np.ndarray


@dataclass(frozen=True)
class LiteralExample:
    abs_delta_bits: float
    dataset_name: str
    doc_preview: str
    model_a_token_boundaries: str
    model_b_token_boundaries: str


@dataclass
class GapAggregate:
    total_loss_a: float = 0.0
    total_loss_b: float = 0.0
    total_bytes: int = 0
    count: int = 0

    def add(self, *, loss_a: float, loss_b: float, num_bytes: int, count: int = 1) -> None:
        self.total_loss_a += loss_a
        self.total_loss_b += loss_b
        self.total_bytes += int(num_bytes)
        self.count += int(count)

    def as_dict(self, name: str) -> dict[str, Any]:
        if self.total_bytes <= 0:
            model_a_bpb = None
            model_b_bpb = None
            gap_bpb = None
        else:
            model_a_bpb = self.total_loss_a * LOG2E / self.total_bytes
            model_b_bpb = self.total_loss_b * LOG2E / self.total_bytes
            gap_bpb = (self.total_loss_a - self.total_loss_b) * LOG2E / self.total_bytes

        return {
            "name": name,
            "documents": int(self.count),
            "bytes": int(self.total_bytes),
            "model_a_bpb": model_a_bpb,
            "model_b_bpb": model_b_bpb,
            "gap_bpb": gap_bpb,
            "delta_bits": (self.total_loss_a - self.total_loss_b) * LOG2E,
        }


@dataclass
class GapReportBuilder:
    model_a_name: str
    model_b_name: str
    output_path: str
    top_k_docs: int = 25
    top_k_segments: int = 40
    top_k_literals: int = 40
    dataset_stats: dict[str, GapAggregate] = field(default_factory=lambda: defaultdict(GapAggregate))
    bucket_stats: dict[str, GapAggregate] = field(default_factory=lambda: defaultdict(GapAggregate))
    literal_stats: dict[tuple[str, str], GapAggregate] = field(default_factory=lambda: defaultdict(GapAggregate))
    literal_examples: dict[tuple[str, str], LiteralExample] = field(default_factory=dict)
    group_to_leaves: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    _top_docs_positive: list[tuple[float, int, dict[str, Any]]] = field(default_factory=list)
    _top_docs_negative: list[tuple[float, int, dict[str, Any]]] = field(default_factory=list)
    _top_segments_positive: list[tuple[float, int, dict[str, Any]]] = field(default_factory=list)
    _top_segments_negative: list[tuple[float, int, dict[str, Any]]] = field(default_factory=list)
    _heap_counter: itertools.count = field(default_factory=itertools.count)

    def register_dataset(self, dataset_name: str, tags: Sequence[str]) -> None:
        self.group_to_leaves[dataset_name].add(dataset_name)
        for tag in tags:
            self.group_to_leaves[tag].add(dataset_name)
            self._register_hierarchy(tag, dataset_name)
        self._register_hierarchy(dataset_name, dataset_name)

    def add_document(
        self,
        *,
        document: RawTextDocument,
        per_byte_loss_a: np.ndarray,
        per_byte_loss_b: np.ndarray,
        tokenized_a: TokenizedDocument | None = None,
        tokenized_b: TokenizedDocument | None = None,
    ) -> None:
        self.register_dataset(document.dataset_name, document.tags)

        num_bytes = len(per_byte_loss_a)
        loss_a = float(per_byte_loss_a.sum())
        loss_b = float(per_byte_loss_b.sum())
        self.dataset_stats[document.dataset_name].add(loss_a=loss_a, loss_b=loss_b, num_bytes=num_bytes)

        delta_bits = (loss_a - loss_b) * LOG2E
        gap_bpb = delta_bits / num_bytes if num_bytes > 0 else 0.0

        if num_bytes <= 0:
            return

        prefix_a = np.concatenate(([0.0], np.cumsum(per_byte_loss_a, dtype=np.float64)))
        prefix_b = np.concatenate(([0.0], np.cumsum(per_byte_loss_b, dtype=np.float64)))
        byte_offsets = char_to_byte_offsets(document.text)
        worst_positive_segment: tuple[float, int, int] | None = None
        worst_negative_segment: tuple[float, int, int] | None = None

        for match in _SEGMENT_RE.finditer(document.text):
            segment = match.group(0)
            if not segment:
                continue

            byte_start = byte_offsets[match.start()]
            byte_end = byte_offsets[match.end()]
            if byte_end <= byte_start:
                continue

            segment_loss_a = float(prefix_a[byte_end] - prefix_a[byte_start])
            segment_loss_b = float(prefix_b[byte_end] - prefix_b[byte_start])
            segment_bytes = int(byte_end - byte_start)
            segment_delta_bits = (segment_loss_a - segment_loss_b) * LOG2E
            if segment_delta_bits > 0.0 and (
                worst_positive_segment is None or segment_delta_bits > worst_positive_segment[0]
            ):
                worst_positive_segment = (segment_delta_bits, match.start(), match.end())
            if segment_delta_bits < 0.0 and (
                worst_negative_segment is None or segment_delta_bits < worst_negative_segment[0]
            ):
                worst_negative_segment = (segment_delta_bits, match.start(), match.end())
            bucket = bucket_for_segment(segment)
            visible = render_visible(segment)

            self.bucket_stats[bucket].add(loss_a=segment_loss_a, loss_b=segment_loss_b, num_bytes=segment_bytes)
            if segment_bytes <= 32:
                literal_key = (bucket, visible)
                self.literal_stats[literal_key].add(
                    loss_a=segment_loss_a,
                    loss_b=segment_loss_b,
                    num_bytes=segment_bytes,
                )
                if tokenized_a is not None and tokenized_b is not None:
                    self._maybe_record_literal_example(
                        literal_key=literal_key,
                        document=document,
                        segment_text=segment,
                        segment_byte_start=byte_start,
                        segment_byte_end=byte_end,
                        segment_char_start=match.start(),
                        segment_char_end=match.end(),
                        segment_delta_bits=segment_delta_bits,
                        tokenized_a=tokenized_a,
                        tokenized_b=tokenized_b,
                    )

            if segment_delta_bits == 0.0:
                continue

            segment_record = {
                "dataset": document.dataset_name,
                "bucket": bucket,
                "bytes": int(segment_bytes),
                "delta_bits": segment_delta_bits,
                "gap_bpb": segment_delta_bits / segment_bytes,
                "text": visible,
                "doc_preview": preview_text_window(document.text, match.start(), match.end()),
            }
            _push_top_positive(
                self._top_segments_positive,
                segment_delta_bits,
                segment_record,
                self.top_k_segments,
                self._heap_counter,
            )
            _push_top_negative(
                self._top_segments_negative,
                segment_delta_bits,
                segment_record,
                self.top_k_segments,
                self._heap_counter,
            )

        preview_span = None
        if delta_bits > 0.0:
            preview_span = worst_positive_segment
        elif delta_bits < 0.0:
            preview_span = worst_negative_segment
        preview = (
            preview_text_window(document.text, preview_span[1], preview_span[2])
            if preview_span is not None
            else preview_text(document.text)
        )
        doc_record = {
            "dataset": document.dataset_name,
            "shard": document.shard_name,
            "row_index": int(document.row_index),
            "bytes": int(num_bytes),
            "gap_bpb": gap_bpb,
            "delta_bits": delta_bits,
            "preview": preview,
        }
        _push_top_positive(self._top_docs_positive, delta_bits, doc_record, self.top_k_docs, self._heap_counter)
        _push_top_negative(self._top_docs_negative, delta_bits, doc_record, self.top_k_docs, self._heap_counter)

    def build_summary(self) -> dict[str, Any]:
        dataset_rows = [stats.as_dict(name) for name, stats in sorted(self.dataset_stats.items())]

        group_rows = []
        for group, leaves in sorted(self.group_to_leaves.items()):
            if group in self.dataset_stats and leaves == {group}:
                continue
            aggregate = GapAggregate()
            for leaf in sorted(leaves):
                leaf_stats = self.dataset_stats.get(leaf)
                if leaf_stats is None:
                    continue
                aggregate.add(
                    loss_a=leaf_stats.total_loss_a,
                    loss_b=leaf_stats.total_loss_b,
                    num_bytes=leaf_stats.total_bytes,
                    count=leaf_stats.count,
                )
            if aggregate.total_bytes > 0:
                group_rows.append(aggregate.as_dict(group))

        bucket_rows = [stats.as_dict(bucket) for bucket, stats in sorted(self.bucket_stats.items())]

        positive_literals = _top_literal_rows(
            self.literal_stats,
            literal_examples=self.literal_examples,
            direction="positive",
            limit=self.top_k_literals,
        )
        negative_literals = _top_literal_rows(
            self.literal_stats,
            literal_examples=self.literal_examples,
            direction="negative",
            limit=self.top_k_literals,
        )

        summary = {
            "model_a": self.model_a_name,
            "model_b": self.model_b_name,
            "datasets": sorted(dataset_rows, key=lambda row: abs(row["delta_bits"]), reverse=True),
            "dataset_groups": sorted(group_rows, key=lambda row: abs(row["delta_bits"]), reverse=True),
            "pattern_buckets": sorted(bucket_rows, key=lambda row: abs(row["delta_bits"]), reverse=True),
            "top_documents": {
                "model_a_worse": _sorted_records(self._top_docs_positive),
                "model_b_worse": _sorted_records(self._top_docs_negative),
            },
            "top_segments": {
                "model_a_worse": _sorted_records(self._top_segments_positive),
                "model_b_worse": _sorted_records(self._top_segments_negative),
            },
            "top_literals": {
                "model_a_worse": positive_literals,
                "model_b_worse": negative_literals,
            },
        }
        return summary

    def write(self) -> dict[str, Any]:
        summary = self.build_summary()
        write_report_files(self.output_path, summary)
        return summary

    def _register_hierarchy(self, tag: str, dataset_name: str) -> None:
        parts = tag.split("/")
        for i in range(1, len(parts)):
            self.group_to_leaves["/".join(parts[:i])].add(dataset_name)

    def _maybe_record_literal_example(
        self,
        *,
        literal_key: tuple[str, str],
        document: RawTextDocument,
        segment_text: str,
        segment_byte_start: int,
        segment_byte_end: int,
        segment_char_start: int,
        segment_char_end: int,
        segment_delta_bits: float,
        tokenized_a: TokenizedDocument,
        tokenized_b: TokenizedDocument,
    ) -> None:
        candidate = LiteralExample(
            abs_delta_bits=abs(segment_delta_bits),
            dataset_name=document.dataset_name,
            doc_preview=preview_text_window(document.text, segment_char_start, segment_char_end),
            model_a_token_boundaries=render_token_boundaries(
                segment_text=segment_text,
                segment_byte_start=segment_byte_start,
                segment_byte_end=segment_byte_end,
                tokenized=tokenized_a,
            ),
            model_b_token_boundaries=render_token_boundaries(
                segment_text=segment_text,
                segment_byte_start=segment_byte_start,
                segment_byte_end=segment_byte_end,
                tokenized=tokenized_b,
            ),
        )
        current = self.literal_examples.get(literal_key)
        if current is None or candidate.abs_delta_bits > current.abs_delta_bits:
            self.literal_examples[literal_key] = candidate


def iter_raw_text_documents(
    datasets: dict[str, DatasetComponent],
    *,
    max_docs_per_dataset: int | None,
    max_doc_bytes: int | None,
) -> Iterator[RawTextDocument]:
    for name, component in datasets.items():
        if component.source is None:
            raise ValueError(f"Dataset {name} has no source; raw gap finding requires raw sources.")
        if not isinstance(component.format, TextLmDatasetFormat):
            raise ValueError(
                f"Dataset {name} uses unsupported format {type(component.format).__name__}. "
                "Gap finding currently supports TextLmDatasetFormat only."
            )

        source = component.source.get_shard_source("validation")
        if source is None:
            continue

        tags = tuple(component.tags or ()) + (name,)
        yield from _iter_dataset_documents(
            dataset_name=name,
            tags=tags,
            source=source,
            text_key=component.format.text_key,
            max_docs=max_docs_per_dataset,
            max_doc_bytes=max_doc_bytes,
        )


def _iter_dataset_documents(
    *,
    dataset_name: str,
    tags: tuple[str, ...],
    source: ShardedDataSource[dict],
    text_key: str,
    max_docs: int | None,
    max_doc_bytes: int | None,
) -> Iterator[RawTextDocument]:
    emitted = 0
    for shard_name in source.shard_names:
        for row_index, record in enumerate(source.open_shard(shard_name)):
            if max_docs is not None and emitted >= max_docs:
                return
            if text_key not in record:
                raise ValueError(f"Dataset {dataset_name} record is missing text field {text_key!r}.")
            text = record[text_key]
            if not isinstance(text, str):
                raise ValueError(f"Dataset {dataset_name} text field {text_key!r} is not a string.")
            if not text:
                continue
            if max_doc_bytes is not None:
                text = _truncate_text_to_byte_limit(text, max_doc_bytes)
            emitted += 1
            yield RawTextDocument(
                dataset_name=dataset_name,
                tags=tags,
                shard_name=shard_name,
                row_index=row_index,
                text=text,
            )


def _truncate_text_to_byte_limit(text: str, max_doc_bytes: int) -> str:
    if max_doc_bytes <= 0:
        raise ValueError(f"max_doc_bytes must be positive, got {max_doc_bytes}.")

    if text.isascii():
        return text[:max_doc_bytes]

    running = 0
    end = 0
    for end, ch in enumerate(text, start=1):
        running += len(ch.encode("utf-8"))
        if running > max_doc_bytes:
            return text[: end - 1]
    return text


def tokenize_text_with_byte_spans(tokenizer: MarinTokenizer, hf_tokenizer: Any, text: str) -> TokenizedDocument:
    parts = _safe_split_for_tokenizer(text)
    ids: list[int] = []
    char_spans: list[tuple[int, int]] = []
    char_offset = 0

    for part in parts:
        if not part:
            continue
        encoded = hf_tokenizer(part, add_special_tokens=False, return_offsets_mapping=True)
        part_ids = list(encoded["input_ids"])
        part_offsets = list(encoded["offset_mapping"])
        for token_id, (start, end) in zip(part_ids, part_offsets, strict=True):
            ids.append(token_id)
            if end <= start:
                char_spans.append((-1, -1))
            else:
                char_spans.append((start + char_offset, end + char_offset))
        char_offset += len(part)

    byte_offsets = char_to_byte_offsets(text)
    byte_starts: list[int] = []
    byte_ends: list[int] = []
    for start, end in char_spans:
        if start < 0 or end <= start:
            byte_starts.append(-1)
            byte_ends.append(-1)
        else:
            byte_starts.append(byte_offsets[start])
            byte_ends.append(byte_offsets[end])

    need_bos, need_eos = manual_special_token_policy(tokenizer)
    if need_bos and tokenizer.bos_token_id is not None:
        ids = [tokenizer.bos_token_id, *ids]
        byte_starts = [-1, *byte_starts]
        byte_ends = [-1, *byte_ends]
    if need_eos and tokenizer.eos_token_id is not None:
        ids = [*ids, tokenizer.eos_token_id]
        byte_starts = [*byte_starts, -1]
        byte_ends = [*byte_ends, -1]

    return TokenizedDocument(
        token_ids=np.asarray(ids, dtype=np.int32),
        byte_starts=np.asarray(byte_starts, dtype=np.int32),
        byte_ends=np.asarray(byte_ends, dtype=np.int32),
        num_bytes=int(byte_offsets[-1]),
    )


def chunk_tokenized_document(
    document: TokenizedDocument,
    max_eval_length: int,
    *,
    doc_index: int = -1,
) -> list[TokenizedChunk]:
    if max_eval_length <= 1:
        raise ValueError(f"max_eval_length must be greater than 1, got {max_eval_length}.")

    chunks: list[TokenizedChunk] = []
    total_tokens = len(document.token_ids)
    stride = max_eval_length - 1
    for start in range(0, total_tokens, stride):
        end = min(start + max_eval_length, total_tokens)
        chunks.append(
            TokenizedChunk(
                doc_index=doc_index,
                token_ids=document.token_ids[start:end],
                byte_starts=document.byte_starts[start:end],
                byte_ends=document.byte_ends[start:end],
            )
        )
        if end == total_tokens:
            break
    return chunks


def char_to_byte_offsets(text: str) -> np.ndarray:
    offsets = np.zeros(len(text) + 1, dtype=np.int32)
    running = 0
    for i, ch in enumerate(text, start=1):
        running += len(ch.encode("utf-8"))
        offsets[i] = running
    return offsets


def manual_special_token_policy(tokenizer: MarinTokenizer) -> tuple[bool, bool]:
    add_bos = tokenizer.bos_token_id is not None
    add_eos = tokenizer.eos_token_id is not None
    if not add_bos and not add_eos:
        return False, False

    probe = tokenizer.encode("hi there", add_special_tokens=True)
    need_bos = add_bos and (not probe or probe[0] != tokenizer.bos_token_id)
    need_eos = add_eos and (not probe or probe[-1] != tokenizer.eos_token_id)
    return need_bos, need_eos


def batch_chunks(
    chunks: Sequence[TokenizedChunk],
    *,
    batch_size: int,
    max_eval_length: int,
) -> Iterator[list[TokenizedChunk]]:
    current: list[TokenizedChunk] = []
    for chunk in chunks:
        if len(chunk.token_ids) > max_eval_length:
            raise ValueError(
                f"Tokenized chunk length {len(chunk.token_ids)} exceeds max_eval_length {max_eval_length}."
            )
        current.append(chunk)
        if len(current) == batch_size:
            yield current
            current = []
    if current:
        yield current


def render_visible(text: str, limit: int = 32) -> str:
    visible = visible_text(text)
    if len(visible) > limit:
        return visible[: limit - 1] + "\u2026"
    return visible


def preview_text(text: str, limit: int = 120) -> str:
    preview = visible_text(text)
    if len(preview) > limit:
        preview = preview[: limit - 1] + "\u2026"
    return preview


def preview_text_window(text: str, start: int, end: int, limit: int = 120) -> str:
    if limit <= 2:
        raise ValueError(f"limit must be greater than 2, got {limit}.")
    if len(text) <= limit:
        return visible_text(text)

    start = min(max(start, 0), len(text))
    end = min(max(end, start), len(text))
    center = (start + end) // 2
    content_limit = limit - 2
    window_start = max(0, center - content_limit // 2)
    window_end = min(len(text), window_start + content_limit)
    window_start = max(0, window_end - content_limit)

    prefix = "\u2026" if window_start > 0 else ""
    suffix = "\u2026" if window_end < len(text) else ""
    return prefix + visible_text(text[window_start:window_end]) + suffix


def visible_text(text: str) -> str:
    return text.translate(VISIBLE_WHITESPACE)


def render_token_boundaries(
    *,
    segment_text: str,
    segment_byte_start: int,
    segment_byte_end: int,
    tokenized: TokenizedDocument,
) -> str:
    segment_bytes = segment_text.encode("utf-8")
    pieces: list[str] = []

    for token_start, token_end in zip(tokenized.byte_starts, tokenized.byte_ends, strict=True):
        if token_start < 0 or token_end <= token_start:
            continue
        if token_end <= segment_byte_start or token_start >= segment_byte_end:
            continue

        overlap_start = max(token_start, segment_byte_start) - segment_byte_start
        overlap_end = min(token_end, segment_byte_end) - segment_byte_start
        if overlap_end <= overlap_start:
            continue

        piece = visible_text(segment_bytes[overlap_start:overlap_end].decode("utf-8"))
        if token_start < segment_byte_start:
            piece = "\u2026" + piece
        if token_end > segment_byte_end:
            piece = piece + "\u2026"
        pieces.append(piece)

    if not pieces:
        return "(no aligned tokens)"

    return "|" + "|".join(pieces) + "|"


def bucket_for_segment(segment: str) -> str:
    if segment.isspace():
        if "\t" in segment or "\r" in segment:
            return "whitespace/tab_or_cr"
        if set(segment) == {" "}:
            return "whitespace/single_space" if len(segment) == 1 else "whitespace/multi_space"
        if set(segment) == {"\n"}:
            return "whitespace/newline" if len(segment) == 1 else "whitespace/multi_newline"
        return "whitespace/mixed"

    if _URL_RE.fullmatch(segment):
        return "text/url"
    if _NUMBER_RE.fullmatch(segment):
        return "text/number"
    if _WORD_RE.fullmatch(segment):
        return "text/non_ascii_word" if not segment.isascii() else "text/word"
    if not segment.isascii():
        return "text/non_ascii"
    return "text/punctuation"


def render_report_markdown(summary: dict[str, Any]) -> str:
    def section(title: str, rows: Sequence[dict[str, Any]], note: str | None = None) -> str:
        if not rows:
            return f"## {title}\n\nNo rows.\n"

        headers = list(rows[0].keys())
        table = [
            "| " + " | ".join(_escape_markdown_table_cell(h) for h in headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in rows:
            table.append("| " + " | ".join(_format_markdown_cell(row[h]) for h in headers) + " |")

        parts = [f"## {title}"]
        if note is not None:
            parts.extend(["", note])
        parts.extend(["", *table, ""])
        return "\n".join(parts)

    literal_note = (
        "Representative token boundaries come from the highest-gap occurrence for each literal. "
        "`|` marks token boundaries for each model; an ellipsis means the token continues outside "
        "the literal boundary in that example."
    )
    parts = [
        "# Perplexity Gap Report\n",
        f"**Model A:** {_escape_markdown_text(str(summary['model_a']))}\n",
        f"**Model B:** {_escape_markdown_text(str(summary['model_b']))}\n",
        section("Datasets", summary["datasets"]),
        section("Dataset Groups", summary["dataset_groups"]),
        section("Pattern Buckets", summary["pattern_buckets"]),
        section("Top Documents: Model A Worse", summary["top_documents"]["model_a_worse"]),
        section("Top Documents: Model B Worse", summary["top_documents"]["model_b_worse"]),
        section("Top Segments: Model A Worse", summary["top_segments"]["model_a_worse"]),
        section("Top Segments: Model B Worse", summary["top_segments"]["model_b_worse"]),
        section("Top Literals: Model A Worse", summary["top_literals"]["model_a_worse"], note=literal_note),
        section("Top Literals: Model B Worse", summary["top_literals"]["model_b_worse"], note=literal_note),
    ]
    return "\n".join(parts)


def write_report_files(output_path: str, summary: dict[str, Any]) -> tuple[str, str]:
    summary_path = os.path.join(output_path, "summary.json")
    report_path = os.path.join(output_path, "report.md")
    fs, _, _ = fsspec.get_fs_token_paths(summary_path)
    fs.makedirs(output_path, exist_ok=True)

    with open_url(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    with open_url(report_path, "w") as f:
        f.write(render_report_markdown(summary))

    return summary_path, report_path


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _format_markdown_cell(value: Any) -> str:
    return _escape_markdown_table_cell(_format_cell(value))


def _escape_markdown_table_cell(value: str) -> str:
    return _escape_markdown_text(value).replace("\r", " ").replace("\n", "<br>")


def _escape_markdown_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|")


def _push_top_positive(
    heap: list[tuple[float, int, dict[str, Any]]],
    score: float,
    record: dict[str, Any],
    limit: int,
    counter: itertools.count,
) -> None:
    if score <= 0:
        return
    item = (score, next(counter), record)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif score > heap[0][0]:
        heapq.heapreplace(heap, item)


def _push_top_negative(
    heap: list[tuple[float, int, dict[str, Any]]],
    score: float,
    record: dict[str, Any],
    limit: int,
    counter: itertools.count,
) -> None:
    if score >= 0:
        return
    item = (-score, next(counter), record)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif -score > heap[0][0]:
        heapq.heapreplace(heap, item)


def _sorted_records(heap: Sequence[tuple[float, int, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [record for _, _, record in sorted(heap, key=lambda item: item[0], reverse=True)]


def _top_literal_rows(
    literal_stats: dict[tuple[str, str], GapAggregate],
    *,
    literal_examples: dict[tuple[str, str], LiteralExample],
    direction: str,
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (bucket, literal), stats in literal_stats.items():
        example = literal_examples.get((bucket, literal))
        stats_row = stats.as_dict(literal)
        row = {
            "name": stats_row["name"],
            "bucket": bucket,
            "example_dataset": example.dataset_name if example is not None else None,
            "example_doc_preview": example.doc_preview if example is not None else None,
            "model_a_token_boundaries": example.model_a_token_boundaries if example is not None else None,
            "model_b_token_boundaries": example.model_b_token_boundaries if example is not None else None,
            "documents": stats_row["documents"],
            "bytes": stats_row["bytes"],
            "model_a_bpb": stats_row["model_a_bpb"],
            "model_b_bpb": stats_row["model_b_bpb"],
            "gap_bpb": stats_row["gap_bpb"],
            "delta_bits": stats_row["delta_bits"],
        }
        rows.append(row)

    if direction == "positive":
        rows = [row for row in rows if row["delta_bits"] > 0]
        rows.sort(key=lambda row: row["delta_bits"], reverse=True)
    else:
        rows = [row for row in rows if row["delta_bits"] < 0]
        rows.sort(key=lambda row: row["delta_bits"])

    return rows[:limit]
