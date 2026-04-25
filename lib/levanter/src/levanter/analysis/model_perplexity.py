# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem import open_url

from levanter.analysis.perplexity_gap import (
    LOG2E,
    _SEGMENT_RE,
    GapReportBuilder,
    RawTextDocument,
    TokenizedDocument,
    bucket_for_segment,
    write_report_files,
)


SCORED_DOCUMENTS_FILENAME = "scored_documents.parquet"
SUMMARY_FILENAME = "summary.json"
TOKEN_COUNTS_FILENAME = "token_counts.parquet"
TOKEN_COUNT_SUMMARY_FILENAME = "token_counts_summary.json"
DEFAULT_RARE_TOKEN_LIMIT = 32


@dataclass(frozen=True)
class ScoredDocument:
    document: RawTextDocument
    per_byte_loss: np.ndarray
    tokenized: TokenizedDocument


@dataclass
class ModelLossAggregate:
    total_loss: float = 0.0
    total_bytes: int = 0
    count: int = 0

    def add(self, *, loss: float, num_bytes: int, count: int = 1) -> None:
        self.total_loss += float(loss)
        self.total_bytes += int(num_bytes)
        self.count += int(count)

    def as_dict(self, name: str) -> dict[str, Any]:
        bpb = None if self.total_bytes <= 0 else self.total_loss * LOG2E / self.total_bytes
        return {
            "name": name,
            "documents": int(self.count),
            "bytes": int(self.total_bytes),
            "bpb": bpb,
            "bits": self.total_loss * LOG2E,
        }


@dataclass
class ModelScoreReportBuilder:
    model_name: str
    dataset_stats: dict[str, ModelLossAggregate] = field(default_factory=lambda: defaultdict(ModelLossAggregate))
    bucket_stats: dict[str, ModelLossAggregate] = field(default_factory=lambda: defaultdict(ModelLossAggregate))
    group_to_leaves: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def register_dataset(self, dataset_name: str, tags: Sequence[str]) -> None:
        self.group_to_leaves[dataset_name].add(dataset_name)
        for tag in tags:
            self.group_to_leaves[tag].add(dataset_name)
            self._register_hierarchy(tag, dataset_name)
        self._register_hierarchy(dataset_name, dataset_name)

    def add_document(self, *, document: RawTextDocument, per_byte_loss: np.ndarray) -> None:
        self.register_dataset(document.dataset_name, document.tags)

        num_bytes = len(per_byte_loss)
        if num_bytes <= 0:
            return

        total_loss = float(per_byte_loss.sum())
        self.dataset_stats[document.dataset_name].add(loss=total_loss, num_bytes=num_bytes)

        prefix = np.concatenate(([0.0], np.cumsum(per_byte_loss, dtype=np.float64)))
        byte_offsets = _char_to_byte_offsets(document.text)
        for match in _SEGMENT_RE.finditer(document.text):
            segment = match.group(0)
            if not segment:
                continue

            byte_start = byte_offsets[match.start()]
            byte_end = byte_offsets[match.end()]
            if byte_end <= byte_start:
                continue

            segment_loss = float(prefix[byte_end] - prefix[byte_start])
            segment_bytes = int(byte_end - byte_start)
            self.bucket_stats[bucket_for_segment(segment)].add(loss=segment_loss, num_bytes=segment_bytes)

    def build_summary(self) -> dict[str, Any]:
        dataset_rows = [stats.as_dict(name) for name, stats in sorted(self.dataset_stats.items())]

        group_rows = []
        for group, leaves in sorted(self.group_to_leaves.items()):
            if group in self.dataset_stats and leaves == {group}:
                continue
            aggregate = ModelLossAggregate()
            for leaf in sorted(leaves):
                leaf_stats = self.dataset_stats.get(leaf)
                if leaf_stats is None:
                    continue
                aggregate.add(loss=leaf_stats.total_loss, num_bytes=leaf_stats.total_bytes, count=leaf_stats.count)
            if aggregate.total_bytes > 0:
                group_rows.append(aggregate.as_dict(group))

        bucket_rows = [stats.as_dict(bucket) for bucket, stats in sorted(self.bucket_stats.items())]

        return {
            "model": self.model_name,
            "datasets": sorted(dataset_rows, key=lambda row: row["bits"], reverse=True),
            "dataset_groups": sorted(group_rows, key=lambda row: row["bits"], reverse=True),
            "pattern_buckets": sorted(bucket_rows, key=lambda row: row["bits"], reverse=True),
        }

    def _register_hierarchy(self, tag: str, dataset_name: str) -> None:
        parts = tag.split("/")
        for i in range(1, len(parts)):
            self.group_to_leaves["/".join(parts[:i])].add(dataset_name)


def write_model_score_files(
    output_path: str,
    summary: dict[str, Any],
    scored_documents: Sequence[ScoredDocument],
    *,
    vocab_size: int | None = None,
    token_id_to_text: dict[int, str] | None = None,
) -> None:
    summary_path = os.path.join(output_path, SUMMARY_FILENAME)
    scored_documents_path = os.path.join(output_path, SCORED_DOCUMENTS_FILENAME)
    token_counts_path = os.path.join(output_path, TOKEN_COUNTS_FILENAME)
    token_count_summary_path = os.path.join(output_path, TOKEN_COUNT_SUMMARY_FILENAME)
    fs, _, _ = fsspec.get_fs_token_paths(summary_path)
    fs.makedirs(output_path, exist_ok=True)

    with open_url(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    table = _scored_documents_table(scored_documents)
    with fsspec.open(scored_documents_path, "wb") as f:
        pq.write_table(table, f)

    token_count_summary, token_count_table = _token_count_artifacts(
        scored_documents,
        vocab_size=vocab_size,
        token_id_to_text=token_id_to_text,
    )
    with open_url(token_count_summary_path, "w") as f:
        json.dump(token_count_summary, f, indent=2, sort_keys=True)
    with fsspec.open(token_counts_path, "wb") as f:
        pq.write_table(token_count_table, f)


def read_model_score_summary(output_path: str) -> dict[str, Any]:
    summary_path = os.path.join(output_path, SUMMARY_FILENAME)
    with open_url(summary_path) as f:
        return json.load(f)


def read_scored_documents(output_path: str) -> list[ScoredDocument]:
    scored_documents_path = os.path.join(output_path, SCORED_DOCUMENTS_FILENAME)
    with fsspec.open(scored_documents_path, "rb") as f:
        table = pq.read_table(f)
    return [_scored_document_from_row(row) for row in table.to_pylist()]


def read_token_count_summary(output_path: str) -> dict[str, Any]:
    token_count_summary_path = os.path.join(output_path, TOKEN_COUNT_SUMMARY_FILENAME)
    with open_url(token_count_summary_path) as f:
        return json.load(f)


def compare_scored_outputs(
    *,
    model_a_name: str,
    model_b_name: str,
    model_a_output_path: str,
    model_b_output_path: str,
    output_path: str,
) -> dict[str, Any]:
    scored_documents_a = read_scored_documents(model_a_output_path)
    scored_documents_b = read_scored_documents(model_b_output_path)
    summary = compare_scored_documents(
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        scored_documents_a=scored_documents_a,
        scored_documents_b=scored_documents_b,
        output_path=output_path,
    )
    write_report_files(output_path, summary)
    return summary


def compare_scored_documents(
    *,
    model_a_name: str,
    model_b_name: str,
    scored_documents_a: Sequence[ScoredDocument],
    scored_documents_b: Sequence[ScoredDocument],
    output_path: str,
) -> dict[str, Any]:
    docs_a_by_key = {_scored_document_key(document): document for document in scored_documents_a}
    docs_b_by_key = {_scored_document_key(document): document for document in scored_documents_b}

    if docs_a_by_key.keys() != docs_b_by_key.keys():
        missing_from_b = sorted(docs_a_by_key.keys() - docs_b_by_key.keys())[:5]
        missing_from_a = sorted(docs_b_by_key.keys() - docs_a_by_key.keys())[:5]
        raise ValueError(
            "Model score outputs cover different document sets. "
            f"Missing from model_b: {missing_from_b}. Missing from model_a: {missing_from_a}."
        )

    report = GapReportBuilder(model_a_name=model_a_name, model_b_name=model_b_name, output_path=output_path)
    for key in sorted(docs_a_by_key):
        scored_a = docs_a_by_key[key]
        scored_b = docs_b_by_key[key]
        if scored_a.document.text != scored_b.document.text or scored_a.document.tags != scored_b.document.tags:
            raise ValueError(f"Document metadata mismatch for scored document {key}.")
        report.add_document(
            document=scored_a.document,
            per_byte_loss_a=scored_a.per_byte_loss,
            per_byte_loss_b=scored_b.per_byte_loss,
            tokenized_a=scored_a.tokenized,
            tokenized_b=scored_b.tokenized,
        )
    return report.build_summary()


def _scored_documents_table(scored_documents: Sequence[ScoredDocument]) -> pa.Table:
    rows = {
        "dataset_name": [doc.document.dataset_name for doc in scored_documents],
        "tags": [list(doc.document.tags) for doc in scored_documents],
        "shard_name": [doc.document.shard_name for doc in scored_documents],
        "row_index": [doc.document.row_index for doc in scored_documents],
        "text": [doc.document.text for doc in scored_documents],
        "token_ids": [doc.tokenized.token_ids.tolist() for doc in scored_documents],
        "per_byte_loss": [doc.per_byte_loss.tolist() for doc in scored_documents],
        "token_byte_starts": [doc.tokenized.byte_starts.tolist() for doc in scored_documents],
        "token_byte_ends": [doc.tokenized.byte_ends.tolist() for doc in scored_documents],
        "num_bytes": [doc.tokenized.num_bytes for doc in scored_documents],
    }
    schema = pa.schema(
        [
            ("dataset_name", pa.string()),
            ("tags", pa.list_(pa.string())),
            ("shard_name", pa.string()),
            ("row_index", pa.int64()),
            ("text", pa.string()),
            ("token_ids", pa.list_(pa.int32())),
            ("per_byte_loss", pa.list_(pa.float64())),
            ("token_byte_starts", pa.list_(pa.int32())),
            ("token_byte_ends", pa.list_(pa.int32())),
            ("num_bytes", pa.int32()),
        ]
    )
    return pa.Table.from_pydict(rows, schema=schema)


def _scored_document_from_row(row: dict[str, Any]) -> ScoredDocument:
    document = RawTextDocument(
        dataset_name=row["dataset_name"],
        tags=tuple(row["tags"]),
        shard_name=row["shard_name"],
        row_index=int(row["row_index"]),
        text=row["text"],
    )
    tokenized = TokenizedDocument(
        token_ids=np.asarray(row["token_ids"], dtype=np.int32),
        byte_starts=np.asarray(row["token_byte_starts"], dtype=np.int32),
        byte_ends=np.asarray(row["token_byte_ends"], dtype=np.int32),
        num_bytes=int(row["num_bytes"]),
    )
    per_byte_loss = np.asarray(row["per_byte_loss"], dtype=np.float64)
    if len(per_byte_loss) != tokenized.num_bytes:
        raise ValueError(
            f"Stored per-byte losses for {document.dataset_name}/{document.row_index} do not match num_bytes."
        )
    return ScoredDocument(document=document, per_byte_loss=per_byte_loss, tokenized=tokenized)


def _scored_document_key(scored_document: ScoredDocument) -> tuple[str, str, int]:
    document = scored_document.document
    return (document.dataset_name, document.shard_name, int(document.row_index))


def _char_to_byte_offsets(text: str) -> np.ndarray:
    offsets = np.zeros(len(text) + 1, dtype=np.int32)
    running = 0
    for i, ch in enumerate(text, start=1):
        running += len(ch.encode("utf-8"))
        offsets[i] = running
    return offsets


def _token_count_artifacts(
    scored_documents: Sequence[ScoredDocument],
    *,
    vocab_size: int | None,
    token_id_to_text: dict[int, str] | None,
) -> tuple[dict[str, Any], pa.Table]:
    dataset_counters = _dataset_token_counters(scored_documents)
    overall_counter: Counter[int] = Counter()
    for counter in dataset_counters.values():
        overall_counter.update(counter)

    summary = {
        "vocab_size": vocab_size,
        "overall": _token_count_stats(
            name="overall",
            counts=overall_counter,
            vocab_size=vocab_size,
            token_id_to_text=token_id_to_text,
        ),
        "datasets": [
            _token_count_stats(
                name=dataset_name,
                counts=counts,
                vocab_size=vocab_size,
                token_id_to_text=token_id_to_text,
            )
            for dataset_name, counts in sorted(dataset_counters.items())
        ],
    }
    return summary, _token_counts_table(dataset_counters, token_id_to_text=token_id_to_text)


def _dataset_token_counters(scored_documents: Sequence[ScoredDocument]) -> dict[str, Counter[int]]:
    counters: dict[str, Counter[int]] = defaultdict(Counter)
    for scored_document in scored_documents:
        token_ids = scored_document.tokenized.token_ids
        if token_ids.size == 0:
            continue
        unique_ids, counts = np.unique(token_ids, return_counts=True)
        dataset_counter = counters[scored_document.document.dataset_name]
        for token_id, count in zip(unique_ids.tolist(), counts.tolist(), strict=True):
            dataset_counter[int(token_id)] += int(count)
    return counters


def _token_count_stats(
    *,
    name: str,
    counts: Counter[int],
    vocab_size: int | None,
    token_id_to_text: dict[int, str] | None,
) -> dict[str, Any]:
    total_tokens = int(sum(counts.values()))
    unique_tokens = int(len(counts))
    singleton_tokens = int(sum(1 for count in counts.values() if count == 1))
    rare_tokens_le_3 = int(sum(1 for count in counts.values() if count <= 3))
    coverage_fraction = None if vocab_size is None or vocab_size <= 0 else unique_tokens / vocab_size
    unseen_tokens = None if vocab_size is None else max(vocab_size - unique_tokens, 0)
    rare_token_examples = [
        {
            "token_id": int(token_id),
            "count": int(count),
            "token_text": _token_text(token_id, token_id_to_text),
        }
        for token_id, count in sorted(counts.items(), key=lambda item: (item[1], item[0]))[:DEFAULT_RARE_TOKEN_LIMIT]
    ]
    return {
        "name": name,
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "singleton_tokens": singleton_tokens,
        "rare_tokens_le_3": rare_tokens_le_3,
        "coverage_fraction": coverage_fraction,
        "unseen_tokens": unseen_tokens,
        "rare_token_examples": rare_token_examples,
    }


def _token_counts_table(
    dataset_counters: dict[str, Counter[int]],
    *,
    token_id_to_text: dict[int, str] | None,
) -> pa.Table:
    rows = {
        "dataset_name": [],
        "token_id": [],
        "count": [],
        "token_text": [],
    }
    for dataset_name in sorted(dataset_counters):
        counts = dataset_counters[dataset_name]
        for token_id, count in sorted(counts.items(), key=lambda item: (item[1], item[0])):
            rows["dataset_name"].append(dataset_name)
            rows["token_id"].append(int(token_id))
            rows["count"].append(int(count))
            rows["token_text"].append(_token_text(token_id, token_id_to_text))

    schema = pa.schema(
        [
            ("dataset_name", pa.string()),
            ("token_id", pa.int32()),
            ("count", pa.int64()),
            ("token_text", pa.string()),
        ]
    )
    return pa.Table.from_pydict(rows, schema=schema)


def _token_text(token_id: int, token_id_to_text: dict[int, str] | None) -> str | None:
    if token_id_to_text is None:
        return None
    return token_id_to_text.get(int(token_id))
