# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import json
import logging
import os
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field

import fsspec
import msgspec
from zephyr import Backend, Dataset, load_file

from experiments.train_test_overlap.ngram_new.utils import (
    build_ngram_offsets,
    collect_eval_file_specs,
    collect_input_files,
    record_id,
    tokenize_with_offsets,
)

logger = logging.getLogger(__name__)

_INDEX_CACHE: dict[str, dict[int, dict[str, list[list]]]] = {}
_META_CACHE: dict[str, list[dict]] = {}


@dataclass(frozen=True)
class OverlapMapConfig:
    """Configuration for the n-gram overlap map stage."""

    input_path: str | list[str]
    eval_dataset_paths: str | list[str]
    output_path: str
    ngram_lengths: list[int] = field(default_factory=lambda: [5, 10, 15])
    stride: int = 0
    tokenizer: str = "default"
    eval_text_field: str = "text"
    text_field: str = "text"
    processes: int = 32
    num_shards: int = 128
    skip_existing: bool = True
    write_test_instance_counts: bool = True
    track_progress: bool = True


def _load_test_index(index_path: str) -> dict[int, dict[str, list[list]]]:
    cached = _INDEX_CACHE.get(index_path)
    if cached is not None:
        return cached
    fs, fs_path = fsspec.core.url_to_fs(index_path)
    with fs.open(fs_path, "rb") as f:
        index = msgspec.msgpack.decode(f.read())
    _INDEX_CACHE[index_path] = index
    return index


def _load_test_meta(meta_path: str) -> list[dict]:
    cached = _META_CACHE.get(meta_path)
    if cached is not None:
        return cached
    fs, fs_path = fsspec.core.url_to_fs(meta_path)
    with fs.open(fs_path, "rb") as f:
        meta = msgspec.msgpack.decode(f.read())
    _META_CACHE[meta_path] = meta
    return meta


def _write_test_instance_counts(output_path: str, counts: dict[str, int]) -> str:
    counts_path = os.path.join(output_path, "tmp", "test_instance_counts.jsonl")
    fs, fs_path = fsspec.core.url_to_fs(counts_path)
    fs.makedirs(os.path.dirname(fs_path), exist_ok=True)
    with fs.open(fs_path, "w") as f:
        for eval_dataset, num_instances in sorted(counts.items()):
            f.write(json.dumps({"eval_dataset": eval_dataset, "num_instances": num_instances}) + "\n")
    return counts_path


def _build_test_index(
    eval_specs: list[dict],
    n_values: list[int],
    stride: int,
    tokenizer_name: str,
    eval_text_field: str,
    output_path: str,
    skip_existing: bool,
    track_progress: bool,
) -> tuple[str, str, dict[str, int]]:
    index_path = os.path.join(output_path, "tmp", "test_index.msgpack")
    meta_path = os.path.join(output_path, "tmp", "test_meta.msgpack")
    fs, fs_path = fsspec.core.url_to_fs(index_path)
    meta_fs, meta_fs_path = fsspec.core.url_to_fs(meta_path)
    if skip_existing and fs.exists(fs_path) and meta_fs.exists(meta_fs_path):
        counts: dict[str, int] = {}
        if track_progress:
            print(f"[overlap_map] test index exists at {index_path}", flush=True)
        return index_path, meta_path, counts

    index: dict[int, dict[str, list[list]]] = {n: defaultdict(list) for n in n_values}
    counts: dict[str, set[str]] = defaultdict(set)
    eval_meta: list[dict] = []

    for spec in eval_specs:
        eval_dataset = spec["eval_dataset"]
        start_time = time.time()
        if track_progress:
            print(
                f"[overlap_map] index start path={spec['path']} eval_dataset={eval_dataset}",
                flush=True,
            )
        record_total = 0
        record_with_text = 0
        record_missing_text = 0
        token_count = 0
        ngram_counts: dict[int, int] = {n: 0 for n in n_values}

        for row_idx, record in enumerate(load_file(spec["path"])):
            record_total += 1
            text = record.get(eval_text_field)
            if text is None:
                record_missing_text += 1
                continue
            record_with_text += 1
            instance_id = record_id(record)
            counts[eval_dataset].add(instance_id)
            eval_meta.append(
                {
                    "eval_dataset": eval_dataset,
                    "eval_path": spec["path"],
                    "eval_row": row_idx,
                    "eval_text": text,
                    "eval_instance_id": instance_id,
                }
            )
            eval_idx = len(eval_meta) - 1
            tokens, offsets = tokenize_with_offsets(text, tokenizer_name)
            token_count += len(tokens)
            for n in n_values:
                ngram_offsets = build_ngram_offsets(tokens, offsets, n, stride)
                ngram_counts[n] += len(ngram_offsets)
                for ngram, spans in ngram_offsets.items():
                    index[n][ngram].append([eval_idx, [[start, end] for start, end in spans]])

        if track_progress:
            elapsed = round(time.time() - start_time, 3)
            print(
                "[overlap_map] index done "
                f"path={spec['path']} eval_dataset={eval_dataset} "
                f"records_total={record_total} records_with_text={record_with_text} "
                f"records_missing_text={record_missing_text} tokens={token_count} "
                f"ngram_counts={ngram_counts} elapsed_sec={elapsed}",
                flush=True,
            )

    index_out: dict[int, dict[str, list[list]]] = {n_val: dict(ngrams) for n_val, ngrams in index.items()}

    fs.makedirs(os.path.dirname(fs_path), exist_ok=True)
    with fs.open(fs_path, "wb") as f:
        f.write(msgspec.msgpack.encode(index_out))
    meta_fs.makedirs(os.path.dirname(meta_fs_path), exist_ok=True)
    with meta_fs.open(meta_fs_path, "wb") as f:
        f.write(msgspec.msgpack.encode(eval_meta))

    counts_out = {dataset: len(instance_ids) for dataset, instance_ids in counts.items()}
    return index_path, meta_path, counts_out


def _process_training_shard(
    specs: Iterator[dict],
    *,
    index_path: str,
    meta_path: str,
    n_values: list[int],
    stride: int,
    tokenizer_name: str,
    text_field: str,
    log_progress: bool,
) -> Iterator[dict]:
    index = _load_test_index(index_path)
    eval_meta = _load_test_meta(meta_path)

    for spec in specs:
        start_time = time.time()
        record_total = 0
        record_with_text = 0
        record_missing_text = 0
        token_count = 0
        overlap_records = 0
        ngram_counts: dict[int, int] = {n: 0 for n in n_values}
        if log_progress:
            print(
                f"[overlap_map] train start path={spec['path']}",
                flush=True,
            )

        for row_idx, record in enumerate(load_file(spec["path"])):
            record_total += 1
            text = record.get(text_field)
            if text is None:
                record_missing_text += 1
                continue
            record_with_text += 1
            train_doc_id = record.get("id")
            train_doc_id = str(train_doc_id) if train_doc_id is not None else spec["path"]
            tokens, offsets = tokenize_with_offsets(text, tokenizer_name)
            token_count += len(tokens)
            for n in n_values:
                index_n = index.get(n)
                if not index_n:
                    continue
                train_ngram_offsets = build_ngram_offsets(tokens, offsets, n, stride)
                ngram_counts[n] += len(train_ngram_offsets)
                for ngram, spans in train_ngram_offsets.items():
                    entries = index_n.get(ngram)
                    if not entries:
                        continue
                    train_offsets = [[start, end] for start, end in spans]
                    for entry in entries:
                        eval_idx = entry[0]
                        eval_offsets = entry[1]
                        eval_info = eval_meta[eval_idx]
                        overlap_records += 1
                        yield {
                            "eval_dataset": eval_info["eval_dataset"],
                            "eval_path": eval_info["eval_path"],
                            "eval_row": eval_info["eval_row"],
                            "eval_text": eval_info["eval_text"],
                            "eval_instance_id": eval_info["eval_instance_id"],
                            "n": n,
                            "ngram": ngram,
                            "eval_offsets": eval_offsets,
                            "train_path": spec["path"],
                            "train_row": row_idx,
                            "train_text": text,
                            "train_ngram": ngram,
                            "train_offsets": train_offsets,
                            "train_doc_id": train_doc_id,
                        }

        if log_progress:
            elapsed = round(time.time() - start_time, 3)
            print(
                "[overlap_map] train done "
                f"path={spec['path']} records_total={record_total} "
                f"records_with_text={record_with_text} records_missing_text={record_missing_text} "
                f"tokens={token_count} ngram_counts={ngram_counts} "
                f"overlap_records={overlap_records} elapsed_sec={elapsed}",
                flush=True,
            )


def run_overlap_map(config: OverlapMapConfig) -> str:
    """Compute overlap events between eval datasets and training data."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    n_values = sorted(set(config.ngram_lengths))
    if not n_values:
        raise ValueError("ngram_lengths must be non-empty")
    if config.num_shards <= 0:
        raise ValueError("num_shards must be positive")

    eval_specs = collect_eval_file_specs(config.eval_dataset_paths)
    train_specs = [{"path": path} for path in collect_input_files(config.input_path)]

    if config.track_progress:
        print(
            "[overlap_map] summary "
            f"num_eval_files={len(eval_specs)} num_train_files={len(train_specs)} "
            f"num_shards={config.num_shards} max_parallelism={config.processes} "
            f"ngram_lengths={n_values} stride={config.stride} tokenizer={config.tokenizer} "
            f"eval_text_field={config.eval_text_field} text_field={config.text_field}",
            flush=True,
        )

    index_path, meta_path, counts = _build_test_index(
        eval_specs,
        n_values,
        config.stride,
        config.tokenizer,
        config.eval_text_field,
        config.output_path,
        config.skip_existing,
        config.track_progress,
    )

    if config.write_test_instance_counts and counts:
        _write_test_instance_counts(config.output_path, counts)

    train_pipeline = (
        Dataset.from_list(train_specs)
        .reshard(num_shards=config.num_shards)
        .map_shard(
            functools.partial(
                _process_training_shard,
                index_path=index_path,
                meta_path=meta_path,
                n_values=n_values,
                stride=config.stride,
                tokenizer_name=config.tokenizer,
                text_field=config.text_field,
                log_progress=config.track_progress,
            )
        )
        .write_jsonl(
            os.path.join(config.output_path, "tmp", "overlap_details-{shard:05d}.jsonl.gz"),
            skip_existing=config.skip_existing,
        )
    )

    Backend.execute(train_pipeline, max_parallelism=config.processes)

    success_path = os.path.join(config.output_path, ".SUCCESS")
    fs, fs_path = fsspec.core.url_to_fs(success_path)
    fs.makedirs(os.path.dirname(fs_path), exist_ok=True)
    with fs.open(fs_path, "w") as f:
        f.write("")

    return config.output_path
