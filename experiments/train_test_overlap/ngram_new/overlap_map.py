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
from zephyr import Backend, Dataset, load_file, load_jsonl

from experiments.train_test_overlap.ngram_new.utils import (
    collect_eval_file_specs,
    collect_input_files,
    get_tokenizer,
    iter_ngrams,
    record_id,
)

logger = logging.getLogger(__name__)

_INDEX_CACHE: dict[str, dict[int, dict[str, list[list[str]]]]] = {}


@dataclass(frozen=True)
class OverlapMapConfig:
    """Configuration for the n-gram overlap map stage."""

    input_path: str | list[str]
    eval_dataset_paths: str | list[str]
    output_path: str
    ngram_lengths: list[int] = field(default_factory=lambda: [5, 10, 15])
    stride: int = 0
    tokenizer: str = "default"
    text_field: str = "text"
    processes: int = 32
    num_shards: int = 128
    skip_existing: bool = True
    write_test_instance_counts: bool = True
    track_progress: bool = True


def _load_test_index(index_path: str) -> dict[int, dict[str, list[list[str]]]]:
    cached = _INDEX_CACHE.get(index_path)
    if cached is not None:
        return cached
    fs, fs_path = fsspec.core.url_to_fs(index_path)
    with fs.open(fs_path, "rb") as f:
        index = msgspec.msgpack.decode(f.read())
    _INDEX_CACHE[index_path] = index
    return index


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
    output_path: str,
    skip_existing: bool,
    track_progress: bool,
) -> tuple[str, dict[str, int]]:
    index_path = os.path.join(output_path, "tmp", "test_index.msgpack")
    fs, fs_path = fsspec.core.url_to_fs(index_path)
    if skip_existing and fs.exists(fs_path):
        counts: dict[str, int] = {}
        if track_progress:
            print(f"[overlap_map] test index exists at {index_path}", flush=True)
        return index_path, counts

    tokenizer = get_tokenizer(tokenizer_name)
    index: dict[int, dict[str, set[tuple[str, str]]]] = {n: {} for n in n_values}
    counts: dict[str, set[str]] = defaultdict(set)

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

        for record in load_jsonl(spec["path"]):
            record_total += 1
            text = record.get("text")
            if text is None:
                record_missing_text += 1
                continue
            record_with_text += 1
            instance_id = record_id(record)
            counts[eval_dataset].add(instance_id)
            tokens = tokenizer.tokenize(text)
            token_count += len(tokens)
            for n in n_values:
                seen: set[str] = set()
                for ngram in iter_ngrams(tokens, n, stride):
                    if ngram in seen:
                        continue
                    seen.add(ngram)
                    ngram_counts[n] += 1
                    entry_set = index[n].setdefault(ngram, set())
                    entry_set.add((eval_dataset, instance_id))

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

    index_out: dict[int, dict[str, list[list[str]]]] = {}
    for n_val, ngrams in index.items():
        index_out[n_val] = {ngram: [list(entry) for entry in entries] for ngram, entries in ngrams.items()}

    fs.makedirs(os.path.dirname(fs_path), exist_ok=True)
    with fs.open(fs_path, "wb") as f:
        f.write(msgspec.msgpack.encode(index_out))

    counts_out = {dataset: len(instance_ids) for dataset, instance_ids in counts.items()}
    return index_path, counts_out


def _process_training_shard(
    specs: Iterator[dict],
    *,
    index_path: str,
    n_values: list[int],
    stride: int,
    tokenizer_name: str,
    text_field: str,
    log_progress: bool,
) -> Iterator[dict]:
    tokenizer = get_tokenizer(tokenizer_name)
    index = _load_test_index(index_path)

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

        for record in load_file(spec["path"]):
            record_total += 1
            text = record.get(text_field)
            if text is None:
                record_missing_text += 1
                continue
            record_with_text += 1
            record_overlaps: dict[tuple[str, int], set[str]] = defaultdict(set)
            tokens = tokenizer.tokenize(text)
            token_count += len(tokens)
            for n in n_values:
                index_n = index.get(n)
                if not index_n:
                    continue
                for ngram in iter_ngrams(tokens, n, stride):
                    ngram_counts[n] += 1
                    entry_keys = index_n.get(ngram)
                    if not entry_keys:
                        continue
                    for eval_dataset, instance_id in entry_keys:
                        record_overlaps[(eval_dataset, n)].add(instance_id)

            if record_overlaps:
                doc_id = record.get("id")
                train_doc_id = str(doc_id) if doc_id is not None else spec["path"]
                for (eval_dataset, n_val), instance_ids in record_overlaps.items():
                    overlap_records += 1
                    yield {
                        "eval_dataset": eval_dataset,
                        "n": n_val,
                        "instance_ids": sorted(instance_ids),
                        "train_path": spec["path"],
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
            f"text_field={config.text_field}",
            flush=True,
        )

    index_path, counts = _build_test_index(
        eval_specs,
        n_values,
        config.stride,
        config.tokenizer,
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
                n_values=n_values,
                stride=config.stride,
                tokenizer_name=config.tokenizer,
                text_field=config.text_field,
                log_progress=config.track_progress,
            )
        )
        .write_jsonl(
            os.path.join(config.output_path, "tmp", "overlap_instances-{shard:05d}.jsonl.gz"),
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
