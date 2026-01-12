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

import json
import logging
import os
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field

import fsspec
import msgspec
from zephyr import Backend, Dataset, load_file, load_jsonl

from experiments.train_test_overlap.ngram_new.utils import collect_eval_file_specs, record_id

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OverlapReduceConfig:
    """Configuration for the n-gram overlap reduce stage."""

    map_output_path: str
    output_path: str
    eval_dataset_paths: str | list[str] | None = None
    eval_text_field: str = "text"
    ngram_lengths: list[int] = field(default_factory=list)
    processes: int = 32
    skip_existing: bool = True
    write_details: bool = False


def _reduce_overlap_group(key: tuple[str, int], items: Iterator[dict]) -> dict:
    eval_dataset, n_val = key
    instance_ids: set[str] = set()
    for item in items:
        instance_id = item.get("eval_instance_id")
        if instance_id is None:
            instance_id = f"{item['eval_path']}:{item['eval_row']}"
        instance_ids.add(instance_id)
    return {"eval_dataset": eval_dataset, "n": n_val, "instance_ids": sorted(instance_ids)}


def _reduce_overlap_by_train_path(key: tuple[str, int, str], items: Iterator[dict]) -> dict:
    eval_dataset, n_val, train_path = key
    instance_ids: set[str] = set()
    doc_ids: set[str] = set()
    for item in items:
        instance_id = item.get("eval_instance_id")
        if instance_id is None:
            instance_id = f"{item['eval_path']}:{item['eval_row']}"
        instance_ids.add(instance_id)
        doc_id = item.get("train_doc_id")
        if doc_id:
            doc_ids.add(doc_id)
    sorted_ids = sorted(instance_ids)
    return {
        "eval_dataset": eval_dataset,
        "n": n_val,
        "train_path": train_path,
        "train_doc_ids": sorted(doc_ids),
        "instance_ids": sorted_ids,
        "overlap_count": len(sorted_ids),
    }


def _load_test_instance_counts(counts_path: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in load_jsonl(counts_path):
        counts[record["eval_dataset"]] = record["num_instances"]
    return counts


def _compute_test_instance_counts(eval_dataset_paths: str | list[str], text_field: str) -> dict[str, int]:
    counts: dict[str, set[str]] = defaultdict(set)
    eval_specs = collect_eval_file_specs(eval_dataset_paths)
    for spec in eval_specs:
        eval_dataset = spec["eval_dataset"]
        for record in load_file(spec["path"]):
            text = record.get(text_field)
            if text is None:
                continue
            counts[eval_dataset].add(record_id(record))
    return {dataset: len(instance_ids) for dataset, instance_ids in counts.items()}


def _compute_test_instance_links(eval_dataset_paths: str | list[str], text_field: str) -> dict[tuple[str, str], str]:
    links: dict[tuple[str, str], str] = {}
    eval_specs = collect_eval_file_specs(eval_dataset_paths)
    for spec in eval_specs:
        eval_dataset = spec["eval_dataset"]
        for record in load_file(spec["path"]):
            text = record.get(text_field)
            if text is None:
                continue
            instance_id = record_id(record)
            links[(eval_dataset, instance_id)] = spec["path"]
    return links


def _attach_instance_links(
    eval_dataset: str, instance_ids: list[str], link_map: dict[tuple[str, str], str]
) -> list[str]:
    return [link_map.get((eval_dataset, instance_id), instance_id) for instance_id in instance_ids]


def _load_test_index_n_values(map_output_path: str) -> list[int]:
    index_path = os.path.join(map_output_path, "tmp", "test_index.msgpack")
    fs, fs_path = fsspec.core.url_to_fs(index_path)
    if not fs.exists(fs_path):
        return []
    with fs.open(fs_path, "rb") as f:
        index = msgspec.msgpack.decode(f.read())
    return sorted(int(n) for n in index.keys())


def run_overlap_reduce(config: OverlapReduceConfig) -> str:
    """Aggregate overlap events into per-dataset statistics."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    stats_dir = os.path.join(config.output_path, "stats")
    stats_path = os.path.join(stats_dir, "overlap_stats.jsonl")
    train_path_stats_path = os.path.join(stats_dir, "overlap_stats_by_train_path.jsonl")
    details_path = os.path.join(stats_dir, "overlap_details.jsonl.gz")
    stats_fs, stats_fs_path = fsspec.core.url_to_fs(stats_path)
    train_fs, train_fs_path = fsspec.core.url_to_fs(train_path_stats_path)
    details_fs, details_fs_path = fsspec.core.url_to_fs(details_path)
    stats_exists = stats_fs.exists(stats_fs_path)
    train_path_exists = train_fs.exists(train_fs_path)
    details_exists = details_fs.exists(details_fs_path) if config.write_details else True
    if config.skip_existing and stats_exists and train_path_exists and details_exists:
        logger.info(
            "Skipping reduce; outputs exist at %s, %s, and %s",
            stats_path,
            train_path_stats_path,
            details_path,
        )
        return config.output_path

    overlaps_pattern = os.path.join(config.map_output_path, "tmp", "overlap_details-*.jsonl.gz")
    overlaps_dataset = Dataset.from_files(overlaps_pattern).flat_map(load_jsonl)

    link_map: dict[tuple[str, str], str] = {}
    if config.eval_dataset_paths is not None:
        link_map = _compute_test_instance_links(config.eval_dataset_paths, config.eval_text_field)

    if config.write_details and (not config.skip_existing or not details_exists):
        details_fs.makedirs(os.path.dirname(details_fs_path), exist_ok=True)
        with fsspec.open(details_path, "wt", compression="infer") as f:
            for record in Backend.execute(overlaps_dataset, max_parallelism=config.processes):
                f.write(json.dumps(record) + "\n")

    if not config.skip_existing or not train_path_exists:
        train_path_pipeline = overlaps_dataset.map(
            lambda r: {
                "eval_dataset": r["eval_dataset"],
                "n": r["n"],
                "train_path": r["train_path"],
                "train_doc_id": r.get("train_doc_id"),
                "eval_instance_id": r.get("eval_instance_id"),
                "eval_path": r.get("eval_path"),
                "eval_row": r.get("eval_row"),
            }
        ).group_by(
            key=lambda r: (r["eval_dataset"], r["n"], r["train_path"]),
            reducer=_reduce_overlap_by_train_path,
            num_output_shards=1,
        )
        train_fs.makedirs(os.path.dirname(train_fs_path), exist_ok=True)
        with train_fs.open(train_fs_path, "w") as f:
            for record in Backend.execute(train_path_pipeline, max_parallelism=config.processes):
                record["instance_links"] = _attach_instance_links(
                    record["eval_dataset"], record["instance_ids"], link_map
                )
                f.write(json.dumps(record) + "\n")

    if not config.skip_existing or not stats_exists:
        overlaps_pipeline = overlaps_dataset.map(
            lambda r: {
                "eval_dataset": r["eval_dataset"],
                "n": r["n"],
                "eval_instance_id": r.get("eval_instance_id"),
                "eval_path": r.get("eval_path"),
                "eval_row": r.get("eval_row"),
            }
        ).group_by(key=lambda r: (r["eval_dataset"], r["n"]), reducer=_reduce_overlap_group, num_output_shards=1)
        overlap_records = list(Backend.execute(overlaps_pipeline, max_parallelism=config.processes))

        counts_path = os.path.join(config.map_output_path, "tmp", "test_instance_counts.jsonl")
        counts_fs, counts_fs_path = fsspec.core.url_to_fs(counts_path)
        if counts_fs.exists(counts_fs_path):
            counts = _load_test_instance_counts(counts_path)
        else:
            if config.eval_dataset_paths is None:
                raise ValueError("eval_dataset_paths is required when test_instance_counts.jsonl is missing")
            counts = _compute_test_instance_counts(config.eval_dataset_paths, config.eval_text_field)

        n_values = (
            sorted(set(config.ngram_lengths))
            if config.ngram_lengths
            else sorted({record["n"] for record in overlap_records})
        )
        if not n_values and counts:
            raise ValueError("ngram_lengths must be provided when no overlaps are present")

        overlap_map: dict[tuple[str, int], list[str]] = {
            (record["eval_dataset"], record["n"]): record["instance_ids"] for record in overlap_records
        }

        output_records: list[dict] = []
        for eval_dataset, num_instances in sorted(counts.items()):
            for n_val in n_values:
                instance_ids = overlap_map.get((eval_dataset, n_val), [])
                output_records.append(
                    {
                        "eval_dataset": eval_dataset,
                        "n": n_val,
                        "num_instances": num_instances,
                        "instance_ids": instance_ids,
                        "instance_links": _attach_instance_links(eval_dataset, instance_ids, link_map),
                    }
                )

        stats_fs.makedirs(os.path.dirname(stats_fs_path), exist_ok=True)
        with stats_fs.open(stats_fs_path, "w") as f:
            for record in output_records:
                f.write(json.dumps(record) + "\n")

    success_path = os.path.join(config.output_path, ".SUCCESS")
    success_fs, success_fs_path = fsspec.core.url_to_fs(success_path)
    success_fs.makedirs(os.path.dirname(success_fs_path), exist_ok=True)
    with success_fs.open(success_fs_path, "w") as f:
        f.write("")

    logger.info("Wrote overlap stats to %s", stats_path)
    if config.write_details:
        logger.info("Wrote overlap details to %s", details_path)
    return config.output_path
