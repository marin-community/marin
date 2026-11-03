# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay selected Focus Crawl UUIDs from bounded original WARC ranges."""

import argparse
import hashlib
import json
from dataclasses import asdict
from importlib.metadata import version

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from iris.cluster.client.job_info import get_job_info
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

from experiments.datakit.store.focus_provider_replay import FocusTarget, fetch_focus_documents, focus_html_extractor


def replay_focus_file(
    warc_filename: str,
    targets: list[FocusTarget],
    output_dir: str,
    *,
    justext_revision: str,
    max_record_bytes: int,
    max_cdx_bytes: int,
) -> dict:
    """Write hash-verified original candidates and a completion marker for one WARC."""
    binding = {
        "warc_filename": warc_filename,
        "targets": [asdict(target) for target in targets],
        "justext_revision": justext_revision,
    }
    extract_html = focus_html_extractor(justext_revision=justext_revision)
    with requests.Session() as session:
        records = fetch_focus_documents(
            warc_filename,
            targets,
            session=session,
            base_url="https://data.commoncrawl.org",
            extract_html=extract_html,
            max_record_bytes=max_record_bytes,
            max_cdx_bytes=max_cdx_bytes,
        )
    fs, key = filesystem_for(prefix_join(output_dir, "candidates.parquet"))
    fs.makedirs(key.rpartition("/")[0], exist_ok=True)
    with fs.open(key, "wb") as stream:
        pq.write_table(pa.Table.from_pylist(records), stream)
    report = {
        "binding": binding,
        "unique_candidates": len(records),
        "runtime_dependencies": {
            name: version(name) for name in ("jusText", "scikit-learn", "numpy", "resiliparse", "lxml", "warcio")
        },
        "model_settings": {"JUSTEXT_MODEL": "sklearn", "JUSTEXT_NO_DOWNLOAD": "1"},
        "text_sha256": {record["id"]: hashlib.sha256(record["text"].encode()).hexdigest() for record in records},
        "scope": "Original WARC candidates; selected occurrence expansion and global coverage require reconciliation.",
    }
    marker_fs, marker_key = filesystem_for(prefix_join(output_dir, "_complete.json"))
    with marker_fs.open(marker_key, "wt") as stream:
        json.dump(report, stream, sort_keys=True)
    return report


def replay_focus_partition(
    targets: dict[str, list[dict]],
    output_dir: str,
    *,
    task_index: int,
    task_count: int,
    justext_revision: str,
    max_record_bytes: int,
    max_cdx_bytes: int,
) -> dict:
    """Replay a disjoint deterministic subset of the frozen WARC target list."""
    if not 0 <= task_index < task_count:
        raise ValueError("Invalid Focus Crawl task partition")
    completed = []
    for index, filename in enumerate(sorted(targets)):
        if index % task_count != task_index:
            continue
        destination = prefix_join(output_dir, f"file={index:06d}")
        report = replay_focus_file(
            filename,
            [FocusTarget(**target) for target in targets[filename]],
            destination,
            justext_revision=justext_revision,
            max_record_bytes=max_record_bytes,
            max_cdx_bytes=max_cdx_bytes,
        )
        completed.append({"file_index": index, "output": destination, "candidates": report["unique_candidates"]})
        print(json.dumps(completed[-1]), flush=True)
    report = {"task_index": task_index, "task_count": task_count, "completed": completed}
    fs, key = filesystem_for(prefix_join(output_dir, f"task-{task_index:05d}-of-{task_count:05d}.json"))
    with fs.open(key, "wt") as stream:
        json.dump(report, stream, sort_keys=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True, help="JSON mapping WARC filenames to selected target lists")
    parser.add_argument("--file-index", type=int, help="Replay one sorted WARC file; otherwise partition the full list")
    parser.add_argument("--output", required=True)
    parser.add_argument("--justext-revision", required=True)
    parser.add_argument("--max-record-bytes", required=True, type=int)
    parser.add_argument("--max-cdx-bytes", required=True, type=int)
    args = parser.parse_args()
    target_fs, target_key = filesystem_for(args.targets)
    with target_fs.open(target_key, "rt") as stream:
        targets = json.load(stream)
    if args.file_index is None:
        info = get_job_info()
        if info is None:
            raise ValueError("Full Focus Crawl replay requires Iris task partition information")
        replay_focus_partition(
            targets,
            args.output,
            task_index=info.task_index,
            task_count=info.num_tasks,
            justext_revision=args.justext_revision,
            max_record_bytes=args.max_record_bytes,
            max_cdx_bytes=args.max_cdx_bytes,
        )
        return
    filename = sorted(targets)[args.file_index]
    report = replay_focus_file(
        filename,
        [FocusTarget(**target) for target in targets[filename]],
        args.output,
        justext_revision=args.justext_revision,
        max_record_bytes=args.max_record_bytes,
        max_cdx_bytes=args.max_cdx_bytes,
    )
    print(json.dumps({"warc_filename": filename, "unique_candidates": report["unique_candidates"]}))


if __name__ == "__main__":
    main()
