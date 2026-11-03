# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Copy only verified selected indices and completed provider-file candidates."""

import argparse
import hashlib
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from marin.datakit.normalize import generate_id
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

from experiments.datakit.scripts.metadata_cluster_reconstruction import (
    CANDIDATE_SCHEMA,
    COPY_CHUNK_BYTES,
    _candidate_digest,
    _scan_binding,
    _transfer,
    _verify_plan,
)
from experiments.datakit.scripts.metadata_cluster_selection import _digest, _read_json, _write_json

EXTERNAL_SOURCES = {"common_corpus/english", "common-crawl-focus-2026-22"}


def _sha256(path: str) -> str:
    fs, key = filesystem_for(path)
    digest = hashlib.sha256()
    with fs.open(key, "rb") as stream:
        while chunk := stream.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _size(path: str) -> int:
    fs, key = filesystem_for(path)
    return fs.size(key)


def checkpoint_inventory(old: dict, new: dict, marker_paths: set[str], workers: int) -> dict:
    """Require unchanged source pins and inventory only published complete checkpoints."""
    _verify_plan(old)
    _verify_plan(new)
    if _digest(old["sources"]) != _digest(new["sources"]):
        raise ValueError("Checkpoint relocation must preserve every frozen source manifest")
    if old["output_prefix"] == new["output_prefix"]:
        raise ValueError("Migration requires a fresh output prefix")

    def source_inventory(source: dict) -> dict:
        name = source["marin_source"]
        root = old["source_roots"][name]
        if new["source_roots"][name] == root:
            raise ValueError("Every migrated source must use a fresh output root")
        index_marker_path = prefix_join(root, "index/_complete.json")
        if index_marker_path not in marker_paths:
            raise ValueError("Required selected index completion is absent: " + name)
        index = _read_json(index_marker_path)
        if (
            index["binding"] != {"source_sha256": source["manifest_sha256"]}
            or index["selected_occurrences"] != source["selected_occurrences"]
            or not index["index_path"].startswith(root + "/index/")
        ):
            raise ValueError("Selected index differs from its frozen source: " + name)
        candidates = []
        candidate_prefix = prefix_join(root, "candidates/")
        for path in sorted(marker_paths):
            if not path.startswith(candidate_prefix):
                continue
            marker = _read_json(path)
            file_index = marker["binding"]["file_index"]
            if (
                not marker["complete"]
                or marker["scope"] != "provider_file_candidates"
                or marker["binding"] != _scan_binding(old, source, file_index, index)
                or path != prefix_join(root, f"candidates/file={file_index:06d}/_complete.json")
                or not marker["candidates"].startswith(prefix_join(root, f"candidates/file={file_index:06d}/"))
            ):
                raise ValueError("Candidate checkpoint does not bind one complete pinned provider file")
            candidates.append({"marker": marker, "size": _size(marker["candidates"])})
        return {
            "marin_source": name,
            "old_root": root,
            "new_root": new["source_roots"][name],
            "index": index,
            "index_size": _size(index["index_path"]),
            "candidates": candidates,
        }

    sources = [s for s in old["sources"] if s["marin_source"] not in EXTERNAL_SOURCES]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        entries = list(pool.map(source_inventory, sources))
    return {
        "old_plan_sha256": old["manifest_sha256"],
        "new_plan_sha256": new["manifest_sha256"],
        "indices": len(entries),
        "candidate_files": sum(len(e["candidates"]) for e in entries),
        "index_bytes": sum(e["index_size"] for e in entries),
        "candidate_bytes": sum(c["size"] for e in entries for c in e["candidates"]),
        "sources": entries,
    }


def migrate_source_checkpoints(entry: dict) -> dict:
    """Publish destination markers only after source and destination content verification."""
    root, index = entry["new_root"], entry["index"]
    destination = prefix_join(root, "index/migrated=" + index["index_sha256"] + "/selected.sqlite")
    fs, key = filesystem_for(destination)
    fs.makedirs(key.rsplit("/", 1)[0], exist_ok=True)
    if (
        _transfer(index["index_path"], destination) != index["index_sha256"]
        or _sha256(destination) != index["index_sha256"]
    ):
        raise ValueError("Selected-index bytes differ from the frozen SHA256")
    new_index = {**index, "index_path": destination}
    marker_path = prefix_join(root, "index/_complete.json")
    _write_json(marker_path, new_index)
    if _read_json(marker_path) != new_index:
        raise ValueError("Destination selected-index marker differs on readback")
    migrated_candidates = []
    for candidate in entry["candidates"]:
        marker = candidate["marker"]
        digest = hashlib.sha256()
        rows = 0
        file_hash = _sha256(marker["candidates"])
        fs, key = filesystem_for(marker["candidates"])
        with fs.open(key, "rb") as stream:
            parquet = pq.ParquetFile(stream)
            if not parquet.schema_arrow.equals(CANDIDATE_SCHEMA):
                raise ValueError("Candidate checkpoint has an unexpected schema")
            for batch in parquet.iter_batches(batch_size=512):
                for row in batch.to_pylist():
                    if generate_id(row["text"]) != row["id"]:
                        raise ValueError("Candidate text does not match its canonical selected ID")
                    _candidate_digest(digest, row["id"], row["text"])
                    rows += 1
        if rows != marker["candidate_rows"] or digest.hexdigest() != marker["candidate_digest"]:
            raise ValueError("Candidate content differs from its completed scan")
        file_index = marker["binding"]["file_index"]
        destination = prefix_join(
            root, f"candidates/file={file_index:06d}/attempt=migrated-{uuid.uuid4().hex}/candidates.parquet"
        )
        fs, key = filesystem_for(destination)
        fs.makedirs(key.rsplit("/", 1)[0], exist_ok=True)
        if _transfer(marker["candidates"], destination) != file_hash or _sha256(destination) != file_hash:
            raise ValueError("Destination candidate bytes differ from verified source bytes")
        new_marker = {**marker, "candidates": destination}
        marker_path = prefix_join(root, f"candidates/file={file_index:06d}/_complete.json")
        _write_json(marker_path, new_marker)
        if _read_json(marker_path) != new_marker:
            raise ValueError("Destination candidate marker differs on readback")
        migrated_candidates.append({"file_index": file_index, "file_sha256": file_hash, "candidate_rows": rows})
    return {
        "marin_source": entry["marin_source"],
        "index_sha256": index["index_sha256"],
        "candidates": migrated_candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inventory", "copy"), required=True)
    parser.add_argument("--old-plan", required=True)
    parser.add_argument("--new-plan", required=True)
    parser.add_argument("--marker-paths", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    if args.mode == "inventory":
        inventory = checkpoint_inventory(
            _read_json(args.old_plan),
            _read_json(args.new_plan),
            set(Path(args.marker_paths).read_text().splitlines()),
            args.workers,
        )
        _write_json(args.inventory, inventory)
        print(json.dumps({k: v for k, v in inventory.items() if k != "sources"}))
        return
    inventory = _read_json(args.inventory)
    old, new = _read_json(args.old_plan), _read_json(args.new_plan)
    _verify_plan(old)
    _verify_plan(new)
    if inventory["old_plan_sha256"] != old["manifest_sha256"] or inventory["new_plan_sha256"] != new["manifest_sha256"]:
        raise ValueError("Migration inventory belongs to different plans")
    entries = {entry["marin_source"]: entry for entry in inventory["sources"]}
    expected = {source["marin_source"] for source in new["sources"]} - EXTERNAL_SOURCES
    if set(entries) != expected or len(entries) != len(inventory["sources"]):
        raise ValueError("Migration inventory must cover every generic source exactly once")
    for name, entry in entries.items():
        if (
            entry["old_root"] != old["source_roots"][name]
            or entry["new_root"] != new["source_roots"][name]
            or entry["new_root"] == entry["old_root"]
            or not entry["new_root"].startswith("s3://marin-us-east-02a/")
        ):
            raise ValueError("Checkpoint relocation must write only the frozen fresh 02a source roots")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        completed = list(pool.map(migrate_source_checkpoints, inventory["sources"]))
    report = {
        "inventory_sha256": _digest(inventory),
        "complete": True,
        "sources": completed,
        "indices": len(completed),
        "candidate_files": sum(len(s["candidates"]) for s in completed),
    }
    _write_json(args.report, report)
    print(json.dumps({k: v for k, v in report.items() if k != "sources"}))


if __name__ == "__main__":
    main()
