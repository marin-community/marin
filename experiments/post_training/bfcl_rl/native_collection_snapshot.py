# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Seal completed native collection selections for the audited batch curator."""

import json
import zipfile
from dataclasses import dataclass

from marin.execution.artifact import Artifact
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.bfcl_rl.collect import ModelSource
from experiments.post_training.bfcl_rl.data import PARTITION_MANIFEST_SHA256
from experiments.post_training.bfcl_rl.offline_curate import validate_snapshot_files
from experiments.post_training.bfcl_rl.recovery_data import generation_collection_receipt, load_audited_partition


@dataclass(frozen=True)
class CollectionSnapshotConfig:
    terminal_uri: str
    data_root: str
    output_path: str


def seal_completed_collection(config: CollectionSnapshotConfig) -> Artifact:
    terminal = json.loads(StoragePath(config.terminal_uri).read_text())
    launch = terminal["config"]
    resolved = json.loads(StoragePath(launch["artifacts"]["resolved_config_uri"]).read_text())
    locator = launch["inputs"]["model"]
    model = ModelSource(locator["tokenizer_uri"], locator["tokenizer_revision"], locator["uri"], "completed")
    receipt = generation_collection_receipt(
        terminal, resolved, model=model, harness="native", partition=load_audited_partition(config.data_root)
    )
    source = StoragePath(launch["artifacts"]["attempts_root"])
    groups = {
        "canonical_results": list((source / "trace_jobs/eval_sessions/*/*/result.json").glob()),
        "literal_logs": list((StoragePath(launch["runtime"]["experiments_dir"]) / "logs/*_literal.jsonl").glob()),
        "archives": list((StoragePath(receipt.trajectory_root) / "schema_v6/archives/**/*.zip").glob()),
    }
    tasks = []
    for path in groups["canonical_results"]:
        trial = json.loads(path.read_text())
        if not trial["finished_at"]:
            raise ValueError("Collection snapshot requires completed canonical trials")
        tasks.append(trial["task_name"])
    if len(tasks) != len(set(tasks)) or set(tasks) != receipt.task_names:
        raise ValueError("Collection snapshot must cover the complete declared task selection")
    if not groups["literal_logs"] or not groups["archives"]:
        raise ValueError("Collection snapshot requires literal logs and retained archives")
    retained = set()
    for path in groups["archives"]:
        archive_fs, archive_key = filesystem_for(str(path))
        with archive_fs.open(archive_key, "rb", block_size=65536, cache_type="none") as stream:
            with zipfile.ZipFile(stream) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                for record in manifest["records"]:
                    if record["record_id"] in retained or archive.getinfo(record["entry"]).file_size != record["bytes"]:
                        raise ValueError("Collection snapshot contains duplicate or incomplete retained records")
                    retained.add(record["record_id"])
    if len(retained) != len(tasks):
        raise ValueError("Collection snapshot retained count differs from its completed tasks")
    manifest = {
        "schema_version": 1,
        "state": "sealed",
        "partition_manifest_sha256": PARTITION_MANIFEST_SHA256,
        "config": launch,
        "resolved": resolved,
        "task_names": sorted(tasks),
        "producer": {"run_id": terminal["result"]["run_id"], "state": terminal["result"]["state"]},
    }
    target = StoragePath(config.output_path)
    for group, paths in groups.items():
        files = []
        for index, path in enumerate(sorted(paths, key=str)):
            origin_fs, origin_key = filesystem_for(str(path))
            destination = target / "files" / group / f"{index}-{path.name}"
            target_fs, target_key = filesystem_for(str(destination))
            bucket, origin_object = origin_key.split("/", 1)
            target_bucket, target_object = target_key.split("/", 1)
            if not str(path).startswith("s3://") or not str(destination).startswith("s3://") or bucket != target_bucket:
                raise ValueError("Native snapshot copies must remain in the same regional S3 bucket")
            before = origin_fs.info(origin_key)
            target_fs.call_s3(
                "copy_object",
                Bucket=bucket,
                Key=target_object,
                CopySource={"Bucket": bucket, "Key": origin_object},
                CopySourceIfMatch=before["ETag"],
            )
            target_fs.invalidate_cache(target_key)
            after = target_fs.info(target_key)
            if (before["size"], before["ETag"]) != (after["size"], after["ETag"]):
                raise ValueError("Native snapshot copy differs from its source object")
            files.append(
                {
                    "uri": str(destination),
                    "bytes": after["size"],
                    "fingerprint_type": "etag",
                    "fingerprint": after["ETag"],
                }
            )
        manifest[group] = files
    validate_snapshot_files([file for group in groups for file in manifest[group]])
    (target / "snapshot.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return Artifact(path=config.output_path)
