# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repair the audited Delphi TPP40 Europe Stack-Edu payload delta."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from marin.transform.stack_edu.hydrate import STACK_EDU_REQUIRED_COLUMNS, stack_edu_record_id
from rigging.filesystem import open_url, url_to_fs
from rigging.filesystem.cluster_config import marin_prefix_for_region, region_from_metadata
from rigging.filesystem.cross_region import cached_marin_region
from zephyr.readers import InputFileSpec, load_jsonl, load_parquet
from zephyr.writers import write_jsonl_file

MANIFEST_SCHEMA_VERSION = 1
BUNDLE_RECORD_OVERHEAD_LIMIT_BYTES = 4_096


@dataclass(frozen=True)
class RepairTask:
    """One source shard and its corresponding Europe target shard."""

    language: str
    source_region: str
    source_shard: str
    target_input_file: str
    target_shard: str
    target_metrics_path: str
    row_start: int
    row_end: int
    add_blob_ids: tuple[str, ...]
    remove_blob_ids: tuple[str, ...]
    baseline_missing_blob_ids: tuple[str, ...]
    expected_metric: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RepairTask:
        return cls(
            language=value["language"],
            source_region=value["source_region"],
            source_shard=value["source_shard"],
            target_input_file=value["target_input_file"],
            target_shard=value["target_shard"],
            target_metrics_path=value["target_metrics_path"],
            row_start=value["row_start"],
            row_end=value["row_end"],
            add_blob_ids=tuple(value["add_blob_ids"]),
            remove_blob_ids=tuple(value["remove_blob_ids"]),
            baseline_missing_blob_ids=tuple(value["baseline_missing_blob_ids"]),
            expected_metric=value["expected_metric"],
        )


@dataclass(frozen=True)
class RepairManifest:
    """Frozen contract for the cross-region Stack-Edu payload repair."""

    repair_id: str
    target_region: str
    expected_add_records: int
    expected_remove_records: int
    expected_transfer_content_bytes: int
    source_record_counts: dict[str, int]
    source_content_bytes: dict[str, int]
    tasks: tuple[RepairTask, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RepairManifest:
        if value["schema_version"] != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"Unsupported repair manifest schema: {value['schema_version']}")
        manifest = cls(
            repair_id=value["repair_id"],
            target_region=value["target_region"],
            expected_add_records=value["expected_add_records"],
            expected_remove_records=value["expected_remove_records"],
            expected_transfer_content_bytes=value["expected_transfer_content_bytes"],
            source_record_counts=value["source_record_counts"],
            source_content_bytes=value["expected_transfer_content_bytes_by_source_region"],
            tasks=tuple(RepairTask.from_dict(task) for task in value["tasks"]),
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        add_ids = [blob_id for task in self.tasks for blob_id in task.add_blob_ids]
        remove_ids = [blob_id for task in self.tasks for blob_id in task.remove_blob_ids]
        if len(add_ids) != self.expected_add_records or len(set(add_ids)) != len(add_ids):
            raise ValueError("Repair manifest additions are incomplete or duplicated")
        if len(remove_ids) != self.expected_remove_records or len(set(remove_ids)) != len(remove_ids):
            raise ValueError("Repair manifest removals are incomplete or duplicated")
        if set(add_ids) & set(remove_ids):
            raise ValueError("A blob id cannot be both added and removed")
        if sum(self.source_content_bytes.values()) != self.expected_transfer_content_bytes:
            raise ValueError("Per-region transfer bytes do not equal the frozen total")

        actual_source_counts: dict[str, int] = {region: 0 for region in self.source_record_counts}
        target_shards: set[str] = set()
        target_metrics_paths: set[str] = set()
        for task in self.tasks:
            if task.source_region not in actual_source_counts:
                raise ValueError(f"Unknown source region: {task.source_region}")
            actual_source_counts[task.source_region] += len(task.add_blob_ids)
            if task.target_shard in target_shards:
                raise ValueError(f"Duplicate target shard: {task.target_shard}")
            target_shards.add(task.target_shard)
            if task.target_metrics_path in target_metrics_paths:
                raise ValueError(f"Duplicate target metrics path: {task.target_metrics_path}")
            target_metrics_paths.add(task.target_metrics_path)
            for name, values in (
                ("add_blob_ids", task.add_blob_ids),
                ("remove_blob_ids", task.remove_blob_ids),
                ("baseline_missing_blob_ids", task.baseline_missing_blob_ids),
            ):
                if len(values) != len(set(values)):
                    raise ValueError(f"Duplicate {name} for {task.target_shard}")
            if not set(task.remove_blob_ids) <= set(task.baseline_missing_blob_ids):
                raise ValueError(f"Removals must be baseline-missing rows: {task.target_shard}")
            expected_count = task.row_end - task.row_start - len(task.baseline_missing_blob_ids)
            if task.expected_metric["count"] != expected_count:
                raise ValueError(f"Unexpected frozen count for {task.target_shard}")
            if task.expected_metric["path"] != task.target_shard:
                raise ValueError(f"Metric path does not match target shard: {task.target_shard}")
            if task.expected_metric["input_file"] != task.target_input_file:
                raise ValueError(f"Metric input does not match target input: {task.target_shard}")
            if task.expected_metric["row_start"] != task.row_start or task.expected_metric["row_end"] != task.row_end:
                raise ValueError(f"Metric row range does not match task: {task.target_shard}")
            if task.expected_metric["missing_blob"] != len(task.baseline_missing_blob_ids):
                raise ValueError(f"Metric missing count does not match task: {task.target_shard}")
        if actual_source_counts != self.source_record_counts:
            raise ValueError(f"Source record counts differ: {actual_source_counts} != {self.source_record_counts}")

    @property
    def add_blob_ids(self) -> set[str]:
        return {blob_id for task in self.tasks for blob_id in task.add_blob_ids}

    @property
    def remove_blob_ids(self) -> set[str]:
        return {blob_id for task in self.tasks for blob_id in task.remove_blob_ids}


@dataclass
class RepairMeasurement:
    """Observed delta state across one apply attempt."""

    inserted_blob_ids: set[str] = field(default_factory=set)
    already_present_blob_ids: set[str] = field(default_factory=set)
    removed_blob_ids: set[str] = field(default_factory=set)
    already_absent_blob_ids: set[str] = field(default_factory=set)

    def validate(self, manifest: RepairManifest) -> None:
        if self.inserted_blob_ids & self.already_present_blob_ids:
            raise ValueError("An addition cannot be both inserted and already present")
        if self.removed_blob_ids & self.already_absent_blob_ids:
            raise ValueError("A removal cannot be both removed and already absent")
        verified_additions = self.inserted_blob_ids | self.already_present_blob_ids
        verified_removals = self.removed_blob_ids | self.already_absent_blob_ids
        if verified_additions != manifest.add_blob_ids:
            raise ValueError("Observed additions do not cover the frozen manifest")
        if verified_removals != manifest.remove_blob_ids:
            raise ValueError("Observed removals do not cover the frozen manifest")

    def result(self) -> dict[str, Any]:
        return {
            "inserted_this_attempt": len(self.inserted_blob_ids),
            "inserted_blob_ids": sorted(self.inserted_blob_ids),
            "already_present": len(self.already_present_blob_ids),
            "already_present_blob_ids": sorted(self.already_present_blob_ids),
            "removed_this_attempt": len(self.removed_blob_ids),
            "removed_blob_ids": sorted(self.removed_blob_ids),
            "already_absent": len(self.already_absent_blob_ids),
            "already_absent_blob_ids": sorted(self.already_absent_blob_ids),
            "verified_additions": len(self.inserted_blob_ids | self.already_present_blob_ids),
            "verified_removals": len(self.removed_blob_ids | self.already_absent_blob_ids),
        }


def load_repair_manifest(path: str, expected_sha256: str) -> RepairManifest:
    """Load a manifest only when its exact byte hash matches the reviewed value."""
    with open_url(path, "rb") as handle:
        payload = handle.read()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Manifest SHA-256 mismatch: {actual_sha256} != {expected_sha256}")
    return RepairManifest.from_dict(json.loads(payload))


def _assert_storage_prefix(path: str, storage_prefix: str) -> None:
    normalized_prefix = storage_prefix.rstrip("/")
    if path != normalized_prefix and not path.startswith(f"{normalized_prefix}/"):
        raise ValueError(f"Path is outside required storage prefix {storage_prefix}: {path}")


def _assert_running_region(expected_region: str) -> None:
    metadata_region = region_from_metadata()
    if metadata_region is None:
        raise ValueError(f"Cannot determine the VM region from GCP metadata; expected {expected_region}")
    if metadata_region != expected_region:
        raise ValueError(f"Repair must run on a VM in {expected_region}, not {metadata_region}")
    actual_region = cached_marin_region()
    if actual_region is None:
        raise ValueError(f"Cannot determine the VM region; expected {expected_region}")
    if actual_region != expected_region:
        raise ValueError(f"Repair must run in {expected_region}, not {actual_region}")


def _read_single_jsonl(path: str) -> dict[str, Any]:
    rows = list(load_jsonl(path))
    if len(rows) != 1:
        raise ValueError(f"Expected one JSONL record at {path}, found {len(rows)}")
    return rows[0]


def _object_fingerprint(path: str) -> dict[str, Any]:
    fs, resolved_path = url_to_fs(path)
    info = fs.info(resolved_path)
    fingerprint: dict[str, Any] = {"path": path, "size": int(info["size"])}
    for key in ("crc32c", "md5Hash", "md5", "etag", "generation", "version_id"):
        if value := info.get(key):
            fingerprint[key] = str(value)
    if len(fingerprint) == 2:
        fingerprint["checksum"] = str(fs.checksum(resolved_path))
    return fingerprint


def _physical_file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    fs, resolved_path = url_to_fs(path)
    with fs.open(resolved_path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_size_limit(content_bytes: int, record_count: int) -> int:
    return content_bytes + BUNDLE_RECORD_OVERHEAD_LIMIT_BYTES * record_count


def _validate_bundle_size(path: str, content_bytes: int, record_count: int) -> int:
    size = int(_object_fingerprint(path)["size"])
    size_limit = _bundle_size_limit(content_bytes, record_count)
    if size > size_limit:
        raise ValueError(f"Repair bundle exceeds transfer ceiling: {size} > {size_limit} bytes")
    return size


def _validate_bundle_records(
    records: list[dict[str, Any]], expected_blob_ids: set[str], expected_content_bytes: int
) -> dict[str, dict[str, Any]]:
    by_blob_id: dict[str, dict[str, Any]] = {}
    content_bytes = 0
    for record in records:
        metadata = record["metadata"]
        blob_id = metadata["blob_id"]
        if blob_id in by_blob_id:
            raise ValueError(f"Duplicate repair record: {blob_id}")
        by_blob_id[blob_id] = record
        content_bytes += int(metadata["length_bytes"])
    if set(by_blob_id) != expected_blob_ids:
        missing = sorted(expected_blob_ids - set(by_blob_id))
        extra = sorted(set(by_blob_id) - expected_blob_ids)
        raise ValueError(f"Repair bundle ids differ; missing={missing}, extra={extra}")
    if content_bytes != expected_content_bytes:
        raise ValueError(f"Repair content bytes differ: {content_bytes} != {expected_content_bytes}")
    return by_blob_id


def extract_repair_bundle(
    manifest: RepairManifest,
    manifest_sha256: str,
    source_region: str,
    output_path: str,
    result_path: str,
) -> dict[str, Any]:
    """Extract only one source region's audited records into a tiny bundle."""
    _assert_running_region(source_region)
    storage_prefix = marin_prefix_for_region(source_region)
    _assert_storage_prefix(output_path, storage_prefix)
    _assert_storage_prefix(result_path, storage_prefix)
    tasks = [task for task in manifest.tasks if task.source_region == source_region and task.add_blob_ids]
    if not tasks:
        raise ValueError(f"No additions assigned to source region {source_region}")
    for task in tasks:
        _assert_storage_prefix(task.source_shard, storage_prefix)

    expected_blob_ids = {blob_id for task in tasks for blob_id in task.add_blob_ids}
    expected_count = manifest.source_record_counts[source_region]
    expected_content_bytes = manifest.source_content_bytes[source_region]
    if len(expected_blob_ids) != expected_count:
        raise ValueError(f"Source assignment count differs: {len(expected_blob_ids)} != {expected_count}")

    output_fs, resolved_output = url_to_fs(output_path)
    result_fs, resolved_result = url_to_fs(result_path)
    if result_fs.exists(resolved_result):
        result = _read_single_jsonl(result_path)
        if result["manifest_sha256"] != manifest_sha256 or result["source_region"] != source_region:
            raise ValueError(f"Existing extraction result is for a different contract: {result_path}")
        records = list(load_jsonl(output_path))
        _validate_bundle_records(records, expected_blob_ids, expected_content_bytes)
        if result["bundle_fingerprint"] != _object_fingerprint(output_path):
            raise ValueError(f"Existing repair bundle fingerprint changed: {output_path}")
        if result["bundle_sha256"] != _physical_file_sha256(output_path):
            raise ValueError(f"Existing repair bundle SHA-256 changed: {output_path}")
        if result["bundle_size_bytes"] != _validate_bundle_size(output_path, expected_content_bytes, expected_count):
            raise ValueError(f"Existing repair bundle size changed: {output_path}")
        return {**result, "skipped": True}
    if output_fs.exists(resolved_output):
        raise ValueError(f"Repair bundle exists without its validated result marker: {output_path}")

    records: list[dict[str, Any]] = []
    found_blob_ids: set[str] = set()
    for task in tasks:
        task_blob_ids = set(task.add_blob_ids)
        for record in load_jsonl(task.source_shard):
            blob_id = record["metadata"]["blob_id"]
            if blob_id not in task_blob_ids:
                continue
            if blob_id in found_blob_ids:
                raise ValueError(f"Duplicate source repair record: {blob_id}")
            found_blob_ids.add(blob_id)
            records.append(record)

    _validate_bundle_records(records, expected_blob_ids, expected_content_bytes)
    records.sort(key=lambda record: record["metadata"]["blob_id"])
    write_jsonl_file(records, output_path)
    bundle_size_bytes = _validate_bundle_size(output_path, expected_content_bytes, expected_count)
    result = {
        "repair_id": manifest.repair_id,
        "manifest_sha256": manifest_sha256,
        "source_region": source_region,
        "record_count": len(records),
        "content_bytes": expected_content_bytes,
        "bundle_size_bytes": bundle_size_bytes,
        "bundle_sha256": _physical_file_sha256(output_path),
        "bundle_fingerprint": _object_fingerprint(output_path),
        "skipped": False,
    }
    write_jsonl_file([result], result_path)
    return result


def _load_extraction_results(
    manifest: RepairManifest,
    manifest_sha256: str,
    result_paths: list[str],
    storage_prefix: str,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for result_path in result_paths:
        _assert_storage_prefix(result_path, storage_prefix)
        result = _read_single_jsonl(result_path)
        source_region = result["source_region"]
        if source_region in results:
            raise ValueError(f"Duplicate extraction result for {source_region}")
        if source_region not in manifest.source_record_counts:
            raise ValueError(f"Unexpected extraction result region: {source_region}")
        if result["repair_id"] != manifest.repair_id or result["manifest_sha256"] != manifest_sha256:
            raise ValueError(f"Extraction result is for a different repair contract: {result_path}")
        if result["record_count"] != manifest.source_record_counts[source_region]:
            raise ValueError(f"Extraction result record count differs for {source_region}")
        if result["content_bytes"] != manifest.source_content_bytes[source_region]:
            raise ValueError(f"Extraction result content bytes differ for {source_region}")
        results[source_region] = result
    if set(results) != set(manifest.source_record_counts):
        raise ValueError(f"Extraction result regions differ: {sorted(results)}")
    return results


def _source_region_by_blob_id(manifest: RepairManifest) -> dict[str, str]:
    return {blob_id: task.source_region for task in manifest.tasks for blob_id in task.add_blob_ids}


def _load_repair_bundles(
    manifest: RepairManifest,
    manifest_sha256: str,
    bundle_paths: list[str],
    extraction_result_paths: list[str],
    storage_prefix: str,
) -> dict[str, dict[str, Any]]:
    extraction_results = _load_extraction_results(manifest, manifest_sha256, extraction_result_paths, storage_prefix)
    source_region_by_blob_id = _source_region_by_blob_id(manifest)
    records: list[dict[str, Any]] = []
    seen_regions: set[str] = set()
    for bundle_path in bundle_paths:
        _assert_storage_prefix(bundle_path, storage_prefix)
        bundle_sha256 = _physical_file_sha256(bundle_path)
        matching_regions = {
            region for region, result in extraction_results.items() if result["bundle_sha256"] == bundle_sha256
        }
        if len(matching_regions) != 1:
            raise ValueError(f"Copied repair bundle SHA-256 does not match one extraction result: {bundle_path}")
        source_region = matching_regions.pop()
        bundle_records = list(load_jsonl(bundle_path))
        bundle_regions = {source_region_by_blob_id.get(record["metadata"]["blob_id"]) for record in bundle_records}
        if None in bundle_regions or len(bundle_regions) != 1:
            raise ValueError(f"Repair bundle does not contain one known source region: {bundle_path}")
        record_source_region = bundle_regions.pop()
        if record_source_region != source_region:
            raise ValueError(f"Repair bundle content does not match extraction region: {bundle_path}")
        if source_region in seen_regions:
            raise ValueError(f"Duplicate repair bundle for {source_region}")
        seen_regions.add(source_region)
        expected_blob_ids = {blob_id for blob_id, region in source_region_by_blob_id.items() if region == source_region}
        expected_count = manifest.source_record_counts[source_region]
        expected_content_bytes = manifest.source_content_bytes[source_region]
        _validate_bundle_records(bundle_records, expected_blob_ids, expected_content_bytes)
        bundle_size_bytes = _validate_bundle_size(bundle_path, expected_content_bytes, expected_count)
        extraction_result = extraction_results[source_region]
        if bundle_size_bytes != extraction_result["bundle_size_bytes"]:
            raise ValueError(f"Copied repair bundle size differs for {source_region}")
        records.extend(bundle_records)
    if seen_regions != set(manifest.source_record_counts):
        raise ValueError(f"Repair bundle regions differ: {sorted(seen_regions)}")
    return _validate_bundle_records(records, manifest.add_blob_ids, manifest.expected_transfer_content_bytes)


def _repair_task_records(
    task: RepairTask,
    repair_records: dict[str, dict[str, Any]],
    measurement: RepairMeasurement,
):
    current_records = iter(load_jsonl(task.target_shard))
    current = next(current_records, None)
    baseline_missing = set(task.baseline_missing_blob_ids)
    task_additions = set(task.add_blob_ids)
    task_removals = set(task.remove_blob_ids)
    verified_task_additions: set[str] = set()
    verified_task_removals: set[str] = set()
    input_spec = InputFileSpec(
        path=task.target_input_file,
        format="parquet",
        columns=STACK_EDU_REQUIRED_COLUMNS,
        row_start=task.row_start,
        row_end=task.row_end,
    )
    output_count = 0
    for row in load_parquet(input_spec):
        expected_id = stack_edu_record_id(task.language, row)
        blob_id = row["blob_id"]
        if blob_id in baseline_missing:
            record_is_present = current is not None and current["id"] == expected_id
            if blob_id in task_removals and record_is_present:
                measurement.removed_blob_ids.add(blob_id)
                current = next(current_records, None)
            elif blob_id in task_removals:
                measurement.already_absent_blob_ids.add(blob_id)
            elif record_is_present:
                raise ValueError(f"Unexpected baseline-missing record is present: {blob_id}")
            if blob_id in task_removals:
                verified_task_removals.add(blob_id)
            continue

        if current is not None and current["id"] == expected_id:
            if blob_id in task_additions:
                if current != repair_records[blob_id]:
                    raise ValueError(f"Existing repaired record differs from source bundle: {blob_id}")
                measurement.already_present_blob_ids.add(blob_id)
                verified_task_additions.add(blob_id)
            output_count += 1
            yield current
            current = next(current_records, None)
            continue

        if blob_id not in task_additions:
            current_id = current["id"] if current is not None else None
            raise ValueError(
                f"Target sequence mismatch for {task.target_shard}: expected {expected_id}, found {current_id}"
            )
        repair_record = repair_records[blob_id]
        if repair_record["id"] != expected_id:
            raise ValueError(f"Repair record id does not match metadata row: {blob_id}")
        measurement.inserted_blob_ids.add(blob_id)
        verified_task_additions.add(blob_id)
        output_count += 1
        yield repair_record

    if current is not None:
        raise ValueError(f"Unexpected trailing target record in {task.target_shard}: {current['id']}")
    expected_count = int(task.expected_metric["count"])
    if output_count != expected_count:
        raise ValueError(f"Repaired row count differs for {task.target_shard}: {output_count} != {expected_count}")
    if verified_task_additions != task_additions:
        raise ValueError(f"Task additions were not all verified: {task.target_shard}")
    if verified_task_removals != task_removals:
        raise ValueError(f"Task removals were not all verified: {task.target_shard}")


def _completion_fingerprints(manifest: RepairManifest) -> list[dict[str, Any]]:
    paths = sorted(
        [task.target_shard for task in manifest.tasks] + [task.target_metrics_path for task in manifest.tasks]
    )
    return [_object_fingerprint(path) for path in paths]


def apply_repair_bundles(
    manifest: RepairManifest,
    manifest_sha256: str,
    bundle_paths: list[str],
    extraction_result_paths: list[str],
    completion_path: str,
) -> dict[str, Any]:
    """Rebuild affected Europe shards in metadata order and update metrics."""
    _assert_running_region(manifest.target_region)
    storage_prefix = marin_prefix_for_region(manifest.target_region)
    _assert_storage_prefix(completion_path, storage_prefix)
    for task in manifest.tasks:
        _assert_storage_prefix(task.target_input_file, storage_prefix)
        _assert_storage_prefix(task.target_shard, storage_prefix)
        _assert_storage_prefix(task.target_metrics_path, storage_prefix)

    completion_fs, resolved_completion = url_to_fs(completion_path)
    if completion_fs.exists(resolved_completion):
        completion = _read_single_jsonl(completion_path)
        if completion["manifest_sha256"] != manifest_sha256:
            raise ValueError(f"Existing completion marker is for a different manifest: {completion_path}")
        current_fingerprints = _completion_fingerprints(manifest)
        if completion["target_fingerprints"] != current_fingerprints:
            raise ValueError("A completed repair target changed after validation")
        return {**completion, "skipped": True}

    repair_records = _load_repair_bundles(
        manifest,
        manifest_sha256,
        bundle_paths,
        extraction_result_paths,
        storage_prefix,
    )
    measurement = RepairMeasurement()
    for task in manifest.tasks:
        result = write_jsonl_file(_repair_task_records(task, repair_records, measurement), task.target_shard)
        if result["count"] != task.expected_metric["count"]:
            raise ValueError(f"Writer count differs for {task.target_shard}")
        write_jsonl_file([task.expected_metric], task.target_metrics_path)

    measurement.validate(manifest)
    completion = {
        "repair_id": manifest.repair_id,
        "manifest_sha256": manifest_sha256,
        "repaired_tasks": len(manifest.tasks),
        "measured_delta": measurement.result(),
        "target_fingerprints": _completion_fingerprints(manifest),
        "skipped": False,
    }
    write_jsonl_file([completion], completion_path)
    return completion


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.set_defaults(command="validate")

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--source-region", required=True)
    extract_parser.add_argument("--output", required=True)
    extract_parser.add_argument("--result", required=True)

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--bundle", action="append", required=True)
    apply_parser.add_argument("--extraction-result", action="append", required=True)
    apply_parser.add_argument("--completion", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = load_repair_manifest(args.manifest, args.expected_manifest_sha256)
    if args.command == "validate":
        result: dict[str, Any] = {
            "repair_id": manifest.repair_id,
            "tasks": len(manifest.tasks),
            "add_records": manifest.expected_add_records,
            "remove_records": manifest.expected_remove_records,
            "transfer_content_bytes": manifest.expected_transfer_content_bytes,
        }
    elif args.command == "extract":
        result = extract_repair_bundle(
            manifest=manifest,
            manifest_sha256=args.expected_manifest_sha256,
            source_region=args.source_region,
            output_path=args.output,
            result_path=args.result,
        )
    else:
        result = apply_repair_bundles(
            manifest=manifest,
            manifest_sha256=args.expected_manifest_sha256,
            bundle_paths=args.bundle,
            extraction_result_paths=args.extraction_result,
            completion_path=args.completion,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
