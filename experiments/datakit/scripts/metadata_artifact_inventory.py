# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover frozen release inputs by reading small artifact metadata only."""

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from iris.cluster.redaction import redact_submit_argv
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.redaction import REDACTED_VALUE, is_sensitive_key_name, redact_value

MAX_METADATA_BYTES = 2 * 1024 * 1024
INPUT_KINDS = ("tokenize", "decontam", "cluster", "quality", "content_type")


def sanitize_metadata(value: Any) -> Any:
    """Remove credential values while retaining provider identifiers and hashes."""
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            if isinstance(key, str) and is_sensitive_key_name(key):
                result[key] = REDACTED_VALUE
            elif key == "command_line" and isinstance(child, list) and all(isinstance(x, str) for x in child):
                result[key] = [sanitize_metadata(argument) for argument in redact_submit_argv(child)]
            else:
                result[key] = sanitize_metadata(child)
        return result
    if isinstance(value, list):
        return [sanitize_metadata(child) for child in value]
    if isinstance(value, str):
        key, separator, _ = value.partition("=")
        if separator and is_sensitive_key_name(key):
            return key + "=" + REDACTED_VALUE
        # Provider revisions, file paths and dataset IDs are identifiers, not
        # entropy-based secret candidates. Still strip known credential formats.
        return redact_value({"path": value})["path"]
    return value


def read_metadata(path: str, *, max_bytes: int = MAX_METADATA_BYTES) -> dict:
    """Read a bounded JSON object; never read corpus data."""
    fs, key = filesystem_for(path)
    payload = fs.cat_file(key, end=max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError(f"Metadata exceeds byte budget: {path}")
    return sanitize_metadata(json.loads(payload))


def _read_record(path: str) -> tuple[str, dict]:
    try:
        record = read_metadata(prefix_join(path, ".artifact.json"))
        return path, {"status": "found", "record": record}
    except FileNotFoundError:
        return path, {"status": "missing"}


def _absolute(prefix: str, path: str) -> str:
    return path if "://" in path else prefix_join(prefix, path)


def inventory(store_record: dict, *, workers: int = 8, cache_dir: str | None = None) -> dict:
    """Read frozen dependency records and recursively trace normalized inputs.

    Missing metadata is reported, never replaced by today's source registry.
    Transitive tracing follows only normalized/raw/processed dependencies.
    """
    store_record = sanitize_metadata(store_record)
    source_names = store_record["result"]["source_names"]
    dep_paths = store_record["dep_paths"]
    if len(dep_paths) != len(source_names) * len(INPUT_KINDS) + 2:
        raise ValueError("Store dependencies do not match five inputs per frozen source plus two dedup roots")
    for index, source_name in enumerate(source_names):
        identity = store_record["deps"][index * len(INPUT_KINDS)]
        actual_name = identity.removeprefix("datakit/tokenize/").rsplit("_", 1)[0]
        if actual_name != source_name:
            raise ValueError(f"Frozen dependency order mismatch: {source_name} != {actual_name}")
    prefix = str(StoragePath(store_record["output_path"]).parent.parent)
    records: dict[str, dict] = {}

    def load(paths: list[str]) -> None:
        pending = sorted(set(paths) - records.keys())

        def read(path: str) -> tuple[str, dict]:
            target = Path(cache_dir) if cache_dir else None
            name = hashlib.sha256(path.encode()).hexdigest() + ".json"
            cached = target / name if target else None
            if cached and cached.exists():
                status = sanitize_metadata(json.loads(cached.read_text()))
                status.pop("path", None)
            else:
                _, status = _read_record(path)
            record = status.get("record")
            if record is not None and "result" not in record:
                try:
                    info = read_metadata(prefix_join(path, ".executor_info"))
                except FileNotFoundError:
                    info = {}
                dependencies = info.get("dependencies", [])
                # Copied artifacts retain original-region executor metadata.
                resolved = [
                    prefix_join(prefix, dependency.split("/marin/", 1)[1]) if "/marin/" in dependency else dependency
                    for dependency in dependencies
                ]
                status["record"] = {
                    "result": record,
                    "config": info.get("config"),
                    "dep_paths": resolved,
                    "executor_info": info,
                    "legacy_payload": True,
                }
            if cached:
                target.mkdir(parents=True, exist_ok=True)
                cached.write_text(json.dumps({"path": path, **status}))
            return path, status

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for path, status in executor.map(read, pending):
                records[path] = status

    load(dep_paths)
    exact = records[dep_paths[-2]].get("record", {}).get("result", {}).get("sources", {})
    fuzzy = records[dep_paths[-1]].get("record", {}).get("result", {}).get("sources", {})
    sources = []
    for index, name in enumerate(source_names):
        start = index * len(INPUT_KINDS)
        roots = dict(zip(INPUT_KINDS, dep_paths[start : start + len(INPUT_KINDS)], strict=True))
        token = records[roots["tokenize"]].get("record", {})
        source_key = token.get("result", {}).get("source_keys", {}).get("train")
        normalized = str(StoragePath(_absolute(prefix, source_key)).parent.parent) if source_key else None
        sources.append(
            {
                "marin_source": name,
                "artifact_roots": roots,
                "source_key": source_key,
                "normalized_artifact": normalized,
                "exact_dedup": exact.get(source_key),
                "verified_fuzzy_dedup": fuzzy.get(source_key),
                "fuzzy_exempt": name in store_record["config"]["fuzzy_dedup_exempt"],
            }
        )
    pending = [source["normalized_artifact"] for source in sources if source["normalized_artifact"]]
    while pending:
        load(pending)
        pending = [
            dependency
            for path in pending
            for dependency in records[path].get("record", {}).get("dep_paths", [])
            if dependency not in records
        ]
    for source in sources:
        record = records.get(source["normalized_artifact"], {}).get("record", {})
        source["normalization_parameters"] = record.get("config")
        source["normalization_input_artifacts"] = record.get("dep_paths", [])
        source["normalized_outputs"] = record.get("result", {})
        source["normalization_provenance"] = record.get("provenance")
        source["normalization_executor_info"] = record.get("executor_info")
        relative = (record.get("config") or {}).get("relative_input_path")
        source["normalization_input_paths"] = [
            prefix_join(dependency, relative) if relative else dependency for dependency in record.get("dep_paths", [])
        ]
        original_dependencies = (record.get("executor_info") or {}).get("dependencies", record.get("dep_paths", []))
        source["original_normalization_input_paths"] = [
            prefix_join(dependency, relative) if relative else dependency for dependency in original_dependencies
        ]
        token = records[source["artifact_roots"]["tokenize"]].get("record", {}).get("result", {})
        source["normalized_shard_count_hint"] = token.get("counters", {}).get("train", {}).get("zephyr/item_count")
        pending_dependencies = source["normalization_input_artifacts"][:]
        seen: set[str] = set()
        providers = []
        while pending_dependencies:
            dependency = pending_dependencies.pop()
            if dependency in seen:
                continue
            seen.add(dependency)
            ancestor = records[dependency].get("record", {})
            config = ancestor.get("config") or {}
            if config.get("hf_dataset_id"):
                providers.append(
                    {"dataset": config["hf_dataset_id"], "revision": config.get("revision"), "artifact": dependency}
                )
            pending_dependencies.extend(ancestor.get("dep_paths", []))
        source["upstream_provider_references"] = providers
    return {
        "store_uri": store_record["output_path"],
        "store_version": store_record["result"]["version"],
        "store_provenance": store_record.get("provenance"),
        "store_config": store_record["config"],
        "bucket_edges": store_record["result"]["bucket_edges"],
        "tokenizer": store_record["result"]["tokenizer"],
        "source_count": len(sources),
        "sources": sources,
        "records": records,
        "metadata_status_counts": dict(Counter(item["status"] for item in records.values())),
    }


def recover_harrier_embeddings(report: dict, *, workers: int = 8) -> dict:
    """Read embedding provenance named by frozen cluster artifacts, without data shards.

    Missing or incomplete metadata is reported separately from document vector
    coverage. Metadata availability does not establish which documents have vectors.
    """
    report = sanitize_metadata(report)
    records = report["records"]
    prefix = str(StoragePath(report["store_uri"]).parent.parent)
    paths = {}
    for source in report["sources"]:
        cluster = records.get(source["artifact_roots"]["cluster"], {}).get("record", {}).get("result", {})
        path = cluster.get("embedding_output_dir")
        paths[source["marin_source"]] = _absolute(prefix, path) if path else None
    pending = sorted({path for path in paths.values() if path and path not in records})
    with ThreadPoolExecutor(max_workers=workers) as executor:
        records.update(executor.map(_read_record, pending))
    facts = ("model_name", "model_revision", "embedding_dim", "quantization_scale", "quantization_range")
    statuses = Counter()
    for source in report["sources"]:
        path = paths[source["marin_source"]]
        status = records.get(path, {})
        metadata = status.get("record", {}).get("result", {})
        provenance = {"artifact": path, **{key: metadata.get(key) for key in facts}} if path else None
        source["harrier_embedding_artifact"] = path
        source["harrier_embedding_provenance"] = provenance
        if path is None:
            availability = "not_recorded"
        elif status.get("status") != "found":
            availability = "missing_metadata"
        elif any(metadata.get(key) is None or metadata.get(key) == "" for key in facts):
            availability = "incomplete_metadata"
        elif metadata.get("source_key") != source["source_key"]:
            availability = "source_key_mismatch"
        else:
            availability = "available"
        source["harrier_embedding_metadata_status"] = availability
        source["harrier_embedding_document_coverage"] = "not_examined"
        statuses[availability] += 1
    report["harrier_embedding_metadata_status"] = dict(statuses)
    report["metadata_status_counts"] = dict(Counter(item["status"] for item in records.values()))
    return report


def audit_provider_routes(
    report: dict, *, fetch_sidecars: bool = False, workers: int = 8, original_prefix: str | None = None
) -> dict:
    """Audit provider file provenance without claiming Dataset Viewer row ordering.

    Download provenance preserves selected provider paths, but does not identify
    Hugging Face configs/splits or establish their global row offsets. Missing
    files and schemas are explicit coverage gaps, never inferred from staging.
    """
    report = sanitize_metadata(report)
    records = report["records"]
    raw_paths = [path for path in records if "/raw/" in path]
    sidecars = report.get("provider_provenance_sidecars", {})
    if fetch_sidecars:

        def read(path: str) -> tuple[str, dict]:
            if sidecars.get(path, {}).get("status") == "found":
                return path, sidecars[path]
            roots = []
            if records[path].get("metadata_lookup_path"):
                roots.append(records[path]["metadata_lookup_path"])
            if original_prefix and "/marin/" in path:
                roots.append(prefix_join(original_prefix, path.split("/marin/", 1)[1]))
            roots.append(path)
            if "/raw/" in path and "/" in path.split("/raw/", 1)[1]:
                roots.extend([str(StoragePath(root).parent) for root in roots])
            for root in dict.fromkeys(roots):
                for filename in (".provenance.json", "provenance.json"):
                    location = prefix_join(root, filename)
                    try:
                        return path, {
                            "status": "found",
                            "metadata": read_metadata(location),
                            "metadata_lookup_path": location,
                        }
                    except FileNotFoundError:
                        continue
            return path, {"status": "missing"}

        with ThreadPoolExecutor(max_workers=workers) as executor:
            sidecars = dict(executor.map(read, raw_paths))
    sources = []
    for source in report["sources"]:
        pending = list(source["normalization_input_artifacts"])
        visited = set()
        providers = []
        missing = []
        while pending:
            path = pending.pop()
            if path in visited:
                continue
            visited.add(path)
            status = records.get(path, {})
            if status.get("status") != "found":
                missing.append(path)
            record = status.get("record", {})
            config = record.get("config") or {}
            config = config.get("attrs", config)
            sidecar = sidecars.get(path, {})
            provenance = sidecar.get("metadata", {})
            dataset = config.get("hf_dataset_id") or provenance.get("dataset")
            if dataset:
                providers.append(
                    {
                        "dataset": dataset,
                        "provider_kind": "hugging_face" if config.get("hf_dataset_id") else "external",
                        "revision": config.get("revision") or provenance.get("version"),
                        "selected_file_patterns": config.get("hf_urls_glob"),
                        "selected_provider_files": provenance.get("links") if config.get("hf_dataset_id") else None,
                        "original_selected_files": provenance.get("links") if config.get("hf_dataset_id") else None,
                        "provenance_links": provenance.get("links"),
                        "metadata_lookup_path": sidecar.get("metadata_lookup_path"),
                        "download_artifact": path,
                        "download_provenance_status": sidecar.get("status", "not_fetched"),
                        "original_download_dependencies": record.get("executor_info", {}).get("dependencies"),
                        "viewer_config": None,
                        "viewer_split": None,
                        "split_order_status": "not_verified",
                    }
                )
            pending.extend(record.get("dep_paths", []))
        sources.append(
            {
                "marin_source": source["marin_source"],
                "provider_routes": providers,
                "missing_ancestor_metadata": missing,
                "normalization_input_paths": source.get("normalization_input_paths"),
                "original_normalization_input_paths": source.get("original_normalization_input_paths"),
                "provider_mapping_status": "partial" if providers else "unresolved",
                "global_split_offset_status": "not_verified",
            }
        )
    return {
        "source_count": len(sources),
        "sources_with_provider_references": sum(bool(source["provider_routes"]) for source in sources),
        "sources_with_verified_split_order": 0,
        "limitations": [
            "Sorted download links are not evidence of Hugging Face config/split row order.",
            "Provider file row counts and dataset builder ordering have not been read or checked.",
            "Source transforms may group/filter rows; offsets must precede those transforms.",
            "A provider dataset reference does not imply its pinned revision supports Dataset Viewer /rows.",
        ],
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-record", type=Path)
    parser.add_argument("--inventory", type=Path, help="Audit an existing inventory instead of rereading all artifacts")
    parser.add_argument("--provider-audit", action="store_true")
    parser.add_argument("--harrier-embeddings", action="store_true", help="Recover frozen embedding metadata only")
    parser.add_argument("--fetch-provider-sidecars", action="store_true")
    parser.add_argument("--original-prefix", help="Explicit original artifact prefix for copied metadata lookups")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.inventory:
        report = sanitize_metadata(json.loads(args.inventory.read_text()))
    else:
        if args.store_record is None:
            parser.error("Supply --store-record or --inventory")
        report = inventory(json.loads(args.store_record.read_text()), workers=args.workers, cache_dir=args.cache_dir)
    if args.harrier_embeddings:
        report = recover_harrier_embeddings(report, workers=args.workers)
    if args.provider_audit:
        report = audit_provider_routes(
            report,
            fetch_sidecars=args.fetch_provider_sidecars,
            workers=args.workers,
            original_prefix=args.original_prefix,
        )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "source_count",
                    "metadata_status_counts",
                    "sources_with_provider_references",
                    "sources_with_verified_split_order",
                    "harrier_embedding_metadata_status",
                )
                if key in report
            }
        )
    )


if __name__ == "__main__":
    main()
