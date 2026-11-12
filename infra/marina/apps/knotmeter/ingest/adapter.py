# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collector bundle v1 -> Knotmeter catalog and lossless relative request timestamps.

Metrics come from the pinned upstream processor's agg.json, never from UI arithmetic.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .replay import RequestRecord, prepare_requests, profile_records

BUNDLE_MANIFEST = "agentx-source.json"


@dataclass(frozen=True)
class ImportRun:
    key: str
    digest: str
    # Preserve opaque upstream settings and aggregate fields across collector versions.
    run: dict[str, Any]
    requests: list[RequestRecord]
    series: dict[str, Any] = field(default_factory=dict)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def prepare_bundle(bundle: Path, source_attempt: dict[str, Any]) -> list[ImportRun]:
    """Validate the saved receipt and preserve upstream metrics and source identities."""
    manifest_path = bundle / BUNDLE_MANIFEST
    submission = read_json(manifest_path)
    source = submission["source"]
    if (
        source["repository"] != "marin-community/marin"
        or source["kind"] != "github-actions"
        or source_attempt["id"] != source["run_id"]
        or source_attempt["run_attempt"] != source["run_attempt"]
        or source_attempt["head_sha"] != source["head_sha"]
        or source_attempt["status"] != "completed"
        or source_attempt["conclusion"] not in {"success", "failure", "timed_out"}
    ):
        raise ValueError("Bundle does not match its completed original GitHub attempt")
    prepared = []
    for aggregate_path in sorted(bundle.glob("bmk_*/agg.json")):
        aggregate = read_json(aggregate_path)
        raw_dir = bundle / aggregate_path.parent.name.removeprefix("bmk_")
        point = next(
            p
            for p in submission["points"]
            if p["manifest"]["profile"] == aggregate["hw"] and p["manifest"]["concurrency"] == aggregate["conc"]
        )
        receipt, manifest = point["receipt"], point["manifest"]
        config = receipt["configuration"]["series"]
        attempt_path = raw_dir / "attempt.json"
        attempt = read_json(attempt_path)
        hardware = config["profiles"][manifest["profile"]]
        hosts = hardware.get("hosts", 1)
        physical_gpus = hosts * hardware["gpus"]
        if (
            receipt["status"] != "valid"
            or attempt["status"] != "succeeded"
            or attempt["iris_attempt_uid"] != receipt["attempt_uid"]
            or attempt["source"] != source
            or aggregate["recipe_fingerprint"] != manifest["recipe_fingerprint"]
            or attempt["recipe_fingerprint"] != manifest["recipe_fingerprint"]
            or receipt["source"] != source
            or manifest["source"] != source
            or aggregate["model"] != config["model"]
            or aggregate["num_gpus"] != physical_gpus
            or aggregate["precision"] != config["precision"]
            or aggregate["precision"] != "bf16"
            or aggregate["scenario_type"] != "agentic-coding"
        ):
            raise ValueError("Aggregate, recipe, physical GPU count or native attempt differs from its receipt")
        files = [manifest_path, aggregate_path, attempt_path]
        if hosts > 1:
            gang_path = raw_dir / "gang.json"
            gang = read_json(gang_path)
            tasks = gang["tasks"]
            if (
                len(tasks) != hosts
                or len({t["iris_node_name"] for t in tasks}) != hosts
                or not {t["iris_node_name"] for t in tasks}.issubset(manifest["placement"]["nodes"])
                or len({gpu for t in tasks for gpu in t["gpu_uuids"]}) != physical_gpus
                or any(
                    t["status"] != "succeeded"
                    or t["source"] != source
                    or t["recipe_fingerprint"] != manifest["recipe_fingerprint"]
                    or len(t["gpu_uuids"]) != hardware["gpus"]
                    or gang["attempts"].get(t["iris_task_id"].rsplit(":", 1)[0]) != t["iris_attempt_uid"]
                    for t in tasks
                )
            ):
                raise ValueError("Multihost receipt does not cover the physical GPU allocation")
            files.append(gang_path)
        profile_path = raw_dir / "aiperf_artifacts/profile_export.jsonl"
        files.append(profile_path)
        origin, requests = prepare_requests(lambda path=profile_path: profile_records(path))
        profiling = sum(r["phase"] == "profiling" and not r["error"] for r in requests)
        validation = receipt["validation"]
        throughput = aggregate["request_metrics"]["throughput"]
        output = throughput["per_gpu"]["output_tput_tps"]
        if (
            len(requests) != aggregate["num_requests_total"]
            or profiling != aggregate["num_requests_successful"]
            or profiling != validation["profiling_requests"]
            or validation["coverage"]["expected_duration_seconds"] != 3600
            or not math.isclose(output, throughput["output"]["tokens_per_second"] / physical_gpus, abs_tol=0.00001)
        ):
            raise ValueError("Request counts, full-hour coverage or physical-GPU normalization differs")
        file_hashes = {}
        for path in files:
            with path.open("rb") as stream:
                file_hashes[str(path.relative_to(bundle))] = hashlib.file_digest(stream, "sha256").hexdigest()
        key = f"{source['run_id']}-{source['run_attempt']}-{aggregate['hw']}-c{aggregate['conc']}"
        dataset = manifest["workload"]["lineage"]
        run = {
            "key": key,
            "date": source_attempt["created_at"][:10],
            "source_created_at": source_attempt["created_at"],
            "source": source,
            "measurement_seconds": 3600,
            "topology": attempt["server_parallelism"],
            "model": aggregate["model"],
            "hardware": hardware["gpu"],
            "physical_gpus": physical_gpus,
            "hosts": hosts,
            "concurrency": aggregate["conc"],
            "precision": aggregate["precision"],
            "scenario": "AgentX",
            "runtime": aggregate["image"],
            "recipe": manifest["recipe_fingerprint"],
            "model_revision": config["model_revision"],
            "dataset": dataset["dataset"],
            "dataset_revision": dataset["dataset_revision"],
            "dataset_sessions": dataset["sessions"],
            "workload_key": manifest["workload"]["key"],
            "request_count": len(requests),
            "profiling_requests": profiling,
            "origin_ns": str(origin),
            "metrics": aggregate["request_metrics"],
            "diagnostic_aggregates": {
                "isl": aggregate["request_metrics"].get("tokens", {}).get("input"),
                "osl": aggregate["request_metrics"].get("tokens", {}).get("output_actual"),
                "kvCacheUtil": None,
                "prefixCacheHitRate": None,
            },
            "framework": aggregate["framework"],
            "aggregate": aggregate,
            "configuration": config,
            "manifest": manifest,
            "validation": validation,
            "source_files": file_hashes,
            "format_version": 2,
        }
        digest = hashlib.sha256(json.dumps([run, requests], sort_keys=True, allow_nan=False).encode()).hexdigest()
        run["content_sha256"] = digest
        prepared.append(ImportRun(key, digest, run, requests))
    if not prepared:
        raise ValueError("No saved benchmark aggregates found")
    return prepared
