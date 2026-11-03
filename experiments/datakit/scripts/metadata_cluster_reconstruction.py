# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct a complete selected cluster through parallel original-provider scans.

Provider-file markers certify only that file's candidate scan. Source completion
requires every pinned file and every selected ID; cluster completion requires
all selected sources. Outputs are private run artifacts, not a public release.
"""

import argparse
import dataclasses
import errno
import hashlib
import json
import logging
import os
import re
import sqlite3
import tempfile
import uuid
from pathlib import Path
from typing import Any

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from fsspec.exceptions import FSTimeoutError
from iris.cluster.client.job_info import get_job_info
from marin.datakit.download.penfever_rollouts import PenfeverRollout
from marin.datakit.normalize import generate_id
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_fsspec_s3, fsspec_s3_conf
from rigging.filesystem.storage_path import prefix_join
from rigging.secrets import get_secret
from rigging.timing import ExponentialBackoff, retry_with_backoff

from experiments.datakit.scripts.metadata_cluster_selection import _digest, _read_json, _write_json
from experiments.datakit.scripts.metadata_focus_provider import FOCUS_SOURCE, focus_provider_evidence
from experiments.datakit.store.davinci_provider_rows import (
    HISTORICAL_BATCH_SIZE,
    DavinciCtxReadAudit,
    davinci_ctx_reader_policy,
    read_davinci_context,
)
from experiments.datakit.store.provider_reconstruction import (
    ProviderTransform,
    ReconstructionRecipe,
    provider_normalizer,
)
from experiments.datakit.store.reconstruct_selected_source import (
    PROVIDER_COLUMNS,
    RECONSTRUCTED_TEXT_SCHEMA,
    provider_rows,
)
from experiments.datakit.store.software_heritage_empty_rows import (
    SoftwareHeritageEmptyAudit,
    empty_reader_policy,
    empty_source_audit,
    read_software_heritage,
    validate_empty_audit,
)

logger = logging.getLogger(__name__)
CANDIDATE_SCHEMA = pa.schema([("id", pa.string()), ("text", pa.string())])
COPY_CHUNK_BYTES = 8 * 1024 * 1024
TEXT_SHARD_BYTES = 256 * 1024 * 1024
# Audited original inventories for the frozen Hero source revisions; regional
# copies of these downloads are incomplete.
FROZEN_ORIGINAL_FILE_COUNTS = {"climblab-ja": 512, "coderforge": 672}


def _recipe(value: dict) -> ReconstructionRecipe:
    penfever = value.get("penfever_dataset")
    return ReconstructionRecipe(
        **{
            **value,
            "transform": ProviderTransform(value["transform"]),
            "drop_fields": tuple(value.get("drop_fields", ())),
            **({"penfever_dataset": PenfeverRollout(**penfever)} if penfever is not None else {}),
        }
    )


def _recipe_manifest(recipe: ReconstructionRecipe) -> dict:
    value = dataclasses.asdict(recipe)
    # Unrelated frozen source recipes retain their original serialized identity.
    if value.get("penfever_dataset") is None:
        value.pop("penfever_dataset", None)
    return value


def _canonical_artifact_path(path: str) -> str:
    return path.replace("s3://marin-na/marin/", "s3://marin-us-east-02a/marin/", 1)


def _provider_kind(source: dict) -> tuple[str, str | None]:
    recipe = _recipe(source["recipe"])
    if (
        recipe.transform is ProviderTransform.SOFTWARE_HERITAGE_EMPTY
        and source["marin_source"] != "nemotron_code_v1/content"
    ):
        return "blocked", "Audited SWH empty filtering is restricted to its frozen v1 source"
    rendered_sources = {
        "molmo2-cap",
        "numinamath-1.5",
        "numinamath-tir",
        "superior-reasoning",
        "swe-rebench-openhands",
        "synthetic-1",
    }
    if recipe.transform is ProviderTransform.DIRECT and (
        source["marin_source"] in rendered_sources or source["marin_source"].startswith("penfever-traces/")
    ):
        return "blocked", "Original structured provider rows require their production text renderer"
    if recipe.transform is ProviderTransform.PENFEVER:
        dataset = recipe.penfever_dataset
        if (
            dataset is None
            or dataset.marin_name != source["marin_source"]
            or any(
                value["dataset"] != dataset.hf_dataset_id or value["revision"] != dataset.revision
                for value in source["provider_references"]
            )
        ):
            return "blocked", "Penfever renderer provenance must match its frozen source and provider revision"
    if source["marin_source"] == FOCUS_SOURCE:
        proof = source.get("original_provider_evidence", {})
        expected = focus_provider_evidence({})
        digest = proof.get("targets_sha256", "")
        expected["targets_sha256"] = digest
        paths = source["provider_paths"]
        if (
            proof != expected
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or len(paths) != 141
            or paths != sorted(set(paths))
            or any(
                not re.fullmatch(r"https://data\.commoncrawl\.org/[A-Za-z0-9_./-]+\.warc\.gz", path) for path in paths
            )
        ):
            return "blocked", "Focus Crawl requires exact original WARC targets and pinned replay evidence"
        return "external_original_warc_replay", None
    structured_transforms = {
        "davinci-dev/env-native": ProviderTransform.DAVINCI_ENV,
        "davinci-dev/ctx-native": ProviderTransform.DAVINCI_CTX,
        "agenttrove": ProviderTransform.AGENTTROVE,
        "coderforge": ProviderTransform.CODERFORGE,
        "gpt-oss-rollouts": ProviderTransform.GPT_OSS,
    }
    expected_transform = structured_transforms.get(source["marin_source"])
    if expected_transform is not None and recipe.transform is not expected_transform:
        return "blocked", "Structured original provider rows require their pinned production text renderer"
    expected_files = FROZEN_ORIGINAL_FILE_COUNTS.get(source["marin_source"])
    if expected_files is not None:
        evidence = source.get("original_provider_evidence") or {}
        root = evidence.get("original_provider_root") or evidence.get("original_normalizer_dependency")
        paths = source["provider_paths"]
        if (
            evidence.get("original_file_count") != expected_files
            or len(paths) != expected_files
            or paths != sorted(set(paths))
            or not root
            or any(not path.startswith(root.rstrip("/") + "/") for path in paths)
        ):
            return "blocked", "The frozen original download inventory must replace its known partial regional copy"
    if recipe.transform is ProviderTransform.BIODIVERSITY:
        return "blocked", "Biodiversity requires a complete input-assembly stage"
    if source["marin_source"] in {"nemotron_code_v1/content", "nemotron_code_v2/content"}:
        if recipe.transform is ProviderTransform.SOFTWARE_HERITAGE_EMPTY:
            if (
                source["marin_source"] != "nemotron_code_v1/content"
                or source.get("original_provider_evidence", {}).get("empty_reader_policy") != empty_reader_policy()
                or recipe.normalization_sha256 != empty_reader_policy()["normalizer_sha256"]
                or (recipe.text_field, recipe.id_field, recipe.bare, recipe.max_whitespace_run_chars, recipe.drop_fields)
                != ("content", "sha1_git", True, 128, ())
            ):
                return "blocked", "Audited SWH empty filtering requires its explicit v1 policy evidence"
            return "software_heritage_audited_empty", None
        if recipe.transform is not ProviderTransform.SOFTWARE_HERITAGE:
            return "blocked", "Software Heritage binary rows require both provider content-hash checks"
        return "software_heritage", None
    if source["marin_source"] == "hplt_v3":
        paths = source["provider_paths"]
        proof = source.get("original_provider_evidence", {})
        if (
            recipe.transform is not ProviderTransform.DIRECT
            or proof.get("provider_revision") != "release three; current bytes must match selected canonical IDs"
            or len(paths) != 1294
            or any(
                not re.fullmatch(r"https://data\.hplt-project\.org/three/sorted/eng_Latn/[0-9]+_\d+\.jsonl\.zst", path)
                for path in paths
            )
        ):
            return "blocked", "HPLT requires its complete original release-file inventory"
        return "hplt_original_release", None
    providers = source["provider_references"]
    if not providers or any(not value.get("dataset") or not value.get("revision") for value in providers):
        return "blocked", "An original-provider snapshot and replay stage have not been established"
    roots = [_canonical_artifact_path(value["artifact"]).rstrip("/") + "/" for value in providers]
    if any(
        not any(_canonical_artifact_path(path).startswith(root) for root in roots) for path in source["provider_paths"]
    ):
        return "blocked", "Provider files are outside their frozen original download roots"
    return "hugging_face_original_snapshot", None


def prepare_reconstruction_plan(
    selection: dict,
    selection_reports: list[dict],
    providers: dict,
    output_prefix: str,
    previous_plan: dict | None = None,
) -> dict:
    """Bind complete selected ledgers and original-file candidates into one immutable plan.

    The selection summary must come from aggregate_selection_reports, whose
    frozen inventory coverage checks precede this stage. Index construction
    rechecks each referenced completion marker and actual occurrence bytes.
    """
    if not selection.get("complete") or not selection.get("complete_cluster"):
        raise ValueError("Reconstruction requires a complete-cluster selection ledger")
    if providers["cluster_id"] != selection["target_cluster"]:
        raise ValueError("Provider and selection plans refer to different clusters")
    task_count = selection["task_count"]
    if sorted(report["config"]["task_index"] for report in selection_reports) != list(range(task_count)):
        raise ValueError("Selection reports do not cover every task exactly once")
    items: dict[str, dict] = {}
    for report in selection_reports:
        config = report["config"]
        if (
            not report["complete"]
            or report.get("sampled")
            or config["inventory_sha256"] != selection["inventory_sha256"]
            or config["target_cluster"] != selection["target_cluster"]
            or config["task_count"] != task_count
        ):
            raise ValueError("Selection task is incomplete or belongs to different frozen inputs")
        for item in report["completed"]:
            if item["status"] == "selected":
                if item["output"] in items:
                    raise ValueError("Repeated selection output in reconstruction inputs")
                items[item["output"]] = item
    source_providers = {source["marin_source"]: source for source in providers["sources"]}
    sources: list[dict[str, Any]] = []
    selected_paths: set[str] = set()
    for name, selected in sorted(selection["sources"].items()):
        if not selected["selected_occurrences"]:
            continue
        source = source_providers[name]
        source = {
            **source,
            "provider_paths": [
                path for path in source["provider_paths"] if not any(part.startswith(".") for part in path.split("/"))
            ],
        }
        if len(set(source["provider_paths"])) != len(source["provider_paths"]):
            raise ValueError("Repeated provider file paths make task coverage ambiguous")
        ledgers = []
        for path in selected["ledgers"]:
            if path in selected_paths or path not in items or items[path]["source"] != name:
                raise ValueError("Selection source outputs are repeated, absent, or assigned to a different source")
            selected_paths.add(path)
            ledgers.append(items[path])
        if sum(item["selected_occurrences"] for item in ledgers) != selected["selected_occurrences"]:
            raise ValueError("Source occurrence counts do not match selected-ledger descriptors")
        kind, blocked = _provider_kind(source)
        if not source["provider_paths"]:
            kind, blocked = "blocked", "No original provider files have been located"
        recipe = _recipe(source["recipe"])
        provider_normalizer(recipe)
        sources.append(
            {
                "marin_source": name,
                "selected_occurrences": selected["selected_occurrences"],
                "ledgers": ledgers,
                "recipe": _recipe_manifest(recipe),
                "provider_paths": source["provider_paths"],
                "provider_references": source["provider_references"],
                "original_provider_evidence": source.get("original_provider_evidence"),
                "provider_kind": kind,
                "blocked_reason": blocked,
                "scratch_estimate_bytes": {
                    "index": selected["selected_occurrences"] * 96 * 3 + 1024**3,
                    "candidate_worker": selected["selected_occurrences"] * 160 * 3 + 1024**3,
                    "source_reducer": selected["selected_occurrences"] * 192 * 3 + 1024**3,
                    "basis": "conservative SQLite bytes per selected occurrence with 3x scratch headroom; no text table",
                },
            }
        )
    if sum(source["selected_occurrences"] for source in sources) != selection["selected_occurrences"]:
        raise ValueError("Source selection totals differ from the complete cluster")
    for source in sources:
        source["manifest_sha256"] = _digest(source)
    previous_sources: dict[str, dict] = {}
    previous_roots: dict[str, str] = {}
    if previous_plan is not None:
        _verify_plan(previous_plan)
        previous_sources = {source["marin_source"]: source for source in previous_plan["sources"]}
        previous_roots = previous_plan.get("source_roots") or migrate_unversioned_source_roots(previous_plan)
    source_roots: dict[str, str] = {}
    for source in sources:
        name = source["marin_source"]
        previous = previous_sources.get(name)
        if previous is not None and previous["manifest_sha256"] == source["manifest_sha256"]:
            source_roots[name] = previous_roots[name]
        else:
            name_hash = hashlib.sha256(name.encode()).hexdigest()[:16]
            source_roots[name] = prefix_join(
                output_prefix, "source=" + name_hash + "/version=" + source["manifest_sha256"]
            )
    plan = {
        "source_roots": source_roots,
        "selection_sha256": _digest(selection),
        "selection_inventory_sha256": selection["inventory_sha256"],
        "provider_manifest_sha256": _digest(providers),
        "target_cluster": selection["target_cluster"],
        "selected_occurrences": selection["selected_occurrences"],
        "output_prefix": output_prefix,
        "sources": sources,
    }
    plan["manifest_sha256"] = _digest(plan)
    return plan


def _verify_plan(plan: dict) -> None:
    if _digest({key: value for key, value in plan.items() if key != "manifest_sha256"}) != plan["manifest_sha256"]:
        raise ValueError("Reconstruction plan bytes differ from their manifest digest")
    for source in plan["sources"]:
        if (
            _digest({key: value for key, value in source.items() if key != "manifest_sha256"})
            != source["manifest_sha256"]
        ):
            raise ValueError("Reconstruction source differs from its manifest digest")


def _source(plan: dict, name: str) -> dict:
    return next(source for source in plan["sources"] if source["marin_source"] == name)


def migrate_unversioned_source_roots(plan: dict) -> dict[str, str]:
    """Recover explicit output roots for the first, already running unversioned plan.

    This migration is only used while preparing a replacement plan. Worker APIs
    require source_roots and never infer a legacy path.
    """
    return {
        source["marin_source"]: prefix_join(
            plan["output_prefix"], "source=" + hashlib.sha256(source["marin_source"].encode()).hexdigest()[:16]
        )
        for source in plan["sources"]
    }


def _source_root(plan: dict, source: dict) -> str:
    if "source_roots" not in plan:
        raise ValueError("Worker plans require explicit source_roots; prepare a replacement with --previous-plan")
    return plan["source_roots"][source["marin_source"]]


def _index_marker(plan: dict, source: dict) -> str:
    return prefix_join(_source_root(plan, source), "index/_complete.json")


def _scan_marker(plan: dict, source: dict, file_index: int) -> str:
    return prefix_join(_source_root(plan, source), f"candidates/file={file_index:06d}/_complete.json")


def _exists(path: str) -> bool:
    fs, key = filesystem_for(path)
    return fs.exists(key)


def _transfer(source: str, destination: str) -> str:
    digest = hashlib.sha256()
    source_fs, source_key = filesystem_for(source)
    destination_fs, destination_key = filesystem_for(destination)
    with source_fs.open(source_key, "rb") as reader, destination_fs.open(destination_key, "wb") as writer:
        while chunk := reader.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
            writer.write(chunk)
    return digest.hexdigest()


def build_selected_index(plan: dict, name: str, batch_size: int = 8192) -> dict:
    """Build one shared on-disk selected-ID/count index after validating ledger bytes."""
    _verify_plan(plan)
    source = _source(plan, name)
    binding = {"source_sha256": source["manifest_sha256"]}
    marker_path = _index_marker(plan, source)
    if _exists(marker_path):
        marker = _read_json(marker_path)
        if marker["binding"] != binding:
            raise ValueError("Selected index checkpoint has different pinned inputs")
        return marker
    with tempfile.TemporaryDirectory(prefix="hero-selected-index-") as temporary:
        local = str(Path(temporary) / "selected.sqlite")
        total = 0
        with sqlite3.connect(local) as index:
            index.execute("PRAGMA journal_mode=OFF")
            index.execute("PRAGMA synchronous=OFF")
            index.execute("CREATE TABLE selected (id BLOB PRIMARY KEY, occurrences INTEGER NOT NULL) WITHOUT ROWID")
            for ledger in source["ledgers"]:
                marker = _read_json(ledger["completion_marker"])
                if (
                    marker["manifest"]["marin_source"] != name
                    or marker["target_cluster"] != plan["target_cluster"]
                    or marker["selected_occurrences"] != ledger["selected_occurrences"]
                    or marker["ordered_occurrences_sha256"] != ledger["ordered_occurrences_sha256"]
                    or marker["output"] != ledger["output"]
                ):
                    raise ValueError("Selected ledger marker differs from the reconstruction plan")
                fs, key = filesystem_for(ledger["output"])
                digest = hashlib.sha256()
                count = 0
                previous = -1
                with fs.open(key, "rb") as stream:
                    parquet = pq.ParquetFile(stream)
                    for batch in parquet.iter_batches(
                        columns=["id", "normalized_row", "normalized_shard", "cluster_id", "marin_source"],
                        batch_size=batch_size,
                    ):
                        rows = batch.to_pylist()
                        ids = []
                        for row in rows:
                            if (
                                row["marin_source"] != name
                                or row["cluster_id"] != plan["target_cluster"]
                                or row["normalized_shard"] != marker["manifest"]["normalized"]
                                or row["normalized_row"] <= previous
                            ):
                                raise ValueError("Selected occurrence is wrong-source, wrong-cluster, or repeated")
                            previous = row["normalized_row"]
                            digest.update(json.dumps([row["id"], previous], separators=(",", ":")).encode() + b"\n")
                            value = bytes.fromhex(row["id"])
                            if len(value) != 16:
                                raise ValueError("Selected document ID is not a canonical 128-bit hash")
                            ids.append((value,))
                        index.executemany(
                            "INSERT INTO selected VALUES (?,1) "
                            "ON CONFLICT(id) DO UPDATE SET occurrences=occurrences+1",
                            ids,
                        )
                        count += len(rows)
                if count != ledger["selected_occurrences"] or digest.hexdigest() != ledger["ordered_occurrences_sha256"]:
                    raise ValueError("Selected ledger counts or ordered occurrence hashes differ")
                total += count
                index.commit()
            distinct = index.execute("SELECT COUNT(*) FROM selected").fetchone()[0]
        if total != source["selected_occurrences"]:
            raise ValueError("Index does not cover every selected occurrence of this source")
        index_path = prefix_join(_source_root(plan, source), f"index/attempt={uuid.uuid4().hex}/selected.sqlite")
        fs, key = filesystem_for(index_path)
        fs.makedirs(key.rsplit("/", 1)[0], exist_ok=True)
        digest = _transfer(local, index_path)
        marker = {
            "binding": binding,
            "index_path": index_path,
            "index_sha256": digest,
            "selected_occurrences": total,
            "distinct_ids": distinct,
        }
        _write_json(marker_path, marker)
        return marker


def _local_index(plan: dict, source: dict, directory: str) -> tuple[str, dict]:
    marker = _read_json(_index_marker(plan, source))
    if marker["binding"] != {"source_sha256": source["manifest_sha256"]}:
        raise ValueError("Selected index belongs to different reconstruction inputs")
    destination = str(Path(directory) / (marker["index_sha256"] + ".sqlite"))
    if not Path(destination).exists():
        partial = destination + ".partial"
        if _transfer(marker["index_path"], partial) != marker["index_sha256"]:
            raise ValueError("Selected index bytes differ from their completion marker")
        Path(partial).replace(destination)
    return destination, marker


def _scan_binding(plan: dict, source: dict, file_index: int, index: dict) -> dict:
    return {
        "source_sha256": source["manifest_sha256"],
        "file_index": file_index,
        "provider_path": source["provider_paths"][file_index],
        "index_sha256": index["index_sha256"],
    }


def _candidate_digest(digest: Any, doc_id: str, text: str) -> None:
    digest.update(bytes.fromhex(doc_id))
    digest.update(hashlib.sha256(text.encode()).digest())


def _validate_ctx_reader_audit(audit: dict | None, provider_path: str, decoded_rows: int) -> dict:
    """Require exact historical filtering and evidence for every excluded raw batch."""
    if audit is None or audit["policy"] != davinci_ctx_reader_policy() or audit["complete"] is not True:
        raise ValueError("CTX provider scan lacks a complete pinned historical reader audit")
    total = audit["total_rows"]
    if (
        audit["provider_path"] != provider_path
        or total != audit["expected_rows"]
        or audit["decoded_rows"] != decoded_rows
        or min(total, decoded_rows, audit["excluded_rows"]) < 0
        or decoded_rows + audit["excluded_rows"] != total
    ):
        raise ValueError("CTX reader audit does not account for the complete original file")
    excluded = 0
    previous_batch = -1
    for batch in audit["excluded_batches"]:
        ordinal = batch["batch_index"]
        start, count = batch["raw_row_start"], batch["raw_row_count"]
        if (
            batch["provider_path"] != provider_path
            or ordinal <= previous_batch
            or start != ordinal * HISTORICAL_BATCH_SIZE
            or not 0 <= start < total
            or count != min(HISTORICAL_BATCH_SIZE, total - start)
            or not batch["error"]
            or not re.fullmatch(r"[0-9a-f]{64}", batch["original_batch_arrow_ipc_sha256"])
            or not batch["faults"]
        ):
            raise ValueError("CTX exclusion is not one audited original 16-row batch")
        previous_batch = ordinal
        excluded += count
        for fault in batch["faults"]:
            prefix = bytes.fromhex(fault["original_prefix_hex"])
            if (
                not start <= fault["raw_row"] < start + count
                or not fault["field"]
                or fault["original_bytes"] <= 0
                or len(prefix) != min(32, fault["original_bytes"])
                or not 0 <= fault["byte_start"] < fault["byte_stop"] <= fault["original_bytes"]
                or not fault["reason"]
                or not re.fullmatch(r"[0-9a-f]{64}", fault["original_bytes_sha256"])
            ):
                raise ValueError("CTX exclusion lacks original malformed-byte evidence")
            if fault["original_bytes"] <= 32 and hashlib.sha256(prefix).hexdigest() != fault["original_bytes_sha256"]:
                raise ValueError("CTX malformed-value bytes differ from their recorded SHA256")
    if excluded != audit["excluded_rows"]:
        raise ValueError("CTX excluded-batch ranges differ from the reader row accounting")
    return audit


def _ctx_reader_audit_summary(source: dict, audits: list[dict]) -> dict:
    if len(audits) != len(source["provider_paths"]):
        raise ValueError("CTX source lacks a historical reader audit for every original file")
    for path, audit in zip(source["provider_paths"], audits, strict=True):
        _validate_ctx_reader_audit(audit, path, audit["decoded_rows"])
    return {
        "policy": davinci_ctx_reader_policy(),
        "complete": True,
        "provider_files": len(audits),
        "total_rows": sum(audit["total_rows"] for audit in audits),
        "decoded_rows": sum(audit["decoded_rows"] for audit in audits),
        "excluded_rows": sum(audit["excluded_rows"] for audit in audits),
        "excluded_batches": [batch for audit in audits for batch in audit["excluded_batches"]],
        "file_audits_sha256": _digest(audits),
        "file_audits": audits,
    }


def scan_provider_file(plan: dict, name: str, file_index: int, index_directory: str, batch_size: int = 512) -> dict:
    """Checkpoint unique selected-text candidates from one complete original provider file."""
    source = _source(plan, name)
    operation = f"provider scan source={name} file_index={file_index} raw_path={source['provider_paths'][file_index]}"
    try:
        return retry_with_backoff(
            lambda: _scan_provider_file(plan, name, file_index, index_directory, batch_size),
            retryable=_is_retryable_provider_read,
            max_attempts=4,
            backoff=ExponentialBackoff(initial=5.0, maximum=30.0, factor=2.0, jitter=0.5),
            operation=operation,
        )
    except Exception as error:
        error.add_note(operation)
        raise


def _is_retryable_provider_read(error: Exception) -> bool:
    return isinstance(error, FSTimeoutError) or (
        isinstance(error, OSError)
        and (
            (error.errno == errno.EBUSY and "Reduce your rate of simultaneous reads on the same object" in str(error))
            or (
                error.errno == errno.EIO
                and str(error).endswith(
                    "An error occurred (429) when calling the HeadObject operation: Too Many Requests"
                )
            )
        )
    )


def _scan_provider_file(plan: dict, name: str, file_index: int, index_directory: str, batch_size: int) -> dict:
    _verify_plan(plan)
    source = _source(plan, name)
    if source["blocked_reason"]:
        raise ValueError(source["blocked_reason"])
    if source["provider_kind"] == "external_original_warc_replay":
        raise ValueError("Original WARC targets require the separate Focus replay and completion stages")
    local_index, index_marker = _local_index(plan, source, index_directory)
    binding = _scan_binding(plan, source, file_index, index_marker)
    marker_path = _scan_marker(plan, source, file_index)
    if _exists(marker_path):
        marker = _read_json(marker_path)
        if marker["binding"] != binding or not marker["complete"]:
            raise ValueError("Provider scan checkpoint differs from its pinned file or selected index")
        if _recipe(source["recipe"]).transform is ProviderTransform.DAVINCI_CTX:
            _validate_ctx_reader_audit(marker.get("reader_audit"), binding["provider_path"], marker["raw_rows"])
        if _recipe(source["recipe"]).transform is ProviderTransform.SOFTWARE_HERITAGE_EMPTY:
            validate_empty_audit(
                marker.get("reader_audit"), binding["provider_path"], marker["raw_rows"], marker["provider_snapshot"]
            )
        return marker
    recipe = _recipe(source["recipe"])
    replay = provider_normalizer(recipe)
    columns = PROVIDER_COLUMNS.get(recipe.transform, [recipe.text_field, recipe.id_field])
    raw_path = source["provider_paths"][file_index]
    logger.info("Scanning provider source=%s file_index=%d raw_path=%s", name, file_index, raw_path)
    raw_fs, raw_key = filesystem_for(raw_path)
    before = raw_fs.info(raw_key)
    snapshot = {key: before.get(key) for key in ("size", "etag", "ETag", "version_id") if before.get(key) is not None}
    destination = prefix_join(
        _source_root(plan, source), f"candidates/file={file_index:06d}/attempt={uuid.uuid4().hex}/candidates.parquet"
    )
    fs, key = filesystem_for(destination)
    fs.makedirs(key.rsplit("/", 1)[0], exist_ok=True)
    digest = hashlib.sha256()
    rows = hits = unique = 0
    pending = []
    reader_audit = DavinciCtxReadAudit()
    empty_audit = SoftwareHeritageEmptyAudit(snapshot)
    if recipe.transform is ProviderTransform.DAVINCI_CTX:
        originals = read_davinci_context(raw_path, reader_audit)
    elif recipe.transform is ProviderTransform.SOFTWARE_HERITAGE_EMPTY:
        originals = read_software_heritage(raw_path, empty_audit, batch_size)
    else:
        originals = provider_rows(raw_path, columns, batch_size)
    with tempfile.TemporaryDirectory(prefix="hero-candidate-") as temporary:
        with (
            sqlite3.connect(Path(local_index).as_uri() + "?mode=ro", uri=True) as selected,
            sqlite3.connect(Path(temporary) / "seen.sqlite") as seen,
            fs.open(key, "wb") as stream,
            pq.ParquetWriter(stream, CANDIDATE_SCHEMA) as writer,
        ):
            seen.execute("PRAGMA journal_mode=OFF")
            seen.execute("CREATE TABLE seen (id BLOB PRIMARY KEY, text_sha256 BLOB) WITHOUT ROWID")
            for raw in originals:
                rows += 1
                for document in replay([raw]):
                    doc_id, text = document["id"], document["text"]
                    identity = bytes.fromhex(doc_id)
                    if selected.execute("SELECT 1 FROM selected WHERE id=?", (identity,)).fetchone() is None:
                        continue
                    hits += 1
                    text_hash = hashlib.sha256(text.encode()).digest()
                    previous = seen.execute("SELECT text_sha256 FROM seen WHERE id=?", (identity,)).fetchone()
                    if previous is not None:
                        if previous[0] != text_hash:
                            raise ValueError("Provider file contains conflicting text for the same selected ID")
                        continue
                    seen.execute("INSERT INTO seen VALUES (?,?)", (identity, text_hash))
                    pending.append({"id": doc_id, "text": text})
                    unique += 1
                    _candidate_digest(digest, doc_id, text)
                    if len(pending) >= batch_size:
                        writer.write_table(pa.Table.from_pylist(pending, schema=CANDIDATE_SCHEMA))
                        pending.clear()
                if recipe.transform is ProviderTransform.SOFTWARE_HERITAGE_EMPTY:
                    empty_audit.verified(raw)
            if pending:
                writer.write_table(pa.Table.from_pylist(pending, schema=CANDIDATE_SCHEMA))
    after = raw_fs.info(raw_key)
    if snapshot != {key: after.get(key) for key in snapshot}:
        raise ValueError("Original provider object changed during reconstruction")
    marker = {
        "binding": binding,
        "complete": True,
        "scope": "provider_file_candidates",
        "provider_snapshot": snapshot,
        "raw_rows": rows,
        "matching_provider_outputs": hits,
        "candidate_rows": unique,
        "candidate_digest": digest.hexdigest(),
        "candidates": destination,
    }
    if recipe.transform is ProviderTransform.DAVINCI_CTX:
        marker["reader_audit"] = _validate_ctx_reader_audit(reader_audit.report(), raw_path, rows)
    if recipe.transform is ProviderTransform.SOFTWARE_HERITAGE_EMPTY:
        marker["reader_audit"] = validate_empty_audit(empty_audit.report(), raw_path, rows, snapshot)
    _write_json(marker_path, marker)
    return marker


class _TextShards:
    def __init__(self, output_prefix: str, byte_limit: int):
        self.prefix = output_prefix
        self.byte_limit = byte_limit
        self.outputs: list[dict] = []
        self.rows: list[dict] = []
        self.bytes = 0

    def append(self, name: str, doc_id: str, text: str) -> None:
        self.rows.append({"marin_source": name, "id": doc_id, "text": text})
        self.bytes += len(text.encode())
        if self.bytes >= self.byte_limit or len(self.rows) >= 65536:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        destination = prefix_join(self.prefix, f"text-{len(self.outputs):06d}.parquet")
        fs, key = filesystem_for(destination)
        fs.makedirs(key.rsplit("/", 1)[0], exist_ok=True)
        with fs.open(key, "wb") as stream:
            pq.write_table(pa.Table.from_pylist(self.rows, schema=RECONSTRUCTED_TEXT_SCHEMA), stream)
        self.outputs.append({"path": destination, "occurrences": len(self.rows)})
        self.rows.clear()
        self.bytes = 0


def reconcile_source(
    plan: dict, name: str, index_directory: str, *, sqlite_cache_mib: int, text_shard_bytes: int = TEXT_SHARD_BYTES
) -> dict:
    """Verify complete source coverage with an explicitly bounded SQLite page cache."""
    if not 1 <= sqlite_cache_mib <= 16384:
        raise ValueError("Reducer SQLite cache must be between 1 MiB and 16 GiB")
    _verify_plan(plan)
    source = _source(plan, name)
    if source["blocked_reason"]:
        raise ValueError(source["blocked_reason"])
    if source["provider_kind"] == "external_original_warc_replay":
        raise ValueError("Original WARC targets require the separate Focus replay and completion stages")
    local_index, index_marker = _local_index(plan, source, index_directory)
    source_marker = prefix_join(_source_root(plan, source), "reconstructed/_complete.json")
    binding = {
        "source_sha256": source["manifest_sha256"],
        "index_sha256": index_marker["index_sha256"],
    }
    is_ctx = _recipe(source["recipe"]).transform is ProviderTransform.DAVINCI_CTX
    is_swh_empty = _recipe(source["recipe"]).transform is ProviderTransform.SOFTWARE_HERITAGE_EMPTY
    previous = _read_json(source_marker) if _exists(source_marker) else None
    if previous is not None:
        if previous["binding"] != binding:
            raise ValueError("Source reconstruction checkpoint belongs to different inputs")
        if not is_ctx and not is_swh_empty:
            return previous
    markers = []
    file_audits = []
    for file_index in range(len(source["provider_paths"])):
        marker = _read_json(_scan_marker(plan, source, file_index))
        if (
            marker["binding"] != _scan_binding(plan, source, file_index, index_marker)
            or not marker["complete"]
            or marker["scope"] != "provider_file_candidates"
        ):
            raise ValueError("Provider-file coverage contains an incomplete or wrong-input candidate scan")
        if is_ctx:
            file_audits.append(
                _validate_ctx_reader_audit(
                    marker.get("reader_audit"), source["provider_paths"][file_index], marker["raw_rows"]
                )
            )
        if is_swh_empty:
            file_audits.append(
                validate_empty_audit(
                    marker.get("reader_audit"),
                    source["provider_paths"][file_index],
                    marker["raw_rows"],
                    marker["provider_snapshot"],
                )
            )
        markers.append(marker)
    reader_summary = _ctx_reader_audit_summary(source, file_audits) if is_ctx else None
    if is_swh_empty:
        reader_summary = empty_source_audit(source["provider_paths"], file_audits)
    if previous is not None:
        if previous.get("reader_audit") != reader_summary:
            raise ValueError("Source checkpoint differs from its complete per-file reader audits")
        return previous
    output = prefix_join(_source_root(plan, source), f"reconstructed/attempt={uuid.uuid4().hex}")
    text_outputs = _TextShards(output, text_shard_bytes)
    emitted = duplicate_candidates = 0
    with tempfile.TemporaryDirectory(prefix="hero-reconcile-") as temporary:
        local = str(Path(temporary) / "selected.sqlite")
        _transfer(local_index, local)
        with sqlite3.connect(local) as selected:
            selected.execute("PRAGMA journal_mode=OFF")
            selected.execute("PRAGMA synchronous=OFF")
            selected.execute(f"PRAGMA cache_size=-{sqlite_cache_mib * 1024}")
            selected.execute("PRAGMA temp_store=FILE")
            selected.execute("ALTER TABLE selected ADD COLUMN text_sha256 BLOB")
            for marker in markers:
                digest = hashlib.sha256()
                rows = 0
                fs, key = filesystem_for(marker["candidates"])
                with fs.open(key, "rb") as stream:
                    parquet = pq.ParquetFile(stream)
                    if not parquet.schema_arrow.equals(CANDIDATE_SCHEMA):
                        raise ValueError("Provider candidate schema differs from the expected wire format")
                    for batch in parquet.iter_batches(batch_size=512):
                        for candidate in batch.to_pylist():
                            doc_id, text = candidate["id"], candidate["text"]
                            if generate_id(text) != doc_id:
                                raise ValueError("Candidate text does not hash to its selected document ID")
                            _candidate_digest(digest, doc_id, text)
                            rows += 1
                            identity = bytes.fromhex(doc_id)
                            row = selected.execute(
                                "SELECT occurrences,text_sha256 FROM selected WHERE id=?", (identity,)
                            ).fetchone()
                            if row is None:
                                raise ValueError("Candidate output contains an ID absent from the selected ledger")
                            occurrences, previous_hash = row
                            current_hash = hashlib.sha256(text.encode()).digest()
                            if previous_hash is not None:
                                if previous_hash != current_hash:
                                    raise ValueError(
                                        "Original provider files contain conflicting text for a selected ID"
                                    )
                                duplicate_candidates += 1
                                continue
                            selected.execute("UPDATE selected SET text_sha256=? WHERE id=?", (current_hash, identity))
                            for _ in range(occurrences):
                                text_outputs.append(name, doc_id, text)
                                emitted += 1
                if rows != marker["candidate_rows"] or digest.hexdigest() != marker["candidate_digest"]:
                    raise ValueError("Provider candidates differ from their completed scan counts or content digest")
                selected.commit()
            missing_ids, missing_occurrences = selected.execute(
                "SELECT COUNT(*),COALESCE(SUM(occurrences),0) FROM selected WHERE text_sha256 IS NULL"
            ).fetchone()
            text_outputs.flush()
    report = {
        "binding": binding,
        "marin_source": name,
        "provider_files": len(markers),
        "selected_occurrences": source["selected_occurrences"],
        "reconstructed_occurrences": emitted,
        "missing_ids": missing_ids,
        "missing_occurrences": missing_occurrences,
        "duplicate_candidate_ids": duplicate_candidates,
        "sqlite_cache_mib": sqlite_cache_mib,
        "historical_occurrence_provenance": "not_established",
        "outputs": text_outputs.outputs,
        "complete": not missing_occurrences and emitted == source["selected_occurrences"],
    }
    if is_ctx or is_swh_empty:
        report["reader_audit"] = reader_summary
    _write_json(prefix_join(output, "report.json"), report)
    if not report["complete"]:
        raise ValueError(f"Original-provider reconstruction left {missing_occurrences} selected occurrences unresolved")
    _write_json(source_marker, report)
    return report


def aggregate_reconstruction(plan: dict) -> dict:
    """Require every selected source before certifying complete-cluster text recovery."""
    _verify_plan(plan)
    completed = []
    output_paths: set[str] = set()
    for source in plan["sources"]:
        marker = _read_json(prefix_join(_source_root(plan, source), "reconstructed/_complete.json"))
        index = _read_json(_index_marker(plan, source))
        expected_binding = {"source_sha256": source["manifest_sha256"], "index_sha256": index["index_sha256"]}
        if (
            not marker["complete"]
            or index["binding"] != {"source_sha256": source["manifest_sha256"]}
            or index["selected_occurrences"] != source["selected_occurrences"]
            or marker["binding"] != expected_binding
            or marker["marin_source"] != source["marin_source"]
            or marker["missing_ids"] != 0
            or marker["missing_occurrences"] != 0
            or marker["selected_occurrences"] != source["selected_occurrences"]
            or marker["reconstructed_occurrences"] != source["selected_occurrences"]
            or marker["provider_files"] != len(source["provider_paths"])
        ):
            raise ValueError("Source reconstruction does not cover the pinned complete source")
        if _recipe(source["recipe"]).transform is ProviderTransform.DAVINCI_CTX:
            audit = marker.get("reader_audit")
            if audit is None or audit != _ctx_reader_audit_summary(source, audit["file_audits"]):
                raise ValueError("CTX source completion lacks its complete historical exclusion audit")
        if _recipe(source["recipe"]).transform is ProviderTransform.SOFTWARE_HERITAGE_EMPTY:
            audit = marker.get("reader_audit")
            if audit is None or audit != empty_source_audit(source["provider_paths"], audit["file_audits"]):
                raise ValueError("SWH source completion lacks its complete historical-compatible empty audit")
        output_rows = 0
        for output in marker["outputs"]:
            if output["path"] in output_paths:
                raise ValueError("Repeated reconstructed output path cannot establish distinct occurrence coverage")
            output_paths.add(output["path"])
            fs, key = filesystem_for(output["path"])
            with fs.open(key, "rb") as stream:
                parquet = pq.ParquetFile(stream)
                if (
                    not parquet.schema_arrow.equals(RECONSTRUCTED_TEXT_SCHEMA)
                    or parquet.metadata.num_rows != output["occurrences"]
                ):
                    raise ValueError(
                        "Reconstructed output schema or occurrence count differs from its source completion"
                    )
                output_rows += parquet.metadata.num_rows
        if output_rows != source["selected_occurrences"]:
            raise ValueError("Reconstructed source outputs do not contain every selected occurrence")
        completed.append(marker)
    total = sum(item["reconstructed_occurrences"] for item in completed)
    if total != plan["selected_occurrences"]:
        raise ValueError("Reconstructed sources do not cover the complete selected cluster")
    result = {
        "plan_sha256": plan["manifest_sha256"],
        "target_cluster": plan["target_cluster"],
        "complete_cluster": True,
        "reconstructed_occurrences": total,
        "sources": len(completed),
        "validation": (
            "original-provider text hashes and selected occurrence counts; "
            "historical occurrence provenance not established"
        ),
    }
    _write_json(prefix_join(plan["output_prefix"], "_complete.json"), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "index", "scan", "reconcile", "aggregate"), required=True)
    parser.add_argument("--plan")
    parser.add_argument("--selection-summary")
    parser.add_argument("--selection-reports-prefix")
    parser.add_argument("--selection-task-count", type=int)
    parser.add_argument("--provider-manifest")
    parser.add_argument("--previous-plan")
    parser.add_argument("--output-prefix")
    parser.add_argument("--source")
    parser.add_argument("--exclude-source", action="append", default=[])
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--task-count", type=int, default=1)
    parser.add_argument("--r2-endpoint")
    parser.add_argument("--sqlite-cache-mib", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    required = ["plan"]
    if args.mode == "prepare":
        required += [
            "selection_summary",
            "selection_reports_prefix",
            "selection_task_count",
            "provider_manifest",
            "output_prefix",
        ]
    if any(getattr(args, name) is None for name in required):
        parser.error("Missing required arguments for mode: " + ", ".join(required))
    if args.mode == "reconcile" and args.sqlite_cache_mib is None:
        parser.error("--sqlite-cache-mib is required for source reconciliation (16384 for the 64 GiB QA worker)")
    if args.r2_endpoint:
        key, secret = get_secret("R2_KEY_ID"), get_secret("R2_KEY_SECRET")
        if not key or not secret:
            raise ValueError("The original R2 snapshot requires namespaced R2 credentials")
        # Iris may export a CW ambient endpoint. Registered CW buckets keep their
        # explicit routing while the unregistered historical R2 bucket uses this one.
        os.environ["AWS_ENDPOINT_URL"] = args.r2_endpoint
        os.environ["FSSPEC_S3"] = json.dumps(fsspec_s3_conf(args.r2_endpoint))
        fsspec.config.conf.pop("s3", None)
        configure_fsspec_s3(args.r2_endpoint, key, secret)
    if args.mode == "prepare":
        selection = _read_json(args.selection_summary)
        reports = [
            _read_json(
                prefix_join(args.selection_reports_prefix, f"task-{i:05d}-of-{args.selection_task_count:05d}.json")
            )
            for i in range(args.selection_task_count)
        ]
        plan = prepare_reconstruction_plan(
            selection,
            reports,
            _read_json(args.provider_manifest),
            args.output_prefix,
            _read_json(args.previous_plan) if args.previous_plan else None,
        )
        _write_json(args.plan, plan)
        print(
            json.dumps(
                {
                    "sources": len(plan["sources"]),
                    "blocked_sources": [s["marin_source"] for s in plan["sources"] if s["blocked_reason"]],
                }
            )
        )
        return
    plan = _read_json(args.plan)
    if args.exclude_source and args.mode != "scan":
        parser.error("--exclude-source is only supported for provider-file scan dispatch")
    known_sources = {source["marin_source"] for source in plan["sources"]}
    excluded = set(args.exclude_source)
    if unknown := excluded - known_sources:
        parser.error("Unknown excluded sources: " + ", ".join(sorted(unknown)))
    if args.source in excluded:
        parser.error("The requested source is also excluded")
    sources = [
        source
        for source in plan["sources"]
        if (args.source is None or source["marin_source"] == args.source) and source["marin_source"] not in excluded
    ]
    if args.source is not None and not sources:
        raise ValueError("Requested source is absent from the reconstruction plan")
    job_info = get_job_info()
    if args.mode in {"scan", "reconcile"}:
        external = [s for s in sources if s["provider_kind"] == "external_original_warc_replay"]
        if args.source is not None and external:
            raise ValueError("Requested source requires the separate original WARC replay and completion stages")
        sources = [s for s in sources if s["provider_kind"] != "external_original_warc_replay"]
        logger.info(json.dumps({"external_sources": len(external)}))
        blocked = [s for s in sources if s["blocked_reason"]]
        if args.source is not None and blocked:
            raise ValueError(blocked[0]["blocked_reason"])
        logger.info(
            json.dumps(
                {"mode": args.mode, "blocked_sources": len(blocked), "ready_sources": len(sources) - len(blocked)}
            )
        )
        sources = [s for s in sources if not s["blocked_reason"]]
    task_index = args.task_index if args.task_index is not None else (job_info.task_index if job_info else 0)
    if not 0 <= task_index < args.task_count:
        raise ValueError("Task index is outside the assigned reconstruction partition")
    with tempfile.TemporaryDirectory(prefix="hero-index-cache-") as directory:
        if args.mode == "index":
            for source in sources[task_index :: args.task_count]:
                build_selected_index(plan, source["marin_source"])
        elif args.mode == "scan":
            tasks = [(s, i) for s in sources if not s["blocked_reason"] for i in range(len(s["provider_paths"]))]
            for source, index in tasks[task_index :: args.task_count]:
                scan_provider_file(plan, source["marin_source"], index, directory)
        elif args.mode == "reconcile":
            for source in sources[task_index :: args.task_count]:
                reconcile_source(plan, source["marin_source"], directory, sqlite_cache_mib=args.sqlite_cache_mib)
        else:
            print(json.dumps(aggregate_reconstruction(plan)))


if __name__ == "__main__":
    main()
