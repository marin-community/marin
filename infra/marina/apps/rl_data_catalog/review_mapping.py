# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Join sealed FULL observations to original reviewer batches without choosing later judgments."""

import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.pipeline.models import ReviewRecord


def receipt_document(uri: str, expected: str | None = None) -> tuple[dict, dict]:
    """Seal a read-only receipt, checking a publisher digest when one survives."""
    path = StoragePath(uri)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"Original receipt differs from its published digest: {uri}")
    if path.read_bytes() != raw:
        raise ValueError(f"Original receipt changed during staging: {uri}")
    return json.loads(raw), {
        "receipt": {"uri": uri, "sha256": digest},
        "hash_basis": "publisher_reference" if expected else "staging_readback",
    }


def referenced_artifacts(batch: dict) -> list[tuple[str, str | None]]:
    """Follow explicit overlay origins instead of treating an overlay as a reviewer output."""
    references = [batch.get("primary_review"), batch.get("repair_receipt"), batch.get("pass2a"), batch.get("pass2b")]
    references.extend(origin.get("receipt") for origin in batch.get("origins", {}).values())
    result = []
    for reference in references:
        if not isinstance(reference, dict):
            continue
        if reference.get("artifact"):
            result.append(
                (reference["artifact"] + "/.artifact.json", reference.get("artifact_sha256") or reference.get("sha256"))
            )
        elif reference.get("kind") == "batch_reviews":
            result.append((reference["path"], reference["sha256"]))
    return result


def matching_batch_rows(batch: dict, rows: dict[str, dict]) -> list[str]:
    """Identify original reviewer lineage by exact task ID and complete typed observation."""
    contracts = {row["task_id"]: row for row in batch["identity"]["contracts"]}
    matched = []
    for observation in batch["reviews"]:
        task = observation["task_id"]
        row = rows.get(task)
        contract = contracts.get(task)
        if row is None or contract is None:
            continue
        if any(not contract.get(field) for field in ("task_id", "task_sha256", "runtime_sha256")):
            continue
        if ReviewRecord.model_validate_json(json.dumps(observation)) == ReviewRecord.model_validate_json(row["review"]):
            matched.append(task)
    return matched


def original_batch_mapping(rows: list[dict], roots: list[str], workers: int) -> dict[str, Any]:
    """Map exact original batches; absent and ambiguous matches remain explicit."""
    by_task = {row["task_id"]: row for row in rows}
    candidates: dict[str, dict[str, dict]] = defaultdict(dict)
    visited: set[str] = set()
    pending = [(root, None) for root in roots]
    receipts: list[dict] = []
    seen_digests: dict[str, str] = {}
    while pending:
        current = []
        for uri, digest in pending:
            if uri in visited:
                if digest and uri in seen_digests and seen_digests[uri] != digest:
                    raise ValueError(f"Published evidence identity changed within the mapping: {uri}")
                continue
            visited.add(uri)
            if not uri.endswith(".json"):
                artifact = StoragePath(uri + "/.artifact.json")
                if artifact.exists():
                    current.append((str(artifact), digest))
                else:
                    paths = StoragePath(uri + "/batches/*/attempts/*.json").glob()
                    paths.extend(StoragePath(uri + "/batches/*.json").glob())
                    current.extend((str(path), None) for path in sorted(paths, key=str))
            else:
                current.append((uri, digest))
        pending = []
        # Only one bounded wave is held; large corpora never accumulate raw batches.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(current), 32):
                wave = current[start : start + 32]
                documents = pool.map(lambda reference: receipt_document(*reference), wave)
                for (uri, _), (document, provenance) in zip(wave, documents, strict=True):
                    receipts.append(provenance)
                    seen_digests[uri] = provenance["receipt"]["sha256"]
                    if "result" in document:
                        pending.extend((value, None) for value in document["result"].get("batch_receipts", []))
                        continue
                    if "identity" not in document or ("reviews" not in document and "origins" not in document):
                        raise ValueError(f"Unrecognized review evidence document: {uri}")
                    if "attempts" not in document:
                        pending.extend(referenced_artifacts(document))
                        pending.extend((uri, None) for uri in document.get("sealed_batch_receipts", []))
                        continue
                    for task in matching_batch_rows(document, by_task):
                        candidates[task][uri] = {
                            "receipt": provenance["receipt"],
                            "hash_basis": provenance["hash_basis"],
                            "reviewer": document["identity"]["reviewer"],
                            "original_contract": next(
                                contract for contract in document["identity"]["contracts"] if contract["task_id"] == task
                            ),
                        }
    entries = []
    for row in rows:
        matches = candidates.get(row["task_id"], {})
        entry = {field: row[field] for field in ("task_id", "task_sha256", "runtime_sha256")}
        if len(matches) == 1:
            entry.update(status="mapped", **next(iter(matches.values())))
        else:
            entry.update(
                status="not_mapped",
                reason=(
                    "No original batch with recorded input identities matches the sealed task ID and complete judgment."
                    if not matches
                    else "Multiple original batches match; original reviewer attempt is ambiguous."
                ),
                matching_receipts=sorted(matches),
            )
        entries.append(entry)
    return {
        "entries": entries,
        "evidence_receipts": receipts,
        "mapped": sum(row["status"] == "mapped" for row in entries),
        "not_mapped": sum(row["status"] == "not_mapped" for row in entries),
    }
