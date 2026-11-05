# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize an immutable combined BioTasks Taskforge snapshot."""

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import fsspec


def _read_bytes(uri: str) -> bytes:
    with fsspec.open(uri, "rb") as stream:
        return stream.read()


def _write_new(uri: str, data: bytes) -> None:
    fs, path = fsspec.core.url_to_fs(uri)
    if fs.exists(path):
        raise FileExistsError(f"refusing to overwrite immutable artifact: {uri}")
    with fs.open(path, "wb") as stream:
        stream.write(data)


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--notebooks", required=True)
    parser.add_argument("--repositories", required=True)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--iris-job", required=True)
    args = parser.parse_args()

    result_bytes = _read_bytes(args.results)
    rows = [json.loads(line) for line in result_bytes.splitlines() if line]
    terminals = Counter(row["terminal"] for row in rows)
    sources = Counter(row["source_campaign"] for row in rows)
    idea_ids = {row["idea"]["idea_id"] for row in rows}
    task_digests = {row["task_digest"] for row in rows}
    if terminals != {"accepted": len(rows)}:
        raise ValueError(f"combined Taskforge results are not all accepted: {terminals}")
    if len(idea_ids) != len(rows) or len(task_digests) != len(rows):
        raise ValueError("combined results contain duplicate idea or TaskSpec identities")

    tasks = [
        {
            "idea_id": row["idea"]["idea_id"],
            "screening_record_sha256": row["screening_record_sha256"],
            "source_campaign": row["source_campaign"],
            "task": json.loads(row["task_json"]),
            "task_digest": row["task_digest"],
            "task_id": row["task_id"],
        }
        for row in rows
    ]
    tasks.sort(key=lambda row: row["task_id"])
    task_bytes = _jsonl(tasks)
    artifacts = {
        "taskforge-results.jsonl": result_bytes,
        "taskspecs.jsonl": task_bytes,
    }
    artifact_manifest = {
        name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in artifacts.items()
    }
    for name, data in artifacts.items():
        _write_new(f"{args.snapshot_root.rstrip('/')}/{name}", data)

    inputs = {}
    for name, uri in (("notebooks", args.notebooks), ("repositories", args.repositories)):
        data = _read_bytes(uri)
        inputs[name] = {"uri": uri, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = {
        "schema_version": "1",
        "campaign_id": "biotasks/combined/v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "code_commit": args.code_commit,
        "iris_job": args.iris_job,
        "release_status": "internal_unreviewed_do_not_publish",
        "selection_rule": "blinded GLM recommended_action=accept from each frozen source campaign",
        "taskforge_acceptance_note": (
            "structural acceptance is not scientific review, rights approval, solver calibration, or release approval"
        ),
        "inputs": inputs,
        "summary": {
            "candidate_count": len(rows),
            "terminal_states": dict(sorted(terminals.items())),
            "source_campaigns": dict(sorted(sources.items())),
            "unique_idea_count": len(idea_ids),
            "unique_task_spec_count": len(task_digests),
        },
        "artifacts": artifact_manifest,
    }
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    _write_new(f"{args.snapshot_root.rstrip('/')}/manifest.json", manifest_bytes)
    print(manifest_bytes.decode(), end="")


if __name__ == "__main__":
    main()
