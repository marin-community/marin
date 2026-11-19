# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage hash-verified attempt evidence from the selected difficulty measurements."""

import hashlib
import io
import json
import tarfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import click
from finestore.rl.difficulty import DifficultyAttempt, DifficultyMeasurement, measurement_from_attempts
from marina.client import DEFAULT_MARINA_URL, AppletQuery, applet_query
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from infra.marina.apps.rl_data_catalog.load import batches, ensure_tables
from infra.marina.apps.rl_data_catalog.role_projection import ManifestReference, manifest_bytes
from infra.marina.apps.rl_data_catalog.server.content_access import content_is_withheld
from infra.marina.apps.rl_data_catalog.server.evidence_content import pack_content
from infra.marina.apps.rl_data_catalog.server.restrictions import declaration_restriction

MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_MEMBER_BYTES = 16 * 1024 * 1024
ROLLOUT_MEMBERS = {
    "conversation": "trial/agent/conversation.json",
    "model_input": "trial/agent/model-input.json",
    "verifier": "trial/verifier/verdict.json",
    "exception": "trial/exception.txt",
}


def rollout_content(raw: bytes) -> dict[str, Any]:
    """Read only named regular evidence files; never extract archive paths or serve credentials/config."""
    content: dict[str, Any] = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        for field, name in ROLLOUT_MEMBERS.items():
            member = members.get(name)
            if member is None:
                content[field] = None
                continue
            if not member.isfile() or member.size > MAX_MEMBER_BYTES:
                raise ValueError(f"Rollout evidence is not a bounded regular file: {name}")
            stream = archive.extractfile(member)
            assert stream is not None
            with stream:
                value = stream.read().decode("utf-8")
            content[field] = json.loads(value) if name.endswith(".json") else value
    return content


def stage_attempt(reference: dict, tasks: set[str], solver: str) -> dict[str, Any]:
    receipt = ManifestReference.model_validate(reference)
    payload = json.loads(manifest_bytes(receipt))
    task_id = payload["task_id"]
    if task_id not in tasks:
        raise ValueError(f"Attempt belongs to a task outside the frozen measurement: {receipt.uri}")
    result = payload["result"]
    started, finished = result.get("started_at"), result.get("finished_at")
    duration = (
        (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds()
        if started and finished
        else None
    )
    archive_ref = ManifestReference(uri=payload["archive"], sha256=payload["archive_sha256"])
    path = StoragePath(archive_ref.uri)
    if path.size() > MAX_ARCHIVE_BYTES:
        raise ValueError(f"Attempt archive exceeds the bounded viewer payload: {archive_ref.uri}")
    content = rollout_content(manifest_bytes(archive_ref))
    attempt_id = hashlib.sha256(receipt.uri.encode()).hexdigest()
    return {
        "attempt_id": attempt_id,
        "task_id": task_id,
        "summary": {
            "attempt_id": attempt_id,
            "task_id": task_id,
            "solver": solver,
            "attempt": receipt.uri.rsplit("/", 2)[-2],
            "trial_id": result["id"],
            "score": payload.get("reward"),
            "status": payload["status"],
            "reason": payload.get("reason"),
            "duration": duration,
            "exception": (result.get("exception_info") or {}).get("exception_type"),
            "has_rollout": True,
            "receipt": reference,
            "archive": archive_ref.model_dump(mode="json"),
        },
        "detail": {
            **content,
            "verifier_result": result.get("verifier_result"),
            "exception_info": result.get("exception_info"),
            "receipt": reference,
            "archive": archive_ref.model_dump(mode="json"),
        },
    }


def stage_rollouts(measurement: dict[str, Any], source: dict[str, Any], workers: int) -> tuple[dict, list[dict]]:
    if content_is_withheld(source):
        raise ValueError("Restricted rollouts remain in their private store; do not stage content")
    reference = ManifestReference.model_validate(measurement["attempts_receipt"])
    aggregate = json.loads(manifest_bytes(reference))
    typed = DifficultyMeasurement.model_validate(measurement)
    counted = measurement_from_attempts(typed, [DifficultyAttempt.model_validate(row) for row in aggregate["attempts"]])
    if (counted.n, counted.solved) != (typed.n, typed.solved):
        raise ValueError("Attempt outcomes disagree with the recorded measurement counts")
    sample = measurement["sample"]
    tasks = {row["task_id"] for row in sample["tasks"]}
    outcomes = aggregate["attempts"]
    if any(row["task_id"] not in tasks for row in outcomes):
        raise ValueError("Aggregate outcomes contain a task outside the frozen measurement sample")
    references = aggregate.get("attempt_receipts", [])
    if len({row["uri"] for row in references}) != len(references):
        raise ValueError("Duplicate attempt evidence references")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda ref: stage_attempt(ref, tasks, measurement["solver"]), references))
    # The aggregate records the qualified task outcome; an immutable raw
    # attempt can retain an earlier classification (for example a prompt-budget
    # failure). Preserve both rather than rewriting the original attempt.
    for row in rows:
        row["summary"]["task_outcomes"] = [outcome for outcome in outcomes if outcome["task_id"] == row["task_id"]]
    covered = {row["task_id"] for row in rows}
    for task_id in sorted({row["task_id"] for row in outcomes} - covered):
        outcome = next(row for row in reversed(outcomes) if row["task_id"] == task_id)
        attempt_id = "outcome:" + hashlib.sha256(task_id.encode()).hexdigest()
        rows.append(
            {
                "attempt_id": attempt_id,
                "task_id": task_id,
                "summary": {
                    "attempt_id": attempt_id,
                    "task_id": task_id,
                    "solver": measurement["solver"],
                    "attempt": "not recorded",
                    "trial_id": None,
                    "score": outcome.get("reward"),
                    "status": outcome["status"],
                    "reason": outcome.get("reason"),
                    "duration": None,
                    "exception": None,
                    "has_rollout": False,
                    "receipt": reference.model_dump(mode="json"),
                },
                "detail": {
                    "status": "unavailable",
                    "reason": "No digest-bound per-attempt artifact is recorded for this outcome.",
                },
            }
        )
    metadata = {
        "data_restriction": declaration_restriction(source).value,
        "recorded_attempt_count": len(references),
        "attempts_receipt": reference.model_dump(mode="json"),
        "attempt_count": len(rows),
        "task_count": len({row["task_id"] for row in rows}),
        "outcomes": outcomes,
        "sample_receipt": measurement["sample_receipt"],
    }
    return metadata, rows


def load_rollouts(measurement: dict, metadata: dict, rows: list[dict], query: AppletQuery) -> dict[str, int]:
    """Archive a complete attempt set, verifying idempotent readback before selecting it."""
    sources = query(
        "SELECT s.declaration FROM atlas_sources s JOIN atlas_current c USING(load_id) "
        "WHERE c.name='latest' AND s.id=:source",
        {"source": measurement["source_id"]},
    )["rows"]
    if len(sources) != 1 or content_is_withheld(sources[0]["declaration"]):
        raise ValueError("Rollout content may only be staged for an open selected source")
    if metadata["data_restriction"] != declaration_restriction(sources[0]["declaration"]).value:
        raise ValueError("Staged rollout restriction differs from its source")
    stored = query(
        "SELECT measurement FROM atlas_difficulty WHERE measurement_id=:id", {"id": measurement["measurement_id"]}
    )["rows"]
    if stored != [{"measurement": measurement}]:
        raise ValueError("Rollout measurement differs from the stored immutable observation")
    tasks = {task["task_id"] for task in measurement["sample"]["tasks"]}
    if metadata["attempts_receipt"] != measurement["attempts_receipt"] or any(
        row["task_id"] not in tasks for row in rows
    ):
        raise ValueError("Staged attempts do not belong to the immutable measurement sample")
    if len({row["attempt_id"] for row in rows}) != len(rows) or len(rows) != metadata["attempt_count"]:
        raise ValueError("Staged attempt IDs or total disagree with the sealed set")
    ensure_tables(query, ("atlas_rollout_sets", "atlas_rollout_attempts"))
    identity = measurement["measurement_id"]
    inserted = 0
    for body in batches({**row, "detail": pack_content(row["detail"])} for row in rows):
        inserted += query(
            "INSERT INTO atlas_rollout_attempts SELECT :id,r.attempt_id,r.task_id,r.summary,r.detail "
            "FROM jsonb_to_recordset(CAST(:rows AS JSONB)) "
            "AS r(attempt_id TEXT,task_id TEXT,summary JSONB,detail JSONB) "
            "ON CONFLICT DO NOTHING",
            {"id": identity, "rows": body},
        )["row_count"]
        expected = json.loads(body)
        actual = query(
            "SELECT attempt_id,task_id,summary,detail FROM atlas_rollout_attempts "
            "WHERE measurement_id=:id AND attempt_id IN "
            "(SELECT value FROM jsonb_array_elements_text(CAST(:ids AS JSONB))) ORDER BY attempt_id",
            {"id": identity, "ids": json.dumps([row["attempt_id"] for row in expected])},
        )["rows"]
        if actual != sorted(expected, key=lambda row: row["attempt_id"]):
            raise ValueError(f"Stored rollout evidence differs for measurement {identity}")
    count = query("SELECT COUNT(*) AS count FROM atlas_rollout_attempts WHERE measurement_id=:id", {"id": identity})[
        "rows"
    ][0]["count"]
    if count != metadata["attempt_count"]:
        raise ValueError("Stored attempt set does not match its sealed count")
    query(
        "INSERT INTO atlas_rollout_sets VALUES (:id,CAST(:metadata AS JSONB)) ON CONFLICT DO NOTHING",
        {"id": identity, "metadata": json.dumps(metadata)},
    )
    if query("SELECT metadata FROM atlas_rollout_sets WHERE measurement_id=:id", {"id": identity})["rows"] != [
        {"metadata": metadata}
    ]:
        raise ValueError("Existing measurement points to different rollout evidence")
    return {"inserted": inserted, "attempts": count}


@click.command(help=__doc__)
@click.option("--measurement-id", required=True)
@click.option("--app-id", required=True)
@click.option("--url", default=DEFAULT_MARINA_URL)
@click.option("--workers", type=click.IntRange(1, 32), default=8)
@click.option("--receipt", type=click.Path(path_type=Path), required=True)
def main(measurement_id: str, app_id: str, url: str, workers: int, receipt: Path) -> None:
    if receipt.exists():
        raise click.ClickException("Choose a new receipt path")
    configure_coreweave_s3()
    query = applet_query(url, app_id)
    records = query("SELECT measurement FROM atlas_difficulty WHERE measurement_id=:id", {"id": measurement_id})["rows"]
    if len(records) != 1:
        raise click.ClickException("Measurement not found")
    measurement = records[0]["measurement"]
    sources = query(
        "SELECT s.declaration FROM atlas_sources s JOIN atlas_current c USING(load_id) "
        "WHERE c.name='latest' AND s.id=:id",
        {"id": measurement["source_id"]},
    )["rows"]
    if len(sources) != 1:
        raise click.ClickException("Source is not in the selected catalog")
    metadata, rows = stage_rollouts(measurement, sources[0]["declaration"], workers)
    result = load_rollouts(measurement, metadata, rows, query)
    receipt.write_text(json.dumps({"measurement_id": measurement_id, **result, "metadata": metadata}, indent=2) + "\n")
    click.echo(json.dumps(result))


if __name__ == "__main__":
    main()
