# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze a separate scored 20-model snapshot, only after complete raw trials."""

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import fsspec
from taskcompendium.models import TaskSpec
from taskforge.content_hash import digest

from experiments.post_training.capability_driven_envs.biotasks_library.finalize_combined_v1 import _write_new
from experiments.post_training.capability_driven_envs.biotasks_library.finalize_openrouter_first_wave_v3 import (
    _read,
    _score,
)
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_first_wave import (
    _model_path,
    _read_completed,
)
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_first_wave_v3 import load_cohort
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_v3_additional20 import (
    CAMPAIGN_ID,
    ROOT,
    load_panel,
    load_pilot_ids,
)

SNAPSHOT = ROOT + "snapshots/complete-1/"


def freeze(commit: str) -> dict[str, Any]:
    """Reject duplicates, unverifiable routes and incomplete task coverage."""
    cohort = load_cohort()
    models, panel_sha = load_panel()
    tasks = {row["task_id"]: TaskSpec.model_validate(row["task"]) for row in cohort.rows}
    pilot_ids = load_pilot_ids(set(tasks))
    results: list[dict[str, Any]] = []
    sources = []
    for model in models:
        prefix = ROOT + f"attempts/{_model_path(model)}/"
        expected = {
            "pilot.jsonl": [tasks[item] for item in pilot_ids],
            "results.jsonl": [tasks[item] for item in sorted(tasks) if item not in pilot_ids],
        }
        raw_by_id: dict[str, tuple[dict[str, Any], str]] = {}
        for filename, subset in expected.items():
            uri = prefix + filename
            if _read_completed(uri, subset, model, panel_sha) is None:
                raise ValueError(f"Missing completed batch: {uri}")
            raw_rows, source = _read(uri)
            sources.append(source)
            for row in raw_rows:
                if row["task_id"] in raw_by_id:
                    raise ValueError("Pilot/full duplicate task trial")
                raw_by_id[row["task_id"]] = (row, uri)
        if set(raw_by_id) != set(tasks):
            raise ValueError(f"Incomplete model-task matrix: {model.model}")
        backfill = {}
        for filename in ("route-metadata-backfill-pilot.jsonl", "route-metadata-backfill.jsonl"):
            uri = prefix + filename
            fs, path = fsspec.core.url_to_fs(uri)
            if fs.exists(path):
                entries, source = _read(uri)
                sources.append(source)
                for entry in entries:
                    if entry["response_id"] in backfill:
                        raise ValueError("Duplicate route evidence")
                    backfill[entry["response_id"]] = entry
        for task_id in sorted(tasks):
            task = tasks[task_id]
            raw, source_uri = raw_by_id[task_id]
            if raw["frozen_task_digest"] != digest(task) or raw.get("campaign_id") != CAMPAIGN_ID:
                raise ValueError("New response identity mismatch")
            evidence = raw.get("provider_evidence", [])
            if any(
                not e.get("route_matches")
                and backfill.get(e.get("response_id"), {}).get("provider_name") != model.provider_name
                for e in evidence
            ):
                raise ValueError("Unverified/mismatched model provider route")
            score, grade_status = _score(task, raw)
            if score is not None and not evidence:
                raise ValueError("Scored response lacks verified generation route")
            results.append(
                {
                    "task_id": task_id,
                    "task_digest": digest(task),
                    "model": model.model,
                    "provider_tag": model.provider_tag,
                    "provider_name": model.provider_name,
                    "score": score,
                    "grade_status": grade_status,
                    "source_uri": source_uri,
                    "stop_reason": raw.get("trial", {}).get("stop_reason"),
                    "response_ids": [e.get("response_id") for e in evidence],
                }
            )
    if len(results) != 20 * 230 or len({(r["model"], r["task_id"]) for r in results}) != len(results):
        raise ValueError("Incomplete additional20 coverage")
    results.sort(key=lambda row: (row["task_id"], row["model"]))
    data = b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in results)
    outcomes = Counter(
        "correct" if row["score"] == 1 else "incorrect" if row["score"] == 0 else row["grade_status"] for row in results
    )
    with fsspec.open(ROOT + "budget.json", "rb") as f:
        budget_bytes = f.read()
    if json.loads(budget_bytes)["panel_sha256"] != panel_sha:
        raise ValueError("Campaign budget/panel mismatch")
    manifest = {
        "schema_version": "1",
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "code_commit": commit,
        "status": "internal_provisional_not_irt_validated",
        "task_snapshot_sha256": "b893bd40fb1c6676619b32cd11a180969605b2b69a7a3b5b6df1974ae4bbb8d6",
        "panel_manifest_sha256": panel_sha,
        "budget_sha256": hashlib.sha256(budget_bytes).hexdigest(),
        "summary": {"items": 230, "models": 20, "pairs": len(results), "outcomes": dict(outcomes)},
        "inputs": sources,
        "artifacts": {"item-responses.jsonl": {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}},
    }
    _write_new(SNAPSHOT + "item-responses.jsonl", data)
    _write_new(SNAPSHOT + "manifest.json", json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.code_commit), indent=2), flush=True)


if __name__ == "__main__":
    main()
