# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze the selected disjoint 20 respondents after both staged campaigns finish."""

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fsspec
from taskcompendium.models import TaskSpec
from taskforge.content_hash import digest

from experiments.post_training.capability_driven_envs.biotasks_library.effective20_batches import full_batches
from experiments.post_training.capability_driven_envs.biotasks_library.finalize_combined_v1 import _write_new
from experiments.post_training.capability_driven_envs.biotasks_library.finalize_openrouter_first_wave_v3 import (
    _read,
    _score,
)
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_first_wave import (
    PinnedModel,
    _model_path,
    _read_completed,
)
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_first_wave_v3 import load_cohort
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_v3_additional20 import (
    CAMPAIGN_ID as INITIAL_CAMPAIGN,
)
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_v3_additional20 import (
    ROOT as INITIAL_ROOT,
)
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_v3_additional20 import (
    load_panel as load_initial,
)
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_v3_additional20 import (
    load_pilot_ids,
)
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_v3_topup16 import (
    CAMPAIGN_ID as TOPUP_CAMPAIGN,
)
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_v3_topup16 import (
    ROOT as TOPUP_ROOT,
)
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_v3_topup16 import (
    load_panel as load_topup,
)

SNAPSHOT = INITIAL_ROOT.replace("openrouter-additional20-v1/", "openrouter-effective20-v2/snapshots/complete-1/")
SELECTION_ID = "biotasks-v3-openrouter-effective20-v2"


def selected_models(path: Path) -> tuple[list[tuple[PinnedModel, str, str, str]], str]:
    """Require 7 + 13 distinct respondents and both original panel digests."""
    content = path.read_bytes()
    manifest = json.loads(content)
    first, first_sha = load_initial()
    second, second_sha = load_topup()
    if (
        manifest["selection_id"] != SELECTION_ID
        or manifest["initial_panel_sha256"] != first_sha
        or manifest["topup_panel_sha256"] != second_sha
        or len(manifest["initial_models"]) != 7
        or len(manifest["topup_models"]) != 13
        or len(set(manifest["initial_models"])) != 7
        or len(set(manifest["topup_models"])) != 13
        or not set(manifest["initial_models"]) <= {m.model for m in first}
        or not set(manifest["topup_models"]) <= {m.model for m in second}
        or set(manifest["batch_sizes"]) != set(manifest["initial_models"]) | set(manifest["topup_models"])
        or any(not isinstance(size, int) or not 1 <= size <= 218 for size in manifest["batch_sizes"].values())
    ):
        raise ValueError("Effective20 selection changed or mismatches source panels")
    chosen: list[tuple[PinnedModel, str, str, str]] = [
        (m, INITIAL_ROOT, INITIAL_CAMPAIGN, first_sha) for m in first if m.model in manifest["initial_models"]
    ] + [(m, TOPUP_ROOT, TOPUP_CAMPAIGN, second_sha) for m in second if m.model in manifest["topup_models"]]
    if len(chosen) != 20 or len({m.model for m, _, _, _ in chosen}) != 20:
        raise ValueError("Effective20 models overlap")
    return chosen, hashlib.sha256(content).hexdigest()


def freeze(commit: str, selection_path: Path) -> dict[str, Any]:
    """Verify attempts, hashes, origins and routes before publishing the matrix."""
    cohort = load_cohort()
    tasks = {row["task_id"]: TaskSpec.model_validate(row["task"]) for row in cohort.rows}
    pilot_ids = load_pilot_ids(set(tasks))
    selected, selection_sha = selected_models(selection_path)
    sizes = json.loads(selection_path.read_text())["batch_sizes"]
    results: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for model, root, campaign_id, source_panel_sha in selected:
        prefix = root + f"attempts/{_model_path(model)}/"
        rest = [tasks[item] for item in sorted(tasks) if item not in pilot_ids]
        expected = [(prefix + "pilot.jsonl", [tasks[item] for item in pilot_ids])]
        expected.extend(full_batches(rest, model.model, prefix, sizes))
        raw_by_id: dict[str, tuple[dict[str, Any], str]] = {}
        for uri, subset in expected:
            if _read_completed(uri, subset, model, source_panel_sha) is None:
                raise ValueError(f"Missing completed batch: {uri}")
            rows, source = _read(uri)
            inputs.append(source)
            for raw in rows:
                if raw["task_id"] in raw_by_id:
                    raise ValueError("Duplicate model/task generation")
                raw_by_id[raw["task_id"]] = (raw, uri)
        if set(raw_by_id) != set(tasks):
            raise ValueError(f"Incomplete raw model coverage: {model.model}")
        backfill = {}
        for batch_uri, _ in expected:
            filename = Path(batch_uri).name
            overlay_name = (
                "route-metadata-backfill.jsonl" if filename == "results.jsonl" else f"route-metadata-backfill-{filename}"
            )
            uri = prefix + overlay_name
            fs, path = fsspec.core.url_to_fs(uri)
            if fs.exists(path):
                entries, source = _read(uri)
                inputs.append(source)
                for entry in entries:
                    if entry["response_id"] in backfill:
                        raise ValueError("Duplicate route evidence")
                    backfill[entry["response_id"]] = entry
        for task_id in sorted(tasks):
            task = tasks[task_id]
            raw, source_uri = raw_by_id[task_id]
            if raw["frozen_task_digest"] != digest(task) or raw.get("campaign_id") != campaign_id:
                raise ValueError("Raw task or campaign identity changed")
            if raw["model"] != model.model or raw["requested_provider_tag"] != model.provider_tag:
                raise ValueError("Model/provider request changed")
            evidence = raw.get("provider_evidence", [])
            if any(
                not entry.get("route_matches")
                and backfill.get(entry.get("response_id"), {}).get("provider_name") != model.provider_name
                for entry in evidence
            ):
                raise ValueError("Missing or mismatched provider verification")
            score, grade_status = _score(task, raw)
            if score is not None and not evidence:
                raise ValueError("Scored trial lacks verified generation route")
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
                    "source_panel_sha256": source_panel_sha,
                    "response_ids": [entry.get("response_id") for entry in evidence],
                }
            )
    if len(results) != 20 * 230 or len({(r["model"], r["task_id"]) for r in results}) != len(results):
        raise ValueError("Incomplete effective20 population")
    results.sort(key=lambda row: (row["task_id"], row["model"]))
    data = b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in results)
    outcomes = Counter(
        "correct" if row["score"] == 1 else "incorrect" if row["score"] == 0 else row["grade_status"] for row in results
    )
    with fsspec.open(INITIAL_ROOT + "budget.json", "rb") as f:
        budget_bytes = f.read()
    if json.loads(budget_bytes)["panel_sha256"] != load_initial()[1]:
        raise ValueError("Shared budget baseline changed")
    manifest = {
        "schema_version": "1",
        "selection_id": SELECTION_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "code_commit": commit,
        "status": "internal_provisional_not_irt_validated",
        "task_snapshot_sha256": "b893bd40fb1c6676619b32cd11a180969605b2b69a7a3b5b6df1974ae4bbb8d6",
        "selection_sha256": selection_sha,
        "shared_budget_sha256": hashlib.sha256(budget_bytes).hexdigest(),
        "summary": {"items": 230, "models": 20, "pairs": len(results), "outcomes": dict(outcomes)},
        "inputs": inputs,
        "artifacts": {"item-responses.jsonl": {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}},
    }
    _write_new(SNAPSHOT + "item-responses.jsonl", data)
    _write_new(SNAPSHOT + "manifest.json", json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.code_commit, args.selection), indent=2), flush=True)


if __name__ == "__main__":
    main()
