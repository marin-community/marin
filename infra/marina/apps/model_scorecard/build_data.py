# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve scorecard.yaml into scorecard.json by reading each cited run's eval record.

Run from the repository root with CoreWeave credentials:

    uv run python infra/marina/apps/model_scorecard/build_data.py
"""

import json
from collections.abc import Callable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml
from rigging.filesystem.buckets import filesystem_for

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "scorecard.yaml"
OUTPUT = ROOT / "scorecard.json"

RECORDS_ROOT = "s3://marin-us-east-02a/marin/evals"
EVALDASH_RUN_URL = "https://marina.oa.dev/evaldash/runs/{run_id}"
FILER_URL = "https://filer.marina.oa.dev/#{encoded_uri}"
HF_URL = "https://huggingface.co/{repo}/tree/{revision}"
SCORE_DECIMALS = 2

FetchRecord = Callable[[str], Mapping[str, Any]]


class Label(StrEnum):
    OFFICIAL = "official"
    UNOFFICIAL = "unofficial"


class LinkKind(StrEnum):
    EVALDASH = "evaldash"
    STORAGE = "storage"


def run_link(run_id: str) -> dict[str, str]:
    return {"kind": LinkKind.EVALDASH, "url": EVALDASH_RUN_URL.format(run_id=run_id)}


def source_link(uri: str) -> dict[str, str]:
    """Link a source URI to its EvalDash run when it lives in a run directory, else to Filer."""
    prefix = f"{RECORDS_ROOT}/"
    if uri.startswith(prefix):
        run_id = uri.removeprefix(prefix).split("/", 1)[0]
        if run_id and run_id != "ablations":
            return run_link(run_id)
    return {"kind": LinkKind.STORAGE, "url": FILER_URL.format(encoded_uri=quote(uri, safe=""))}


def headline_score(record: Mapping[str, Any]) -> float:
    """The record's primary metric on a 0-100 scale."""
    run_id = record["run_id"]
    if record["status"] != "succeeded":
        raise ValueError(f"Run {run_id} has status {record['status']!r}")
    tasks = record["eval"]["tasks"]
    metrics = record["canonical_metrics"]
    if len(tasks) != 1 or len(metrics) != 1:
        raise ValueError(f"Run {run_id} must report exactly one task, found {len(tasks)} tasks and {len(metrics)}")
    primary = tasks[0]["benchmark"]["primary_metric"]
    (task_metrics,) = metrics.values()
    if primary not in task_metrics:
        raise ValueError(f"Run {run_id} has no canonical value for primary metric {primary!r}")
    return round(100 * float(task_metrics[primary]), SCORE_DECIMALS)


def resolve_cell(eval_name: str, spec: Mapping[str, Any] | None, fetch_record: FetchRecord) -> dict[str, Any]:
    if spec is None:
        return {"missing": ""}
    note = spec.get("note")
    if "missing" in spec:
        return {"missing": spec["missing"]}
    label = Label(spec["label"])
    if "run_id" in spec:
        run_id = spec["run_id"]
        record = fetch_record(run_id)
        if record["eval"]["name"] != eval_name:
            raise ValueError(f"Run {run_id} evaluates {record['eval']['name']!r}, not {eval_name!r}")
        cell = {"score": headline_score(record), "label": label, "link": run_link(run_id), "run_id": run_id}
    elif "source" in spec:
        score = float(spec["score"])
        if not 0 <= score <= 100:
            raise ValueError(f"{eval_name} source score {score} is outside 0-100")
        cell = {"score": round(score, SCORE_DECIMALS), "label": label, "link": source_link(spec["source"])}
    else:
        raise ValueError(f"{eval_name} cell needs run_id, source, or missing: {dict(spec)}")
    if note:
        cell["note"] = note
    return cell


def build_scorecard(manifest: Mapping[str, Any], fetch_record: FetchRecord) -> dict[str, Any]:
    evals = list(manifest["evals"])
    if len(set(evals)) != len(evals):
        raise ValueError("scorecard.yaml lists an eval twice")
    names = [model["name"] for model in manifest["models"]]
    if len(set(names)) != len(names):
        raise ValueError("scorecard.yaml lists a model twice")
    models = []
    for model in manifest["models"]:
        cells = model.get("cells", {})
        unknown = set(cells) - set(evals)
        if unknown:
            raise ValueError(f"{model['name']} has cells for unlisted evals: {sorted(unknown)}")
        models.append(
            {
                "name": model["name"],
                "hf_repo": model["hf_repo"],
                "url": HF_URL.format(repo=model["hf_repo"], revision=model["revision"]),
                "cells": {name: resolve_cell(name, cells.get(name), fetch_record) for name in evals},
            }
        )
    return {"evals": evals, "models": models}


def fetch_eval_record(run_id: str) -> Mapping[str, Any]:
    fs, path = filesystem_for(f"{RECORDS_ROOT}/{run_id}/record.json")
    return json.loads(fs.cat_file(path))


def main() -> None:
    manifest = yaml.safe_load(MANIFEST.read_text())
    scorecard = build_scorecard(manifest, fetch_eval_record)
    OUTPUT.write_text(json.dumps(scorecard, indent=2) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
