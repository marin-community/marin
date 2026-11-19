# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve scorecard.yaml into scorecard.json by reading each cited run's eval record.

Run from the repository root with CoreWeave credentials:

    uv run python infra/marina/apps/model_scorecard/build_data.py
"""

import dataclasses
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml
from evalrecords.eval_measurements import measurement_from_record
from evalrecords.records import CW_RECORDS_PREFIX, EvalRunRecord, RunStatus, read_record, record_path
from marin.evaluation.harbor.runner import DEFAULT_MIN_COMPLETION_RATE
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.step_status import STATUS_SUCCESS

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "scorecard.yaml"
OUTPUT = ROOT / "scorecard.json"

EVALDASH_RUN_URL = "https://marina.oa.dev/evaldash/runs/{run_id}"
FILER_URL = "https://filer.marina.oa.dev/#{encoded_uri}"
HF_URL = "https://huggingface.co/{repo}/tree/{revision}"
ABLATIONS_DIR = "ablations"
SCORE_DECIMALS = 2

FetchRecord = Callable[[str], EvalRunRecord]


class Label(StrEnum):
    OFFICIAL = "official"
    UNOFFICIAL = "unofficial"


class LinkKind(StrEnum):
    EVALDASH = "evaldash"
    STORAGE = "storage"


@dataclass(frozen=True)
class Link:
    kind: LinkKind
    url: str


@dataclass(frozen=True)
class Trials:
    """Agentic trial outcomes behind a score, which averages reward over the scored trials.

    Trials that hit the agent time limit are scored with reward 0; infrastructure failures are unscored.
    """

    attempted: int
    scored: int
    timeouts: int


@dataclass(frozen=True)
class ScoredCell:
    score: float
    label: Label
    link: Link
    run_id: str | None = None
    note: str | None = None
    suspect: str | None = None
    """Why the score may not reflect the model, e.g. a known grading or protocol problem."""
    trials: Trials | None = None


@dataclass(frozen=True)
class MissingCell:
    missing: str
    """Why the cell has no score; empty when the manifest gives no reason."""


Cell = ScoredCell | MissingCell


def run_link(run_id: str) -> Link:
    return Link(LinkKind.EVALDASH, EVALDASH_RUN_URL.format(run_id=run_id))


def source_link(uri: str) -> Link:
    """Link a source URI to its EvalDash run when it lives in a run directory, else to Filer."""
    prefix = f"{CW_RECORDS_PREFIX}/"
    if uri.startswith(prefix):
        run_id = uri.removeprefix(prefix).split("/", 1)[0]
        if run_id and run_id != ABLATIONS_DIR:
            return run_link(run_id)
    return Link(LinkKind.STORAGE, FILER_URL.format(encoded_uri=quote(uri, safe="")))


def headline_score(record: EvalRunRecord) -> float:
    """The published benchmark measurement on a 0-100 scale."""
    if record.status != RunStatus.SUCCEEDED:
        raise ValueError(f"Run {record.run_id} has status {record.status.value!r}")
    if record.step_status not in (None, STATUS_SUCCESS):
        raise ValueError(f"Run {record.run_id} has step status {record.step_status!r}")
    measurement = measurement_from_record(record)
    if measurement is None:
        raise ValueError(f"Run {record.run_id} has no published measurement")
    return round(100 * measurement.value, SCORE_DECIMALS)


def model_identity(model: Mapping[str, Any]) -> tuple[str, str | None]:
    """The (location, revision) a model row's runs must evaluate: a Hugging Face revision or a checkpoint URI."""
    if ("hf_repo" in model) == ("location" in model):
        raise ValueError(f"{model['name']} needs exactly one of hf_repo or location")
    if "location" in model:
        if "revision" in model:
            raise ValueError(f"{model['name']} location rows take no revision")
        return model["location"], None
    return model["hf_repo"], model["revision"]


def model_link(model: Mapping[str, Any]) -> str:
    if "location" in model:
        return FILER_URL.format(encoded_uri=quote(model["location"], safe=""))
    return HF_URL.format(repo=model["hf_repo"], revision=model["revision"])


def parse_trials(spec: Mapping[str, int] | None) -> Trials | None:
    if spec is None:
        return None
    trials = Trials(**spec)
    if not 0 <= trials.timeouts <= trials.scored <= trials.attempted:
        raise ValueError(f"Trial counts need 0 <= timeouts <= scored <= attempted: {dict(spec)}")
    return trials


def resolve_cell(
    eval_name: str, spec: Mapping[str, Any] | None, model: Mapping[str, Any], fetch_record: FetchRecord
) -> Cell:
    if spec is None:
        return MissingCell("")
    if "missing" in spec:
        return MissingCell(spec["missing"])
    label = Label(spec["label"])
    note = spec.get("note")
    suspect = spec.get("suspect")
    trials = parse_trials(spec.get("trials"))
    if "run_id" in spec:
        record = fetch_record(spec["run_id"])
        if record.evaluation.name != eval_name:
            raise ValueError(f"Run {record.run_id} evaluates {record.evaluation.name!r}, not {eval_name!r}")
        revision = record.model.config.revision if record.model.config is not None else None
        location, expected_revision = model_identity(model)
        # A checkpoint directory is recorded with or without its trailing slash depending on the serving config.
        if (record.model.location.rstrip("/"), revision) != (location.rstrip("/"), expected_revision):
            raise ValueError(
                f"Run {record.run_id} evaluates {record.model.location!r}@{revision!r}, "
                f"not {location!r}@{expected_revision!r}"
            )
        return ScoredCell(headline_score(record), label, run_link(record.run_id), record.run_id, note, suspect, trials)
    if "source" in spec:
        score = float(spec["score"])
        if not 0 <= score <= 100:
            raise ValueError(f"{eval_name} source score {score} is outside 0-100")
        return ScoredCell(
            round(score, SCORE_DECIMALS), label, source_link(spec["source"]), note=note, suspect=suspect, trials=trials
        )
    raise ValueError(f"{eval_name} cell needs run_id, source, or missing: {dict(spec)}")


def cell_json(cell: Cell) -> dict[str, Any]:
    return dataclasses.asdict(cell, dict_factory=lambda items: {key: value for key, value in items if value is not None})


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
        location, _ = model_identity(model)
        models.append(
            {
                "name": model["name"],
                "location": location,
                "url": model_link(model),
                "cells": {name: cell_json(resolve_cell(name, cells.get(name), model, fetch_record)) for name in evals},
            }
        )
    return {"evals": evals, "models": models, "min_completion": DEFAULT_MIN_COMPLETION_RATE}


def fetch_eval_record(run_id: str) -> EvalRunRecord:
    return read_record(record_path(CW_RECORDS_PREFIX, run_id))


def main() -> None:
    configure_coreweave_s3()
    manifest = yaml.safe_load(MANIFEST.read_text())
    scorecard = build_scorecard(manifest, fetch_eval_record)
    OUTPUT.write_text(json.dumps(scorecard, indent=2) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
