# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Restore missing Harbor trajectories from the kept final-attempt trial tree."""

import json
from dataclasses import dataclass

from finestore.eval import ARCHIVE_SAMPLES_TABLE, EvaluationStore, SampleKind, sample_from_archive_row
from finestore.reader import ReadView
from marin.evaluation.harbor.trajectory import archive_trajectory
from marin.evaluation.rollouts import normalize_rollouts
from rigging.filesystem.storage_path import StoragePath

_ATTEMPTS_SUBDIR = "attempts"
_TRAJECTORY_SUFFIX = "agent/trajectory.json"


@dataclass(frozen=True)
class BackfillResult:
    """The number of restored and still-missing Harbor trajectories."""

    restored: int
    missing: int


def _trajectory_path(trial_dir: StoragePath, trial_uri: str | None) -> StoragePath | None:
    """Use the recorded final attempt, even when the job tree has been relocated."""
    if trial_uri:
        segments = trial_uri.rstrip("/").split("/")
        matches = [index for index, segment in enumerate(segments) if segment == trial_dir.name]
        if matches:
            tail = segments[matches[-1] + 1 :]
            if len(tail) == 2 and tail[0] == _ATTEMPTS_SUBDIR and tail[1].isdigit():
                candidate = trial_dir / _ATTEMPTS_SUBDIR / tail[1] / _TRAJECTORY_SUFFIX
                if candidate.exists():
                    return candidate
    legacy = trial_dir / _TRAJECTORY_SUFFIX
    return legacy if legacy.exists() else None


def backfill_harbor_trajectories(results_path: str) -> BackfillResult:
    """Patch Harbor samples without a trajectory using their kept native result and final attempt."""
    reader = ReadView(results_path)
    missing_rows = {
        (row["task"], row["doc_id"], row["trial_id"]): row
        for row in reader.iter_rows(ARCHIVE_SAMPLES_TABLE)
        if row["kind"] == SampleKind.AGENTIC and not row.get("trajectory_uri")
    }
    if not missing_rows:
        return BackfillResult(restored=0, missing=0)

    restored = 0
    jobs = StoragePath.parse(results_path) / "harbor_jobs"
    with EvaluationStore.open(results_path, writer_id="harbor-trajectory-backfill") as store:
        for result_file in (jobs / "*/*/result.json").glob():
            trial_dir = result_file.parent
            result = json.loads(result_file.read_text())
            doc_id = result.get("task_name", trial_dir.name)
            matches = [key for key in missing_rows if key[1:] == (doc_id, trial_dir.name)]
            if not matches:
                continue
            trajectory = _trajectory_path(trial_dir, result.get("trial_uri"))
            if trajectory is None:
                continue
            raw = trajectory.read_bytes()
            for key in matches:
                row = missing_rows.pop(key)
                sample = sample_from_archive_row(row)
                stored = archive_trajectory(store, raw, task=sample.task, doc_id=doc_id, trial_id=trial_dir.name)
                store.add_sample(sample.model_copy(update={"trajectory_uri": stored.uri}), trial_id=trial_dir.name)
                restored += 1
        if restored:
            store.seal()
    if restored:
        normalize_rollouts(results_path, writer_id="harbor-trajectory-backfill-rollouts")
    return BackfillResult(restored=restored, missing=len(missing_rows))
