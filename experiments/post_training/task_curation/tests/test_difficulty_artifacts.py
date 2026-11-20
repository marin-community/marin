# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Relocated locks reach the original runtime path without changing frozen tasks."""

import hashlib
import json
from pathlib import Path

import pytest
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.local import EnvironmentManifest, dump_manifest, local_runtime

from experiments.post_training.task_curation import difficulty_artifacts, difficulty_preparation
from experiments.post_training.task_curation.difficulty_preparation import prepare_source
from experiments.post_training.task_curation.tests.prebuilt_fixtures import prebuilt_task


@pytest.mark.parametrize("corrupt_archive", [False, True])
def test_frozen_grader_path_restores_verified_regional_artifacts(
    tmp_path, monkeypatch, difficulty_source, corrupt_archive
):
    regional = tmp_path / "regional"
    original = tmp_path / "frozen" / difficulty_artifacts.RELOCATED_LOCK
    remote = regional / difficulty_artifacts.RELOCATED_LOCK
    remote.parent.mkdir(parents=True)
    remote.write_bytes(b"locked-original-requirements\n")
    archive = b"original-archive-payload"
    manifest = EnvironmentManifest(files=("/fixture.py",), files_sha256=hashlib.sha256(archive).hexdigest())
    (remote.parent / "environment.json").write_text(dump_manifest(manifest))
    (remote.parent / "files.tar").write_bytes(b"changed" if corrupt_archive else archive)
    monkeypatch.setattr(difficulty_artifacts, "REGIONAL_ENVIRONMENT_ROOT", str(regional))
    task = TaskSpec.model_validate(json.loads(prebuilt_task("relocated")["lowered_task_spec"])["task"])
    task = task.model_copy(
        update={
            "grader": task.grader.model_copy(
                update={
                    "environment": task.grader.environment.model_copy(
                        update={"packages_lock": str(original), "docker_image": None}
                    )
                }
            )
        }
    )
    frozen = task.model_dump_json()
    source = difficulty_source(task)
    monkeypatch.setattr(difficulty_preparation, "panel_task_runtimes", lambda panel: ([task], [{}]))
    if corrupt_archive:
        with pytest.raises(ValueError, match="archive differs"):
            prepare_source(source, tmp_path / "prepared")
        assert not original.exists()
        return
    prepared = prepare_source(source, tmp_path / "prepared")
    assert prepared.tasks[0].reason is None
    assert original.read_bytes() == remote.read_bytes()
    assert (original.parent / "files.tar").read_bytes() == archive
    assert local_runtime(str(original)).lock_sha256 == hashlib.sha256(remote.read_bytes()).hexdigest()
    assert local_runtime(str(original)).manifest == manifest
    assert task.model_dump_json() == frozen
    emitted = TaskSpec.model_validate_json((Path(prepared.tasks[0].path) / "task-spec.json").read_text())
    assert emitted.grader.environment.packages_lock == str(original)
