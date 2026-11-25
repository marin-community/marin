# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pyarrow.parquet as pq
import pytest
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.models import TASK_ROLLOUT_COLUMN, TaskRolloutSpec
from shellbox.machine import NetworkPolicy
from taskcompendium.models import AgentMachine, ConversationInput, ScriptGrader, TextMessage

from experiments.post_training.rl_checkpoint_eval import (
    CHECKPOINT_TASKS_FILENAME,
    checkpoint_cohorts,
    write_checkpoint_tasks,
)
from experiments.post_training.task_curation.harbor_packages import write_package_archives

IMAGE = "fixture/checkpoint:1"
DIGEST = "sha256:" + "1" * 64


def test_checkpoint_task_artifact_preserves_cohort_and_caps_package_verifier_deadlines(tmp_path):
    cohort = next(cohort for cohort in checkpoint_cohorts() if cohort.name == "openthoughts_tblite")
    downloaded = tmp_path / "downloaded"
    output = tmp_path / "output"
    packages = []
    for index, package in enumerate(cohort.packages):
        directory = tmp_path / package
        packages.append(directory)
        (directory / "tests").mkdir(parents=True)
        timeout = 180 if index % 2 else 4000
        (directory / "task.toml").write_text(
            f"[verifier]\ntimeout_sec = {timeout}\n"
            f'[environment]\ndocker_image = "{IMAGE}"\ncpus = 2\nallow_internet = false\n'
        )
        (directory / "instruction.md").write_text(f"Solve {package}.")
        (directory / "tests" / "test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n")
    write_package_archives(packages, StoragePath(str(downloaded)))
    write_checkpoint_tasks(cohort=cohort, image_digests={IMAGE: DIGEST}, downloaded=str(downloaded), output=str(output))

    rows = pq.read_table(output / CHECKPOINT_TASKS_FILENAME).to_pylist()
    specs = [TaskRolloutSpec.model_validate_json(row[TASK_ROLLOUT_COLUMN]) for row in rows]
    assert [spec.task.id for spec in specs] == [f"openthoughts_tblite/{package}" for package in cohort.packages]
    assert [spec.limits.verifier_timeout for spec in specs] == [3600, 180] * 15
    for spec, package in zip(specs, cohort.packages, strict=True):
        assert spec.task.context == ConversationInput(events=(TextMessage(role="user", content=f"Solve {package}."),))
        assert isinstance(spec.task.grader, ScriptGrader)
        assert isinstance(spec.task.grader.placement, AgentMachine)
        assert not spec.task.resources.worker
        assert len(spec.task.resources.verifier) == 1
        assert spec.task.environment_requirements.docker_image == f"{IMAGE}@{DIGEST}"
        assert spec.task_machine is not None and spec.task_machine.network == NetworkPolicy.DENY

    (downloaded / "tasks" / f"{cohort.packages[-1]}.tar.gz").unlink()
    incomplete = tmp_path / "incomplete"
    with pytest.raises(FileNotFoundError):
        write_checkpoint_tasks(
            cohort=cohort, image_digests={IMAGE: DIGEST}, downloaded=str(downloaded), output=str(incomplete)
        )
    assert not (incomplete / CHECKPOINT_TASKS_FILENAME).exists()
