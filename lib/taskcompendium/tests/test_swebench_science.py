# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

from taskcompendium.importers.harbor.science import lower_science_package
from taskcompendium.models import GraderMachine, MissingArtifactPolicy, ScriptGrader, Source
from taskcompendium.runtime.grading import _reward_value

FIXTURE = Path(__file__).parent / "fixtures/swebench_science/task_002"
SOURCE = Source(
    dataset="OpenMOSS-Team/SWE-bench-Science",
    revision="997408298e8e4d09978b1120b95c4540d1a71bcb",
    row="tasks/task_002",
    importer_revision="test",
)


def test_native_verifier_entrypoint_and_hidden_tests_remain_image_owned():
    lowered = lower_science_package(FIXTURE, task_id="science-002", source=SOURCE)
    task = lowered.task
    config = tomllib.loads((FIXTURE / "task.toml").read_text())
    assert task.context.events[0].content == (FIXTURE / "instruction.md").read_text()
    assert task.environment_requirements.docker_image == config["environment"]["docker_image"].replace(":v0.1.0@", "@")
    assert "network" not in task.environment_requirements.capabilities
    assert not task.resources.worker
    assert not task.resources.oracle
    assert [resource.path for resource in task.resources.verifier] == ["grader.py"]
    assert (
        base64.b64decode(task.resources.verifier[0].source.content_base64) == (FIXTURE / "tests/grader.py").read_bytes()
    )
    grader = task.grader
    assert isinstance(grader, ScriptGrader)
    assert grader.argv == ("bash", "/tests/test.sh")
    assert isinstance(grader.placement, GraderMachine)
    assert grader.placement.environment.docker_image == config["verifier"]["environment"]["docker_image"].replace(
        ":v0.1.0@", "@"
    )
    assert "network" not in grader.placement.environment.capabilities
    assert len(grader.placement.artifacts) == 1
    assert grader.placement.artifacts[0].missing == MissingArtifactPolicy.ERROR
    assert lowered.episode.agent_timeout == 5400
    assert lowered.episode.verifier_timeout == 1800
    assert (lowered.episode.cpus, lowered.episode.memory_mb, lowered.episode.storage_mb) == (2, 8192, 20480)


def test_unexpected_patch_artifact_is_rejected(tmp_path):
    package = tmp_path / "package"
    shutil.copytree(FIXTURE, package)
    config = package / "task.toml"
    config.write_text(config.read_text().replace("/logs/artifacts/model.patch", "/logs/artifacts/other.patch"))
    with pytest.raises(ValueError, match="sole Science artifact"):
        lower_science_package(package, task_id="science-002", source=SOURCE)


@pytest.mark.parametrize("reward", [0, 1])
def test_nested_native_report_returns_only_the_numeric_reward(reward):
    lowered = lower_science_package(FIXTURE, task_id="science-002", source=SOURCE)
    report = json.dumps({"reward": reward, "task_id": "002", "private": {"passed": 3, "collected": 4}}).encode()
    file = lowered.task.grader.reward.files[0]
    assert _reward_value(report, file.format, file.key) == (float(reward), None)


def test_native_collect_includes_committed_and_untracked_candidate_changes(tmp_path):
    package = tmp_path / "package"
    shutil.copytree(FIXTURE, package)
    repo, artifacts = tmp_path / "repo", tmp_path / "artifacts"
    repo.mkdir()
    env = {**os.environ, "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig"), "ENV_ARTIFACTS_PATH": str(artifacts)}

    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True)

    git("init")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.invalid")
    (repo / "source.py").write_text("baseline\n")
    git("add", ".")
    git("commit", "-m", "baseline")
    baseline = git("rev-parse", "HEAD").stdout.decode().strip()
    config = package / "task.toml"
    config.write_text(config.read_text().replace("/app/task_002", str(repo)))
    lowered = lower_science_package(package, task_id="science-002", source=SOURCE)
    hook = lowered.task.grader.placement.collect[0]
    subprocess.run(hook.argv, env=env, check=True, capture_output=True)
    assert (artifacts / "model.patch").read_bytes() == b""
    (repo / "source.py").write_text("candidate\n")
    git("add", ".")
    git("commit", "-m", "candidate")
    (repo / "new.bin").write_bytes(b"\x00candidate\xff")
    subprocess.run(hook.argv, env=env, check=True, capture_output=True)
    git("reset", "--hard", baseline)
    git("clean", "-fd")
    git("apply", "--binary", str(artifacts / "model.patch"))
    assert (repo / "source.py").read_text() == "candidate\n"
    assert (repo / "new.bin").read_bytes() == b"\x00candidate\xff"
