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
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.importers.harbor.package import JobSettings
from taskcompendium.models import GraderMachine, MissingArtifactPolicy, ScriptGrader, Source, TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.grading import _reward_value

import experiments.post_training.task_curation.harbor_packages as packages_module
from experiments.evaluation.benchmarks import swebench_science
from experiments.evaluation.benchmarks.swebench_science import lower_science_package
from experiments.post_training.task_curation.harbor_packages import package_archive, stage_repository_packages

# Task 002's MIT-licensed manifest and collection hook are copied from DATASET at REVISION.
FIXTURE = Path(__file__).parent / "fixtures/harbor/swebench_science"
SOURCE = Source(
    dataset=swebench_science.DATASET,
    revision=swebench_science.REVISION,
    row="tasks/task_002",
    importer_revision="test",
)


ROW = RawRow("science-002", SOURCE, {})


@pytest.fixture
def science_package(tmp_path):
    package = tmp_path / "task_002"
    tests = package / "tests"
    tests.mkdir(parents=True)
    for name in ("task.toml", "pre_artifacts.sh"):
        shutil.copyfile(FIXTURE / name, package / name)
    (package / "instruction.md").write_text("Repair the example implementation.")
    (tests / "grader.py").write_text("from private_tests import grade\ngrade()\n")
    (tests / "test.sh").write_text("exit 99\n")  # This stub must never replace the image's verifier.
    return package


def test_science_corpus_sets_solver_resources_and_keeps_the_verifier_image(science_package):
    row = RawRow(ROW.id, SOURCE, {"name": science_package.name, "archive": package_archive(science_package)})
    [source] = swebench_science.benchmarks()[0].resolve("pi-xhigh")
    task = source.config.recipe.convert(row, ConversionContext({}, None))
    assert isinstance(task, TaskSpec)
    assert task.environment_requirements.docker_build is not None
    assert isinstance(task.grader, ScriptGrader) and isinstance(task.grader.placement, GraderMachine)
    assert task.grader.placement.environment.docker_build is None
    episode = source.config.item_builder(task).episode
    assert episode is not None
    assert (episode.cpus, episode.memory_mb, episode.storage_mb) == (2, 8192, 10240)


def test_native_verifier_entrypoint_and_hidden_tests_remain_image_owned(science_package):
    lowered = lower_science_package(science_package, ROW, JobSettings())
    task = lowered.task
    config = tomllib.loads((science_package / "task.toml").read_text())
    assert task.context.events[0].content == (science_package / "instruction.md").read_text()
    assert task.environment_requirements.docker_image == config["environment"]["docker_image"].replace(":v0.1.0@", "@")
    assert "network" not in task.environment_requirements.capabilities
    assert not task.resources.worker
    assert not task.resources.oracle
    assert [resource.path for resource in task.resources.verifier] == ["grader.py"]
    assert (
        base64.b64decode(task.resources.verifier[0].source.content_base64)
        == (science_package / "tests/grader.py").read_bytes()
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


def test_unexpected_patch_artifact_is_rejected(science_package):
    package = science_package
    config = package / "task.toml"
    config.write_text(config.read_text().replace("/logs/artifacts/model.patch", "/logs/artifacts/other.patch"))
    with pytest.raises(ValueError, match="sole Science artifact"):
        lower_science_package(package, ROW, JobSettings())


@pytest.mark.parametrize("reward", [0, 1])
def test_nested_native_report_returns_only_the_numeric_reward(reward, science_package):
    lowered = lower_science_package(science_package, ROW, JobSettings())
    report = json.dumps({"reward": reward, "task_id": "002", "private": {"passed": 3, "collected": 4}}).encode()
    file = lowered.task.grader.reward.files[0]
    assert _reward_value(report, file.format, file.key) == (float(reward), None)


def test_native_collect_includes_committed_and_untracked_candidate_changes(tmp_path, science_package):
    package = science_package
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
    lowered = lower_science_package(package, ROW, JobSettings())
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


def test_nested_huggingface_selection_stages_only_requested_tasks(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    for name in ("task_001", "task_002", "task_003"):
        directory = repository / "tasks" / name
        directory.mkdir(parents=True)
        (directory / "task.toml").write_text(name)
    monkeypatch.setattr(packages_module, "snapshot_download", lambda *_args, **_kwargs: str(repository))
    output = tmp_path / "staged"
    stage_repository_packages(
        "org/fixture",
        swebench_science.REVISION,
        StoragePath(str(output)),
        root="tasks",
        selected=frozenset({"task_001", "task_002"}),
    )
    assert sorted(path.name for path in (output / "tasks").iterdir()) == ["task_001.tar.gz", "task_002.tar.gz"]
    with pytest.raises(ValueError, match="task_004"):
        stage_repository_packages(
            "org/fixture",
            swebench_science.REVISION,
            StoragePath(str(tmp_path / "missing")),
            root="tasks",
            selected=frozenset({"task_004"}),
        )
