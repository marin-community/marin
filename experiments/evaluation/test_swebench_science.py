# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.importers.harbor.dockerfile import dockerfile_images
from taskcompendium.models import GraderMachine, MissingArtifactPolicy, ScriptGrader, Source
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow

from experiments.evaluation.benchmarks import swebench_science
from experiments.post_training.task_curation.harbor_packages import package_archive

# Task 002's MIT-licensed manifest and collection hook are copied from DATASET at REVISION.
FIXTURE = Path(__file__).parent / "fixtures/harbor/swebench_science"
SOURCE = Source(
    dataset=swebench_science.DATASET,
    revision=swebench_science.REVISION,
    row="tasks/task_002",
    importer_revision="test",
)


ROW = RawRow("swebench-science-002", SOURCE, {})


@pytest.fixture
def swebench_science_package(tmp_path):
    package = tmp_path / "task_002"
    tests = package / "tests"
    tests.mkdir(parents=True)
    for name in ("task.toml", "pre_artifacts.sh"):
        shutil.copyfile(FIXTURE / name, package / name)
    (package / "instruction.md").write_text("Repair the example implementation.")
    (tests / "grader.py").write_text("from private_tests import grade\ngrade()\n")
    (tests / "test.sh").write_text("exit 99\n")  # This stub must never replace the image's verifier.
    return package


def test_corpus_omits_held_pycistopic_package(tmp_path):
    [source] = swebench_science.benchmarks()[0].resolve("pi-xhigh")
    rows = []
    for name in ("task_018", "task_019", "task_020", "task_119"):
        archive = tmp_path / f"{name}.tar.gz"
        archive.write_bytes(b"package")
        rows.extend(source.config.recipe.source.read(StoragePath(str(archive)), ConversionContext({}, None)))
    assert [row["name"] for row in rows] == ["task_018", "task_020", "task_119"]


def test_corpus_sets_solver_disk_and_keeps_the_verifier_image(swebench_science_package):
    row = RawRow(
        ROW.id, SOURCE, {"name": swebench_science_package.name, "archive": package_archive(swebench_science_package)}
    )
    [source] = swebench_science.benchmarks()[0].resolve("pi-xhigh")
    task = source.config.recipe.convert(row, ConversionContext({}, None))
    build = task.environment_requirements.docker_build
    assert build is not None
    dockerfile = next(file for file in build.files if file.path == "Dockerfile")
    manifest = tomllib.loads((swebench_science_package / "task.toml").read_text())
    assert dockerfile_images(base64.b64decode(dockerfile.source.content_base64).decode()) == (
        manifest["environment"]["docker_image"].replace(":v0.1.0@", "@"),
    )
    assert isinstance(task.grader, ScriptGrader) and isinstance(task.grader.placement, GraderMachine)
    assert task.grader.placement.environment.docker_build is None
    episode = source.config.item_builder(task).episode
    assert episode is not None
    assert episode.storage_mb == 10240


def test_native_verifier_entrypoint_and_hidden_tests_remain_image_owned(swebench_science_package):
    task = swebench_science.lower_package(swebench_science_package, ROW)
    config = tomllib.loads((swebench_science_package / "task.toml").read_text())
    assert task.context.events[0].content == (swebench_science_package / "instruction.md").read_text()
    assert "network" not in task.environment_requirements.capabilities
    assert not task.resources.worker
    assert not task.resources.oracle
    assert [resource.path for resource in task.resources.verifier] == ["grader.py"]
    assert (
        base64.b64decode(task.resources.verifier[0].source.content_base64)
        == (swebench_science_package / "tests/grader.py").read_bytes()
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


def test_unexpected_patch_artifact_is_rejected(swebench_science_package):
    package = swebench_science_package
    config = package / "task.toml"
    config.write_text(config.read_text().replace("/logs/artifacts/model.patch", "/logs/artifacts/other.patch"))
    with pytest.raises(ValueError, match=r"/logs/artifacts/model\.patch"):
        swebench_science.lower_package(package, ROW)


def test_native_collect_includes_committed_and_untracked_candidate_changes(tmp_path, swebench_science_package):
    package = swebench_science_package
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
    task = swebench_science.lower_package(package, ROW)
    hook = task.grader.placement.collect[0]
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
