# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Environment identity, the build-and-record step, and the artifact a declaration resolves."""

import hashlib
import json
import re
from dataclasses import replace

import pytest
from marin.execution.lazy import run
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.images.build import (
    IDENTITY_CHARS,
    PLATFORM,
    EnvironmentBuild,
    MissingEnvironmentArtifact,
    build_environment,
    built_environment,
    environment_artifact,
    identity_digest,
)
from experiments.post_training.task_curation.tests.image_builds import (
    REPOSITORY,
    install_fake_build_tools,
    tracked_lock,
)

COMPILED = "numpy==2.3.5 \\\n    --hash=sha256:" + hashlib.sha256(b"numpy==2.3.5").hexdigest() + "\n"
"""What the ``uv`` stand-in compiles ``numpy==2.3.5`` to."""


@pytest.fixture
def docker_log(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "prefix"))
    return install_fake_build_tools(tmp_path, monkeypatch)


@pytest.fixture
def lock(tmp_path):
    return tracked_lock(tmp_path)


def test_apt_packages_beyond_the_worker_image_push_an_image_and_record_its_digest(docker_log):
    environment = Environment(pypi=("numpy==2.3.5",), apt=("jq",), data=("nltk:punkt_tab",))
    (built,) = run(environment_artifact(environment, REPOSITORY))
    identity = identity_digest(environment)
    tag = f"{REPOSITORY}:task-curation-env-{identity[:IDENTITY_CHARS]}"
    assert built.tag == tag
    assert built.image == f"{REPOSITORY}@sha256:{hashlib.sha256(tag.encode()).hexdigest()}"
    assert StoragePath(built.lock_url).read_text() == COMPILED
    assert built.lock_sha256 == hashlib.sha256(COMPILED.encode()).hexdigest()
    assert built.path.endswith(f"images/env-{identity[:IDENTITY_CHARS]}/2026.10.08")
    build = next(line for line in docker_log.read_text().splitlines() if line.startswith("buildx build"))
    assert {f"--platform={PLATFORM}", "--push", f"--tag={tag}"} <= set(build.split())
    assert built_environment(environment) == built


@pytest.mark.parametrize(
    "apt", [pytest.param((), id="no-apt"), pytest.param(("build-essential",), id="worker-image-apt")]
)
def test_a_worker_environment_records_its_lock_without_docker(lock, docker_log, apt):
    environment = Environment(lock=lock, apt=apt)
    (built,) = run(environment_artifact(environment, REPOSITORY))
    assert built.image is None
    assert StoragePath(built.lock_url).read_bytes() == lock.read_bytes()
    assert docker_log.read_text() == ""


def test_an_unchanged_environment_resolves_without_building(lock, docker_log, monkeypatch):
    environment = Environment(lock=lock, apt=("jq",))
    (first,) = run(environment_artifact(environment, REPOSITORY))
    calls = docker_log.read_text()
    monkeypatch.setenv("PATH", "/nonexistent")
    (second,) = run(environment_artifact(environment, REPOSITORY))
    assert second.image == first.image
    assert docker_log.read_text() == calls


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda environment: (environment.lock.write_text("numpy==2.3.4\n"), environment)[1], id="lock"),
        pytest.param(lambda environment: replace(environment, lock=None, pypi=("numpy==2.3.5",)), id="pypi"),
        pytest.param(lambda environment: replace(environment, apt=("build-essential",)), id="apt"),
        pytest.param(lambda environment: replace(environment, data=("nltk:wordnet",)), id="data"),
    ],
)
def test_declaration_changes_rename_the_environment_artifact(lock, change):
    environment = Environment(lock=lock)
    original = environment_artifact(environment).name
    assert environment_artifact(change(environment)).name != original


def test_the_repository_is_where_a_build_pushes_not_identity(lock):
    environment = Environment(lock=lock, apt=("jq",))
    assert environment_artifact(environment, "other.invalid/images").name == environment_artifact(environment).name


def test_build_refuses_a_lock_git_does_not_track(lock, docker_log, tmp_path):
    untracked = lock.parent / "untracked.lock"
    untracked.write_text(lock.read_text())
    environment = Environment(lock=untracked)
    with pytest.raises(ValueError, match="tracked by git"):
        build_environment(environment, EnvironmentBuild(identity_digest(environment), REPOSITORY, str(tmp_path / "out")))


def test_an_image_build_requires_repository_credentials(lock, docker_log, tmp_path):
    (tmp_path / "docker-config" / "config.json").write_text(json.dumps({"auths": {}}))
    environment = Environment(lock=lock, apt=("jq",))
    with pytest.raises(RuntimeError, match=re.escape("docker login registry.invalid")):
        build_environment(environment, EnvironmentBuild(identity_digest(environment), REPOSITORY, str(tmp_path / "out")))


def test_an_unbuilt_environment_names_the_build_command(lock, tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "prefix"))
    environment = Environment(lock=lock)
    identity = identity_digest(environment)[:IDENTITY_CHARS]
    with pytest.raises(MissingEnvironmentArtifact, match=re.escape(f"images --identity {identity}")):
        built_environment(environment)
