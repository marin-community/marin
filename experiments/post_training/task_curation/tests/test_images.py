# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Environment identity, the build-and-record step, and the artifact a declaration resolves."""

import hashlib
import io
import json
import re
import shlex
import tarfile
from dataclasses import replace

import pytest
from click.testing import CliRunner
from marin.execution.lazy import run
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.images import __main__ as images
from experiments.post_training.task_curation.images import build
from experiments.post_training.task_curation.images.build import (
    IDENTITY_CHARS,
    EnvironmentBuild,
    build_environment,
    built_environment,
    dockerfile,
    environment_artifact,
    identity_digest,
)
from experiments.post_training.task_curation.tests.image_builds import (
    COMPILED_LOCK,
    DATA_FILES,
    DATA_PATH,
    REPOSITORY,
    REQUIREMENTS,
    TOOLCHAIN_FRAGMENT,
    install_fake_archive,
    install_fake_build_tools,
    tracked_fragment,
    tracked_repository,
)


@pytest.fixture
def docker_log(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "prefix"))
    return install_fake_build_tools(tmp_path, monkeypatch)


@pytest.fixture
def repository(tmp_path):
    return tracked_repository(tmp_path)


@pytest.fixture
def fragment(repository):
    return tracked_fragment(repository)


def test_apt_packages_beyond_the_worker_image_push_an_image_and_record_its_digest(docker_log):
    environment = Environment(requirements=REQUIREMENTS, apt=("jq",))
    (built,) = run(environment_artifact(environment, REPOSITORY))
    identity = identity_digest(environment)
    tag = f"{REPOSITORY}:task-curation-env-{identity[:IDENTITY_CHARS]}"
    assert built.image == f"{REPOSITORY}@sha256:{hashlib.sha256(tag.encode()).hexdigest()}"
    assert StoragePath(built.lock_url).read_text() == COMPILED_LOCK
    assert built_environment(environment) == built
    commands = [shlex.split(line) for line in docker_log.read_text().splitlines()]
    export = next(command for command in commands if command[:2] == ["buildx", "build"])
    parameters = dict(
        option.split("=", 1)
        for option in next(arg for arg in export if arg.startswith("--output=")).removeprefix("--output=").split(",")
    )
    assert parameters["compression"] == "gzip"
    assert parameters["force-compression"] == "true"


def test_image_serialization_changes_invalidate_built_images_but_preserve_worker_environments(monkeypatch):
    worker = Environment(requirements=REQUIREMENTS)
    image = replace(worker, apt=("jq",))
    worker_identity = environment_artifact(worker).name
    image_identity = environment_artifact(image).name
    monkeypatch.setattr(build, "IMAGE_COMPRESSION", "zstd")
    assert environment_artifact(worker).name == worker_identity
    assert environment_artifact(image).name != image_identity


@pytest.mark.parametrize(
    "apt", [pytest.param((), id="no-apt"), pytest.param(("build-essential",), id="worker-image-apt")]
)
def test_a_worker_environment_records_its_lock_without_docker(docker_log, apt):
    environment = Environment(requirements=REQUIREMENTS, apt=apt)
    (built,) = run(environment_artifact(environment, REPOSITORY))
    assert built.image is None
    assert StoragePath(built.lock_url).read_text() == COMPILED_LOCK
    assert docker_log.read_text() == ""


def test_an_unchanged_environment_resolves_without_building(docker_log, monkeypatch):
    environment = Environment(requirements=REQUIREMENTS, apt=("jq",))
    (first,) = run(environment_artifact(environment, REPOSITORY))
    calls = docker_log.read_text()
    monkeypatch.setenv("PATH", "/nonexistent")
    (second,) = run(environment_artifact(environment, REPOSITORY))
    assert second.image == first.image
    assert docker_log.read_text() == calls


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda environment, _: replace(environment, requirements=("numpy==2.3.4",)), id="requirements"),
        pytest.param(lambda environment, _: replace(environment, apt=("build-essential",)), id="apt"),
        pytest.param(
            lambda environment, archive: replace(environment, files={DATA_PATH: replace(archive, member="german")}),
            id="files",
        ),
        pytest.param(
            lambda environment, _: replace(environment, dockerfile=tracked_fragment(environment.dockerfile.parent)),
            id="dockerfile",
        ),
    ],
)
def test_declaration_changes_rename_the_environment_artifact(repository, monkeypatch, change):
    archive = install_fake_archive(monkeypatch)
    plain = repository / "plain.Dockerfile"
    plain.write_text("RUN true\n")
    environment = Environment(requirements=REQUIREMENTS, files={DATA_PATH: archive}, dockerfile=plain)
    original = environment_artifact(environment).name
    assert environment_artifact(change(environment, archive)).name != original


def test_editing_a_dockerfile_fragment_renames_the_environment_artifact(fragment):
    environment = Environment(requirements=REQUIREMENTS, dockerfile=fragment)
    original = environment_artifact(environment).name
    fragment.write_text(TOOLCHAIN_FRAGMENT + "RUN touch /opt/toolchain/patched\n")
    assert environment_artifact(environment).name != original


@pytest.mark.parametrize(
    "instruction",
    ["FROM debian:bookworm", "USER 65534", "WORKDIR /lean4", "COPY lean-toolchain /lean4/", 'ENTRYPOINT ["lake"]'],
)
def test_a_dockerfile_fragment_cannot_redirect_the_generated_steps_after_it(fragment, instruction):
    fragment.write_text(TOOLCHAIN_FRAGMENT + f"# {instruction}\nRUN true \\\n    && true\n{instruction}\n")
    with pytest.raises(ValueError, match=instruction.split()[0]):
        Environment(requirements=REQUIREMENTS, dockerfile=fragment)


def test_declared_files_and_env_are_stored_for_the_worker_and_enter_the_image_before_the_lock(docker_log, monkeypatch):
    archive = install_fake_archive(monkeypatch)
    environment = Environment(
        requirements=REQUIREMENTS, apt=("jq",), files={DATA_PATH: archive}, env={"FIXTURE_DATA": "/srv/fixture"}
    )
    (built,) = run(environment_artifact(environment, REPOSITORY))
    stored = StoragePath(built.path)
    archive_bytes = (stored / "files.tar").read_bytes()
    assert json.loads((stored / "environment.json").read_text()) == {
        "files": [DATA_PATH],
        "files_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "env": {"FIXTURE_DATA": "/srv/fixture"},
    }
    with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as tar:
        members = {member.name: tar.extractfile(member).read() for member in tar.getmembers() if member.isfile()}
    assert members == {
        f"{DATA_PATH.removeprefix('/')}/{name.removeprefix('english/')}": content for name, content in DATA_FILES.items()
    }
    lines = dockerfile(environment).splitlines()
    apt_step = next(index for index, line in enumerate(lines) if line.startswith("RUN apt-get"))
    lock_copy = next(index for index, line in enumerate(lines) if line.startswith("COPY requirements.lock"))
    files_copy = lines.index("COPY files/ /")
    assert apt_step < files_copy < lock_copy
    assert lines[files_copy + 1] == 'ENV FIXTURE_DATA="/srv/fixture"'


def test_a_worker_environment_without_files_or_env_stores_no_manifest(docker_log):
    (built,) = run(environment_artifact(Environment(requirements=REQUIREMENTS), REPOSITORY))
    assert not (StoragePath(built.path) / "environment.json").exists()


def test_an_archive_whose_bytes_differ_from_the_declared_digest_is_not_built(docker_log, monkeypatch, tmp_path):
    archive = replace(install_fake_archive(monkeypatch), sha256="0" * 64)
    environment = Environment(requirements=REQUIREMENTS, files={DATA_PATH: archive})
    with pytest.raises(ValueError, match="SHA-256"):
        build_environment(environment, EnvironmentBuild(identity_digest(environment), REPOSITORY, str(tmp_path / "out")))
    assert not (tmp_path / "out").exists()


def test_the_repository_is_where_a_build_pushes_not_identity():
    environment = Environment(requirements=REQUIREMENTS, apt=("jq",))
    assert environment_artifact(environment, "other.invalid/images").name == environment_artifact(environment).name


def test_build_from_an_untracked_dockerfile_in_a_worktree_records_the_environment(repository, docker_log, tmp_path):
    untracked = repository / "untracked.Dockerfile"
    untracked.write_text(TOOLCHAIN_FRAGMENT)
    environment = Environment(requirements=REQUIREMENTS, dockerfile=untracked)
    built = build_environment(
        environment, EnvironmentBuild(identity_digest(environment), REPOSITORY, str(tmp_path / "out"))
    )
    assert StoragePath(built.lock_url).read_text() == COMPILED_LOCK
    assert built.image is not None
    assert built.identity == identity_digest(environment)


def test_build_from_a_gitless_source_bundle_records_the_environment(docker_log, tmp_path):
    fragment = tmp_path / "bundle.Dockerfile"
    fragment.write_text(TOOLCHAIN_FRAGMENT)
    environment = Environment(requirements=REQUIREMENTS, dockerfile=fragment)
    built = build_environment(
        environment, EnvironmentBuild(identity_digest(environment), REPOSITORY, str(tmp_path / "out"))
    )
    assert StoragePath(built.lock_url).read_text() == COMPILED_LOCK
    assert built.image is not None
    assert built.identity == identity_digest(environment)


def test_an_image_build_requires_repository_credentials(docker_log, tmp_path):
    (tmp_path / "docker-config" / "config.json").write_text(json.dumps({"auths": {}}))
    environment = Environment(requirements=REQUIREMENTS, apt=("jq",))
    with pytest.raises(RuntimeError, match=re.escape("docker login registry.invalid")):
        build_environment(environment, EnvironmentBuild(identity_digest(environment), REPOSITORY, str(tmp_path / "out")))


def test_registered_environment_build_command_resolves_new_grader(tmp_path, monkeypatch):
    install_fake_build_tools(tmp_path, monkeypatch)
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    environment = Environment(requirements=REQUIREMENTS)
    monkeypatch.setattr(images, "BUILDABLE_ENVIRONMENTS", (environment,))
    identity = identity_digest(environment)[:IDENTITY_CHARS]
    result = CliRunner().invoke(images.main, ["--identity", identity, "--repository", REPOSITORY])
    assert result.exit_code == 0, result.output
    assert StoragePath(built_environment(environment).lock_url).read_text().startswith("numpy==2.3.5")
