# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Image identity, the build-and-record step, and the artifact a declaration resolves."""

import hashlib
import json
import re

import pytest
from marin.execution.lazy import run

from experiments.post_training.task_curation.images.build import (
    IDENTITY_CHARS,
    PLATFORM,
    ImageBuild,
    MissingImageArtifact,
    base_image,
    build_image,
    built_image,
    identity_digest,
    image_artifact,
)
from experiments.post_training.task_curation.tests.image_builds import (
    BASE_IMAGE,
    REPOSITORY,
    install_fake_docker,
    tracked_recipe,
)


@pytest.fixture
def recipe(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "prefix"))
    return tracked_recipe(tmp_path)


@pytest.fixture
def docker_log(tmp_path, monkeypatch):
    return install_fake_docker(tmp_path, monkeypatch)


def test_build_pushes_the_identity_tag_and_records_the_pushed_digest(recipe, docker_log):
    (image,) = run(image_artifact(recipe, REPOSITORY))
    identity = identity_digest(recipe)
    tag = f"{REPOSITORY}:task-curation-fixture-{identity[:IDENTITY_CHARS]}"
    assert image.tag == tag
    assert image.image == f"{REPOSITORY}@sha256:{hashlib.sha256(tag.encode()).hexdigest()}"
    assert image.base_image == BASE_IMAGE
    assert image.lock_sha256 == hashlib.sha256(b"numpy==2.3.5\n").hexdigest()
    assert [file.path for file in image.context_files] == ["Dockerfile", "requirements.lock"]
    assert [file.path for file in image.package_files["fixturepkg"]] == ["__init__.py"]
    assert image.platform == PLATFORM
    assert image.path.endswith(f"images/fixture-{identity[:IDENTITY_CHARS]}/2026.10.07")
    build = next(line for line in docker_log.read_text().splitlines() if line.startswith("buildx build"))
    assert f"--platform={PLATFORM}" in build.split()
    assert "--push" in build.split()
    assert f"--build-context=fixturepkg={recipe.packages[0]}" in build.split()
    assert built_image(recipe) == image


def test_an_unchanged_recipe_resolves_without_docker(recipe, docker_log, monkeypatch):
    (first,) = run(image_artifact(recipe, REPOSITORY))
    calls = docker_log.read_text()
    monkeypatch.setenv("PATH", "/nonexistent")
    (second,) = run(image_artifact(recipe, REPOSITORY))
    assert second.image == first.image
    assert docker_log.read_text() == calls


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda recipe: (recipe.context / "requirements.lock").write_text("numpy==2.3.4\n"), id="lock"),
        pytest.param(lambda recipe: (recipe.packages[0] / "__init__.py").write_text("VALUE = 2\n"), id="package"),
        pytest.param(lambda recipe: (recipe.context / "extra.txt").write_text("extra\n"), id="new-file"),
        pytest.param(lambda recipe: (recipe.context / "requirements.lock").chmod(0o755), id="mode"),
        pytest.param(
            lambda recipe: (recipe.context / "Dockerfile").write_text(
                (recipe.context / "Dockerfile").read_text().replace("1" * 64, "2" * 64)
            ),
            id="base-image",
        ),
    ],
)
def test_recipe_changes_rename_the_image_artifact(recipe, change):
    original = image_artifact(recipe).name
    change(recipe)
    assert image_artifact(recipe).name != original


def test_bytecode_caches_do_not_enter_the_identity(recipe):
    original = image_artifact(recipe).name
    cache = recipe.packages[0] / "__pycache__"
    cache.mkdir()
    (cache / "__init__.cpython-312.pyc").write_bytes(b"\0")
    assert image_artifact(recipe).name == original


def test_the_repository_is_where_a_build_pushes_not_identity(recipe):
    assert image_artifact(recipe, "other.invalid/images").name == image_artifact(recipe, REPOSITORY).name


@pytest.mark.parametrize(
    "change,message",
    [
        pytest.param(
            lambda recipe: (recipe.context / "notes.txt").write_text("untracked\n"), "untracked", id="untracked"
        ),
        pytest.param(lambda recipe: (recipe.context / "requirements.lock").chmod(0o755), "executable", id="exec"),
    ],
)
def test_build_refuses_files_a_workspace_bundle_would_not_reproduce(recipe, docker_log, tmp_path, change, message):
    change(recipe)
    with pytest.raises(ValueError, match=message):
        build_image(recipe, ImageBuild(identity_digest(recipe), REPOSITORY, str(tmp_path / "output")))
    assert "buildx build" not in docker_log.read_text()


def test_build_requires_repository_credentials(recipe, docker_log, tmp_path):
    (tmp_path / "docker-config" / "config.json").write_text(json.dumps({"auths": {}}))
    with pytest.raises(RuntimeError, match=re.escape("docker login registry.invalid")):
        build_image(recipe, ImageBuild(identity_digest(recipe), REPOSITORY, str(tmp_path / "output")))


def test_a_declared_recipe_without_a_built_artifact_names_the_build_command(recipe):
    with pytest.raises(MissingImageArtifact, match=re.escape("images --recipe fixture")):
        built_image(recipe)


def test_dockerfile_must_name_one_digest_pinned_base(recipe):
    (recipe.context / "Dockerfile").write_text("FROM python:3.12\n")
    with pytest.raises(ValueError, match="pinned by digest"):
        identity_digest(recipe)


def test_python_imports_in_run_continuations_are_not_base_images(recipe):
    base = "ghcr.io/example/base@sha256:" + "a" * 64
    (recipe.context / "Dockerfile").write_text(
        f"FROM {base}\nRUN python -c 'import nltk; \\\n    from sympy.parsing.latex import parse_latex'\n"
    )
    assert base_image(recipe.context / "Dockerfile") == base
