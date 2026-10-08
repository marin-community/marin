# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A git-tracked fixture recipe and a ``docker`` stand-in, so image builds run without a daemon or registry."""

import json
import subprocess
from pathlib import Path

from experiments.post_training.task_curation.images.recipes import ImageRecipe

BASE_IMAGE = "ghcr.io/marin-community/iris-task@sha256:" + "1" * 64
REPOSITORY = "registry.invalid/fixture/images"

# Records each invocation and answers `imagetools inspect` with a digest derived from the tag, so a
# changed recipe pushes a different digest.
FAKE_DOCKER = """#!/bin/sh
echo "$*" >> "$FAKE_DOCKER_LOG"
case "$1 $2" in
  "buildx version") echo "github.com/docker/buildx v0.0.0-fixture" ;;
  "buildx build") ;;
  "buildx imagetools") printf '{"digest": "sha256:%s"}' "$(printf %s "$4" | sha256sum | cut -d' ' -f1)" ;;
  *) exit 1 ;;
esac
"""


def install_fake_docker(root: Path, monkeypatch) -> Path:
    """Put a ``docker`` stand-in first on PATH and credentials for ``REPOSITORY`` in DOCKER_CONFIG; return its log."""
    bin_dir, config_dir, log = root / "bin", root / "docker-config", root / "docker.log"
    bin_dir.mkdir()
    config_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(FAKE_DOCKER)
    docker.chmod(0o755)
    (config_dir / "config.json").write_text(json.dumps({"auths": {REPOSITORY.split("/")[0]: {}}}))
    log.touch()
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setenv("DOCKER_CONFIG", str(config_dir))
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(log))
    return log


def track_all(repository: Path) -> None:
    subprocess.run(["git", "add", "-A"], cwd=repository, check=True)


def tracked_recipe(root: Path) -> ImageRecipe:
    """A recipe whose context and package live in a git repository with every file tracked."""
    repository = root / "repository"
    context, package = repository / "fixture", repository / "fixturepkg"
    context.mkdir(parents=True)
    package.mkdir()
    (context / "Dockerfile").write_text(f"FROM {BASE_IMAGE}\nCOPY --from=fixturepkg . /opt/fixturepkg\n")
    (context / "requirements.lock").write_text("numpy==2.3.5\n")
    (package / "__init__.py").write_text("VALUE = 1\n")
    subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True)
    track_all(repository)
    return ImageRecipe("fixture", context, packages=(package,))
