# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``docker``, ``uv`` and archive-download stand-ins and a git repository, so builds run offline without a daemon."""

import hashlib
import io
import json
import subprocess
import zipfile
from collections.abc import Mapping
from pathlib import Path

from experiments.post_training.task_curation.environment import Archive
from experiments.post_training.task_curation.images import build

REPOSITORY = "registry.invalid/fixture/images"
REQUIREMENTS = ("numpy==2.3.5",)
COMPILED_LOCK = "numpy==2.3.5 \\\n    --hash=sha256:" + hashlib.sha256(b"numpy==2.3.5").hexdigest() + "\n"
"""What the ``uv`` stand-in compiles ``REQUIREMENTS`` to."""
TOOLCHAIN_FRAGMENT = "ENV TOOLCHAIN_HOME=/opt/toolchain\nRUN mkdir -p /opt/toolchain && touch /opt/toolchain/ready\n"
ARCHIVE_URL = "https://archives.invalid/fixture/data.zip"
DATA_PATH = "/srv/fixture/english"
DATA_FILES = {"english/abbrev.tab": b"etc\n", "english/sentence_starters.txt": b"The\n"}
"""A zip of a directory member, as the NLTK data archives are laid out; ``DATA_PATH`` receives ``english``."""

# Records each invocation and answers `imagetools inspect` with a digest derived from the tag, so a
# changed environment pushes a different digest.
FAKE_DOCKER = """#!/bin/sh
echo "$*" >> "$FAKE_DOCKER_LOG"
case "$1 $2" in
  "buildx version") echo "github.com/docker/buildx v0.0.0-fixture" ;;
  "buildx build") ;;
  "buildx imagetools") printf '{"digest": "sha256:%s"}' "$(printf %s "$4" | sha256sum | cut -d' ' -f1)" ;;
  *) exit 1 ;;
esac
"""

# Answers `uv pip compile ... --output-file OUT IN` offline: each input pin with a hash derived from it.
FAKE_UV = """#!/bin/sh
[ "$1 $2" = "pip compile" ] || exit 1
shift 2
while [ $# -gt 0 ]; do
  case "$1" in
    --output-file) output="$2"; shift 2 ;;
    *) source="$1"; shift ;;
  esac
done
while IFS= read -r requirement; do
  printf '%s \\\\\\n    --hash=sha256:%s\\n' "$requirement" "$(printf %s "$requirement" | sha256sum | cut -d' ' -f1)"
done < "$source" > "$output"
"""


def install_fake_build_tools(root: Path, monkeypatch) -> Path:
    """Put ``docker`` and ``uv`` stand-ins first on PATH and credentials for ``REPOSITORY`` in DOCKER_CONFIG.

    Returns the log of docker invocations.
    """
    bin_dir, config_dir, log = root / "bin", root / "docker-config", root / "docker.log"
    bin_dir.mkdir()
    config_dir.mkdir()
    for name, script in (("docker", FAKE_DOCKER), ("uv", FAKE_UV)):
        (bin_dir / name).write_text(script)
        (bin_dir / name).chmod(0o755)
    (config_dir / "config.json").write_text(json.dumps({"auths": {REPOSITORY.split("/")[0]: {}}}))
    log.touch()
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setenv("DOCKER_CONFIG", str(config_dir))
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(log))
    return log


def zip_bytes(files: Mapping[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in sorted(files.items()):
            archive.writestr(name, content)
    return buffer.getvalue()


def install_fake_archive(monkeypatch, files: Mapping[str, bytes] = DATA_FILES, member: str = "english") -> Archive:
    """Serve a zip of ``files`` at ``ARCHIVE_URL`` instead of downloading, and declare its ``member``."""
    data = zip_bytes(files)

    def serve(url: str) -> bytes:
        if url != ARCHIVE_URL:
            raise AssertionError(f"Unexpected download: {url}")
        return data

    monkeypatch.setattr(build, "download", serve)
    return Archive(ARCHIVE_URL, hashlib.sha256(data).hexdigest(), member)


def track_all(repository: Path) -> None:
    subprocess.run(["git", "add", "-A"], cwd=repository, check=True)


def tracked_repository(root: Path) -> Path:
    """An empty git repository, for files the build requires git to track."""
    repository = root / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True)
    return repository


def tracked_fragment(repository: Path) -> Path:
    fragment = repository / "toolchain.Dockerfile"
    fragment.write_text(TOOLCHAIN_FRAGMENT)
    track_all(repository)
    return fragment
