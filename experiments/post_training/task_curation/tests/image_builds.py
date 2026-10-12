# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``docker`` and ``uv`` stand-ins and a git-tracked lock, so builds run without a daemon, registry or index."""

import json
import subprocess
from pathlib import Path

REPOSITORY = "registry.invalid/fixture/images"
LOCK = "numpy==2.3.5 \\\n    --hash=sha256:" + "a" * 64 + "\n"

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


def track_all(repository: Path) -> None:
    subprocess.run(["git", "add", "-A"], cwd=repository, check=True)


def tracked_lock(root: Path) -> Path:
    """A lock file in a git repository that tracks it."""
    repository = root / "repository"
    repository.mkdir()
    lock = repository / "fixture.lock"
    lock.write_text(LOCK)
    subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True)
    track_all(repository)
    return lock
