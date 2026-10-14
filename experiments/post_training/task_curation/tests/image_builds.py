# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``docker`` and ``uv`` stand-ins and a git-tracked lock, so builds run without a daemon, registry or index."""

import json
import subprocess
import sys
from pathlib import Path

REPOSITORY = "registry.invalid/fixture/images"
LOCK = "numpy==2.3.5 \\\n    --hash=sha256:" + "a" * 64 + "\n"
TOOLCHAIN_FRAGMENT = "ENV TOOLCHAIN_HOME=/opt/toolchain\nRUN mkdir -p /opt/toolchain && touch /opt/toolchain/ready\n"

# Records each invocation and answers `imagetools inspect` with a digest derived from the tag, so a
# changed environment pushes a different digest.
FAKE_DOCKER = """import hashlib
import json
import os
import sys

with open(os.environ["FAKE_DOCKER_LOG"], "a") as log:
    log.write(" ".join(sys.argv[1:]) + "\\n")
match sys.argv[1:3]:
    case ["buildx", "version"]:
        print("github.com/docker/buildx v0.0.0-fixture")
    case ["buildx", "build"]:
        pass
    case ["buildx", "imagetools"]:
        print(json.dumps({"digest": "sha256:" + hashlib.sha256(sys.argv[4].encode()).hexdigest()}))
    case _:
        sys.exit(1)
"""

# Answers `uv pip compile ... --output-file OUT IN` offline: each input pin with a hash derived from it.
FAKE_UV = r"""import hashlib
import sys
from pathlib import Path

assert sys.argv[1:3] == ["pip", "compile"]
output = Path(sys.argv[sys.argv.index("--output-file") + 1])
source = Path(sys.argv[-1])
with output.open("w") as compiled:
    for requirement in source.read_text().splitlines():
        digest = hashlib.sha256(requirement.encode()).hexdigest()
        compiled.write(f"{requirement} \\\n    --hash=sha256:{digest}\n")
"""


def install_fake_build_tools(root: Path, monkeypatch) -> Path:
    """Put ``docker`` and ``uv`` stand-ins first on PATH and credentials for ``REPOSITORY`` in DOCKER_CONFIG.

    Returns the log of docker invocations.
    """
    bin_dir, config_dir, log = root / "bin", root / "docker-config", root / "docker.log"
    bin_dir.mkdir()
    config_dir.mkdir()
    for name, script in (("docker", FAKE_DOCKER), ("uv", FAKE_UV)):
        (bin_dir / name).write_text(f"#!{sys.executable}\n{script}")
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


def tracked_fragment(lock: Path) -> Path:
    fragment = lock.parent / "toolchain.Dockerfile"
    fragment.write_text(TOOLCHAIN_FRAGMENT)
    track_all(lock.parent)
    return fragment
