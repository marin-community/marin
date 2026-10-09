# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import io
import tarfile
from pathlib import Path

import pytest
from taskcompendium.models import EnvironmentRequirements, TaskResource
from taskcompendium.runtime.resources import inline_resource

from taskforge.sandbox.images import (
    BuildLimits,
    BuildTooLarge,
    DockerBuild,
    build_context_archive,
    build_digest,
    docker_build_from_directory,
    pinned_image,
)

LIMITS = BuildLimits(max_files=16, max_file_bytes=1024, max_total_bytes=2048)


def resource(path: str, content: bytes, mode: str | None = None) -> TaskResource:
    return inline_resource(path, content).model_copy(update={"mode": mode})


def write_context(root: Path) -> None:
    (root / "bin").mkdir(parents=True)
    (root / "Dockerfile").write_text("FROM busybox\nCOPY bin /opt/bin\n")
    (root / "bin" / "run.sh").write_text("#!/bin/sh\necho ok\n")
    (root / "bin" / "run.sh").chmod(0o755)


def test_directory_round_trips_through_the_build_archive(tmp_path):
    write_context(tmp_path / "ctx")
    build = docker_build_from_directory(tmp_path / "ctx", LIMITS)

    assert {f.path: f.mode for f in build.files} == {"Dockerfile": "644", "bin/run.sh": "755"}
    archive = build_context_archive(build)
    assert archive == build_context_archive(DockerBuild(files=tuple(reversed(build.files))))
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(archive))) as tar:
        tar.extractall(tmp_path / "out", filter="data")
    assert (tmp_path / "out" / "bin" / "run.sh").read_text() == "#!/bin/sh\necho ok\n"
    assert (tmp_path / "out" / "bin" / "run.sh").stat().st_mode & 0o777 == 0o755
    assert build_digest(docker_build_from_directory(tmp_path / "out", LIMITS)) == build_digest(build)


def test_build_digest_tracks_content_mode_path_and_dockerfile():
    dockerfile = resource("Dockerfile", b"FROM x\n")
    script = resource("a.sh", b"a", "755")
    other = resource("x/Dockerfile", b"FROM y\n")
    base = DockerBuild(files=(dockerfile, script))
    assert build_digest(base) == build_digest(DockerBuild(files=(script, dockerfile)))
    # An unset mode is the archive's default mode, so it digests the same as an explicit 644.
    assert build_digest(base) == build_digest(DockerBuild(files=(resource("Dockerfile", b"FROM x\n", "644"), script)))
    variants = [
        DockerBuild(files=(dockerfile, resource("a.sh", b"b", "755"))),
        DockerBuild(files=(dockerfile, resource("a.sh", b"a", "644"))),
        DockerBuild(files=(dockerfile, resource("b.sh", b"a", "755"))),
        DockerBuild(files=(dockerfile, script, other)),
        DockerBuild(files=(dockerfile, script, other), dockerfile="x/Dockerfile"),
    ]
    assert len({build_digest(base), *(build_digest(v) for v in variants)}) == 1 + len(variants)


def test_build_must_contain_its_dockerfile():
    with pytest.raises(ValueError, match="no 'Dockerfile'"):
        DockerBuild(files=(resource("app/run.sh", b"echo\n"),))


def test_context_over_the_total_limit_is_refused(tmp_path):
    write_context(tmp_path)
    (tmp_path / "blob.bin").write_bytes(b"x" * 1024)
    (tmp_path / "blob2.bin").write_bytes(b"x" * 1024)
    with pytest.raises(BuildTooLarge, match="limit is 2048"):
        docker_build_from_directory(tmp_path, LIMITS)


def test_symlinks_are_refused(tmp_path):
    write_context(tmp_path)
    (tmp_path / "link").symlink_to(tmp_path / "Dockerfile")
    with pytest.raises(ValueError, match="regular files"):
        docker_build_from_directory(tmp_path, LIMITS)


def test_pinned_image_is_a_task_docker_image():
    digest = "sha256:" + "a" * 64
    image = pinned_image("registry.example", "capability-infra/task", digest)
    assert EnvironmentRequirements(docker_image=image).docker_image == f"registry.example/capability-infra/task@{digest}"
    with pytest.raises(ValueError, match="manifest digest"):
        pinned_image("registry.example", "capability-infra/task", "latest")
