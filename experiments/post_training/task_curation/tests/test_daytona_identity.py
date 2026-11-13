# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snapshot counting runs through the production identity and spec path."""

import base64
import gzip
import io
import tarfile

import pytest

pytest.importorskip("harbor")  # the counter runs the production Harbor spec mapping

from daytona import Resources
from shellbox.backends.daytona.snapshot import snapshot_name
from shellbox.image import DockerfileSource, RegistryImage

from experiments.post_training.task_curation.daytona_identity import (
    count_lowered_spec_identities,
    count_registry_identities,
    count_task_identities,
)


@pytest.fixture
def taskcompendium_environment():
    pytest.importorskip("taskcompendium")


def _native_tar(dockerfile: str, files: dict[str, bytes]) -> bytes:
    members = {
        "environment/Dockerfile": dockerfile.encode(),
        **{f"environment/{name}": content for name, content in files.items()},
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return gzip.compress(buffer.getvalue(), mtime=0)


def test_identical_native_tars_share_one_snapshot_and_add_tars_fail():
    shared = _native_tar("FROM ubuntu:24.04\nCOPY files/data.csv /app/data.csv\n", {"files/data.csv": b"a,b\n1,2\n"})
    count = count_task_identities([("task-a", shared), ("task-b", shared)])
    assert count.distinct == 1 and not count.failures
    assert next(iter(count.sizes.values())) == 2

    add_tar = _native_tar("FROM ubuntu:24.04\nADD files/data.csv /app/data.csv\n", {"files/data.csv": b"a,b\n1,2\n"})
    count = count_task_identities([("task-add", add_tar), ("task-a", shared)])
    assert count.distinct == 1
    assert count.failures == [("task-add", "Daytona Dockerfiles with ADD require a prebuilt registry image")]


def test_data_baked_into_contexts_splits_identities():
    first = _native_tar("FROM ubuntu:24.04\nCOPY files/ /app/files\n", {"files/a.csv": b"1\n"})
    second = _native_tar("FROM ubuntu:24.04\nCOPY files/ /app/files\n", {"files/a.csv": b"2\n"})
    count = count_task_identities([("a", first), ("b", second)])
    assert count.distinct == 2 and not count.failures


@pytest.mark.usefixtures("taskcompendium_environment")
def test_lowered_specs_collapse_and_cite_the_registry_image():
    def spec(dockerfile: str, mode: str | None) -> dict:
        resource = {
            "path": "Dockerfile",
            "source": {"kind": "inline_file", "content_base64": base64.b64encode(dockerfile.encode()).decode()},
        }
        if mode is not None:
            resource["mode"] = mode
        requirements = {"command_semantics": "linux_process", "docker_build": {"files": [resource]}}
        return {
            "task": {"environment_requirements": requirements, "grader": {"environment": requirements}},
            "runtime": {"task_machine": {}, "verifier_machine": {}},
        }

    shared = spec("FROM aweaiteam/calibforge:tb2-verifier-base-v1@sha256:0f61\nWORKDIR /app\n", "644")
    null_mode = spec("FROM aweaiteam/calibforge:tb2-verifier-base-v1@sha256:0f61\nWORKDIR /app\n", None)
    count = count_lowered_spec_identities([("a", shared), ("b", shared), ("c", null_mode)])
    assert count.distinct == 1 and not count.failures
    assert next(iter(count.sizes.values())) == 6  # verifier + task machine per task on one shared identity


def test_registry_identities_count_digest_pinned_bases():
    count = count_registry_identities(["ubuntu:24.04", "ubuntu:24.04", "python:3.11-slim-bookworm"])
    assert count.distinct == 2 and not count.failures


def test_counter_names_match_the_production_name_for_the_staged_tree(tmp_path):
    """Executable bits and file contents in the task tar reach the identity."""
    dockerfile = "FROM ubuntu:24.04\nCOPY setup_repo.sh /app/setup_repo.sh\n"
    members = {
        "environment/Dockerfile": (dockerfile.encode(), 0o644),
        "environment/setup_repo.sh": (b"#!/bin/sh\ngit checkout abc123\n", 0o755),
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, (content, mode) in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = mode
            archive.addfile(info, io.BytesIO(content))
    payload = gzip.compress(buffer.getvalue(), mtime=0)

    staged = tmp_path / "staged"
    staged.mkdir()
    for name, (content, mode) in members.items():
        _write = staged / name[len("environment/") :]
        _write.parent.mkdir(parents=True, exist_ok=True)
        _write.write_bytes(content)
        _write.chmod(mode)
    expected = snapshot_name(DockerfileSource(staged, staged / "Dockerfile"), Resources())

    count = count_task_identities([("task", payload)])
    assert list(count.sizes) == [expected]

    non_executable = io.BytesIO()
    with tarfile.open(fileobj=non_executable, mode="w") as archive:
        for name, (content, mode) in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o644 if name.endswith("setup_repo.sh") else mode
            archive.addfile(info, io.BytesIO(content))
    plain = count_task_identities([("task", gzip.compress(non_executable.getvalue(), mtime=0))])
    assert list(plain.sizes) != [expected]  # the executable bit is part of the identity


@pytest.mark.usefixtures("taskcompendium_environment")
def test_a_task_needing_a_container_runtime_is_refused_on_daytona():
    """Snapshot count is not enough: an unsupported capability blocks the verdict."""
    image = "ubuntu:24.04@sha256:" + "a" * 64
    requirements = {"capabilities": ["docker"], "command_semantics": "linux_process", "docker_image": image}
    spec = {
        "task": {"environment_requirements": requirements},
        "runtime": {"task_machine": {}},
    }
    count = count_lowered_spec_identities([("container-runtime", spec)])
    assert count.distinct == 1
    assert [task_id for task_id, _ in count.execution_failures] == ["container-runtime"]


@pytest.mark.usefixtures("taskcompendium_environment")
def test_grader_environment_is_counted_independently_and_lock_refusals_are_retained():
    actor = "actor@sha256:" + "a" * 64
    grader = "grader@sha256:" + "b" * 64
    requirements = {"docker_image": actor, "command_semantics": "linux_process"}
    spec = {
        "task": {
            "environment_requirements": requirements,
            "grader": {"environment": {**requirements, "docker_image": grader}},
        },
        "runtime": {"task_machine": {}, "verifier_machine": {}},
    }
    separate = count_lowered_spec_identities([("separate", spec)])
    assert set(separate.sizes) == {
        snapshot_name(RegistryImage(actor), Resources()),
        snapshot_name(RegistryImage(grader), Resources()),
    }
    spec["task"]["grader"]["environment"] = {
        "packages_lock": "https://example.test/lock",
        "command_semantics": "linux_process",
    }
    locked = count_lowered_spec_identities([("locked", spec)])
    assert locked.distinct == 1
    assert [task_id for task_id, _ in locked.execution_failures] == ["locked"]


@pytest.mark.usefixtures("taskcompendium_environment")
def test_actor_capacity_keeps_separate_grader_snapshot_resources():
    environment = {"command_semantics": "linux_process", "docker_image": "example/image@sha256:" + "0" * 64}
    task = {
        "id": "independent",
        "environment_requirements": environment,
        "grader": {"environment": environment},
        "execution_requirements": {
            "cpus": 2,
            "memory_mb": 4096,
            "storage_mb": 10240,
            "agent_timeout_sec": 900,
            "verifier_timeout_sec": 600,
        },
    }
    runtime = {
        "task_machine": {"cpus": 2, "memory_mb": 4096, "storage_mb": 10240},
        "verifier_machine": {"cpus": 4, "memory_mb": 8192, "storage_mb": 20480},
    }
    measured = count_lowered_spec_identities([("independent", {"task": task, "runtime": runtime})])
    assert measured.distinct == 2 and not measured.execution_failures
