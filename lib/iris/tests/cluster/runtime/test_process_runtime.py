# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for ProcessRuntime mount resolution and TMPFS cleanup."""

import sys
from pathlib import Path

from iris.cluster.runtime.process import ProcessRuntime
from iris.cluster.runtime.types import ContainerConfig, ContainerPhase, MountKind, MountSpec
from iris.rpc import job_pb2
from iris.test_util import wait_for_condition


def _make_config(
    mounts: list[MountSpec],
    *,
    args: list[str],
    env: dict[str, str] | None = None,
    workdir_host_path: Path | None = None,
) -> ContainerConfig:
    return ContainerConfig(
        image="test:latest",
        entrypoint=job_pb2.RuntimeEntrypoint(
            run_command=job_pb2.CommandEntrypoint(
                argv=[
                    sys.executable,
                    "-c",
                    (
                        "import os,sys; print('|'.join(sys.argv[1:])); "
                        "print(os.getenv('PATH_A', '')); print(os.getenv('PATH_B', ''))"
                    ),
                    *args,
                ]
            )
        ),
        env=env or {},
        mounts=mounts,
        workdir_host_path=workdir_host_path,
        task_id="test-task",
    )


def _run(runtime: ProcessRuntime, config: ContainerConfig):
    handle = runtime.create_container(config)
    handle.run()
    wait_for_condition(lambda: handle.status().phase == ContainerPhase.STOPPED)
    lines = [line.data for line in handle.log_reader().read_all() if line.source == "stdout"]
    assert handle.status().exit_code == 0
    return handle, lines


def test_tmpfs_mounts_across_containers_use_distinct_host_directories(tmp_path):
    runtime = ProcessRuntime(cache_dir=tmp_path)
    mounts = [MountSpec("tmp", "/tmp", kind=MountKind.TMPFS)]
    config = _make_config(mounts, args=[], env={"PATH_A": "/tmp"})

    first, first_lines = _run(runtime, config)
    second, second_lines = _run(runtime, config)
    first_path = Path(first_lines[1])
    second_path = Path(second_lines[1])

    assert first_path != second_path
    assert first_path.is_dir()
    assert second_path.is_dir()
    first.cleanup()
    second.cleanup()


def test_tmpfs_and_cache_mounts_resolve_to_independent_host_directories(tmp_path):
    runtime = ProcessRuntime(cache_dir=tmp_path)
    config = _make_config(
        [
            MountSpec("tmp", "/tmp", kind=MountKind.TMPFS),
            MountSpec("root-cache-uv", "/root/.cache/uv", kind=MountKind.CACHE),
        ],
        args=[],
        env={"PATH_A": "/tmp", "PATH_B": "/root/.cache/uv"},
    )

    handle, lines = _run(runtime, config)
    tmpfs_path, cache_path = Path(lines[1]), Path(lines[2])

    assert tmpfs_path != cache_path
    assert tmpfs_path.is_dir()
    assert cache_path.is_dir()
    handle.cleanup()


def test_process_handle_cleanup_removes_observed_tmpfs_directory(tmp_path):
    runtime = ProcessRuntime(cache_dir=tmp_path)
    config = _make_config(
        [MountSpec("tmp", "/tmp", kind=MountKind.TMPFS)],
        args=[],
        env={"PATH_A": "/tmp"},
    )
    handle, lines = _run(runtime, config)
    tmpfs_path = Path(lines[1])
    assert tmpfs_path.is_dir()

    handle.cleanup()

    assert not tmpfs_path.exists()


def test_process_runtime_remaps_arguments_below_cache_mounts(tmp_path):
    runtime = ProcessRuntime(cache_dir=tmp_path)
    handle, lines = _run(
        runtime,
        _make_config(
            [
                MountSpec("cargo", "/cargo", kind=MountKind.CACHE),
                MountSpec("uv", "/uv/cache", kind=MountKind.CACHE),
            ],
            args=["/cargo", "/cargo/target", "/uv/cache/python"],
        ),
    )
    cargo, target, python = map(Path, lines[0].split("|"))

    assert cargo.parent == tmp_path
    assert target == cargo / "target"
    assert python.parent.parent == tmp_path
    assert python.name == "python"
    handle.cleanup()


def test_process_runtime_leaves_unmounted_arguments_unchanged(tmp_path):
    runtime = ProcessRuntime(cache_dir=tmp_path)
    handle, lines = _run(
        runtime,
        _make_config(
            [MountSpec("app", "/app", kind=MountKind.CACHE)],
            args=["/etc/hosts", "--flag=value", "/apps/other"],
        ),
    )

    assert lines[0] == "/etc/hosts|--flag=value|/apps/other"
    handle.cleanup()


def test_process_runtime_prefers_nested_mount_for_argument_remapping(tmp_path):
    runtime = ProcessRuntime(cache_dir=tmp_path)
    handle, lines = _run(
        runtime,
        _make_config(
            [
                MountSpec("app", "/app", kind=MountKind.CACHE),
                MountSpec("data", "/app/data", kind=MountKind.CACHE),
            ],
            args=["/app/data/x", "/app/other"],
        ),
    )
    nested, parent = map(Path, lines[0].split("|"))

    assert nested.parent.parent == tmp_path
    assert nested.name == "x"
    assert parent.parent.parent == tmp_path
    assert parent.name == "other"
    assert nested.parent != parent.parent
    handle.cleanup()
