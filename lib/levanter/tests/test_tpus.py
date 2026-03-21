# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import shlex
from collections.abc import Sequence
from typing import Any

import pytest

from levanter.infra import tpus


def _run_create_with_capacity(monkeypatch: pytest.MonkeyPatch, capacity_type: str) -> tuple[Any, ...]:
    commands: list[tuple[Any, ...]] = []
    describe_calls = 0

    def fake_describe_tpu_queued_resource(*_args, **_kwargs):
        nonlocal describe_calls
        describe_calls += 1
        if describe_calls == 1:
            return None
        return {"state": {"state": "ACTIVE"}}

    def fake_run_command(*args, **_kwargs):
        commands.append(args)

    monkeypatch.setattr(tpus, "describe_tpu_queued_resource", fake_describe_tpu_queued_resource)
    monkeypatch.setattr(tpus, "run_command", fake_run_command)
    monkeypatch.setattr(tpus.time, "sleep", lambda _seconds: None)

    tpus.start_tpu_vm_queued_resources(
        "exp2039-test",
        tpu_type="v6e-8",
        capacity_type=capacity_type,
        version="v2-alpha-tpuv6e",
        zone="europe-west4-a",
        node_count=1,
    )

    assert len(commands) >= 2
    return commands[1]


def _contains_arg(command: tuple[Any, ...], arg: str) -> bool:
    return any(str(token) == arg for token in command)


def _contains_arg_prefix(command: tuple[Any, ...], prefix: str) -> bool:
    return any(str(token).startswith(prefix) for token in command)


@pytest.mark.parametrize(
    ("capacity_type", "expected_flag"),
    [
        ("spot", "--spot"),
        ("preemptible", "--spot"),
        ("best-effort", "--best-effort"),
        ("reserved", "--reserved"),
    ],
)
def test_start_tpu_vm_queued_resources_uses_expected_capacity_flag(
    monkeypatch: pytest.MonkeyPatch, capacity_type: str, expected_flag: str
):
    create_command = _run_create_with_capacity(monkeypatch, capacity_type)
    assert _contains_arg(create_command, expected_flag)
    assert not _contains_arg_prefix(create_command, "--provisioning-model=")


def test_start_tpu_vm_queued_resources_on_demand_uses_no_special_capacity_flags(monkeypatch: pytest.MonkeyPatch):
    create_command = _run_create_with_capacity(monkeypatch, "on-demand")
    assert not _contains_arg(create_command, "--spot")
    assert not _contains_arg(create_command, "--best-effort")
    assert not _contains_arg(create_command, "--reserved")


def test_describe_tpu_workers_flattens_multislice_workers(monkeypatch: pytest.MonkeyPatch):
    descriptions = {
        "pod-0": {"networkEndpoints": [{"ipAddress": "10.0.0.1"}, {"ipAddress": "10.0.0.2"}]},
        "pod-1": {"networkEndpoints": [{"ipAddress": "10.0.1.1"}]},
    }

    def fake_check_output(args: Sequence[str], **_kwargs):
        return json.dumps(descriptions[args[6]]).encode("utf-8")

    monkeypatch.setattr(tpus.subprocess, "check_output", fake_check_output)

    workers = tpus.describe_tpu_workers("pod", "us-central2-b", 2)

    assert [(worker.ordinal, worker.slice_name, worker.worker_index, worker.ip_address) for worker in workers] == [
        (0, "pod-0", 0, "10.0.0.1"),
        (1, "pod-0", 1, "10.0.0.2"),
        (2, "pod-1", 0, "10.0.1.1"),
    ]


def test_ssh_tpu_worker_uses_resolved_slice_and_worker(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        tpus,
        "describe_tpu_workers",
        lambda *_args, **_kwargs: [
            tpus.TpuWorker(ordinal=0, slice_index=0, slice_name="pod-0", worker_index=0, ip_address=None),
            tpus.TpuWorker(ordinal=1, slice_index=0, slice_name="pod-0", worker_index=1, ip_address=None),
            tpus.TpuWorker(ordinal=2, slice_index=1, slice_name="pod-1", worker_index=0, ip_address=None),
        ],
    )
    monkeypatch.setattr(tpus, "add_ssh_key", lambda *_args, **_kwargs: None)

    commands: list[tuple[Any, ...]] = []

    def fake_run_command(*args, **_kwargs):
        commands.append(args)

    monkeypatch.setattr(tpus, "run_command", fake_run_command)

    tpus.ssh_tpu_worker("pod", "us-central2-b", 2, 2, "echo", "hello")

    assert commands == [
        (
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            "pod-1",
            "--quiet",
            "--worker=0",
            "--zone=us-central2-b",
            "--command=echo hello",
        )
    ]


def test_run_container_on_worker_builds_named_docker_command(monkeypatch: pytest.MonkeyPatch):
    docker_calls: list[tuple[str, Sequence[str], dict[str, str], bool, str]] = []
    ssh_calls: list[tuple[Any, ...]] = []

    def fake_make_docker_run_command(image_id, command, *, env, foreground, name):
        docker_calls.append((image_id, tuple(command), env, foreground, name))
        return ["docker", "run", "--name=test", image_id, *command]

    def fake_ssh_tpu_worker(tpu_name, zone, node_count, worker_ordinal, *args, ignore_failure=False):
        ssh_calls.append((tpu_name, zone, node_count, worker_ordinal, args, ignore_failure))

    monkeypatch.setattr(tpus, "make_docker_run_command", fake_make_docker_run_command)
    monkeypatch.setattr(tpus, "ssh_tpu_worker", fake_ssh_tpu_worker)

    tpus.run_container_on_worker(
        tpu_name="pod",
        zone="us-central2-b",
        node_count=1,
        worker_ordinal=0,
        full_image_id="repo/image:tag",
        command=["python", "entry.py"],
        env={"RUN_ID": "abc"},
        foreground=False,
        name="marin-alt-sampler",
    )

    assert docker_calls == [("repo/image:tag", ("python", "entry.py"), {"RUN_ID": "abc"}, False, "marin-alt-sampler")]
    assert ssh_calls == [
        (
            "pod",
            "us-central2-b",
            1,
            0,
            ("docker", "run", "--name=test", "repo/image:tag", "python", "entry.py"),
            False,
        )
    ]


def test_stop_container_on_worker_uses_shell_cleanup_command(monkeypatch: pytest.MonkeyPatch):
    ssh_calls: list[tuple[Any, ...]] = []

    def fake_ssh_tpu_worker(tpu_name, zone, node_count, worker_ordinal, *args, ignore_failure=False):
        ssh_calls.append((tpu_name, zone, node_count, worker_ordinal, args, ignore_failure))

    monkeypatch.setattr(tpus, "ssh_tpu_worker", fake_ssh_tpu_worker)

    tpus.stop_container_on_worker("pod", "us-central2-b", 1, 0, name="marin-alt-sampler")

    assert ssh_calls == [
        (
            "pod",
            "us-central2-b",
            1,
            0,
            ("bash", "-lc", shlex.quote("sudo docker rm -f marin-alt-sampler || true")),
            False,
        )
    ]


def test_container_exists_on_worker_interprets_presence(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(tpus, "add_ssh_key", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        tpus,
        "describe_tpu_workers",
        lambda *_args, **_kwargs: [
            tpus.TpuWorker(ordinal=0, slice_index=0, slice_name="pod", worker_index=0, ip_address=None)
        ],
    )

    outputs = [b"present\n", b"absent\n"]

    def fake_check_output(_args: Sequence[str], **_kwargs):
        return outputs.pop(0)

    monkeypatch.setattr(tpus.subprocess, "check_output", fake_check_output)

    assert tpus.container_exists_on_worker("pod", "us-central2-b", 1, 0, name="marin-alt-sampler") is True
    assert tpus.container_exists_on_worker("pod", "us-central2-b", 1, 0, name="marin-alt-sampler") is False


def test_run_python_on_worker_requires_exactly_one_entrypoint(monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(ValueError, match="Exactly one of module or script_path"):
        tpus.run_python_on_worker(
            "pod",
            "us-central2-b",
            1,
            0,
            python_executable="python",
        )

    with pytest.raises(ValueError, match="Exactly one of module or script_path"):
        tpus.run_python_on_worker(
            "pod",
            "us-central2-b",
            1,
            0,
            python_executable="python",
            module="pkg.entry",
            script_path="entry.py",
        )
