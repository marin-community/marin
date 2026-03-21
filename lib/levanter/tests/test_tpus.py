# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

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
