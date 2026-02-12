# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for iris.cluster.types — Entrypoint, EnvironmentSpec, and constraint helpers."""

import pytest

from iris.cluster.types import Entrypoint, JobName


def _add(a, b):
    return a + b


def test_entrypoint_from_callable_resolve_roundtrip():
    ep = Entrypoint.from_callable(_add, 3, b=4)
    fn, args, kwargs = ep.resolve()
    assert fn(*args, **kwargs) == 7


def test_entrypoint_proto_roundtrip_preserves_bytes():
    """Bytes survive to_proto -> from_proto without deserialization."""
    ep = Entrypoint.from_callable(_add, 1, 2)
    original_files = ep.workdir_files

    proto = ep.to_proto()
    ep2 = Entrypoint.from_proto(proto)

    assert ep2.workdir_files == original_files
    fn, args, kwargs = ep2.resolve()
    assert fn(*args, **kwargs) == 3


def test_entrypoint_command():
    ep = Entrypoint.from_command("echo", "hello")
    assert not ep.workdir_files
    assert ep.command == ["echo", "hello"]


def test_entrypoint_callable_has_workdir_files():
    ep = Entrypoint.from_callable(_add, 1, 2)
    assert "_callable.pkl" in ep.workdir_files
    assert "_callable_runner.py" in ep.workdir_files
    assert ep.command is not None


def test_job_name_roundtrip_and_hierarchy():
    job = JobName.root("root")
    child = job.child("child")
    task = child.task(0)

    assert str(job) == "/root"
    assert str(child) == "/root/child"
    assert str(task) == "/root/child/0"
    assert task.parent == child
    assert child.parent == job
    assert job.parent is None

    parsed = JobName.from_string("/root/child/0")
    assert parsed == task
    assert parsed.namespace == "root"
    assert parsed.is_task
    assert parsed.task_index == 0
    assert JobName.root("root").is_ancestor_of(parsed)
    assert not parsed.is_ancestor_of(JobName.root("root"), include_self=False)


@pytest.mark.parametrize(
    "value",
    ["", "root", "/root//child", "/root/ ", "/root/child/", "/root/child//0"],
)
def test_job_name_rejects_invalid_inputs(value: str):
    with pytest.raises(ValueError):
        JobName.from_string(value)


def test_job_name_require_task_errors_on_non_task():
    with pytest.raises(ValueError):
        JobName.from_string("/root/child").require_task()


def test_job_name_to_safe_token_and_deep_nesting():
    job = JobName.from_string("/a/b/c/d/e/0")
    assert job.to_safe_token() == "job__a__b__c__d__e__0"
    assert job.require_task()[1] == 0
