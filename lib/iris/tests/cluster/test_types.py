# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    original_bytes = ep.callable_bytes

    proto = ep.to_proto()
    ep2 = Entrypoint.from_proto(proto)

    assert ep2.callable_bytes == original_bytes
    fn, args, kwargs = ep2.resolve()
    assert fn(*args, **kwargs) == 3


def test_entrypoint_command():
    ep = Entrypoint.from_command("echo", "hello")
    assert ep.is_command
    assert not ep.is_callable
    assert ep.command == ["echo", "hello"]


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
