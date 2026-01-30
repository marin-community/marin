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

from iris.cluster.types import Entrypoint


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
