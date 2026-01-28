"""Tests for iris.cluster.types — Entrypoint and EnvironmentSpec."""

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


