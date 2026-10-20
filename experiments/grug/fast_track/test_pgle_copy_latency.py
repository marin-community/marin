# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``pgle_copy_latency`` turns a profile's large ``copy-start`` costs into start-to-done latencies, decoded here with
XLA's ``ProfiledInstructionsProto`` schema, and leaves every other entry as it was."""

from pathlib import Path

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from experiments.grug.fast_track.pgle_copy_latency import COPY_START_COST_US, copy_latency_profile

_PGLE = Path(__file__).parent / "pgle" / "mfl-hm-ow.pb"


def _profile_class():
    """``tensorflow.profiler.ProfiledInstructionsProto`` (tsl/profiler/protobuf/profiled_instructions.proto)."""
    f = descriptor_pb2.FileDescriptorProto(name="profiled_instructions_test.proto", package="tensorflow.profiler")
    top = f.message_type.add(name="ProfiledInstructionsProto")
    cost = top.nested_type.add(name="InstructionCost")
    cost.field.add(name="name", number=1, type=9, label=1)
    cost.field.add(name="cost_us", number=2, type=1, label=1)
    latency = top.nested_type.add(name="Latency")
    latency.field.add(name="source", number=1, type=9, label=1)
    latency.field.add(name="target", number=2, type=9, label=1)
    latency.field.add(name="latency_us", number=3, type=1, label=1)
    prefix = ".tensorflow.profiler.ProfiledInstructionsProto."
    top.field.add(name="costs", number=1, type=11, label=3, type_name=prefix + "InstructionCost")
    top.field.add(name="latencies", number=2, type=11, label=3, type_name=prefix + "Latency")
    pool = descriptor_pool.DescriptorPool()
    pool.Add(f)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("tensorflow.profiler.ProfiledInstructionsProto"))


def test_large_copy_costs_become_latencies():
    cls = _profile_class()
    raw = _PGLE.read_bytes()
    before = cls.FromString(raw)
    rewritten, moved = copy_latency_profile(raw, min_us=50_000)
    after = cls.FromString(rewritten)

    old_costs = {c.name: c.cost_us for c in before.costs}
    assert {name for name, _ in moved} == {"copy-start.40.0", "copy-start.41.0", "copy-start.114.0", "copy-start.115.0"}
    assert [(lat.source, lat.target, lat.latency_us) for lat in after.latencies] == [
        (name, name.replace("copy-start", "copy-done"), old_costs[name]) for name, _ in moved
    ]
    new_costs = {c.name: c.cost_us for c in after.costs}
    assert new_costs.keys() == old_costs.keys()
    for name, cost in old_costs.items():
        assert new_costs[name] == (COPY_START_COST_US if name in dict(moved) else cost)
