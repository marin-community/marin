# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build an XLA:GPU PGLE latency profile from one XProf trace of the hero train step.

Pass the result with ``--xla_gpu_pgle_profile_file_or_directory_path`` to schedule the same
program with measured instruction costs. XLA's default estimator gives every collective the same
small latency, so it hides a multi-millisecond ragged transport under a single GEMM. The profile
matches instructions by name, so regenerate it whenever the compiled program changes.

    uv run python -m experiments.grug.moe_hero_ep.pgle_profile <xplane.pb URI> <out.pbtxt>
"""

import os
import sys
import tempfile

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory, text_format
from jax.experimental import profiler
from rigging.filesystem.buckets import filesystem_for


def _profile_message_class():
    """``tensorflow.profiler.ProfiledInstructionsProto``, which jaxlib does not export."""
    proto = descriptor_pb2.FileDescriptorProto(name="profiled_instructions.proto", package="tensorflow.profiler")
    message = proto.message_type.add(name="ProfiledInstructionsProto")
    string, double, optional, repeated = (
        descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
        descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE,
        descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
        descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
    )
    cost = message.nested_type.add(name="InstructionCost")
    cost.field.add(name="name", number=1, type=string, label=optional)
    cost.field.add(name="cost_us", number=2, type=double, label=optional)
    latency = message.nested_type.add(name="Latency")
    latency.field.add(name="source", number=1, type=string, label=optional)
    latency.field.add(name="target", number=2, type=string, label=optional)
    latency.field.add(name="latency_us", number=3, type=double, label=optional)
    message_ref = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    message.field.add(
        name="costs",
        number=1,
        type=message_ref,
        label=repeated,
        type_name=".tensorflow.profiler.ProfiledInstructionsProto.InstructionCost",
    )
    message.field.add(
        name="latencies",
        number=2,
        type=message_ref,
        label=repeated,
        type_name=".tensorflow.profiler.ProfiledInstructionsProto.Latency",
    )
    pool = descriptor_pool.DescriptorPool()
    pool.Add(proto)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("tensorflow.profiler.ProfiledInstructionsProto"))


def main(xplane_uri: str, out_path: str) -> None:
    with tempfile.TemporaryDirectory() as root:
        # The converter reads a TensorBoard profile directory: <run>/*.xplane.pb.
        run_dir = os.path.join(root, "plugins", "profile", "run")
        os.makedirs(run_dir)
        # Route by the bucket's declared backend, so CoreWeave and R2 buckets get their credentials.
        fs, path = filesystem_for(xplane_uri)
        fs.get(path, os.path.join(run_dir, "trace.xplane.pb"))
        serialized = profiler.get_profiled_instructions_proto(run_dir)
    profile = _profile_message_class()()
    profile.ParseFromString(serialized)
    if not profile.costs:
        raise ValueError(f"no GPU instruction costs in {xplane_uri}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(text_format.MessageToString(profile))
    print(f"wrote {len(profile.costs)} instruction costs to {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
