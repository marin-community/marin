# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Proto conversion helpers for rigging timing types.

These functions bridge rigging's Duration/Timestamp (which have no proto
dependency) with iris's time_pb2 proto messages.
"""

from iris.rpc import time_pb2
from rigging.timing import Duration, Timestamp


def duration_to_proto(d: Duration) -> time_pb2.Duration:
    return time_pb2.Duration(milliseconds=d.to_ms())


def duration_from_proto(proto: time_pb2.Duration) -> Duration:
    return Duration(proto.milliseconds)


def timestamp_to_proto(ts: Timestamp) -> time_pb2.Timestamp:
    return time_pb2.Timestamp(epoch_ms=ts.epoch_ms())


def timestamp_from_proto(proto: time_pb2.Timestamp) -> Timestamp:
    return Timestamp(proto.epoch_ms)
