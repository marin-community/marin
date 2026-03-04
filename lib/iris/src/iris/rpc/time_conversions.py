# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Conversions between rigging time types and iris protobuf messages."""

from iris.rpc import time_pb2
from rigging.time_utils import Duration, Timestamp


def duration_to_proto(d: Duration) -> time_pb2.Duration:
    """Convert a Duration to a proto Duration message."""
    return time_pb2.Duration(milliseconds=d.to_ms())


def duration_from_proto(proto: time_pb2.Duration) -> Duration:
    """Create a Duration from a proto Duration message."""
    return Duration.from_ms(proto.milliseconds)


def timestamp_to_proto(ts: Timestamp) -> time_pb2.Timestamp:
    """Convert a Timestamp to a proto Timestamp message."""
    return time_pb2.Timestamp(epoch_ms=ts.epoch_ms())


def timestamp_from_proto(proto: time_pb2.Timestamp) -> Timestamp:
    """Create a Timestamp from a proto Timestamp message."""
    return Timestamp.from_ms(proto.epoch_ms)
