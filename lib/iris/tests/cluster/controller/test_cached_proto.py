# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``CachedProto`` TypeDecorator on the SA Core ``schema``."""

from iris.cluster.controller.schema import CachedProto
from iris.rpc.job_pb2 import GetCurrentUserResponse


def _make_user_blob(user_id: str) -> bytes:
    msg = GetCurrentUserResponse()
    msg.user_id = user_id
    return msg.SerializeToString()


def test_bind_param_serializes_proto():
    decoder = CachedProto(GetCurrentUserResponse)
    msg = GetCurrentUserResponse()
    msg.user_id = "abc"
    assert decoder.process_bind_param(msg, None) == msg.SerializeToString()


def test_result_value_round_trips_proto():
    decoder = CachedProto(GetCurrentUserResponse)
    blob = _make_user_blob("hello")
    decoded = decoder.process_result_value(blob, None)
    assert decoded is not None
    assert decoded.user_id == "hello"
