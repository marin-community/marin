# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``CachedProto`` TypeDecorator on the SA Core ``schema``."""

from iris.cluster.controller.schema import CachedProto
from iris.rpc.job_pb2 import LoginRequest


def _make_login_blob(token: str) -> bytes:
    msg = LoginRequest()
    msg.identity_token = token
    return msg.SerializeToString()


def test_bind_param_serializes_proto():
    decoder = CachedProto(LoginRequest)
    msg = LoginRequest()
    msg.identity_token = "abc"
    assert decoder.process_bind_param(msg, None) == msg.SerializeToString()


def test_result_value_round_trips_proto():
    decoder = CachedProto(LoginRequest)
    blob = _make_login_blob("hello")
    decoded = decoder.process_result_value(blob, None)
    assert decoded is not None
    assert decoded.identity_token == "hello"
