# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the compact JSON codec installed via :mod:`iris.rpc.codecs`."""

from iris.rpc import codecs as iris_codecs
from iris.rpc.job_pb2 import JobStatus


def test_encode_omits_indentation_and_newlines() -> None:
    """Compact codec must produce single-line JSON; the upstream default emits
    pretty-printed JSON via ``MessageToJson`` (which defaults to ``indent=2``)."""
    msg = JobStatus(job_id="/alice/job", state=1)
    encoded = iris_codecs.CompactProtoJSONCodec().encode(msg).decode()
    assert "\n" not in encoded
    assert "  " not in encoded
