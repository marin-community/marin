# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the compact JSON codec installed via :mod:`iris.rpc.codecs`."""

from __future__ import annotations

from connectrpc._codec import (
    CODEC_NAME_JSON,
    CODEC_NAME_JSON_CHARSET_UTF8,
    get_codec,
    get_proto_json_codec,
)
from iris.rpc import codecs as iris_codecs
from iris.rpc.job_pb2 import JobStatus


def test_compact_codec_replaces_default_singletons() -> None:
    """Importing iris.rpc.codecs swaps the connectrpc-registered JSON codec."""
    assert isinstance(get_proto_json_codec(), iris_codecs.CompactProtoJSONCodec)
    assert isinstance(get_codec(CODEC_NAME_JSON), iris_codecs.CompactProtoJSONCodec)
    assert isinstance(get_codec(CODEC_NAME_JSON_CHARSET_UTF8), iris_codecs.CompactProtoJSONCodec)


def test_encode_omits_indentation_and_newlines() -> None:
    """Compact codec must produce single-line JSON; the upstream default emits
    pretty-printed JSON via ``MessageToJson`` (which defaults to ``indent=2``)."""
    msg = JobStatus(job_id="/alice/job", state=1)
    encoded = iris_codecs.CompactProtoJSONCodec().encode(msg).decode()
    assert "\n" not in encoded
    assert "  " not in encoded


def test_install_compact_json_codec_is_idempotent() -> None:
    """Calling install twice doesn't replace the singleton with a fresh instance."""
    first = get_proto_json_codec()
    iris_codecs.install_compact_json_codec()
    iris_codecs.install_compact_json_codec()
    assert get_proto_json_codec() is first
