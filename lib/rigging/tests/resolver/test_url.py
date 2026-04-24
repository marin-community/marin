# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure-parsing tests for :class:`rigging.resolver.ServiceURL`."""

import pytest

from rigging.resolver import ServiceURL


def test_parse_iris_with_endpoint_query():
    url = ServiceURL.parse("iris://marin?endpoint=/system/log-server")
    assert url.scheme == "iris"
    assert url.host == "marin"
    assert url.query == {"endpoint": "/system/log-server"}


def test_parse_gcp_no_query():
    url = ServiceURL.parse("gcp://log-server")
    assert url.scheme == "gcp"
    assert url.host == "log-server"
    assert url.query == {}


def test_parse_missing_scheme_raises():
    with pytest.raises(ValueError, match="missing scheme"):
        ServiceURL.parse("marin?endpoint=/x")


def test_parse_missing_host_raises():
    with pytest.raises(ValueError, match="missing host"):
        ServiceURL.parse("iris://")


def test_query_missing_key_absent_not_none():
    url = ServiceURL.parse("iris://marin")
    # Plan §url.py: missing key must be absent from the dict, not present-as-None.
    assert "endpoint" not in url.query
    with pytest.raises(KeyError):
        _ = url.query["endpoint"]


def test_query_duplicate_key_first_wins():
    url = ServiceURL.parse("iris://marin?endpoint=/a&endpoint=/b")
    # We don't have multi-value query semantics; first occurrence wins.
    assert url.query == {"endpoint": "/a"}


def test_query_multiple_distinct_keys():
    url = ServiceURL.parse("gcp://vm?zone=us-central1-a&port=10002")
    assert url.query == {"zone": "us-central1-a", "port": "10002"}
