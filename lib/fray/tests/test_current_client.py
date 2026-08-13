# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for current_client() and set_current_client()."""

from typing import cast

import pytest
from fray.current_client import current_client, set_current_client
from fray.iris_backend import FrayIrisClient
from fray.local_backend import LocalClient
from iris.client.client import IrisClient, IrisContext, iris_ctx_scope


def test_default_returns_local_client():
    client = current_client()
    assert isinstance(client, LocalClient)


def test_set_current_client_context_manager():
    explicit = LocalClient(max_threads=2)
    with set_current_client(explicit) as c:
        assert c is explicit
        assert current_client() is explicit
    assert current_client() is not explicit


def test_set_current_client_restores_on_exception():
    explicit = LocalClient(max_threads=2)
    with pytest.raises(RuntimeError):
        with set_current_client(explicit):
            raise RuntimeError("boom")
    assert current_client() is not explicit


def test_iris_auto_detection_with_context():
    iris_client = cast(IrisClient, object())
    with iris_ctx_scope(IrisContext(job_id=None, client=iris_client)):
        client = current_client()
    assert isinstance(client, FrayIrisClient)


def test_explicit_client_overrides_auto_detection():
    iris_client = cast(IrisClient, object())
    explicit = LocalClient(max_threads=1)
    with iris_ctx_scope(IrisContext(job_id=None, client=iris_client)):
        with set_current_client(explicit):
            assert current_client() is explicit
