# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared platform resource helpers."""

from pathlib import Path

import pytest
from iris.cluster.platforms.types import find_free_port, reserve_free_port


def test_reserve_free_port_releases_lock_on_exit_and_error():
    start = find_free_port()

    with reserve_free_port(start) as port:
        lock = Path(f"/tmp/iris/port_{port}")
        assert lock.exists()
    assert not lock.exists()

    with pytest.raises(RuntimeError):
        with reserve_free_port(start) as port:
            lock = Path(f"/tmp/iris/port_{port}")
            assert lock.exists()
            raise RuntimeError("tunnel failed")
    assert not lock.exists()
