# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import pytest
from marina.testing import test_database


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    with test_database() as url:
        yield url


@pytest.fixture
def rl_progress_client(tmp_path, database_url, monkeypatch):
    from infra.marina.tests import rl_progress_fixtures  # noqa: PLC0415

    yield from rl_progress_fixtures.rl_progress_client(tmp_path, database_url, monkeypatch)
