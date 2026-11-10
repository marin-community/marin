# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import pytest
from marina.testing import test_database

from infra.marina.tests.rl_progress_fixtures import rl_progress_client as rl_progress_client


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    with test_database() as url:
        yield url
