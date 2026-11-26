# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Unit tests under apps/*/tests share the throwaway database.
from collections.abc import Iterator

import pytest
from marina.testing import test_database


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    with test_database() as url:
        yield url
