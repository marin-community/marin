# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Journeys under apps/*/journeys use the kernel and browser fixtures; unit tests under
# apps/*/tests share the throwaway database.
pytest_plugins = ["marina.journey_plugin"]

from collections.abc import Iterator  # noqa: E402

import pytest  # noqa: E402
from marina.testing import test_database  # noqa: E402


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    with test_database() as url:
        yield url
