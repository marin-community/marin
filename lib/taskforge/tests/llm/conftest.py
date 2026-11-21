# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the GLM transport tests.

``fake_glm`` and ``other_fake_glm`` are independent scripted GLM routers (``fake_glm.FakeGlm``), the
second for an endpoint that moves. ``fast_retry`` is a retry policy whose backoff is a millisecond, so
retries and holds are quick.
"""

from collections.abc import Iterator

import pytest
from lib.taskforge.tests.llm.fake_glm import FakeGlm, serving

from taskforge.llm.policy import RetryPolicy


@pytest.fixture
def fast_retry() -> RetryPolicy:
    return RetryPolicy(unavailable_wait=60.0, backoff_initial=0.001, backoff_maximum=0.001, backoff_factor=1.0)


@pytest.fixture
def fake_glm() -> Iterator[FakeGlm]:
    with serving() as fake:
        yield fake


@pytest.fixture
def other_fake_glm() -> Iterator[FakeGlm]:
    with serving() as fake:
        yield fake
