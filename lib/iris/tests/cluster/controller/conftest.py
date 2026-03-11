# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from iris.cluster.controller.transitions import ControllerTransitions


@pytest.fixture
def make_transitions():
    """Factory fixture that creates ControllerTransitions and closes them on teardown."""
    created: list[ControllerTransitions] = []

    def _make(**kwargs) -> ControllerTransitions:
        t = ControllerTransitions(**kwargs)
        created.append(t)
        return t

    yield _make
    for t in created:
        t.close()
