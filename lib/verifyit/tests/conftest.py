# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest


@pytest.fixture
def importable_grading_modules(monkeypatch):
    # Spawned grading workers must import the trusted test scorer package.
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
