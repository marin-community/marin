# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.interp.snowball_identity.harness import SnowballHarness
from experiments.interp.snowball_identity.tiny_model import tiny_models


@pytest.fixture(scope="module")
def models():
    return tiny_models()


@pytest.fixture(scope="module")
def harness(models) -> SnowballHarness:
    return models[0]
