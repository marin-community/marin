# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training.snowball_tests import fixtures
from infra.marina.tests.rl_progress_fixtures import rl_progress_client as rl_progress_client

selection = pytest.fixture(fixtures.selection)
source_provenance = pytest.fixture(fixtures.source_provenance)
recorded_runs = pytest.fixture(fixtures.recorded_runs)
