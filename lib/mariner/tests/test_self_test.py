# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from mariner.cli import main


@pytest.mark.data_integration
@pytest.mark.timeout(900)
def test_self_test_passes_on_release_snapshot(snapshot):
    assert main(["self-test", "--data", str(snapshot)]) == 0
