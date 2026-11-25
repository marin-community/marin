# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from lib.taskforge.tests.proposal_fakes import sample_proposal

from taskforge.model.proposal import TaskProposal


@pytest.fixture
def proposal() -> TaskProposal:
    return sample_proposal()
