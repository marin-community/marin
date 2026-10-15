# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The release tests read sources that exist only in the research checkout: the locally excluded reference outputs and
the sibling mixture-selection repository. Without them the whole package skips; each module keeps its own guard."""

import pytest

from experiments.domain_phase_mix.release import paths


@pytest.fixture(scope="session", autouse=True)
def release_sources():
    if not (paths.RO.exists() and paths.MIXSEL.exists()):
        pytest.skip("release sources absent: reference_outputs or the mixture-selection checkout")
