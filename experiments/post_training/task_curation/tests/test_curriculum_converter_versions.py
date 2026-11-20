# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The curriculum converters' declared versions match their measured census receipts."""

import pytest

from experiments.post_training.task_curation.datasets.daytona_measurements import actor_census_receipts
from experiments.post_training.task_curation.datasets.tasktrove.python_tests import SOURCES, sources


@pytest.mark.data_integration
def test_every_curriculum_converter_version_matches_its_census_receipt():
    """The code's converter version is the authority (ruling 2026-11-19).

    curriculum-easy uses the PythonTestsSource default (v5): a census receipt measured at
    another version (the v6 re-measure on the repair-seal branch's pointers) was taken
    with the wrong converter and is superseded by the target's pointer state, which pins
    the v5 receipt.
    """
    receipts = actor_census_receipts()
    assert receipts, "the census receipts must resolve"
    checked = 0
    for source in sources():
        census = receipts.get(source.info.id)
        assert census is not None, f"no census receipt for {source.info.id}"
        assert census.input.converter_version == source.version, (
            f"{source.info.id}: the census receipt measured converter_version "
            f"{census.input.converter_version} but the converter code declares {source.version}"
        )
        checked += 1
    assert checked >= len(SOURCES)
