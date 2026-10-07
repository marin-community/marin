# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Atlas metadata preserves eligibility and historical provenance."""

from dataclasses import replace

from experiments.post_training.task_curation.sources import atlas_metadata


def test_archive_input_pin_is_separate_from_upstream_lineage():
    archive = atlas_metadata()["Task Trove:DCAgent__code-contests-noblock"]
    upstream_changed = replace(archive, dataset_revision="different-upstream-lineage")
    archive_changed = replace(archive, archive_revision="different-archive-input")
    assert upstream_changed.input_revision == archive.input_revision == archive.archive_revision
    assert archive_changed.input_revision != archive.input_revision
    direct = atlas_metadata()["MarinSkyRL:math500"]
    direct_changed = replace(direct, dataset_revision="different-direct-input")
    assert direct_changed.input_revision != direct.input_revision
    assert direct_changed.input_revision == direct_changed.dataset_revision
