# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The structural-closure cohorts declare their first retained release pins."""

from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.structure_closure import (
    CONVERTER_VERSION,
    RELEASES,
    sources,
)


def test_structure_closure_declares_the_retained_release_pins():
    declared = {source.info.id: source for source in sources()}
    assert set(declared) == {"OpenAthena:structure-closure", "OpenAthena:structure-closure/validation"}
    for _split, (release, revision, rows, intended_use) in RELEASES.items():
        source = declared["OpenAthena:structure-closure" + ("/validation" if intended_use == IntendedUse.EVAL else "")]
        config = source.config
        assert config is not None
        assert (config.repo, config.revision, config.expected_rows) == (release, revision, rows)
        assert config.version == CONVERTER_VERSION
        assert config.intended_use == intended_use
        assert source.pipeline is not None
        assert source.dataset.revision == revision
        assert source.info.count == rows


def test_validation_split_is_a_separate_eval_population():
    train, validation = {source.info.id: source for source in sources()}.values()
    assert train.info.subset is None and train.config.intended_use == IntendedUse.TRAIN
    assert validation.info.subset == "validation" and validation.config.intended_use == IntendedUse.EVAL
    assert train.dataset.name != validation.dataset.name
