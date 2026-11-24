# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generated source identity: ids and titles derive from the upstream keys."""

import pytest
from finestore.rl.curation import SourceOrigin, source_identifier, source_title
from finestore.rl.status_taxonomy import DataRestriction

from experiments.post_training.task_curation.source import SourceInfo


def source(**overrides) -> SourceInfo:
    fields = {
        "data_restriction": DataRestriction.OPEN,
        "upstream_dataset": "math",
        "upstream_name": "math",
        "build_path": "test",
        "origin": SourceOrigin.MARIN,
    }
    return SourceInfo(**{**fields, **overrides})


def test_id_and_title_are_generated_from_the_upstream_identity():
    info = source(subset="validation")
    assert info.id == source_identifier(SourceOrigin.MARIN, "math", "validation", None)
    assert info.title == source_title("math", "validation", None)


def test_a_variant_changes_the_generated_identity():
    info = source(upstream_dataset="hardmath", upstream_name="hardmath", variant="rhs")
    assert info.id == "Marin:hardmath@rhs"
    assert info.title == "hardmath · rhs"


def test_the_origin_must_be_a_source_origin():
    with pytest.raises(ValueError, match="upstream SourceOrigin"):
        source(origin="Marin")
