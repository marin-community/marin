# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generated source identity: aliases resolve, mismatches fail validation."""

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


def test_a_declared_historical_id_resolves_through_the_alias_table():
    info = source(
        id="Task Trove:laion__codeforces-v3",
        upstream_dataset="laion__codeforces-v3",
        upstream_name="laion__codeforces-v3",
        build_path="Task Trove",
        origin=SourceOrigin.TASKTROVE,
    )
    assert info.id == "TaskTrove:laion__codeforces-v3"


def test_a_declared_id_outside_the_generated_identity_is_rejected():
    with pytest.raises(ValueError, match="generated identity"):
        source(id="Marin:other")


def test_a_declared_title_outside_the_generated_title_is_rejected():
    with pytest.raises(ValueError, match="generated title"):
        source(title="Other")
