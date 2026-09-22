# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for repository-qualified Echo file identities."""

import pytest

from echo import repository_identity, search_config


def test_same_path_references_are_distinct_and_round_trip():
    marin, vllm = search_config.REPOSITORY_TARGETS[:2]

    references = [repository_identity.repository_file_reference(target, "README.md") for target in (marin, vllm)]

    assert [reference.result_id for reference in references] == [
        "file:marin-community/marin@main:README.md",
        "file:marin-community/vllm@main:README.md",
    ]
    assert references[0] != references[1]
    assert [repository_identity.parse_repository_file_id(reference.result_id) for reference in references] == references


@pytest.mark.parametrize(
    "value",
    [
        "file:README.md",
        "file:marin-community/vllm@dev:README.md",
        "file:marin-community/unknown@main:README.md",
        "wiki:marin-community/marin@main:README.md",
        "file:marin-community/marin@main:../README.md",
    ],
)
def test_file_identity_rejects_unqualified_or_unconfigured_targets(value):
    with pytest.raises(ValueError):
        repository_identity.parse_repository_file_id(value)
