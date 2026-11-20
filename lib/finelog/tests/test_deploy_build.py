# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finelog image source-revision tests."""

from finelog.deploy.build import finelog_source_build_args
from rigging.provenance import Provenance


def test_finelog_source_build_args_match_dockerfile_contract() -> None:
    provenance = Provenance(
        tree_hash="0123456",
        base_commit="89abcde",
        dirty=True,
        branch="main",
        built_by="finelog-test",
    )

    assert finelog_source_build_args(provenance) == {
        "SOURCE_COMMIT": "89abcde",
        "SOURCE_TREE": "0123456",
        "SOURCE_DIRTY": "true",
    }
