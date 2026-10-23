# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from rigging.filesystem.path_validation import validate_relative_file_paths


@pytest.mark.parametrize(
    "paths,collision",
    [
        (["tests/a", "tests/b", "tests/a"], "tests/a"),
        (["tests/a", "tests/b", "tests"], "tests"),
        (["tests", "tests-extra/a", "tests/a"], "tests/a"),
        (["tests/a/b", "tests/a/c", "tests/a"], "tests/a"),
    ],
)
def test_file_collisions_reject_the_conflicting_path_after_valid_siblings(paths, collision):
    with pytest.raises(ValueError, match=f"^Path collision: {collision}$"):
        validate_relative_file_paths(iter(paths))
