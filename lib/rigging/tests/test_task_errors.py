# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from rigging.task_errors import error_category


@pytest.mark.parametrize(
    "name, expected",
    json.loads((Path(__file__).parent / "fixtures/task_error_categories.json").read_text()).items(),
)
def test_existing_error_category_is_preserved(name, expected):
    assert error_category(name).value == expected
