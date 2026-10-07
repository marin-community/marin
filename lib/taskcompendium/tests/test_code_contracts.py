# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Malformed APPS tests are source defects without aborting the source panel."""

import json

import pytest

from taskcompendium.datasets.code_contracts import normalize_apps
from taskcompendium.grader import grader_config
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection, RawRow


def apps_row(encoded_tests: str) -> RawRow:
    return RawRow(
        id="apps-row",
        source=Source(dataset="codeparrot/apps", revision="pinned", row="0", importer_revision="1"),
        data={
            "id": 0,
            "question": "Read two integers and print their sum.",
            "input_output": encoded_tests,
            "solutions": '["print(sum(map(int, input().split())))"]',
            "difficulty": "introductory",
            "url": "https://example.org/problem",
            "starter_code": "",
        },
    )


@pytest.mark.parametrize("encoded_tests", ["", "{", "null", "[]", '{"inputs": "1 2", "outputs": ["3"]}'])
def test_apps_panel_rejects_malformed_tests_and_preserves_valid_private_contract(encoded_tests):
    valid_tests = json.dumps({"inputs": ["1 2\n"], "outputs": ["3\n"]})
    malformed, valid = (normalize_apps(row) for row in (apps_row(encoded_tests), apps_row(valid_tests)))
    assert isinstance(malformed, ImportRejection)
    assert malformed.kind == ImportFailureKind.SOURCE_DEFECT
    assert malformed.reason == "invalid_test_contract"
    assert isinstance(valid, TaskSpec)
    assert valid.context.events[0].content == "Read two integers and print their sum."
    retained = grader_config(valid)
    assert retained["contract"]["input_output"] == valid_tests
