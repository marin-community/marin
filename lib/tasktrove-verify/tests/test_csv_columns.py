# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import csv

import pytest
from tasktrove_verify.modes import csv_columns
from tasktrove_verify.reward import InvalidTask, Status
from tasktrove_verify.spec import CsvColumnsSpec

ORDERS = 'name,email,quantity\nAda,ada@example.com,3\n"Grace, C.",grace@example.com,1\n'


@pytest.fixture
def workspace(tmp_path):
    directory = tmp_path / "app"
    directory.mkdir()
    return directory


def answer(workspace, text):
    (workspace / "answer.txt").write_text(text)


def grade(workspace, spec):
    return csv_columns.grade(spec, workspace.parent / "tests", workspace)


def test_header_with_every_required_column_scores_one(workspace):
    answer(workspace, ORDERS)
    reward = grade(workspace, CsvColumnsSpec(required=("name", "email", "quantity")))
    assert (reward.reward, reward.status) == (1.0, Status.SCORED)
    assert reward.detail["rows"] == 2


def test_missing_required_column_scores_zero_and_names_it(workspace):
    answer(workspace, ORDERS)
    reward = grade(workspace, CsvColumnsSpec(required=("name", "phone")))
    assert reward.reward == 0.0
    assert reward.detail["missing"] == ["phone"]


def test_header_alone_scores_zero(workspace):
    answer(workspace, "name,email,quantity\n")
    reward = grade(workspace, CsvColumnsSpec(required=("name",)))
    assert reward.reward == 0.0
    assert reward.detail["reason"] == "too_few_rows"


def test_blank_rows_do_not_count_as_data(workspace):
    answer(workspace, "name,email\n\n,\n")
    assert grade(workspace, CsvColumnsSpec(required=("name",))).reward == 0.0


def test_surrounding_whitespace_in_a_header_cell_is_ignored(workspace):
    answer(workspace, " name , email \nAda,ada@example.com\n")
    assert grade(workspace, CsvColumnsSpec(required=("name", "email"))).reward == 1.0


def test_document_inside_a_code_fence_is_unwrapped(workspace):
    answer(workspace, f"Here is the table:\n\n```csv\n{ORDERS}```\n")
    assert grade(workspace, CsvColumnsSpec(required=("name",))).reward == 1.0


def test_any_of_needs_one_column_present(workspace):
    answer(workspace, ORDERS)
    assert grade(workspace, CsvColumnsSpec(any_of=("quantity", "phone"))).reward == 1.0
    reward = grade(workspace, CsvColumnsSpec(any_of=("phone", "address")))
    assert reward.reward == 0.0
    assert reward.detail["expected"] == ["phone", "address"]


def test_prose_that_is_not_a_table_scores_zero(workspace):
    answer(workspace, "I could not produce the table, sorry.")
    assert grade(workspace, CsvColumnsSpec(required=("name",))).reward == 0.0


def test_unreadable_text_scores_zero_without_raising(workspace):
    """An unterminated quote swallows the rest of the answer and trips csv's field size limit."""
    answer(workspace, 'name,email\nAda,"' + "x" * (csv.field_size_limit() + 1))
    reward = grade(workspace, CsvColumnsSpec(required=("name",)))
    assert (reward.reward, reward.status) == (0.0, Status.SCORED)
    assert reward.detail["reason"] == "parse_error"


@pytest.mark.parametrize("text", [None, "", "   \n\n"])
def test_absent_or_blank_output_scores_zero_with_no_output(workspace, text):
    if text is not None:
        answer(workspace, text)
    reward = grade(workspace, CsvColumnsSpec(required=("name",)))
    assert reward.reward == 0.0
    assert reward.detail == {"reason": "no_output"}


def test_spec_naming_nothing_is_an_invalid_task(workspace):
    answer(workspace, ORDERS)
    with pytest.raises(InvalidTask, match="required or any_of"):
        grade(workspace, CsvColumnsSpec())
