# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mode csv-columns: the answer must be CSV whose header names the fields the task asks for.

The document is read with ``csv.reader``, rows whose cells are all blank are dropped, and a header
row plus at least one data row must remain. ``required`` names must all appear as headers;
``any_of`` names, used where the task marks nothing required, need one hit. The mode checks header
structure and the presence of data rows. It does not inspect cell values.
"""

import csv
import io
from pathlib import Path

from tasktrove_verify.grade import InvalidTask, Reward, read_output, scored
from tasktrove_verify.modes.extract import unwrap_fence
from tasktrove_verify.spec import CsvColumnsSpec

MAX_REPORTED_NAMES = 8
MIN_ROWS = 2
"""A header row and one data row: a header alone carries no answer."""


def data_rows(text: str) -> list[list[str]]:
    """The CSV rows that hold at least one non-blank cell. Raises ``csv.Error`` on unreadable text."""
    return [row for row in csv.reader(io.StringIO(text)) if any(cell.strip() for cell in row)]


def grade(spec: CsvColumnsSpec, tests_dir: Path, workspace: Path) -> Reward:
    if not spec.required and not spec.any_of:
        raise InvalidTask("csv-columns expects required or any_of names")

    text = read_output(spec, workspace)
    if text is None:
        return scored(0.0, reason="no_output")
    try:
        rows = data_rows(unwrap_fence(text).strip())
    except csv.Error as error:
        return scored(0.0, reason="parse_error", error=str(error))
    if len(rows) < MIN_ROWS:
        return scored(0.0, reason="too_few_rows", rows=len(rows))

    headers = {cell.strip() for cell in rows[0]}
    missing = [name for name in spec.required if name not in headers]
    if missing:
        return scored(0.0, reason="missing_columns", missing=missing[:MAX_REPORTED_NAMES])
    if spec.any_of and headers.isdisjoint(spec.any_of):
        return scored(0.0, reason="no_expected_column", expected=list(spec.any_of[:MAX_REPORTED_NAMES]))
    return scored(1.0, reason="columns_present", rows=len(rows) - 1)
