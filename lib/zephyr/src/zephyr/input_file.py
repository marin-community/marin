# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Input-file specifications shared by datasets, plans, and readers."""

from dataclasses import dataclass
from typing import Literal

from zephyr.expr import Expr

DEFAULT_FILE_PATH_COLUMN = "__file_path"


@dataclass
class InputFileSpec:
    """Describe a file or row range for a Zephyr reader."""

    path: str
    format: Literal["parquet", "jsonl", "vortex", "auto"] = "auto"
    columns: list[str] | None = None
    row_start: int | None = None
    row_end: int | None = None
    filter_expr: Expr | None = None
