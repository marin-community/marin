# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SQL string literal encoding for Finelog queries."""


def sql_string(value: str) -> str:
    """Quote a string value as a SQL literal."""
    return "'" + value.replace("'", "''") + "'"
