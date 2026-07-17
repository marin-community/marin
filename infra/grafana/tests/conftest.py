# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the bridge tests."""

import pyarrow as pa


def finelog_result(**columns: list) -> pa.Table:
    """Build an Arrow table from column name -> values, shaped like a finelog query result.

    Types are inferred, so a list of ``datetime`` becomes a timestamp column and a
    list of JSON strings becomes the ``labels`` column the bridge flattens.
    """
    return pa.table(dict(columns))
