# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the bridge tests."""

import json

import pyarrow as pa


def result_table(rows: list[tuple[str, float, dict[str, str], int]]) -> pa.Table:
    """Build a table shaped like the bridge's SELECT: (metric, value, labels, collected_us).

    ``labels`` is JSON-encoded the way infra/probes writes it, and ``collected_us``
    is epoch microseconds the way an arrow_cast of finelog's timestamp column
    comes back.
    """
    return pa.table(
        {
            "metric": [r[0] for r in rows],
            "value": [r[1] for r in rows],
            "labels": [json.dumps(r[2], sort_keys=True) for r in rows],
            "collected_us": [r[3] for r in rows],
        }
    )
