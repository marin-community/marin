# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shape a finelog query result for Grafana's Infinity datasource.

The bridge passes SQL straight to finelog, which gates it to SELECT and enforces
a server-side deadline. The only reshaping needed on the way back:

- turn the Arrow result into JSON rows, since Infinity reads JSON;
- render timestamps as epoch milliseconds, which is what Grafana plots;
- flatten the EAV ``labels`` column into ``label_<key>`` fields, since DataFusion
  has no JSON functions and a panel cannot group by a label in SQL.

Panels reference the window through the ``{{from}}`` / ``{{to}}`` macros rather
than embedding the absolute millis Grafana would interpolate, so a relative range
keeps one cache key as its edges drift between refreshes.
"""

import json
import logging
from datetime import UTC, datetime

import pyarrow as pa

logger = logging.getLogger(__name__)

# Window macros a panel writes into its SQL; the bridge substitutes tz-naive UTC
# TIMESTAMP literals before running the query.
FROM_MACRO = "{{from}}"
TO_MACRO = "{{to}}"

# infra/probes writes its label set as a JSON object string. The bridge expands it
# into columns named with this prefix so a panel can select one as its series.
LABELS_COLUMN = "labels"
LABEL_PREFIX = "label_"


def sql_timestamp(at: datetime) -> str:
    """Format ``at`` as the tz-naive UTC literal finelog's timestamp columns compare against.

    finelog stores timestamps tz-naive in UTC, so a tz-aware literal raises a
    comparison error in the engine rather than returning nothing.
    """
    return at.strftime("%Y-%m-%d %H:%M:%S")


def substitute_time_macros(sql: str, start: datetime | None, end: datetime | None) -> str:
    """Replace ``{{from}}`` / ``{{to}}`` with TIMESTAMP literals.

    Raises ``ValueError`` if the SQL uses a macro without the matching bound, so a
    panel that forgets to pass the window fails loudly instead of scanning
    everything.
    """
    for macro, at in ((FROM_MACRO, start), (TO_MACRO, end)):
        if macro in sql:
            if at is None:
                raise ValueError(f"SQL uses {macro} but no matching time bound was supplied")
            sql = sql.replace(macro, f"TIMESTAMP '{sql_timestamp(at)}'")
    return sql


def _json_safe(value: object) -> object:
    """Coerce one Arrow cell into a JSON-serializable value.

    Timestamps become epoch milliseconds (naive cells are read as UTC); bytes
    become text. Everything else passes through.
    """
    if isinstance(value, datetime):
        at = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return round(at.timestamp() * 1000)
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", "replace")
    return value


def _flatten_labels(row: dict[str, object]) -> dict[str, object]:
    """Expand a JSON ``labels`` cell into ``label_<key>`` fields, dropping the raw cell.

    A cell that is not a JSON object is left in place and logged; one malformed
    row is schema drift, not a reason to fail the query.
    """
    raw = row.get(LABELS_COLUMN)
    if raw is None:
        return row
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("labels is not a JSON object")
    except (ValueError, TypeError):
        logger.warning("row has unparseable labels: %.200r", raw)
        return row
    flattened = {key: value for key, value in row.items() if key != LABELS_COLUMN}
    for key, value in parsed.items():
        flattened[f"{LABEL_PREFIX}{key}"] = value
    return flattened


def rows_to_json(table: pa.Table) -> list[dict[str, object]]:
    """Turn a finelog Arrow result into Infinity's JSON rows.

    Flattens any ``labels`` column and renders every cell JSON-safe.
    """
    has_labels = LABELS_COLUMN in table.column_names
    rows: list[dict[str, object]] = []
    for row in table.to_pylist():
        if has_labels:
            row = _flatten_labels(row)
        rows.append({key: _json_safe(value) for key, value in row.items()})
    return rows
