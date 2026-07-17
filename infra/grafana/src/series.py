# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn finelog metric rows into Grafana time series.

finelog's query engine is DataFusion, which has no JSON functions, so the
``labels`` column — a JSON object string written by ``infra/probes`` — cannot be
filtered or grouped in SQL. Label selection therefore happens here, after the
rows come back: ``build_sql`` narrows by metric and time (the only predicates the
engine can serve), and ``to_series`` decodes labels per row to filter them and to
name each row's series.

Because the SQL is generated here rather than supplied by a caller, identifiers
are validated against :data:`IDENT` rather than escaped; anything that is not a
bare identifier is rejected, so no caller-controlled text reaches the engine.
"""

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import pyarrow as pa

logger = logging.getLogger(__name__)

# Metric names and label keys are bare identifiers. Anything else is rejected
# rather than quoted: the generated SQL then has no interpolation an argument
# could escape from.
IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Label values are compared in Python, never interpolated into SQL, so they only
# need to be sane — not identifier-shaped (e.g. "us-east5-a", "v6e-4").
_MAX_LABEL_VALUE_CHARS = 256

# finelog stores its logical TIMESTAMP_MS columns at microsecond precision, so an
# arrow_cast to Int64 yields epoch micros. Grafana plots milliseconds.
_MICROS_PER_MILLI = 1000

# The row shape every query selects. `collected_us` is the cast time axis.
_COLUMNS = ("metric", "value", "labels", "collected_us")

# Series name for rows that carry no value for the requested group_by label.
UNLABELLED = "(none)"


@dataclass(frozen=True)
class Point:
    """One plotted sample: epoch milliseconds, the value, and its series name."""

    time_ms: int
    series: str
    value: float


def _check_ident(kind: str, value: str) -> str:
    if not IDENT.match(value):
        raise ValueError(f"{kind} must be a bare identifier, got {value!r}")
    return value


def sql_timestamp(at: datetime) -> str:
    """Format ``at`` as the tz-naive UTC literal finelog's timestamp columns compare against.

    finelog stores timestamps tz-naive in UTC, so a tz-aware literal raises a
    comparison error in the engine rather than returning nothing.
    """
    return at.strftime("%Y-%m-%d %H:%M:%S")


def build_sql(namespace: str, metric: str, start: datetime, end: datetime, *, limit: int) -> str:
    """Build the narrowing query for one metric over ``[start, end)``.

    Metric and time are the only predicates DataFusion can serve for this schema,
    so the returned rows still carry every label combination. ``limit`` bounds the
    scan so a wide time range cannot pull an unbounded result through the server.
    """
    if not IDENT.match(namespace.replace(".", "_")):
        raise ValueError(f"namespace must be dotted identifiers, got {namespace!r}")
    _check_ident("metric", metric)
    if end <= start:
        raise ValueError(f"end {end} must be after start {start}")

    return (
        f"SELECT metric, value, labels, arrow_cast(collected_at, 'Int64') AS collected_us FROM \"{namespace}\" "
        f"WHERE metric = '{metric}' "
        f"AND collected_at >= TIMESTAMP '{sql_timestamp(start)}' "
        f"AND collected_at < TIMESTAMP '{sql_timestamp(end)}' "
        f"ORDER BY collected_at LIMIT {int(limit)}"
    )


def decode_labels(raw: str) -> Mapping[str, str]:
    """Decode a ``labels`` cell into a flat string mapping.

    Raises ``ValueError``/``JSONDecodeError`` if the cell is not a JSON object.
    """
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"labels must be a JSON object, got {type(parsed).__name__}")
    return {str(k): str(v) for k, v in parsed.items()}


def validate_grouping(group_by: str | None, match: Mapping[str, str] | None) -> None:
    """Check the grouping arguments, raising ``ValueError`` on anything unusable.

    Separate from :func:`to_series` so a caller can reject bad input before paying
    for a query.
    """
    if group_by is not None:
        _check_ident("group_by", group_by)
    for key, value in (match or {}).items():
        _check_ident("label filter key", key)
        if len(value) > _MAX_LABEL_VALUE_CHARS:
            raise ValueError(f"label filter {key!r} value too long ({len(value)} chars)")


def to_series(
    table: pa.Table,
    *,
    group_by: str | None = None,
    match: Mapping[str, str] | None = None,
) -> list[Point]:
    """Decode, filter, and group finelog rows into plottable points.

    ``match`` keeps only rows whose labels equal every given pair; ``group_by``
    names the label whose value becomes the series name (all rows collapse to one
    series named for the metric when it is ``None``). Rows whose labels do not
    parse are dropped with a warning rather than sinking the whole panel — a
    single malformed cell is schema drift, not a reason to blank a dashboard.
    """
    validate_grouping(group_by, match)

    missing = [c for c in _COLUMNS if c not in table.column_names]
    if missing:
        raise ValueError(f"query result is missing columns {missing}; got {table.column_names}")

    points: list[Point] = []
    for row in table.select(_COLUMNS).to_pylist():
        try:
            labels = decode_labels(row["labels"])
        except (ValueError, TypeError, json.JSONDecodeError):
            logger.warning("dropping row with unparseable labels: %.200r", row["labels"])
            continue

        if match and any(labels.get(k) != v for k, v in match.items()):
            continue

        series = row["metric"] if group_by is None else labels.get(group_by, UNLABELLED)
        points.append(
            Point(
                time_ms=round(row["collected_us"] / _MICROS_PER_MILLI),
                series=str(series),
                value=float(row["value"]),
            )
        )
    return points


def to_json_rows(points: Sequence[Point]) -> list[dict[str, object]]:
    """Render points as the long-format rows the Infinity datasource parses.

    One row per sample with an explicit series column; Grafana splits it into one
    line per distinct ``series``.
    """
    return [{"time": p.time_ms, "series": p.series, "value": p.value} for p in points]
