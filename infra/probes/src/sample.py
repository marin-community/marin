# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``Sample`` telemetry record — the single shape every collector emits.

A health check and a gauge are the same thing here: a named metric with a float
value and a JSON label set, stamped with the cycle's collection time. The runner
fans samples to the sinks; ``Sample`` doubles as the finelog table schema
(``FinelogTableSink`` derives its columns from these fields).

Its own module so the runner and the sinks can share it without importing the
``python -m`` entrypoint (which would re-import it as a second ``__main__`` copy).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar


@dataclass(frozen=True)
class Sample:
    # The finelog ordering key. register_table requires the schema to declare a
    # key_column naming an existing column (or carry an implicit `timestamp_ms`
    # column). Our time axis is collected_at, so name it explicitly — the server
    # checks presence, not type, and accepts a TIMESTAMP_MS column. Without this
    # the metrics table never registers and FinelogTableSink drops every row.
    key_column: ClassVar[str] = "collected_at"

    # labels is a JSON object string (e.g. '{"zone":"us-east5-a"}') rather than a
    # dict so the finelog schema is a flat STRING column and the JSONL stays one
    # line per sample; DuckDB's json_extract slices it at query time.
    # collected_at is None only transiently between Sample.of() and the runner
    # stamping it with the cycle time; every recorded sample has it set.
    metric: str
    value: float
    labels: str
    collected_at: datetime | None = None

    @classmethod
    def of(cls, metric: str, value: float, /, **labels: str) -> Sample:
        """Build an unstamped Sample with JSON-encoded labels.

        The runner stamps ``collected_at`` with the cycle's start time so all
        samples from one collection share a timestamp. Labels are sorted for
        stable encoding.
        """
        return cls(metric=metric, value=float(value), labels=json.dumps(labels, sort_keys=True))
