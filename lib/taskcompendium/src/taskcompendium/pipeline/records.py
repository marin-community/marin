# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read and atomically replace local curation records."""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]
