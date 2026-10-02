# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read local curation JSONL records."""

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]
