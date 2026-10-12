# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read and write deterministic compressed campaign analysis artifacts."""

import gzip
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def write_json(path: Path, value: dict) -> None:
    raw = (json.dumps(value, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw)
