# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Merge immutable source-result attempts into one complete campaign snapshot."""

import argparse
import json
from typing import Any

import fsspec


def _read(uri: str) -> list[dict[str, Any]]:
    with fsspec.open(uri, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _identity(row: dict[str, Any]) -> str:
    return str(row["source"]["source_sha256"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--replacement", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-sources", type=int, required=True)
    args = parser.parse_args()

    rows = {_identity(row): row for row in _read(args.base)}
    for uri in args.replacement:
        for row in _read(uri):
            identity = _identity(row)
            if identity not in rows:
                raise ValueError(f"replacement source is absent from base campaign: {identity}")
            rows[identity] = row
    if len(rows) != args.expected_sources:
        raise ValueError(f"merged source count {len(rows)} != expected {args.expected_sources}")
    ordered = sorted(rows.values(), key=lambda row: str(row["source"].get("source_key", row["source"].get("path"))))
    fs, path = fsspec.core.url_to_fs(args.output)
    if fs.exists(path):
        raise FileExistsError(f"refusing to overwrite immutable snapshot: {args.output}")
    with fs.open(path, "wt") as stream:
        for row in ordered:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
