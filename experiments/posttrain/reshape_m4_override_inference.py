#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert MARIN-format M4 override inference shards to generations JSONL."""

from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path
from typing import Any

import fsspec


def input_pattern(path: str) -> str:
    stripped = path.rstrip("/")
    if any(ch in stripped for ch in "*?["):
        return stripped
    if stripped.endswith((".jsonl", ".jsonl.gz")):
        return stripped
    return f"{stripped}/shard_*.jsonl.gz"


def iter_records(path: str):
    files = fsspec.open_files(input_pattern(path), mode="rb")
    if not files:
        raise FileNotFoundError(f"no input shards matched {path!r}")
    for file in files:
        with file as raw:
            if str(file.path).endswith(".gz"):
                with (
                    gzip.GzipFile(fileobj=raw, mode="rb") as gzipped,
                    io.TextIOWrapper(gzipped, encoding="utf-8") as handle,
                ):
                    for line in handle:
                        if line.strip():
                            yield json.loads(line)
                continue
            for line in raw:
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if line.strip():
                    yield json.loads(line)


def to_generation(record: dict[str, Any], model_label: str) -> dict[str, Any]:
    sample_idx = int(record.get("sample_idx", 0))
    custom_id = str(record["m4_custom_id"])
    return {
        "custom_id": f"gen::{custom_id}::s{sample_idx:02d}",
        "model": model_label,
        "sample_idx": sample_idx,
        "contract": record["m4_contract"],
        "statement_id": record["m4_statement_id"],
        "authority_level": record["m4_authority_level"],
        "statement_type": record["m4_statement_type"],
        "system_prompt": record["system_prompt"],
        "user_prompt": record["user_message"],
        "response": record["response_text"],
        "expected_chosen_behavior": record["m4_expected_chosen_behavior"],
        "expected_rejected_failure": record["m4_expected_rejected_failure"],
        "statement_text": record["m4_statement_text"],
        "usage": record.get("usage", {}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Shard path, directory, or glob. Supports gs://.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [to_generation(record, args.model_label) for record in iter_records(args.input)]
    rows.sort(key=lambda row: (row["contract"], row["statement_id"], row["custom_id"], row["sample_idx"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(json.dumps({"output": str(args.output), "records": len(rows), "model_label": args.model_label}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
