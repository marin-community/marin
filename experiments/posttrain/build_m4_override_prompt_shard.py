#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert M4 override draft records into a MARIN inference prompt shard."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path("experiments/posttrain/stage4_output/m4_override_draft/override_prompts.jsonl")
DEFAULT_OUTPUT = Path("experiments/posttrain/stage4_output/m4_override_draft/marin_prompts/shard_00000.jsonl.gz")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            records.append(record)
    return records


def config_id(record: dict[str, Any]) -> str:
    custom_id = str(record["custom_id"])
    parts = custom_id.split("::")
    if len(parts) < 5:
        raise ValueError(f"bad custom_id: {custom_id!r}")
    return f"{parts[-2]}{parts[-1]}"


def prompt_record(record: dict[str, Any]) -> dict[str, Any]:
    contract = str(record["contract"])
    statement_id = str(record["statement_id"])
    return {
        "behavior_id": f"m4::{contract}::{statement_id}",
        "config_id": config_id(record),
        "system_prompt": str(record["system_prompt"]),
        "user_message": str(record["user_prompt"]),
        "rubric": "",
        "m4_custom_id": str(record["custom_id"]),
        "m4_contract": contract,
        "m4_statement_id": statement_id,
        "m4_authority_level": str(record["authority_level"]),
        "m4_statement_type": str(record["statement_type"]),
        "m4_expected_chosen_behavior": str(record["expected_chosen_behavior"]),
        "m4_expected_rejected_failure": str(record["expected_rejected_failure"]),
        "m4_statement_text": str(record["statement_text"]),
    }


def write_prompt_shard(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(prompt_record(record), ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} already exists; pass --force to overwrite")
    records = load_jsonl(args.input)
    write_prompt_shard(args.output, records)
    print(f"wrote {len(records)} prompts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
