# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for repairing a completed rollout file with targeted reruns."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from typing import Any

import fsspec


def read_jsonl(path: str) -> Iterator[dict[str, Any]]:
    with fsspec.open(path, "rt") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str, rows: Iterator[dict[str, Any]]) -> None:
    with fsspec.open(path, "wt") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cmd_failed_ids(args: argparse.Namespace) -> None:
    seen: set[str] = set()
    ids: list[str] = []
    for row in read_jsonl(args.rollouts):
        if row.get("finish_reason") != args.finish_reason:
            continue
        prompt_id = row["prompt_id"]
        if prompt_id in seen:
            continue
        seen.add(prompt_id)
        ids.append(prompt_id)

    with fsspec.open(args.output, "wt") as f:
        for prompt_id in ids:
            f.write(prompt_id + "\n")
    print(json.dumps({"finish_reason": args.finish_reason, "n_prompt_ids": len(ids), "output": args.output}))


def cmd_merge(args: argparse.Namespace) -> None:
    repairs = {row["prompt_id"]: row for row in read_jsonl(args.repair)}
    used_repairs: set[str] = set()

    def merged_rows() -> Iterator[dict[str, Any]]:
        for row in read_jsonl(args.base):
            prompt_id = row["prompt_id"]
            replacement = repairs.get(prompt_id)
            if replacement is None:
                yield row
                continue
            used_repairs.add(prompt_id)
            yield replacement
        for prompt_id, row in repairs.items():
            if prompt_id not in used_repairs:
                yield row

    write_jsonl(args.output, merged_rows())
    print(
        json.dumps(
            {
                "base": args.base,
                "repair": args.repair,
                "output": args.output,
                "n_repairs": len(repairs),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)

    failed_ids = sub.add_parser("failed-ids")
    failed_ids.add_argument("--rollouts", required=True)
    failed_ids.add_argument("--finish-reason", default="http_400")
    failed_ids.add_argument("--output", required=True)
    failed_ids.set_defaults(func=cmd_failed_ids)

    merge = sub.add_parser("merge")
    merge.add_argument("--base", required=True)
    merge.add_argument("--repair", required=True)
    merge.add_argument("--output", required=True)
    merge.set_defaults(func=cmd_merge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
