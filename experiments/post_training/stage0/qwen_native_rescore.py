# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Re-score saved Qwen replies under current math and protocol rules."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from rigging.filesystem.buckets import filesystem_for

from experiments.post_training.stage0.qwen_native_gate import (
    STOP_TOKENS,
    THINK_PREFILL,
    gate_summary,
    math_correct,
    native_facts,
    read_json,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="generation directory containing report.json")
    parser.add_argument("--think-prompts", required=True)
    parser.add_argument("--out", required=True, help="separate JSON report; original receipts remain unchanged")
    args = parser.parse_args()
    report = read_json(f"{args.source}/report.json")
    fs, path = filesystem_for(args.think_prompts)
    with fs.open(path, "rt") as handle:
        prompts = [json.loads(line) for line in handle]
    references = {row["probe_id"]: str(row["answer"]) for row in prompts}
    grades = {}
    for key, summary in report["sets"].items():
        is_math = key.startswith("P_think-")
        fs, path = filesystem_for(f"{args.source}/{key}")
        urls = [fs.unstrip_protocol(name) for name in fs.glob(f"{path}/*.json")]
        with ThreadPoolExecutor(16) as pool:
            records = list(pool.map(read_json, urls))
        by_id = {record["probe_id"]: record for record in records}
        if (
            len(records) != summary["count"]
            or len(by_id) != len(records)
            or (is_math and by_id.keys() != references.keys())
        ):
            raise ValueError(f"{key}: saved replies do not match the complete prompt set")
        records = (
            [by_id[row["probe_id"]] for row in prompts] if is_math else sorted(records, key=lambda r: r["probe_id"])
        )
        grades[key] = []
        for record in records:
            prefill = "" if "-no-thinking-" in key else THINK_PREFILL
            stop_reason = record["stop_reason"]
            if stop_reason is None and record["finish_reason"] == "stop":
                stop_reason = record["last_token_id"]
            record["facts"] = native_facts(
                record["text"],
                record["finish_reason"],
                stop_reason,
                prefill,
                [report["stop_ids"][token] for token in STOP_TOKENS],
            )
            correct = math_correct(references[record["probe_id"]], record["text"]) if is_math else None
            grades[key].append(
                {"probe_id": record["probe_id"], "original_correct": record["correct"], "correct": correct}
            )
            record["correct"] = correct
        report["sets"][key] = gate_summary(records)
    report["scoring"] = {
        "version": "open-think-cap-exclusion-v2",
        "source_report": f"{args.source}/report.json",
        "grades": grades,
    }
    write_json(args.out, report)


if __name__ == "__main__":
    main()
