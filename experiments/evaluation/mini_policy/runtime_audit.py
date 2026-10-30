# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit matched full/subset requests and preserve bounded paired trace evidence."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from finestore.reader import ReadView
from rigging.filesystem.buckets import filesystem_for

BASE = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/runtime-20261030"
MICRO = BASE + (
    "/records/mini-policy-qwen36-math500-micro-canary-20261030-v2-math500-random-"
    "micro-decfb3d148be5916bec92beb20495e4c/results"
)
FULL = (
    BASE
    + "/records/mini-policy-qwen36-math500-full-reference-20261030-v2-math500-92e55b2316ef594fb4d9f0e9f99087be/results"
)


def scan(view, table, columns=None, where=None):
    rows = view.scan(table, columns=columns, where=where)
    return rows.to_pylist(maps_as_pydicts="strict") if rows is not None else []


def operation_map(rows):
    result = {}
    for row in rows:
        payload = json.loads(row["payload"])
        key = (row["identity"], row["operation"])
        if key in result:
            raise ValueError("Duplicate checkpoint operation")
        result[key] = payload["input_digest"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", default=MICRO)
    parser.add_argument("--expected-tasks", type=int, default=10)
    parser.add_argument("--label", default="micro")
    args = parser.parse_args()
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    micro = ReadView(args.subset_root)
    full = ReadView(FULL)
    columns = [
        "task",
        "doc_id",
        "trial_id",
        "filter",
        "prompt_text",
        "prompt_messages",
        "output",
        "target_text",
        "grading",
        "metrics",
        "doc",
    ]
    small = scan(micro, "samples", columns)
    identifiers = sorted({row["doc_id"] for row in small})
    if len(identifiers) != len(small):
        raise ValueError("Unexpected multiple samples per canary task")
    large_scores = scan(full, "samples", ["doc_id", "grading"])
    matched = scan(full, "samples", columns, where=[("doc_id", "in", identifiers)]) if identifiers else []
    large = {row["doc_id"]: row for row in matched}
    mini_ops = operation_map(scan(micro, "evalchemy_operations"))
    full_ops = operation_map(scan(full, "evalchemy_operations"))
    missing_ops = [key for key in mini_ops if key not in full_ops]
    changed_ops = [key for key, value in mini_ops.items() if key in full_ops and full_ops[key] != value]
    pairs = []
    for row in small:
        other = large.get(row["doc_id"])
        pairs.append(
            {
                "doc_id": row["doc_id"],
                "micro": row,
                "full": other,
                "same_prompt": (
                    other is not None
                    and (row["prompt_text"], row["prompt_messages"]) == (other["prompt_text"], other["prompt_messages"])
                ),
                "same_output": other is not None and row["output"] == other["output"],
                "same_score": other is not None and row["grading"]["score"] == other["grading"]["score"],
            }
        )
    paired = all(pair["full"] is not None for pair in pairs) and len(pairs) == args.expected_tasks
    canonical = scan(micro, "evalchemy_results", ["document", "scored"])
    sources = {
        name: {
            "root": root,
            "sample_metadata": view.table_metadata_path("samples"),
            "operation_metadata": view.table_metadata_path("evalchemy_operations"),
        }
        for name, root, view in [(args.label, args.subset_root, micro), ("full", FULL, full)]
    }
    # Spec identity hashes include source row, original ordinal, requests, and grader.
    # Operation input hashes cover prompt, generation settings, context, and template settings.
    receipt = {
        "subset_label": args.label,
        "expected_subset_tasks": args.expected_tasks,
        "sources": sources,
        "micro_samples": len(small),
        "full_samples": len(large_scores),
        "micro_mean": sum(row["grading"]["score"] for row in small) / len(small) if small else None,
        "micro_aggregate": canonical,
        "paired_samples": sum(pair["full"] is not None for pair in pairs),
        "micro_operations": len(mini_ops),
        "missing_operation_identities": missing_ops,
        "changed_operation_inputs": changed_ops,
        "same_prompt_count": sum(pair["same_prompt"] for pair in pairs),
        "same_output_count": sum(pair["same_output"] for pair in pairs),
        "same_score_count": sum(pair["same_score"] for pair in pairs),
        "canary_input_parity_pass": (
            paired
            and len(mini_ops) > 0
            and not missing_ops
            and not changed_ops
            and all(pair["same_prompt"] for pair in pairs)
        ),
        "full_reference_has_500_samples": len(large_scores) == 500,
    }
    evidence = json.dumps(pairs, indent=2, default=str).encode()
    receipt["paired_evidence_sha256"] = hashlib.sha256(evidence).hexdigest()
    for name, data in [
        (f"math500-{args.label}-paired-traces.json", evidence),
        (f"math500-{args.label}-runtime-audit.json", json.dumps(receipt, indent=2).encode()),
    ]:
        (output / name).write_bytes(data)
        fs, key = filesystem_for(BASE + "/audit/" + name)
        fs.pipe_file(key, data)
    print(
        json.dumps({key: value for key, value in receipt.items() if key not in {"sources", "micro_aggregate"}}),
        flush=True,
    )


if __name__ == "__main__":
    main()
