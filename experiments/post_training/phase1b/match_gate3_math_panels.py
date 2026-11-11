# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Restrict the Phase 1b reverse math panels to the same task IDs."""

import argparse
import hashlib
import json

import fsspec
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.probe_sets.continuations import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--named-pair", nargs=3, action="append", required=True, metavar=("NAME", "REFERENCE", "S38"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    configure_coreweave_s3()

    inputs = {name: (reference, teacher) for name, reference, teacher in args.named_pair}
    if len(inputs) != len(args.named_pair):
        raise ValueError("duplicate panel names")

    id_sets = {}
    for name, (reference, _) in inputs.items():
        rows = read_jsonl(reference)
        ids = [row["sample_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{name}: duplicate sample ID")
        id_sets[name] = set(ids)
    common = set.intersection(*id_sets.values())
    if not common:
        raise ValueError("matched panel is empty")

    output = {}
    for name, (reference, teacher) in inputs.items():
        reference_rows = read_jsonl(reference)
        teacher_rows = read_jsonl(teacher)
        if len(reference_rows) != len(teacher_rows):
            raise ValueError(f"{name}: unequal reference and teacher rows")
        selected_reference = []
        selected_teacher = []
        for row, aligned in zip(reference_rows, teacher_rows, strict=True):
            if row["sample_id"] != aligned["sample_id"]:
                raise ValueError(f"{name}: misaligned sample ID")
            if row["sample_id"] not in common:
                continue
            if row["token_ids"][row["prompt_tokens"] :] != aligned["token_ids"][aligned["prompt_tokens"] :]:
                raise ValueError(f"{name}: response token mismatch at {row['sample_id']}")
            selected_reference.append(row)
            selected_teacher.append(aligned)
        if len(selected_reference) != len(common):
            raise ValueError(f"{name}: selected {len(selected_reference)} of {len(common)} common IDs")
        root = f"{args.out.rstrip('/')}/{name}"
        output[name] = {
            "source_rows": len(reference_rows),
            "reference": reference,
            "teacher": teacher,
            "reference_sha256": write_jsonl(f"{root}/reference.jsonl", selected_reference),
            "teacher_sha256": write_jsonl(f"{root}/S38.jsonl", selected_teacher),
        }

    summary = {
        "rows": len(common),
        "sample_ids_sha256": hashlib.sha256("\n".join(sorted(common)).encode()).hexdigest(),
        "panels": output,
    }
    with fsspec.open(f"{args.out.rstrip('/')}/manifest.json", "wt") as handle:
        json.dump(summary, handle, sort_keys=True)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
