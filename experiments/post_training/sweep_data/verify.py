# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independently re-read the final LR-sweep artifact and check it against its own manifest.

Checks: file hashes; row counts and padding to the batch size; row shapes; binary loss weights; zero loss on padding
rows and on positions after each row's content; supervised and context tokens per family against train-index.jsonl
and the manifest; validation counts per family; no held-out task appears in training (task identity).
"""

import gzip
import hashlib
import json
import sys
from collections import Counter

from rigging.filesystem.storage_path import StoragePath


def check_rows(data: bytes, rows: int, padding: int, context: int, pad: int, index: list[dict] | None) -> Counter:
    supervised = Counter()
    lines = gzip.decompress(data).decode().splitlines()
    if len(lines) != rows + padding or (rows + padding) % 16:
        raise ValueError(f"Expected {rows} + {padding} rows, found {len(lines)}")
    for number, line in enumerate(lines):
        row = json.loads(line)
        ids, weights = row["input_ids"], row["loss_weights"]
        if len(ids) != context or len(weights) != context or not set(weights) <= {0, 1}:
            raise ValueError(f"Malformed row {number}")
        if number >= rows:
            if any(weights) or set(ids) != {pad}:
                raise ValueError(f"Padding row {number} is not empty")
            continue
        length = context - next((i for i, token in enumerate(reversed(ids)) if token != pad), context)
        if index is not None:
            entry = index[number]
            if entry["row"] != number or entry["context_tokens"] < length:
                raise ValueError(f"Index entry {number} does not match its row")
            length = entry["context_tokens"]
            if sum(weights) != entry["supervised_tokens"]:
                raise ValueError(f"Row {number} supervision differs from the index")
        if any(weights[length - 1 :]):
            raise ValueError(f"Row {number} has loss past its content")
        supervised[index[number]["family"] if index is not None else "all"] += sum(weights)
    return supervised


def main() -> None:
    root = StoragePath(sys.argv[1])
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    files = {}
    for name, digest in manifest["output_hashes"].items():
        files[name] = (root / name).read_bytes()
        if hashlib.sha256(files[name]).hexdigest() != digest:
            raise ValueError(f"Hash mismatch: {name}")
    context, pad = manifest["max_sequence_tokens"], manifest["pad_token_id"]
    index = [json.loads(line) for line in files["train-index.jsonl"].decode().splitlines()]
    if len(index) != manifest["training_rows"]:
        raise ValueError("Index length differs from training_rows")
    supervised = check_rows(
        files["train-prebuilt.jsonl.gz"], manifest["training_rows"], manifest["padding_rows"], context, pad, index
    )
    if (
        dict(supervised) != manifest["family_supervised_tokens"]
        or sum(supervised.values()) != manifest["supervised_tokens"]
    ):
        raise ValueError("Per-family supervised tokens differ from the manifest")
    if manifest["steps_per_epoch"] * 16 != manifest["training_rows"] + manifest["padding_rows"]:
        raise ValueError("steps_per_epoch inconsistent")
    validation = {}
    for family, name in manifest["validation_files"].items():
        counted = check_rows(
            files[name],
            manifest["validation_rows"][family],
            manifest["validation_padding_rows"][family],
            context,
            pad,
            None,
        )
        if counted["all"] != manifest["validation_supervised_tokens"][family]:
            raise ValueError(f"Validation supervision differs for {family}")
        validation[family] = counted["all"]
    heldout = json.loads(files["heldout-ids.json"])
    trained = {(entry["family"], entry["task"]) for entry in index if entry["family"] != "replay"}
    leaked = [(family, task) for family, tasks in heldout.items() for task in tasks if (family, task) in trained]
    if leaked:
        raise ValueError(f"Held-out tasks in training: {leaked[:5]}")
    audit = [json.loads(line) for line in files["token-audit.jsonl"].decode().splitlines()]
    for family, tasks in heldout.items():
        validation_tasks = sorted(row["task"] for row in audit if row["split"] == f"validation-{family}")
        if validation_tasks != sorted(tasks):
            raise ValueError(f"Validation rows do not match held-out ids for {family}")
    report = {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "files_checked": sorted(files),
        "training_rows": manifest["training_rows"],
        "padding_rows": manifest["padding_rows"],
        "steps_per_epoch": manifest["steps_per_epoch"],
        "family_supervised_tokens": dict(supervised),
        "validation_supervised_tokens": validation,
        "heldout_tasks": {family: len(tasks) for family, tasks in heldout.items()},
        "heldout_in_training": 0,
    }
    print("SWEEP_DATA_VERIFIED " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
