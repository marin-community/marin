# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify production arm prefixes against their manifests and against plain (task #234).

For each prefix: _SUCCESS present and no DRY_RUN_NOT_FOR_TRAINING marker; every file's sha256 equals the manifest's
output_hashes; each epoch file's decompressed sha256 equals splits[epoch].rows_sha256; rows plus padding equal
padded_rows, a multiple of 16, with steps = padded_rows / 16; padding rows are pad-only with zero loss; per-family
supervised and context tokens equal the split's totals; no training row comes from a held-out task (pool v5 ids in the
manifest). Across arms: replay rows in every epoch file are the same multiset as plain's epoch 0. Plain: the three
epoch files hold the same rows (as a multiset) in different orders. RS-multi: each epoch's agentic supervised tokens
are within 1% of plain's B, on plain's task slots.

    python -m experiments.post_training.sft_shootout_rows.verify_arms <plain prefix> [<other arm prefix> ...]
"""

import collections
import gzip
import hashlib
import json
import sys

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.sweep_data.prepare_agentic import PAD_TOKEN
from experiments.post_training.sweep_data.select import CONTEXT

EPOCHS = 3


def check_prefix(uri: str) -> tuple[dict, list[dict], dict]:
    root = StoragePath(uri)
    if not (root / "_SUCCESS").exists():
        raise ValueError(f"{uri}: no _SUCCESS marker")
    if (root / "DRY_RUN_NOT_FOR_TRAINING").exists():
        raise ValueError(f"{uri}: carries a dry-run marker")
    manifest = json.loads((root / "manifest.json").read_text())
    files = {}
    for name, digest in manifest["output_hashes"].items():
        files[name] = (root / name).read_bytes()
        if hashlib.sha256(files[name]).hexdigest() != digest:
            raise ValueError(f"{uri}/{name}: hash differs from the manifest")
    index = [json.loads(line) for line in files["train-index.jsonl"].decode().splitlines()]
    heldout = {task for tasks in manifest["heldout_task_ids"].values() for task in tasks}
    leaked = sorted({item["task_id"] for item in index if item.get("task_id") in heldout})
    if leaked:
        raise ValueError(f"{uri}: held-out tasks in training: {leaked[:5]}")
    epochs = {}
    for epoch in range(EPOCHS):
        split = manifest["splits"][f"epoch-{epoch}"]
        data = gzip.decompress(files[f"train-epoch-{epoch}-prebuilt.jsonl.gz"])
        if hashlib.sha256(data).hexdigest() != split["rows_sha256"]:
            raise ValueError(f"{uri}: epoch {epoch} rows_sha256 differs")
        lines = data.decode().splitlines()
        entries = [item for item in index if item["epoch"] == epoch]
        if len(lines) != split["padded_rows"] or split["padded_rows"] % 16 or len(entries) != split["rows"]:
            raise ValueError(f"{uri}: epoch {epoch} row counts differ")
        if split["steps"] * 16 != split["padded_rows"]:
            raise ValueError(f"{uri}: epoch {epoch} steps differ")
        supervised, context = collections.Counter(), collections.Counter()
        replay = collections.Counter()
        agentic = collections.Counter()
        for number, line in enumerate(lines):
            row = json.loads(line)
            ids, weights = row["input_ids"], row["loss_weights"]
            if len(ids) != CONTEXT or len(weights) != CONTEXT or not set(weights) <= {0, 1}:
                raise ValueError(f"{uri}: epoch {epoch} row {number} malformed")
            if number >= len(entries):
                if any(weights) or set(ids) != {PAD_TOKEN}:
                    raise ValueError(f"{uri}: epoch {epoch} padding row {number} not empty")
                continue
            entry = entries[number]
            if entry["row"] != number or sum(weights) != entry["supervised_tokens"]:
                raise ValueError(f"{uri}: epoch {epoch} row {number} differs from its index entry")
            if any(weights[entry["context_tokens"] - 1 :]):
                raise ValueError(f"{uri}: epoch {epoch} row {number} has loss past its content")
            supervised[entry["family"]] += entry["supervised_tokens"]
            context[entry["family"]] += entry["context_tokens"]
            digest = hashlib.sha256(line.encode()).hexdigest()
            (replay if entry["family"] == "replay" else agentic)[digest] += 1
        if dict(supervised) != split["supervised_tokens"] or dict(context) != split["context_tokens"]:
            raise ValueError(f"{uri}: epoch {epoch} token totals differ from the manifest")
        epochs[epoch] = {"replay": replay, "agentic": agentic, "supervised": dict(supervised)}
    report = {
        "prefix": uri,
        "arm": manifest["arm"],
        "B_supervised_tokens": manifest["B_supervised_tokens"],
        "epochs": {
            e: {
                "rows": manifest["splits"][f"epoch-{e}"]["rows"],
                "padded_rows": manifest["splits"][f"epoch-{e}"]["padded_rows"],
                "steps": manifest["splits"][f"epoch-{e}"]["steps"],
                "supervised_tokens": epochs[e]["supervised"],
                "rows_sha256": manifest["splits"][f"epoch-{e}"]["rows_sha256"],
            }
            for e in epochs
        },
        "heldout_rows": 0,
    }
    return manifest, index, epochs | {"report": report}


def main() -> None:
    plain_manifest, plain_index, plain = check_prefix(sys.argv[1])
    if plain_manifest["arm"] != "plain":
        raise ValueError("The first prefix must be plain")
    if not (plain[0]["agentic"] == plain[1]["agentic"] == plain[2]["agentic"]):
        raise ValueError("Plain epoch files do not hold the same agentic rows")
    orders = [
        [item.get("trial_id") or item.get("byte_offset") for item in plain_index if item["epoch"] == e] for e in range(3)
    ]
    if len({tuple(order) for order in orders}) != EPOCHS:
        raise ValueError("Plain epochs share an order")
    reports = [plain["report"] | {"epochs_same_rows_different_order": True}]
    b = plain_manifest["B_supervised_tokens"]
    for uri in sys.argv[2:]:
        manifest, index, arm = check_prefix(uri)
        if manifest["replay"]["rows_sha256"] != plain_manifest["replay"]["rows_sha256"]:
            raise ValueError(f"{uri}: replay cut differs from plain's")
        for epoch in range(EPOCHS):
            if arm[epoch]["replay"] != plain[0]["replay"]:
                raise ValueError(f"{uri}: epoch {epoch} replay rows differ from plain's")
        report = arm["report"]
        if manifest["arm"] == "rs-multi":
            slots = set(plain_manifest["task_slots"])
            if {item["task_id"] for item in index if item.get("task_id")} - slots:
                raise ValueError(f"{uri}: rows outside plain's task slots")
            deviations = {}
            for epoch in range(EPOCHS):
                agentic = sum(v for family, v in arm[epoch]["supervised"].items() if family != "replay")
                deviations[epoch] = agentic / b - 1
                if abs(deviations[epoch]) > 0.01:
                    raise ValueError(f"{uri}: epoch {epoch} agentic supervised {agentic} is not within 1% of B {b}")
            report["agentic_supervised_vs_B"] = deviations
        reports.append(report)
    print("ARMS_VERIFIED " + json.dumps(reports), flush=True)


if __name__ == "__main__":
    main()
