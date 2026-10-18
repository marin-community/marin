# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combine the LR-sweep agentic and replay components into the final sweep training set.

Training rows from both components are shuffled together with a fixed seed and padded with zero-loss rows to a
multiple of the batch size. Validation files are copied unchanged. train-index.jsonl gives each training row's
family and source identity, so per-family accounting never depends on row order assumptions.
"""

import argparse
import gzip
import hashlib
import json
import random
import tempfile
from collections import Counter
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.sweep_data.prepare_agentic import (
    BATCH,
    PAD_TOKEN,
    output_hashes,
    read_component,
    upload_checked,
    write_prebuilt,
)
from experiments.post_training.sweep_data.select import CONTEXT, SEED


def real_rows(compressed: bytes, rows: int, padding: int) -> list[str]:
    lines = gzip.decompress(compressed).decode().splitlines()
    if len(lines) != rows + padding:
        raise ValueError("Row count differs from manifest")
    for line in lines[rows:]:
        if any(json.loads(line)["loss_weights"]):
            raise ValueError("Component padding carries loss")
    return lines[:rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agentic", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = StoragePath(args.output)
    if output.exists():
        raise ValueError("Use a fresh output prefix")
    agentic_names = (
        "train-prebuilt.jsonl.gz",
        "validation-tmax-prebuilt.jsonl.gz",
        "validation-swesmith-prebuilt.jsonl.gz",
        "token-audit.jsonl",
        "heldout-ids.json",
        "row-probes.jsonl",
    )
    agentic, agentic_bytes, agentic_files = read_component(StoragePath(args.agentic), agentic_names)
    replay, replay_bytes, replay_files = read_component(
        StoragePath(args.replay), ("replay-prebuilt.jsonl.gz", "replay-rows.jsonl")
    )
    for key in ("student_revision", "pad_token_id", "max_sequence_tokens", "batch_size"):
        if agentic[key] != replay[key]:
            raise ValueError(f"Components disagree on {key}")
    # The replay slice is one frozen artifact for every arm. It was sized at R = B/4 against an earlier agentic
    # build; its own manifest records that B, and the achieved ratio against this build is reported below.

    audits = [json.loads(line) for line in agentic_files["token-audit.jsonl"].decode().splitlines()]
    agentic_index = [row for row in audits if row["split"] == "train"]
    replay_index = []
    for line in replay_files["replay-rows.jsonl"].decode().splitlines():
        entry = json.loads(line)
        # Parquet sources record their row within the row group as "row", which would clash with the index's row.
        if "row" in entry:
            entry["row_in_group"] = entry.pop("row")
        replay_index.append({"split": "train", "family": "replay"} | entry)
    rows = real_rows(agentic_files["train-prebuilt.jsonl.gz"], agentic["training_rows"], agentic["padding_rows"])
    rows += real_rows(replay_files["replay-prebuilt.jsonl.gz"], replay["rows"], replay["padding_rows"])
    index = agentic_index + replay_index
    if len(rows) != len(index):
        raise ValueError("Row index and rows differ in length")
    order = list(range(len(rows)))
    random.Random(SEED).shuffle(order)
    padding = -len(rows) % BATCH

    with tempfile.TemporaryDirectory() as temporary:
        out = Path(temporary)
        supervised_by_family = Counter()
        context_by_family = Counter()

        def checked_rows():
            for position in order:
                row = json.loads(rows[position])
                weights, ids = row["loss_weights"], row["input_ids"]
                if len(ids) != CONTEXT or len(weights) != CONTEXT or not set(weights) <= {0, 1}:
                    raise ValueError("Malformed row")
                family = index[position]["family"]
                if sum(weights) != index[position]["supervised_tokens"]:
                    raise ValueError("Row supervision differs from its index entry")
                supervised_by_family[family] += sum(weights)
                context_by_family[family] += index[position]["context_tokens"]
                yield rows[position]

        if write_prebuilt(out / "train-prebuilt.jsonl.gz", checked_rows()) != (len(rows), padding):
            raise ValueError("Written row count differs")
        (out / "train-index.jsonl").write_text(
            "".join(json.dumps({"row": row} | index[position]) + "\n" for row, position in enumerate(order))
        )
        for name in agentic_names[1:]:
            (out / name).write_bytes(agentic_files[name])
        (out / "replay-rows.jsonl").write_bytes(replay_files["replay-rows.jsonl"])

        supervised = sum(supervised_by_family.values())
        family_rows = Counter(entry["family"] for entry in index)
        manifest = {
            "task": 202,
            "description": (
                "Snowball v0 plain-SFT LR-sweep pilot set: TMax + Kimi SWE-smith traces plus nemotron_sft_v3 replay"
            ),
            "student_model": agentic["student_model"],
            "student_revision": agentic["student_revision"],
            "student_weights_identical_to_revision": agentic["student_weights_identical_to_revision"],
            "student_regional_copy": agentic["student_regional_copy"],
            "student_regional_copy_exceptions": agentic["student_regional_copy_exceptions"],
            "tokenizer_files_sha256": agentic["tokenizer_files_sha256"],
            "pad_token_id": PAD_TOKEN,
            "max_sequence_tokens": CONTEXT,
            "batch_size": BATCH,
            "training_rows": len(rows),
            "padding_rows": padding,
            "steps_per_epoch": (len(rows) + padding) // BATCH,
            "supervised_tokens": supervised,
            "context_tokens": sum(context_by_family.values()),
            "family_rows": dict(family_rows),
            "family_supervised_tokens": dict(supervised_by_family),
            "family_context_tokens": dict(context_by_family),
            "family_supervised_share": {family: count / supervised for family, count in supervised_by_family.items()},
            "replay_to_agentic_ratio": (
                supervised_by_family["replay"] / (supervised_by_family["tmax"] + supervised_by_family["swesmith"])
            ),
            "validation_files": agentic["validation_files"],
            "validation_rows": agentic["validation_rows"],
            "validation_padding_rows": agentic["validation_padding_rows"],
            "validation_supervised_tokens": agentic["validation_supervised_tokens"],
            "validation_scope": "agentic only, one fixed trajectory per held-out task; no replay validation",
            "components": {
                "agentic": {"uri": args.agentic, "manifest_sha256": hashlib.sha256(agentic_bytes).hexdigest()},
                "replay": {"uri": args.replay, "manifest_sha256": hashlib.sha256(replay_bytes).hexdigest()},
            },
            "agentic": {key: agentic[key] for key in ("families", "sources", "selection", "loss_mask", "rendering")},
            "replay": (
                {
                    key: replay[key]
                    for key in (
                        "source",
                        "sampled_subset",
                        "excluded_partitions",
                        "seed",
                        "agentic_supervised_tokens_B",
                        "agentic_manifest_sha256",
                        "target_supervised_tokens",
                        "achieved_ratio_to_B",
                        "partition_quotas",
                        "rendering",
                        "loss_mask",
                    )
                }
                | {
                    "partitions": {
                        name: {k: v for k, v in r.items() if k != "files"} for name, r in replay["partitions"].items()
                    }
                }
            ),
            "shuffle_seed": SEED,
            "train_index": (
                "train-index.jsonl: row -> family, task/partition, trajectory or source position, token counts"
            ),
            "test_exposed": False,
            "output_hashes": output_hashes(out),
        }
        upload_checked(out, output, manifest)
        manifest_hash = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
        print(
            "SWEEP_DATA_PREPARED "
            + json.dumps(
                {"manifest_sha256": manifest_hash}
                | {k: v for k, v in manifest.items() if k not in ("agentic", "replay")}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
