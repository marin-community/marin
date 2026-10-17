# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize the independently verified E_2 candidate without modifying sources."""

import argparse
import gzip
import hashlib
import json
import tempfile
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath
from transformers import AutoTokenizer

from experiments.post_training.cruxeval_transfer.constants import STUDENT_MODEL as MODEL
from experiments.post_training.cruxeval_transfer.constants import STUDENT_REVISION as MODEL_REVISION
from experiments.post_training.cruxeval_transfer.rendering import END_TURN_ID, token_counts, training_row
from experiments.post_training.second_expert.constants import CANONICAL, CANONICAL_SHA256, CAPTURE, SUBSET_SHA256
from experiments.post_training.second_expert.verify_candidate import indexed_rows

VERIFICATION = "s3://marin-us-east-02a/marin/scratch/karan/mopd2/task162-candidate-verification-v4"
CONTEXT = 32768
BATCH_SIZE = 16


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = StoragePath(args.output)
    if output.exists():
        raise ValueError("Use a new immutable output prefix")
    report = json.loads((StoragePath(VERIFICATION) / "report.json").read_text())
    if (report["canonical_sha256"], report["subset_sha256"]) != (CANONICAL_SHA256, SUBSET_SHA256):
        raise ValueError("Verification pins differ")
    canonical_bytes = (StoragePath(CANONICAL) / "canonical.jsonl").read_bytes()
    subset_bytes = (StoragePath(CANONICAL) / "subset-groups-1600-seed20261009.json").read_bytes()
    if hashlib.sha256(canonical_bytes).hexdigest() != CANONICAL_SHA256:
        raise ValueError("Canonical hash mismatch")
    if hashlib.sha256(subset_bytes).hexdigest() != SUBSET_SHA256:
        raise ValueError("Subset hash mismatch")
    canonical = indexed_rows(canonical_bytes)
    subset = json.loads(subset_bytes)
    sources = {}
    for name in ("requests.jsonl", "raw-responses.jsonl"):
        content = (StoragePath(CAPTURE) / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != report["source_hashes"][name]:
            raise ValueError(f"Verified source changed: {name}")
        sources[name] = indexed_rows(content)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    template = tokenizer.get_chat_template()
    if hashlib.sha256(template.encode()).hexdigest() != report["template_sha256"]:
        raise ValueError("Template hash mismatch")
    rows, audits, probes = [], [], []
    with tempfile.TemporaryDirectory() as directory:
        local = Path(directory)
        with gzip.open(local / "train-prebuilt.jsonl.gz", "wt") as stream:
            for identity in sorted(subset["custom_ids"]):
                record = canonical[identity]
                if not record["usable_canonical"]:
                    continue
                message = sources["raw-responses.jsonl"][identity]["body"]["choices"][0]["message"]
                row = training_row(
                    {
                        "id": identity,
                        "problem_id": str(record["question_id"]),
                        "messages": [
                            *sources["requests.jsonl"][identity]["b_request"]["messages"],
                            {
                                "role": "assistant",
                                "content": message["content"],
                                "reasoning_content": message.get("reasoning_content") or message.get("reasoning"),
                            },
                        ],
                    }
                )
                total, supervised = token_counts(row, tokenizer)
                if total > CONTEXT:
                    raise ValueError(f"Overlength row: {identity}")
                encoded = tokenizer.apply_chat_template(
                    row["messages"],
                    tokenize=True,
                    return_dict=True,
                    return_assistant_tokens_mask=True,
                    **row["chat_template_kwargs"],
                )
                padding = CONTEXT - total
                prebuilt = {
                    "input_ids": encoded["input_ids"] + [END_TURN_ID] * padding,
                    "loss_weights": encoded["assistant_masks"][1:] + [0] * (padding + 1),
                }
                assert len(prebuilt["loss_weights"]) == CONTEXT
                assert sum(prebuilt["loss_weights"]) == supervised
                stream.write(json.dumps(prebuilt, separators=(",", ":")) + "\n")
                rows.append(row)
                audits.append({"id": identity, "tokens": total, "supervised_tokens": supervised})
                if len(probes) < 4:
                    probes.append(
                        row | {"input_ids": encoded["input_ids"], "loss_weights": prebuilt["loss_weights"][:total]}
                    )
            padding_rows = -len(rows) % BATCH_SIZE
            empty = {"input_ids": [END_TURN_ID] * CONTEXT, "loss_weights": [0] * CONTEXT}
            for _ in range(padding_rows):
                stream.write(json.dumps(empty, separators=(",", ":")) + "\n")
        assert len(rows) == report["selected_usable"] == 1359
        assert sum(row["supervised_tokens"] for row in audits) == report["selected_supervised_tokens"]
        assert sum(row["tokens"] for row in audits) == report["selected_input_tokens"]
        for name, values in (("train.jsonl", rows), ("token-audit.jsonl", audits), ("row-probes.jsonl", probes)):
            (local / name).write_text("".join(json.dumps(row) + "\n" for row in values))
        (local / "chat_template.jinja").write_text(template)
        manifest = report | {
            "test_exposed": True,
            "student": MODEL,
            "student_revision": MODEL_REVISION,
            "verification_prefix": VERIFICATION,
            "training_rows": len(rows),
            "batch_size": BATCH_SIZE,
            "padding_rows": padding_rows,
            "steps_per_epoch": (len(rows) + padding_rows) // BATCH_SIZE,
            "max_sequence_tokens": CONTEXT,
            "supervised_tokens": report["selected_supervised_tokens"],
            "input_tokens": report["selected_input_tokens"],
            "selection_status": "materialized; GPU training not authorized",
            "output_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in local.iterdir()},
        }
        output.mkdirs()
        for path in local.iterdir():
            (output / path.name).write_bytes(path.read_bytes())
            uploaded = (output / path.name).read_bytes()
            if hashlib.sha256(uploaded).hexdigest() != manifest["output_hashes"][path.name]:
                raise ValueError(f"Uploaded artifact hash differs: {path.name}")
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("DATA_PREPARED " + json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
