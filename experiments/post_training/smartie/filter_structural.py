# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exclude held-out prompts from a versioned structural training cohort."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

STRUCTURAL_TASK_PREFIX = "structure-closure:"


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def filter_structural(tokens: Path, native: Path, output: Path) -> dict:
    """Preserve validation and remove training matches by text hash or exact tokens."""
    output.mkdir(parents=True, exist_ok=False)
    lines = {split: (tokens / f"{split}.jsonl").read_text().splitlines() for split in ("train", "validation")}
    rows = {split: [json.loads(line) for line in values] for split, values in lines.items()}
    val_hashes = {row["prompt_sha256"] for row in rows["validation"]}
    val_tokens = {tuple(row["prompt_token_ids"]) for row in rows["validation"]}
    if {row["task_id"] for row in rows["train"]} & {row["task_id"] for row in rows["validation"]}:
        raise ValueError("Source task identities overlap across splits")
    excluded = {
        row["case_id"]
        for row in rows["train"]
        if row["prompt_sha256"] in val_hashes or tuple(row["prompt_token_ids"]) in val_tokens
    }
    report = {
        "rule": "exclude validation prompt text hashes and exact token sequences from training",
        "builder_sha256": file_hash(Path(__file__)),
        "inputs": {},
        "outputs": {},
        "excluded_train_case_ids": sorted(excluded),
    }
    for split in ("train", "validation"):
        token_path = tokens / f"{split}.jsonl"
        native_path = native / f"{split}.parquet"
        table = pq.read_table(native_path)
        native_ids = [json.loads(value)["task"]["id"] for value in table["lowered_task_spec"].to_pylist()]
        expected_ids = [STRUCTURAL_TASK_PREFIX + row["case_id"] for row in rows[split]]
        if len(set(expected_ids)) != len(expected_ids) or set(native_ids) != set(expected_ids):
            raise ValueError(f"Native/token identity mismatch in {split}")
        if len(native_ids) != len(expected_ids):
            raise ValueError(f"Duplicate native identities in {split}")
        rejected = {STRUCTURAL_TASK_PREFIX + case_id for case_id in excluded} if split == "train" else set()
        selected = [
            (line, row)
            for line, row in zip(lines[split], rows[split], strict=True)
            if STRUCTURAL_TASK_PREFIX + row["case_id"] not in rejected
        ]
        filtered = table.filter(pa.array([task_id not in rejected for task_id in native_ids]))
        output_tokens = output / f"{split}.jsonl"
        output_native = output / f"{split}.parquet"
        if split == "validation":
            output_tokens.write_bytes(token_path.read_bytes())
            output_native.write_bytes(native_path.read_bytes())
        else:
            output_tokens.write_text("".join(line + "\n" for line, _ in selected))
            pq.write_table(filtered, output_native)
        restored = pq.read_table(output_native)
        if not restored.equals(filtered):
            raise ValueError(f"Native output readback differs in {split}")
        restored_tokens = [json.loads(line) for line in output_tokens.read_text().splitlines()]
        if restored_tokens != [row for _, row in selected]:
            raise ValueError(f"Token output readback differs in {split}")
        if split == "train" and any(
            row["prompt_sha256"] in val_hashes or tuple(row["prompt_token_ids"]) in val_tokens for row in restored_tokens
        ):
            raise ValueError("Held-out prompt survived filtering")
        report["inputs"][split] = {
            "tokens": str(token_path),
            "tokens_sha256": file_hash(token_path),
            "native": str(native_path),
            "native_sha256": file_hash(native_path),
        }
        report["outputs"][split] = {
            "rows": len(selected),
            "tokens_sha256": file_hash(output_tokens),
            "native_sha256": file_hash(output_native),
            "family_controls": dict(
                sorted(Counter(row["family"] + "/" + row["control"] for _, row in selected).items())
            ),
        }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = filter_structural(args.tokens, args.native, args.output)
    print(json.dumps({split: value["rows"] for split, value in result["outputs"].items()}))
