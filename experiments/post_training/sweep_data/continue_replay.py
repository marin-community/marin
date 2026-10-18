# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extend the frozen LR-sweep replay slice with the next rows of the same seeded sampler (a new artifact version).

The frozen slice stopped each partition's seeded draw stream ('<SEED>:<partition>') once its quota was met. This replays
every stream from the start with the same sampler (prepare_replay.sample_partition), but to quotas scaled to
--target-supervised with the same partition shares. The rows each stream yields up to the old quota must equal the
frozen slice's rows byte for byte (same source position, same source sha256, same rendered row); the build fails
otherwise. The rows after that point are the continuation.

Output (a new prefix): replay-prebuilt.jsonl.gz holds the frozen slice's rows unchanged and in their frozen order,
then the continuation rows (seeded shuffle '<SEED>:continuation'), then zero-loss padding to a multiple of 16.
replay-rows.jsonl has one entry per row (continuation rows marked). manifest.json records base_rows, so the row
builder orders the frozen rows exactly as it orders the frozen slice and appends the continuation after them: any cut
the frozen slice can reach is unchanged.

Usage:
    python -m experiments.post_training.sweep_data.continue_replay --base <frozen replay prefix> \\
        --target-supervised 10000000 --output <new prefix>
"""

import argparse
import collections
import gzip
import hashlib
import itertools
import json
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfFileSystem
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.sweep_data.prepare_agentic import (
    TOKENIZER_FILES,
    audit_hashes,
    load_tokenizer,
    output_hashes,
    read_component,
    row_line,
    upload_checked,
    verify_rows,
    write_prebuilt,
    write_rows,
)
from experiments.post_training.sweep_data.prepare_replay import SEED, quotas_for, sample_partition
from experiments.post_training.sweep_data.select import TrainingRecord

BASE_FILES = ("replay-prebuilt.jsonl.gz", "replay-rows.jsonl")


def position_key(position: dict) -> tuple:
    return tuple(position.get(key) for key in ("partition", "file", "byte_offset", "row_group", "row", "sha256"))


class Base:
    """The frozen slice: its manifest, its real row lines in frozen order and its per-row index."""

    def __init__(self, root: StoragePath):
        self.manifest, self.manifest_bytes, files = read_component(root, BASE_FILES)
        self.lines = gzip.decompress(files["replay-prebuilt.jsonl.gz"]).decode().splitlines()[: self.manifest["rows"]]
        self.index = [json.loads(line) for line in files["replay-rows.jsonl"].decode().splitlines()]
        if len(self.index) != len(self.lines):
            raise ValueError("Base index and rows differ in length")
        self.quotas = self.manifest["partition_quotas"]
        self.line_by_key = {position_key(entry): line for entry, line in zip(self.index, self.lines, strict=True)}
        self.rows_by_partition = collections.Counter(entry["partition"] for entry in self.index)


def continuation_quotas(base: Base, target: int) -> dict[str, int]:
    """Quotas for `target`; the shared quota rule must reproduce the base quotas and the new ones must cover them."""
    if quotas_for(base.manifest["target_supervised_tokens"]) != base.quotas:
        raise ValueError("The quota rule does not reproduce the base slice's quotas")
    quotas = quotas_for(target)
    if any(quotas[name] < base.quotas[name] for name in base.quotas) or set(quotas) != set(base.quotas):
        raise ValueError("New quotas must cover the base quotas, partition by partition")
    return quotas


def continue_partition(
    tokenizer, fs: HfFileSystem, base: Base, name: str, quota: int
) -> tuple[str, list[TrainingRecord], dict]:
    """Replay one partition's stream to `quota`; check its first rows equal the base rows and return the rest."""
    records, report = sample_partition(tokenizer, fs, name, quota)
    # The base stream stopped at the first row that brought its supervised total to the base quota.
    supervised, split = 0, 0
    while supervised < base.quotas[name]:
        supervised += sum(records[split]["mask"][1:])
        split += 1
    if split != base.rows_by_partition[name]:
        raise ValueError(
            f"{name}: replay reaches the base quota after {split} rows, base has {base.rows_by_partition[name]}"
        )
    for record in records[:split]:
        line = base.line_by_key.get(position_key(record["position"]))
        if line is None or line != row_line(record):
            raise ValueError(f"{name}: re-drawn row {record['trajectory']} differs from the frozen slice")
    continuation = records[split:]
    summary = {
        "name": name,
        "base_quota": base.quotas[name],
        "quota": quota,
        "base_rows_matched": split,
        "continuation_rows": len(continuation),
        "continuation_supervised_tokens": sum(sum(r["mask"][1:]) for r in continuation),
        "draws": report["draws"],
        "rejected": report["rejected"],
        "normalized": report["normalized"],
    }
    print("CONTINUE_PARTITION " + json.dumps(summary), flush=True)
    return name, continuation, summary


def write_continued(out: Path, base: Base, continuation: list[TrainingRecord]) -> tuple[int, int, int]:
    """Write the base rows, then the continuation, then padding; re-read and check both parts.

    Returns (continuation rows, padding rows, continuation supervised tokens).
    """
    prebuilt = out / "replay-prebuilt.jsonl.gz"
    _, padding = write_prebuilt(prebuilt, itertools.chain(base.lines, (row_line(record) for record in continuation)))
    # Independent re-read: the base rows are byte-identical and in order; continuation rows match their rendering.
    with gzip.open(prebuilt, "rt") as stream:
        written = stream.read().splitlines()
    if written[: len(base.lines)] != base.lines:
        raise ValueError("Base rows changed on write")
    scratch = out / "continuation-check.jsonl.gz"
    rows, check_padding = write_rows(scratch, continuation)
    continuation_supervised = verify_rows(scratch, continuation, check_padding)
    scratch.unlink()
    if written[len(base.lines) : len(base.lines) + rows] != [row_line(record) for record in continuation]:
        raise ValueError("Continuation rows differ from their rendering")
    return rows, padding, continuation_supervised


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="frozen replay slice prefix (prepare_replay output)")
    parser.add_argument("--target-supervised", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = StoragePath(args.output)
    if output.exists():
        raise ValueError("Use a fresh output prefix")
    base = Base(StoragePath(args.base))
    quotas = continuation_quotas(base, args.target_supervised)
    expected = audit_hashes()

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        (directory / "tokenizer").mkdir()
        tokenizer = load_tokenizer(directory / "tokenizer", expected)
        fs = HfFileSystem()

        with ThreadPoolExecutor(8) as pool:
            results = list(
                pool.map(lambda name: continue_partition(tokenizer, fs, base, name, quotas[name]), sorted(quotas))
            )
        continuation = [record for _, records, _ in results for record in records]
        random.Random(f"{SEED}:continuation").shuffle(continuation)
        out = directory / "out"
        out.mkdir()
        rows, padding, continuation_supervised = write_continued(out, base, continuation)
        base_supervised = sum(entry["supervised_tokens"] for entry in base.index)
        index = [entry | {"continuation": False} for entry in base.index] + [
            record["position"]
            | {
                "supervised_tokens": sum(record["mask"][1:]),
                "context_tokens": len(record["ids"]),
                "continuation": True,
            }
            for record in continuation
        ]
        (out / "replay-rows.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in index))
        manifest = {
            key: base.manifest[key]
            for key in (
                "task",
                "component",
                "source",
                "sampled_subset",
                "excluded_partitions",
                "seed",
                "student_model",
                "student_revision",
                "pad_token_id",
                "max_sequence_tokens",
                "batch_size",
                "rendering",
                "loss_mask",
            )
        } | {
            "version": 2,
            "continuation_of": {
                "uri": args.base,
                "manifest_sha256": hashlib.sha256(base.manifest_bytes).hexdigest(),
                "rows": len(base.lines),
                "supervised_tokens": base_supervised,
            },
            "base_rows": len(base.lines),
            "rule": (
                "each partition's seeded stream replayed from its start with the #202 sampler to quotas with the same "
                "shares; the rows up to the base quota equal the frozen slice byte for byte, the rows after it are the "
                "continuation; frozen rows first in frozen order, continuation after (seeded shuffle)"
            ),
            "continuation_seed": f"{SEED}:continuation",
            "target_supervised_tokens": args.target_supervised,
            "partition_quotas": quotas,
            "base_partition_quotas": base.quotas,
            "partitions": {name: summary for name, _, summary in results},
            "tokenizer_files_sha256": {name: expected[name] for name in TOKENIZER_FILES},
            "rows": len(base.lines) + rows,
            "padding_rows": padding,
            "continuation_rows": rows,
            "supervised_tokens": base_supervised + continuation_supervised,
            "continuation_supervised_tokens": continuation_supervised,
            "context_tokens": sum(entry["context_tokens"] for entry in index),
            "output_hashes": output_hashes(out),
        }
        upload_checked(out, output, manifest)
        print(
            "REPLAY_CONTINUED "
            + json.dumps({k: v for k, v in manifest.items() if k not in ("partitions", "rendering")})
            + " manifest_sha256="
            + hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest(),
            flush=True,
        )


if __name__ == "__main__":
    main()
