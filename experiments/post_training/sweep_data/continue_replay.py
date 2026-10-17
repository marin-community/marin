# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extend the frozen task #202 replay slice with the next rows of the same seeded sampler (a new artifact version).

The #202 slice stopped each partition's seeded draw stream ('<SEED>:<partition>') once its quota was met. This replays
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
import json
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
from huggingface_hub import HfFileSystem
from marin.datakit.sft_sources import NEMOTRON_SFT_V3_TOKEN_COUNTS_B
from rigging.filesystem.storage_path import StoragePath
from transformers import AutoTokenizer

from experiments.post_training.sweep_data.prepare_agentic import (
    BATCH,
    PAD_TOKEN,
    STUDENT_REGIONAL_COPY,
    TOKENIZER_FILES,
    verify_rows,
    write_rows,
)
from experiments.post_training.sweep_data.prepare_replay import EXCLUDED, SEED, sample_partition
from experiments.post_training.sweep_data.render import training_row
from experiments.post_training.sweep_data.select import CONTEXT


def position_key(position: dict) -> tuple:
    return tuple(position.get(key) for key in ("partition", "file", "byte_offset", "row_group", "row", "sha256"))


def quotas_for(target: int) -> dict[str, int]:
    """prepare_replay's quota rule: shares proportional to NEMOTRON_SFT_V3_TOKEN_COUNTS_B, at least 1."""
    weights = {name: size for name, size in NEMOTRON_SFT_V3_TOKEN_COUNTS_B.items() if name not in EXCLUDED}
    total = sum(weights.values())
    return {name: max(1, round(target * size / total)) for name, size in sorted(weights.items())}


def row_line(record: dict) -> str:
    return json.dumps(training_row(record["ids"], record["mask"], CONTEXT, PAD_TOKEN), separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="frozen #202 replay prefix")
    parser.add_argument("--target-supervised", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = StoragePath(args.output)
    if output.exists():
        raise ValueError("Use a fresh output prefix")
    base = StoragePath(args.base)
    base_manifest_bytes = (base / "manifest.json").read_bytes()
    base_manifest = json.loads(base_manifest_bytes)
    files = {name: (base / name).read_bytes() for name in ("replay-prebuilt.jsonl.gz", "replay-rows.jsonl")}
    for name, data in files.items():
        if hashlib.sha256(data).hexdigest() != base_manifest["output_hashes"][name]:
            raise ValueError(f"Base file changed: {name}")
    base_lines = gzip.decompress(files["replay-prebuilt.jsonl.gz"]).decode().splitlines()[: base_manifest["rows"]]
    base_index = [json.loads(line) for line in files["replay-rows.jsonl"].decode().splitlines()]
    if len(base_index) != len(base_lines):
        raise ValueError("Base index and rows differ in length")
    base_quotas = base_manifest["partition_quotas"]
    if quotas_for(base_manifest["target_supervised_tokens"]) != base_quotas:
        raise ValueError("The quota rule does not reproduce the base slice's quotas")
    quotas = quotas_for(args.target_supervised)
    if any(quotas[name] < base_quotas[name] for name in base_quotas) or set(quotas) != set(base_quotas):
        raise ValueError("New quotas must cover the base quotas, partition by partition")
    base_line_by_key = {position_key(entry): line for entry, line in zip(base_index, base_lines, strict=True)}
    base_rows_by_partition = collections.Counter(entry["partition"] for entry in base_index)

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        (directory / "tokenizer").mkdir()
        tokenizer_hashes = {}
        for name in TOKENIZER_FILES:
            data = fsspec.open(f"{STUDENT_REGIONAL_COPY}/{name}", "rb").open().read()
            tokenizer_hashes[name] = hashlib.sha256(data).hexdigest()
            (directory / "tokenizer" / name).write_bytes(data)
        tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer")
        fs = HfFileSystem()

        def run(name: str) -> tuple[str, list[dict], dict]:
            records, report = sample_partition(tokenizer, fs, name, quotas[name])
            # The base stream stopped at the first row that brought its supervised total to the base quota.
            supervised, split = 0, 0
            while supervised < base_quotas[name]:
                supervised += sum(records[split]["mask"][1:])
                split += 1
            if split != base_rows_by_partition[name]:
                raise ValueError(
                    f"{name}: replay reaches the base quota after {split} rows, base has "
                    f"{base_rows_by_partition[name]}"
                )
            for record in records[:split]:
                line = base_line_by_key.get(position_key(record["position"]))
                if line is None or line != row_line(record):
                    raise ValueError(f"{name}: re-drawn row {record['trajectory']} differs from the frozen slice")
            continuation = records[split:]
            summary = {
                "name": name,
                "base_quota": base_quotas[name],
                "quota": quotas[name],
                "base_rows_matched": split,
                "continuation_rows": len(continuation),
                "continuation_supervised_tokens": sum(sum(r["mask"][1:]) for r in continuation),
                "draws": report["draws"],
                "rejected": report["rejected"],
                "normalized": report["normalized"],
            }
            print("CONTINUE_PARTITION " + json.dumps(summary), flush=True)
            return name, continuation, summary

        with ThreadPoolExecutor(8) as pool:
            results = list(pool.map(run, sorted(quotas)))
        continuation = [record for _, records, _ in results for record in records]
        random.Random(f"{SEED}:continuation").shuffle(continuation)
        out = directory / "out"
        out.mkdir()
        padding = -(len(base_lines) + len(continuation)) % BATCH
        prebuilt = out / "replay-prebuilt.jsonl.gz"
        with gzip.open(prebuilt, "wt", compresslevel=6) as stream:
            for line in base_lines:
                stream.write(line + "\n")
            for record in continuation:
                stream.write(row_line(record) + "\n")
            empty = json.dumps(
                {"input_ids": [PAD_TOKEN] * CONTEXT, "loss_weights": [0] * CONTEXT}, separators=(",", ":")
            )
            for _ in range(padding):
                stream.write(empty + "\n")
        # Independent re-read: the base rows are byte-identical and in order; continuation rows match their rendering.
        with gzip.open(prebuilt, "rt") as stream:
            written = stream.read().splitlines()
        if written[: len(base_lines)] != base_lines:
            raise ValueError("Base rows changed on write")
        scratch = out / "continuation-check.jsonl.gz"
        rows, check_padding = write_rows(scratch, continuation)
        continuation_supervised = verify_rows(scratch, continuation, check_padding)
        scratch.unlink()
        if written[len(base_lines) : len(base_lines) + rows] != [row_line(record) for record in continuation]:
            raise ValueError("Continuation rows differ from their rendering")
        base_supervised = sum(entry["supervised_tokens"] for entry in base_index)
        index = [entry | {"continuation": False} for entry in base_index] + [
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
            key: base_manifest[key]
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
                "manifest_sha256": hashlib.sha256(base_manifest_bytes).hexdigest(),
                "rows": len(base_lines),
                "supervised_tokens": base_supervised,
            },
            "base_rows": len(base_lines),
            "rule": (
                "each partition's seeded stream replayed from its start with the #202 sampler to quotas with the same "
                "shares; the rows up to the base quota equal the frozen slice byte for byte, the rows after it are the "
                "continuation; frozen rows first in frozen order, continuation after (seeded shuffle)"
            ),
            "continuation_seed": f"{SEED}:continuation",
            "target_supervised_tokens": args.target_supervised,
            "partition_quotas": quotas,
            "base_partition_quotas": base_quotas,
            "partitions": {name: summary for name, _, summary in results},
            "tokenizer_files_sha256": tokenizer_hashes,
            "rows": len(base_lines) + rows,
            "padding_rows": padding,
            "continuation_rows": rows,
            "supervised_tokens": base_supervised + continuation_supervised,
            "continuation_supervised_tokens": continuation_supervised,
            "context_tokens": sum(entry["context_tokens"] for entry in index),
            "output_hashes": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(out.iterdir())
            },
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        output.mkdirs()
        for path in sorted(out.iterdir(), key=lambda path: path.name == "manifest.json"):
            (output / path.name).write_bytes(path.read_bytes())
            if (
                path.name != "manifest.json"
                and hashlib.sha256((output / path.name).read_bytes()).hexdigest() != manifest["output_hashes"][path.name]
            ):
                raise ValueError(f"Upload hash mismatch: {path.name}")
        print(
            "REPLAY_CONTINUED "
            + json.dumps({k: v for k, v in manifest.items() if k not in ("partitions", "rendering")})
            + " manifest_sha256="
            + hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest(),
            flush=True,
        )


if __name__ == "__main__":
    main()
