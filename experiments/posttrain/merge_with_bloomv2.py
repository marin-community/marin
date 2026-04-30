#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Merge pilot tension pairs into a new bloomv2-style preference dataset.

Default strategy: preserve the original bloomv2 shards unchanged and append a
new "pilot" shard to each of train/ and val_deduped/. This avoids re-encoding
~500k records and keeps the original schema bit-for-bit intact on disk.

Train/val split: variant-held-out.
    variant_idx == val_variant_idx → val
    otherwise → train.

Target dataset structure:
    train/shard-00000.jsonl.gz   (copy of bloomv2 shard 0)
    train/shard-00001.jsonl.gz   (copy of bloomv2 shard 1)
    ...
    train/shard-000XX.jsonl.gz   (copy of bloomv2 shard XX)
    train/shard-pilot.jsonl.gz   (pilot train pairs)
    val_deduped/shard-00000.jsonl.gz   (copy of bloomv2 val)
    val_deduped/shard-pilot.jsonl.gz   (pilot val pairs)

Record schema (bloomv2-compatible):
    {chosen, rejected, hash, prompt, statement_id, question_id}
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

BLOOMV2_SOURCE = "gs://marin-us-central1/preference/bloom_openai_model_spec_v2_gpt41_vs_mixtral_opposite/"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_chat_turns(prompt: str, response: str) -> list[dict]:
    return [
        {"role": "user", "content": prompt, "name": None, "tool_calls": None, "tool_call_id": None},
        {"role": "assistant", "content": response, "name": None, "tool_calls": None, "tool_call_id": None},
    ]


def pair_to_bloomv2_record(pair: dict) -> dict:
    prompt = pair["prompt"]
    chosen = pair["chosen_response"]
    rejected = pair["rejected_response"]
    h = hashlib.sha256((prompt + "|C|" + chosen + "|R|" + rejected).encode("utf-8")).hexdigest()[:24]
    qid = (
        f"tension::{pair['pair_id']}"
        f"::tp{pair['tension_point_idx']:03d}"
        f"::v{pair['variant_idx']:02d}"
        f"::c{pair.get('chosen_draw_idx', 0):02d}"
        f"::r{pair['rejected_sample_idx']:02d}"
    )
    return {
        "chosen": build_chat_turns(prompt, chosen),
        "rejected": build_chat_turns(prompt, rejected),
        "hash": h,
        "prompt": prompt,
        "statement_id": pair["pair_id"],
        "question_id": qid,
    }


def split_pairs(pairs: list[dict], val_variant_idx: int) -> tuple[list[dict], list[dict]]:
    train = [p for p in pairs if p["variant_idx"] != val_variant_idx]
    val = [p for p in pairs if p["variant_idx"] == val_variant_idx]
    return train, val


def write_shard(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def dedup(records: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for r in records:
        if r["hash"] in seen:
            continue
        seen.add(r["hash"])
        out.append(r)
    return out


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {cmd}\nstdout:{p.stdout}\nstderr:{p.stderr}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-pairs", type=Path, default=Path("experiments/posttrain/stage4_output/tier_b/pairs_tier_b.jsonl")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/posttrain/stage4_output/tier_b/dataset"))
    parser.add_argument("--dataset-name", type=str, default="bloomv2_m2")
    parser.add_argument("--val-variant-idx", type=int, default=0)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--upload-region", type=str, default="us-central1")
    parser.add_argument(
        "--skip-bloomv2-copy",
        action="store_true",
        help="Only write pilot shards locally; don't fetch or copy bloomv2 base.",
    )
    args = parser.parse_args()

    pilot_pairs = load_jsonl(args.pilot_pairs)
    print(f"Loaded {len(pilot_pairs)} pilot pairs from {args.pilot_pairs}")

    train_pilot, val_pilot = split_pairs(pilot_pairs, args.val_variant_idx)
    train_records = dedup([pair_to_bloomv2_record(p) for p in train_pilot])
    val_records = dedup([pair_to_bloomv2_record(p) for p in val_pilot])
    print(f"Pilot after dedup: {len(train_records)} train, {len(val_records)} val")

    ds_dir = args.output_dir / args.dataset_name
    train_dir = ds_dir / "train"
    val_dir = ds_dir / "val_deduped"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    train_pilot_shard = train_dir / "shard-pilot.jsonl.gz"
    val_pilot_shard = val_dir / "shard-pilot.jsonl.gz"
    write_shard(train_records, train_pilot_shard)
    write_shard(val_records, val_pilot_shard)
    print(f"Wrote pilot train shard to {train_pilot_shard}")
    print(f"Wrote pilot val   shard to {val_pilot_shard}")

    # README with provenance.
    (ds_dir / "README.md").write_text(
        f"# {args.dataset_name}\n\n"
        f"Base: {BLOOMV2_SOURCE}\n"
        f"Added: {len(train_records)} train + {len(val_records)} val tension-corner\n"
        f"preference pairs from the Tier B pilot\n"
        f"(.agents/logbooks/claude_stress_testing.md Experiments 14-16).\n\n"
        f"Original bloomv2 shards are copied unchanged. Pilot pairs live in\n"
        f"`train/shard-pilot.jsonl.gz` and `val_deduped/shard-pilot.jsonl.gz`.\n"
    )

    if args.upload:
        gcs_prefix = f"gs://marin-{args.upload_region}/preference/{args.dataset_name}/"

        if not args.skip_bloomv2_copy:
            # Server-side copy of bloomv2 shards (fast, no local download).
            print(f"Server-side copy bloomv2 train shards → {gcs_prefix}train/")
            run(["gcloud", "storage", "cp", f"{BLOOMV2_SOURCE}train/*.jsonl.gz", f"{gcs_prefix}train/"])
            print(f"Server-side copy bloomv2 val shards   → {gcs_prefix}val_deduped/")
            run(["gcloud", "storage", "cp", f"{BLOOMV2_SOURCE}val_deduped/*.jsonl.gz", f"{gcs_prefix}val_deduped/"])

        # Upload the pilot shards + README.
        print(f"Uploading pilot shards + README → {gcs_prefix}")
        run(["gcloud", "storage", "cp", str(train_pilot_shard), f"{gcs_prefix}train/"])
        run(["gcloud", "storage", "cp", str(val_pilot_shard), f"{gcs_prefix}val_deduped/"])
        run(["gcloud", "storage", "cp", str(ds_dir / "README.md"), gcs_prefix])
        print("Uploaded.")
        # Verify.
        print("Verifying GCS upload:")
        run(["gcloud", "storage", "ls", f"{gcs_prefix}train/"])
        run(["gcloud", "storage", "ls", f"{gcs_prefix}val_deduped/"])
    else:
        print("(--upload not set; local only.)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
