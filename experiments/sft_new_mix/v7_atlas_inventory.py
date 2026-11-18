# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the V7 comp-prog Atlas inventory from the rebuilt source caches.

Reads the pinned V7 build receipts for the two comp-prog sources and the
pinned V6 apply-time inventory, and writes:

- a two-source V7 inventory for the Atlas population projection and scan;
- the full fifty-source apply inventory with the forty-eight unchanged V6
  source entries copied byte-identically so their population digest checks
  keep passing, plus the two new comp-prog entries.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath

from experiments.sft_new_mix.exp3_runtime import configure_storage, read_pinned

logger = logging.getLogger(__name__)

COMP_PROG_SOURCES = (
    "nemotron_sft_v3/competitive_programming_v2/competitive_coding_cpp",
    "nemotron_sft_v3/competitive_programming_v2/competitive_coding_python",
)
EXPECTED_JOBS = 101


def shard_entry(shard: str) -> dict:
    receipt_path = StoragePath(shard) / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("complete") is not True:
        raise ValueError(f"Unfinished V7 shard receipt: {receipt_path}")
    job = receipt["job"]
    lengths = receipt.get("lengths") or []
    if len(lengths) != receipt["output_rows"]:
        raise ValueError(f"V7 shard receipt length mismatch: {receipt_path}")
    return {
        "cache_uri": str(StoragePath(shard) / "cache"),
        "decisions_uri": str(StoragePath(shard) / "decisions.jsonl"),
        "original_input_path": job["input_path"],
        "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "receipt_uri": str(receipt_path),
        "rows": receipt["output_rows"],
        "tokens": receipt["output_tokens"],
        "supervised_prediction_positions": receipt["supervised_prediction_positions"],
    }


def build_source_entry(name: str, shards: list[dict], source_revision: str) -> dict:
    input_rows = sum(entry["rows"] for entry in shards)
    input_tokens = sum(entry["tokens"] for entry in shards)
    supervised = sum(entry["supervised_prediction_positions"] for entry in shards)
    return {
        "source": name,
        "component": "sft/source/" + name,
        "counts": {
            "completed_input_documents": input_rows,
            "completed_input_tokens": input_tokens,
            "final_stage_input_documents": input_rows,
            "final_stage_input_tokens": input_tokens,
            "input_documents": input_rows,
            "input_tokens": input_tokens,
            "output_documents": input_rows,
            "output_tokens": input_tokens,
            "supervised_prediction_positions": supervised,
        },
        "expected_shards": len(shards),
        "loss_mode": "chat_assistant_only",
        "loss_weights_indexing": "prediction",
        "loss_weights_key": "loss_weights",
        "source_revision": source_revision,
        "shards": shards,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-prefix", required=True, help="The V7 comp-prog build prefix")
    parser.add_argument("--build-plan", required=True)
    parser.add_argument("--build-plan-sha256", required=True)
    parser.add_argument("--v6-inventory", required=True)
    parser.add_argument("--v6-inventory-sha256", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    plan = json.loads(read_pinned(args.build_plan, args.build_plan_sha256))
    revisions = {}
    for job in plan["jobs"]:
        if job["source"] in COMP_PROG_SOURCES:
            revisions.setdefault(job["source"], set()).add(job["source_revision"])
    if set(revisions) != set(COMP_PROG_SOURCES) or any(len(values) != 1 for values in revisions.values()):
        raise ValueError(f"V7 plan source revisions are not unique per source: {revisions}")
    revisions = {name: values.pop() for name, values in revisions.items()}

    entries = []
    for name in COMP_PROG_SOURCES:
        source_hash = hashlib.sha256(name.encode()).hexdigest()
        shards_root = StoragePath(args.build_prefix) / "components" / source_hash / "shards"
        shard_dirs = sorted(str(entry) for entry in shards_root.ls())
        shards = [shard_entry(shard) for shard in shard_dirs]
        if not shards:
            raise ValueError(f"No V7 shard receipts under {shards_root}")
        entries.append(build_source_entry(name, shards, revisions[name]))
        logger.info("%s: %d shards, %d rows", name, len(shards), sum(s["rows"] for s in shards))
    if len({entry["shards"][0]["cache_uri"].rsplit("/shards/", 1)[0] for entry in entries}) != len(COMP_PROG_SOURCES):
        raise ValueError("V7 shard caches do not split cleanly per source")

    tokenizer_uri = plan["jobs"][0]["tokenizer_uri"]
    tokenizer_sha256 = plan["jobs"][0]["tokenizer_sha256"]

    two_source_inventory = {
        "complete": True,
        "completed_shards": EXPECTED_JOBS,
        "expected_shards": EXPECTED_JOBS,
        "missing_receipts": [],
        "scope": "V7 option-G comp-prog candidate for the Atlas lexical scan",
        "sources": entries,
        "tokenizer_sha256": tokenizer_sha256,
        "tokenizer_uri": tokenizer_uri,
    }
    v6_inventory = json.loads(read_pinned(args.v6_inventory, args.v6_inventory_sha256))
    replaced = {entry["source"]: entry for entry in entries}
    sources = []
    seen = set()
    for source in v6_inventory["sources"]:
        name = source.get("source")
        if name in replaced:
            sources.append(replaced[name])
            seen.add(name)
        else:
            sources.append(source)
    if seen != set(COMP_PROG_SOURCES):
        raise ValueError(f"The V6 inventory does not carry the comp-prog entries: found {sorted(seen)}")
    full_inventory = {**v6_inventory, "sources": sources}

    destination = StoragePath(args.destination)
    artifacts = {}
    for name, inventory in (
        ("v7-comp-prog-inventory.json", two_source_inventory),
        ("v7-full-inventory.json", full_inventory),
    ):
        blob = json.dumps(inventory, sort_keys=True, indent=2).encode()
        (destination / name).write_bytes(blob)
        (output / name).write_bytes(blob)
        artifacts[name] = hashlib.sha256(blob).hexdigest()
    receipt = {
        "build_plan_sha256": args.build_plan_sha256,
        "v6_inventory_sha256": args.v6_inventory_sha256,
        "inventories_sha256": artifacts,
        "tokenizer_sha256": tokenizer_sha256,
        "rows": {entry["source"]: entry["counts"]["output_documents"] for entry in entries},
        "tokens": {entry["source"]: entry["counts"]["output_tokens"] for entry in entries},
    }
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2))
    (destination / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
