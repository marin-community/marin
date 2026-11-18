# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract the think-length drop manifest for the S5 measurement sources.

Reads the pinned scripted-v3 decision ledgers for ``math_v3`` and
``openthoughts4-code-glm-5.2-n4`` and writes, per source, the dropped rows
(part path, part-local row, token count) plus the kept-row length histogram.
The measurement stage (see ``s5_think_loss_measure.py``) consumes the manifest
in the cache region.
"""

import argparse
import gzip
import hashlib
import json
import logging
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath

from experiments.sft_new_mix.exp3_runtime import configure_storage, read_pinned

logger = logging.getLogger(__name__)

SOURCES = {
    "math_v3": "nemotron_sft_v3/math_v3/train",
    "openthoughts": "openthoughts4-code-glm-5.2-n4",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scripted-prefix", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    plan = json.loads(read_pinned(args.plan, args.plan_sha256))
    by_source = {}
    for job in plan["jobs"]:
        if job["source"] in SOURCES.values():
            by_source.setdefault(job["source"], []).append(job)

    receipt = {"plan_sha256": args.plan_sha256, "sources": {}}
    for key, source in SOURCES.items():
        jobs = by_source[source]
        if not jobs:
            raise ValueError(f"No scripted jobs for {source}")
        source_hash = hashlib.sha256(source.encode()).hexdigest()
        shards_root = StoragePath(args.scripted_prefix) / "components" / source_hash / "shards"
        entries = sorted(str(entry) for entry in shards_root.ls())
        shard_dirs = [entry for entry in entries if (StoragePath(entry) / "receipt.json").isfile()]
        if not shard_dirs:
            raise ValueError(f"No scripted shards for {source}")
        drops = []
        kept_lengths = []
        stage_totals: dict[str, int] = {}
        covered_rows = 0
        covered_jobs = 0
        for shard in shard_dirs:
            stored = json.loads((StoragePath(shard) / "receipt.json").read_text())
            job = stored["job"]
            if stored["complete"] is not True or job["source"] != source:
                raise ValueError(f"Scripted receipt mismatch in {shard}")
            kept_lengths.extend(int(length) for length in stored["lengths"])
            covered_rows += job["row_stop"] - job["row_start"]
            covered_jobs += 1
            for line in (StoragePath(shard) / "decisions.jsonl").read_text().splitlines():
                record = json.loads(line)
                if record["stage"] == "retained":
                    continue
                stage = record["stage"]
                stage_totals[stage] = stage_totals.get(stage, 0) + 1
                if stage == "think_length":
                    drops.append(
                        {"input_path": job["input_path"], "row": int(record["row"]), "tokens": int(record["tokens"])}
                    )
        if covered_jobs != len(jobs) or covered_rows != sum(job["expected_rows"] for job in jobs):
            raise ValueError(f"Scripted shard coverage {covered_jobs} jobs / {covered_rows} rows differs from the plan")
        table = "\n".join(json.dumps(drop, sort_keys=True) for drop in drops)
        blob = gzip.compress(table.encode())
        path = output / f"{key}-think-length-drops.jsonl.gz"
        path.write_bytes(blob)
        receipt["sources"][key] = {
            "source": source,
            "jobs": len(jobs),
            "input_rows": sum(job["expected_rows"] for job in jobs),
            "stage_totals": stage_totals,
            "think_length_drops": len(drops),
            "think_length_drop_tokens": sum(drop["tokens"] for drop in drops),
            "kept_rows": len(kept_lengths),
            "kept_tokens": sum(kept_lengths),
            "manifest_sha256": hashlib.sha256(blob).hexdigest(),
        }
        logger.info("%s: %d think-length drops", key, len(drops))

    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2))
    destination = StoragePath(args.destination)
    for key in SOURCES:
        (destination / f"{key}-think-length-drops.jsonl.gz").write_bytes(
            (output / f"{key}-think-length-drops.jsonl.gz").read_bytes()
        )
    (destination / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2))
    print(json.dumps(receipt["sources"], sort_keys=True))


if __name__ == "__main__":
    main()
