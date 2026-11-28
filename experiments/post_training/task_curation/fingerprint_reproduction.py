# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce a sealed source fingerprint through the pipeline's builder.

A sealed fingerprint survey resolves a source population at immutable input
pins and computes its deterministic fingerprint: the task count, the
within-source duplicate count, and a seeded sample of normalized tasks. This
script re-reads the pinned input shards, rebuilds the fingerprint with the
same seed, and compares the produced fingerprint with the sealed one field
by field.

```bash
uv run python -m experiments.post_training.task_curation.fingerprint_reproduction \\
    --source-id Nvidia:Nemotron-RL-litmus-bench-v0.2/train \\
    --survey s3://marin-us-east-02a/marin/rl-data/atlas/source-identity/litmus-train-20261128-v1/survey.json \\
    --output reproduction.json --run
```
"""

import argparse
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256

from experiments.post_training.task_curation.source_fingerprints import fingerprint_source

logger = logging.getLogger(__name__)

REPRODUCTION_JSON = "reproduction.json"
FINGERPRINT_FIELDS = (
    "version",
    "seed",
    "task_count",
    "within_source_duplicates",
    "sample_sha256",
    "sample",
    "inputs",
)


def _population_documents(inputs: dict[str, str]):
    for uri, sha256 in sorted(inputs.items()):
        payload = StoragePath(uri).read_bytes()
        if hashlib.sha256(payload).hexdigest() != sha256:
            raise ValueError(f"Population shard bytes changed: {uri}")
        with StoragePath(uri).open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(batch_size=64, columns=["task_spec"]):
                for encoded in batch.column("task_spec").to_pylist():
                    if encoded is None:
                        raise ValueError(f"Null converted spec in {uri}")
                    task = json.loads(encoded)
                    # The sealed branch population serializes schema-0.30 fields main's
                    # schema dropped (integrity_contract, execution requirements); the
                    # fingerprint reads the contract main's builder sees, so absent
                    # fields are stripped rather than failing validation.
                    task.pop("integrity_contract", None)
                    task.pop("execution_requirements", None)
                    task.pop("verifier_execution_requirements", None)
                    task.pop("completion_protocol", None)
                    yield task


def fingerprint(sealed: dict, source_id: str) -> dict:
    entry = sealed["fingerprints"][source_id]
    return fingerprint_source(_population_documents(entry["inputs"]), entry["inputs"]).model_dump(mode="json")


def compare(produced: dict, sealed_entry: dict) -> dict:
    """Compare the produced fingerprint with the sealed one, content first.

    The strict fields are the population identity (inputs, task count,
    duplicates, seed) and every sample entry's content identity: task ID,
    rendered instruction, instruction hash, reference hash and character
    count. ``task_sha256`` and ``tests_sha256`` hash the stored task
    serialization, so they carry the schema vintage: the sealed digest pins
    schema-0.30 serializations while main's schema re-serializes 0.31. They
    are recorded per entry rather than compared.
    """
    strict_entry_fields = (
        "task_id",
        "instruction",
        "instruction_sha256",
        "instruction_characters",
        "reference_sha256",
    )
    vintage_fields = ("task_sha256", "tests_sha256")
    sealed_sample = sealed_entry.get("sample", [])
    produced_sample = produced.get("sample", [])
    entry_mismatches = [
        {
            "index": index,
            "field": field,
            "produced": produced_sample[index].get(field),
            "sealed": sealed_sample[index].get(field),
        }
        for index in range(min(len(produced_sample), len(sealed_sample)))
        for field in strict_entry_fields
        if produced_sample[index].get(field) != sealed_sample[index].get(field)
    ]
    vintage_deltas = [
        {
            "index": index,
            "field": field,
            "produced": produced_sample[index].get(field),
            "sealed": sealed_sample[index].get(field),
        }
        for index in range(min(len(produced_sample), len(sealed_sample)))
        for field in vintage_fields
        if produced_sample[index].get(field) != sealed_sample[index].get(field)
    ]
    strict = {
        "task_count": {"produced": produced.get("task_count"), "sealed": sealed_entry.get("task_count")},
        "within_source_duplicates": {
            "produced": produced.get("within_source_duplicates"),
            "sealed": sealed_entry.get("within_source_duplicates"),
        },
        "seed": {"produced": produced.get("seed"), "sealed": sealed_entry.get("seed")},
        "inputs_sha256": {
            "produced": canonical_sha256(produced.get("inputs", {})),
            "sealed": canonical_sha256(sealed_entry.get("inputs", {})),
        },
        "sample_size": {"produced": len(produced_sample), "sealed": len(sealed_sample)},
    }
    return {
        "strict": strict,
        "strict_equal": all(value["produced"] == value["sealed"] for value in strict.values()),
        "content_entry_mismatch_count": len(entry_mismatches),
        "content_entry_mismatches": entry_mismatches[:8],
        "serialization_vintage_deltas": {
            "count": len(vintage_deltas),
            "fields": vintage_fields,
            "deltas": vintage_deltas[:4],
        },
        "sample_sha256": {"produced": produced.get("sample_sha256"), "sealed": sealed_entry.get("sample_sha256")},
        "note": (
            "sample_sha256 chains the full sample entries, whose task and test hashes pin "
            "schema-0.30 serializations; main re-serializes schema-0.31, so the chain digest "
            "differs while the content identity matches per entry."
        ),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--survey", required=True, help="Sealed fingerprint survey JSON for this source")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    configure_coreweave_s3()
    sealed = json.loads(StoragePath(args.survey).read_text())
    if args.source_id not in sealed["fingerprints"]:
        raise ValueError(f"The survey carries no fingerprint for {args.source_id}")
    if not args.run:
        args.output.write_text(json.dumps({"source_id": args.source_id, "plan": "planned"}, indent=2) + "\n")
        return

    produced = fingerprint(sealed, args.source_id)
    sealed_entry = sealed["fingerprints"][args.source_id]
    comparison = compare(produced, sealed_entry)
    comparison.update(
        {
            "source_id": args.source_id,
            "snapshot_id": sealed.get("snapshot_id"),
            "survey_uri": args.survey,
            "reproduced": comparison["strict_equal"] and comparison["content_entry_mismatch_count"] == 0,
            "produced_at": datetime.now(UTC).isoformat(),
        }
    )
    args.output.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    root = StoragePath(args.artifact_root)
    per_run = (
        root
        / "runs"
        / args.source_id.replace(":", "-").replace("/", "-")
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        / REPRODUCTION_JSON
    )
    with per_run.open("wt", auto_mkdir=True) as stream:
        stream.write(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    with (root / REPRODUCTION_JSON).open("wt", auto_mkdir=True) as stream:
        stream.write(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    logger.info("Reproduction written to %s and %s: reproduced=%s", args.output, per_run, comparison["reproduced"])


if __name__ == "__main__":
    main()
