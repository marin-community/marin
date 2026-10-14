# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize audited BFCL training tasks after reserving protocol diagnostics."""

import argparse
import gzip
import hashlib
import json
import zipfile
from dataclasses import asdict
from pathlib import Path

from tokenizers import Tokenizer

from experiments.post_training.bfcl_rl.data import BFCLPartition, TaskIdentity, materialize_complement, task_identity
from experiments.post_training.skill_acquisition.data import ReplayCase, replay_prefixes


def build_training_replay_cases(
    archive_root: Path,
    partition_file: Path,
    diagnostic_manifest: Path,
    tokenizer_file: Path,
    output: Path,
    *,
    version: str,
    excluded_records: dict[str, str],
) -> dict:
    """Select one exact historical prefix per training task under the fixed split."""
    if output.exists():
        raise FileExistsError(output)
    partition_bytes = partition_file.read_bytes()
    diagnostic_bytes = diagnostic_manifest.read_bytes()
    partition = json.loads(partition_bytes)
    diagnostic = json.loads(diagnostic_bytes)
    if hashlib.sha256(partition_bytes).hexdigest() != diagnostic["source_partition_sha256"]:
        raise ValueError("Replay split refers to a different BFCL partition")
    if hashlib.sha256(tokenizer_file.read_bytes()).hexdigest() != diagnostic["tokenizer_sha256"]:
        raise ValueError("Replay tokenizer differs from the diagnostic tokenizer")
    names = {item["name"]: item["source_id"] for item in partition["complement"]}
    train = set(diagnostic["train_task_ids"])
    reserved = set(diagnostic["diagnostic_reserved_task_ids"])
    parity = set(diagnostic["parity_task_ids"])
    if train & reserved or (train | reserved) != set(names.values()) or (train | reserved) & parity:
        raise ValueError("Replay training split overlaps or does not partition the approved complement")
    if parity != {item["source_id"] for item in partition["parity"]}:
        raise ValueError("Replay BFCL-Parity identities differ from the pinned partition")
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    cases: dict[str, ReplayCase] = {}
    failures = []
    exclusions = []
    selected_rows = 0
    oversized = 0
    candidates = 0
    for source in diagnostic["sources"]:
        path = archive_root / Path(source["path"]).name
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != source["sha256"]:
                raise ValueError(f"Replay archive changed: {path}")
        with zipfile.ZipFile(path) as archive:
            for member in sorted(archive.namelist()):
                if not member.endswith(".json.gz"):
                    continue
                row = json.loads(gzip.decompress(archive.read(member)))
                task_id = names.get(row["trajectory"]["instance_id"])
                if task_id not in train:
                    continue
                selected_rows += 1
                prompt = row["prompt"]["token_ids"]
                ids = prompt + row["response"]["token_ids"]
                mask = [0] * len(prompt) + row["response"]["loss_mask"]
                try:
                    prefixes = replay_prefixes(ids, mask, tokenizer)
                except ValueError as error:
                    entry = {"file": path.name, "record": row["record_id"], "error": str(error)}
                    if excluded_records.get(row["record_id"]) == str(error):
                        exclusions.append(entry)
                    else:
                        failures.append(entry)
                    continue
                for prefix, status in prefixes:
                    if len(prefix) > diagnostic["max_prefix_tokens"]:
                        oversized += 1
                        continue
                    candidates += 1
                    identity = hashlib.sha256(json.dumps(prefix, separators=(",", ":")).encode()).hexdigest()
                    candidate = ReplayCase(
                        task_id,
                        path.name,
                        row["record_id"],
                        row["phase"],
                        row["provenance"]["model_source_identity"],
                        hashlib.sha256(json.dumps([ids, mask], separators=(",", ":")).encode()).hexdigest(),
                        prefix,
                        identity,
                        status,
                    )
                    if task_id not in cases or identity < cases[task_id].prefix_sha256:
                        cases[task_id] = candidate
    if failures:
        raise ValueError(json.dumps({"conversion_failures": failures}, indent=2))
    if {entry["record"] for entry in exclusions} != set(excluded_records):
        raise ValueError("Declared replay exclusions were not reproduced")
    if not cases:
        raise ValueError("No training replay cases found")
    output.mkdir(parents=True)
    case_file = output / "cases.jsonl"
    with case_file.open("x") as stream:
        for task_id in sorted(cases):
            stream.write(json.dumps(asdict(cases[task_id]), separators=(",", ":")) + "\n")
    manifest = {
        **diagnostic,
        "version": version,
        "split_manifest_sha256": hashlib.sha256(diagnostic_bytes).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection": "one prefix per training task; smallest prefix SHA-256",
        "training_cases": len(cases),
        "training_rows": selected_rows,
        "candidate_prefixes_within_budget": candidates,
        "oversized_prefixes": oversized,
        "exclusions": exclusions,
        "cases_sha256": hashlib.sha256(case_file.read_bytes()).hexdigest(),
    }
    manifest.pop("diagnostic_cases")
    manifest.pop("selected_rows")
    with (output / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return manifest


def materialize_training_tasks(source: Path, partition_file: Path, diagnostic_manifest: Path, output: Path) -> dict:
    """Copy the unchanged training subset with immutable split provenance."""
    if output.exists():
        raise FileExistsError(output)
    partition_bytes = partition_file.read_bytes()
    diagnostic_bytes = diagnostic_manifest.read_bytes()
    partition = json.loads(partition_bytes)
    diagnostic = json.loads(diagnostic_bytes)
    if hashlib.sha256(partition_bytes).hexdigest() != diagnostic["source_partition_sha256"]:
        raise ValueError("Diagnostic reservations refer to a different BFCL partition")
    approved = tuple(TaskIdentity(**item) for item in partition["complement"])
    parity = tuple(TaskIdentity(**item) for item in partition["parity"])
    approved_ids = {item.source_id for item in approved}
    parity_ids = {item.source_id for item in parity}
    train_ids = set(diagnostic["train_task_ids"])
    reserved_ids = set(diagnostic["diagnostic_reserved_task_ids"])
    if (
        train_ids & reserved_ids
        or (train_ids | reserved_ids) != approved_ids
        or approved_ids & parity_ids
        or parity_ids != set(diagnostic["parity_task_ids"])
    ):
        raise ValueError("Training and diagnostic IDs must partition the complement and exclude BFCL-Parity")
    selected = tuple(item for item in approved if item.source_id in train_ids)
    failures = []
    for expected in selected:
        try:
            actual = task_identity(source / expected.name)
        except (OSError, ValueError, KeyError) as error:
            failures.append({"task": expected.name, "error": str(error)})
            continue
        if actual != expected:
            failures.append({"task": expected.name, "error": "task identity differs from pinned partition"})
    if failures:
        raise ValueError(json.dumps({"conversion_failures": failures}, indent=2))
    materialize_complement(BFCLPartition(partition["dataset_commit"], selected, parity), source, output / "tasks")
    manifest = {
        "diagnostic_version": diagnostic["version"],
        "source_partition_sha256": hashlib.sha256(partition_bytes).hexdigest(),
        "diagnostic_manifest_sha256": hashlib.sha256(diagnostic_bytes).hexdigest(),
        "source_dataset_commit": partition["dataset_commit"],
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_tasks": len(selected),
        "train_task_ids": sorted(train_ids),
        "diagnostic_reserved_task_ids": sorted(reserved_ids),
        "parity_task_ids": sorted(parity_ids),
        "transformation": "unchanged task files; exclude diagnostic reservations",
    }
    with (output / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--diagnostic-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = materialize_training_tasks(args.source, args.partition, args.diagnostic_manifest, args.output)
    print(json.dumps({"training_tasks": manifest["training_tasks"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
