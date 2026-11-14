# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the repaired CalibForge release to the long-loop data and heldouts."""

import argparse
import hashlib
import io
import json
import tarfile
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from taskcompendium.models import TaskSpec

from experiments.post_training.long_loop.rl_dataset_admission import (
    archived_task,
    exclusion_reasons,
    file_digest,
    matching_question,
    question_anchors,
    task_text,
    write_rows,
)

REVISION = "3b6c5b5533f1b702163f365b7ce869b834cb2994"
COHORT = "long-loop-smartie-calibforge"


def replace_cohort(previous: Path, release: Path, universes: Path, output: Path) -> dict:
    """Preserve released splits and exclude prior heldouts and evaluation matches."""
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((release / "manifest.json").read_text())
    for local, member in (("tasks.parquet", "tasks.parquet"), ("splits-manifest.json", "splits/manifest.json")):
        if file_digest(release / local) != manifest["files"][member]["sha256"]:
            raise ValueError(f"Release file digest mismatch: {member}")
    split_manifest = json.loads((release / "splits-manifest.json").read_text())
    membership = {}
    for split in ("train", "validation"):
        for identifier in split_manifest["splits"][split]["task_ids"]:
            source_row = identifier.split(":", 1)[1]
            if source_row in membership:
                raise ValueError("Released train/validation identities overlap")
            membership[source_row] = split
    original_validation = pq.read_table(previous / "validation.parquet").to_pylist()
    old_holdouts = {
        json.loads(row["task_spec"])["source"]["row"]
        for row in original_validation
        if json.loads(row["task_spec"])["source"]["dataset"] == COHORT
    }
    validation = [r for r in original_validation if json.loads(r["task_spec"])["source"]["dataset"] != COHORT]
    train = pq.read_table(previous / "swesmith-train.parquet").to_pylist()
    index = json.loads((previous / "evaluation-exclusions.json").read_text())
    for key in ("ids", "repos", "text_sha256"):
        index[key] = set(index[key])
    questions = []
    for path in sorted(universes.glob("*.json")):
        if path.name not in {"manifest.json", "mrcr.json", "NUPA.json"}:
            questions.extend(json.loads(path.read_text()))
    anchors = question_anchors(questions)
    records = []
    seen = set()
    for ordinal, row in enumerate(pq.read_table(release / "tasks.parquet").to_pylist()):
        identifier = f"{row['source']}/{row['path']}"
        if identifier not in membership or identifier in seen:
            raise ValueError(f"Release row has no unique split: {identifier}")
        seen.add(identifier)
        split = membership[identifier]
        with tarfile.open(fileobj=io.BytesIO(row["task_binary"]), mode="r:*") as archive:
            instruction = archive.extractfile("instruction.md").read().decode()
        reasons = exclusion_reasons(identifier, instruction, index, check_repo=False)
        if matching_question(instruction, anchors):
            reasons.append("exact_upstream_question_substring")
        if split == "train" and identifier in old_holdouts:
            reasons.append("prior_long_loop_validation_identity")
        record = {
            "source_row": ordinal,
            "id": identifier,
            "release_split": split,
            "archive_sha256": hashlib.sha256(row["task_binary"]).hexdigest(),
            "exclusion_reasons": reasons,
        }
        if reasons:
            record["disposition"] = "excluded"
        else:
            task = archived_task(row, cohort="calibforge", revision=REVISION)
            normalized = {"task_spec": task.model_dump_json()}
            record.update(
                task_id=task.id,
                normalized_task_sha256=hashlib.sha256(normalized["task_spec"].encode()).hexdigest(),
                disposition=split,
            )
            (train if split == "train" else validation).append(normalized)
        records.append(record)
    if seen != set(membership) or len(seen) != 324:
        raise ValueError("Release rows and declared split identities differ")
    train_contexts = {task_text(TaskSpec.model_validate_json(r["task_spec"])) for r in train}
    validation_contexts = {task_text(TaskSpec.model_validate_json(r["task_spec"])) for r in validation}
    if train_contexts & validation_contexts:
        raise ValueError("Mixed train/validation instructions overlap")
    write_rows(output / "agentic-train.parquet", train)
    write_rows(output / "validation.parquet", validation)
    summary = {
        "source_repository": "open-athena/task-trove",
        "source_revision": REVISION,
        "source_prefix": "releases/calibforge-shared-20261017-v2",
        "source_manifest_sha256": file_digest(release / "manifest.json"),
        "released_train": 259,
        "released_validation": 65,
        "calibforge_train": sum(r["disposition"] == "train" for r in records),
        "calibforge_validation": sum(r["disposition"] == "validation" for r in records),
        "swesmith_train": len(train) - sum(r["disposition"] == "train" for r in records),
        "agentic_train": len(train),
        "validation": len(validation),
        "exclusion_counts": dict(Counter(reason for r in records for reason in r["exclusion_reasons"])),
        "old_calibforge_holdouts_protected": sorted(old_holdouts),
        "records": records,
        "files": {},
        "limitations": [
            "Exact identities, normalized text and exact upstream question substrings checked; "
            "semantic overlap not certified."
        ],
    }
    for name in ("agentic-train.parquet", "validation.parquet"):
        summary["files"][name] = {"sha256": file_digest(output / name), "bytes": (output / name).stat().st_size}
    (output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("previous", "release", "universes", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = replace_cohort(args.previous, args.release, args.universes, args.output)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"records", "old_calibforge_holdouts_protected"}}, indent=2
        )
    )
