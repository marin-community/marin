# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay corrected native grade adapters on saved scores without regenerating responses.

The original checkers already ran. This changes only an absent pass flag, using
the same full-credit contract that subsequent native evaluations will execute.
Original dumps remain immutable; receipts hash each source and derived file.
"""

import argparse
import hashlib
import json
from collections import Counter

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from skyrl_train.rollouts.task_projections import verification_result
from taskcompendium.evaluation.corpus import parquet_rows
from taskcompendium.grading import grade_result
from taskcompendium.models import FileReward, ScriptGrader, StdoutReward, TaskSpec, VerifyitGrader, verifyit_spec
from taskcompendium.runtime.grading import file_grade, stdout_grade
from verifyit.grade import Reward, Status

from experiments.post_training.mochi_doom_ftpo.grades import full_credit_task


def replay_row(row: dict, task: TaskSpec) -> dict:
    """Preserve evidence and checker scores, filling only the native pass metadata."""
    original = row.get("verification") or {}
    if original.get("status") != "verified":
        return row
    if original.get("score_min", 0.0) != 0.0 or original.get("score_max", 1.0) != 1.0:
        raise ValueError("Campaign checkers must have their declared native [0, 1] range")
    score = original["score"]
    if not 0.0 <= score <= 1.0:
        raise ValueError("Campaign checker returned a score outside [0, 1]")
    grader = full_credit_task(task).grader
    if isinstance(grader, VerifyitGrader):
        grade = grade_result(verifyit_spec(grader), Reward(score, Status.SCORED))
    else:
        assert isinstance(grader, ScriptGrader)
        if isinstance(grader.reward, StdoutReward):
            grade = stdout_grade(str(score).encode(), grader.reward, {})
        else:
            assert isinstance(grader.reward, FileReward)
            grade = file_grade(str(score).encode(), grader.reward.files[0], grader.reward, {})
    replayed = verification_result(grade)
    if original.get("passed") is not None and original["passed"] is not replayed.passed:
        raise ValueError(f"Explicit native pass disagrees for {task.id}")
    return {**row, "verification": {**original, "passed": replayed.passed}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, help="Frozen full_validation.parquet")
    parser.add_argument("--input", action="append", required=True, help="PROFILE=JSONL_GLOB")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    tasks = {}
    for item in parquet_rows(StoragePath(args.tasks)):
        task = TaskSpec.model_validate_json(item["task_spec"])
        tasks[task.id] = task
    root = StoragePath(args.output_dir)
    receipts = []
    for item in args.input:
        profile, pattern = item.split("=", 1)
        for source in sorted(StoragePath(pattern).glob(), key=str):
            if source.name == "aggregated_results.jsonl":
                continue
            target = root / profile / source.name
            if target.exists():
                raise FileExistsError(str(target))
            source_hash = hashlib.sha256()
            target_hash = hashlib.sha256()
            counts = Counter()
            with source.open("rb") as reader, target.open("wb") as writer:
                for line in reader:
                    source_hash.update(line)
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    updated = replay_row(row, tasks[row["task_id"]])
                    counts[str(updated["verification"].get("passed"))] += 1
                    payload = (json.dumps(updated, ensure_ascii=False) + "\n").encode()
                    writer.write(payload)
                    target_hash.update(payload)
            receipts.append({"source": str(source), "source_sha256": source_hash.hexdigest(),
                             "output": str(target), "output_sha256": target_hash.hexdigest(), "counts": dict(counts)})
    receipt = {"tasks": args.tasks, "policy": "native full credit; Verifyit=1, StdoutReward>=1, FileReward>=1",
               "operation": "adapter replay only; original checker scores and generation evidence preserved",
               "files": receipts}
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
