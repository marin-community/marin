# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze an immutable v2 BioTasks bank with independently verified v1 corrections.

Unresolved GLM audit flags are retained, not silently 'fixed'. Run locally with
private S3 credentials; the output manifest is written last.
"""

import argparse
import base64
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from taskcompendium.models import TaskSpec
from taskforge.content_hash import digest

from experiments.post_training.capability_driven_envs.biotasks_library.finalize_combined_v1 import (
    _read_bytes,
    _write_new,
)

ROOT = "s3://marin-us-east-02a/marin4sc/taskforge/task-library/campaigns/biotasks/combined/"
V1_TASKS = ROOT + "v1/snapshots/complete-1/taskspecs.jsonl"
FIRST_AUDIT = ROOT + "v1/audits/reference-recomputation-glm-v1/results.jsonl"
SECOND_AUDIT = ROOT + "v1/audits/reference-recomputation-glm-v1/adjudication.jsonl"
V2_ROOT = ROOT + "v2/snapshots/verified-corrections-1/"
CORRECTED_IDS = {
    "biotasks-read_counting_modes-78a95b29f211f753",
    "biotasks-loss_function_computation-771b920f400a1b98",
}


def _lines(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.splitlines() if line]


def _artifact(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in rows)


def _resources(task: TaskSpec) -> dict[str, str]:
    data = task.model_dump(mode="json")
    return {
        row["path"]: base64.b64decode(row["source"]["content_base64"]).decode("utf-8")
        for row in data["resources"]["worker"]
    }


def _fix_dice(task: TaskSpec) -> dict[str, Any]:
    resource = json.loads(_resources(task)["workspace/data/predictions_and_targets.json"])
    probabilities = resource["probabilities"]
    targets = resource["targets"]
    if set(probabilities) != set(targets) or len(probabilities) != 3:
        raise ValueError("Unexpected Dice channel contract")
    result: dict[str, Any] = {}
    scores = []
    for key, label in (("channel_0_wt", "dice_wt"), ("channel_1_tc", "dice_tc"), ("channel_2_et", "dice_et")):
        p = [Decimal(str(v)) for v in probabilities[key]]
        y = [Decimal(str(v)) for v in targets[key]]
        if len(p) != 8 or len(y) != 8:
            raise ValueError("Unexpected Dice channel size")
        score = 2 * sum((a * b for a, b in zip(p, y, strict=True)), Decimal(0)) / (sum(p) + sum(y))
        scores.append(score)
        result[label] = float(score.quantize(Decimal(".000001"), rounding=ROUND_HALF_UP))
    result["dice_loss"] = float((1 - sum(scores) / 3).quantize(Decimal(".000001"), rounding=ROUND_HALF_UP))
    if result != {"dice_wt": 0.724832, "dice_tc": 0.702179, "dice_et": 0.686275, "dice_loss": 0.295571}:
        raise ValueError("Dice recomputation did not match independently checked values")
    return result


def _fix_read_count(task: TaskSpec) -> dict[str, Any]:
    resources = _resources(task)
    features = [line.split("\t") for line in resources["workspace/data/features.tsv"].splitlines()[1:]]
    reads = [line.split("\t") for line in resources["workspace/data/reads.tsv"].splitlines()[1:]]
    e = next(row for row in reads if row[0] == "e")
    if e[1:] != ["chr1", "4000", "300M", "+"]:
        raise ValueError("Read e contract changed")
    overlapping = {
        group
        for group, chrom, start, width, strand in features
        if chrom == e[1] and strand == e[4] and int(start) <= 4000 and int(start) + int(width) - 1 >= 4299
    }
    if overlapping != {"E", "F"}:
        raise ValueError("Read e must be strictly contained in both E and F")
    expected = task.grader.parameters["expected"]
    if expected["intersection_strict"]["F"] != 1:
        raise ValueError("Expected original erroneous strict F count")
    corrected = json.loads(json.dumps(expected))
    corrected["intersection_strict"]["F"] = 0
    return corrected


def _patch_task(task: TaskSpec) -> TaskSpec:
    if task.id not in CORRECTED_IDS:
        raise ValueError("Only verified corrections may change TaskSpecs")
    reference = _fix_dice(task) if "loss_function_computation" in task.id else _fix_read_count(task)
    grader = task.grader.model_copy(update={"parameters": {**task.grader.parameters, "expected": reference}})
    fixed = task.model_copy(update={"id": f"{task.id}-v2", "grader": grader})
    return TaskSpec.model_validate(fixed.model_dump(mode="json"))


def freeze(
    *,
    output: str,
    code_commit: str,
    v1_tasks: str = V1_TASKS,
    first_audit: str = FIRST_AUDIT,
    second_audit: str = SECOND_AUDIT,
) -> dict[str, Any]:
    sources = {"v1_tasks": v1_tasks, "first_audit": first_audit, "second_audit": second_audit}
    inputs = {name: _read_bytes(uri) for name, uri in sources.items()}
    old = _lines(inputs["v1_tasks"])
    first = {row["task_id"]: row for row in _lines(inputs["first_audit"])}
    second = {row["task_id"]: row for row in _lines(inputs["second_audit"])}
    if len(old) != 305 or len(first) != 305 or len(second) != 60:
        raise ValueError("Frozen v1 cohort/audit cardinality changed")
    if set(second) != {
        key for key, row in first.items() if row.get("audit", {}).get("reference_assessment") != "confirmed"
    }:
        raise ValueError("Second pass does not cover every unresolved audit")
    if CORRECTED_IDS - set(second):
        raise ValueError("Verified correction missing from second audit")

    tasks = []
    changes = []
    dispositions = Counter()
    for row in old:
        prior = TaskSpec.model_validate(row["task"])
        if prior.id != row["task_id"] or digest(prior) != row["task_digest"]:
            raise ValueError(f"Invalid v1 task identity: {row['task_id']}")
        if first[prior.id]["task_digest"] != row["task_digest"]:
            raise ValueError("First audit task digest mismatch")
        if prior.id in second and second[prior.id]["task_digest"] != row["task_digest"]:
            raise ValueError("Second audit task digest mismatch")
        fixed = _patch_task(prior) if prior.id in CORRECTED_IDS else prior
        status = (
            "verified_reference_corrected"
            if prior.id in CORRECTED_IDS
            else ("audit_disputed_unresolved" if prior.id in second else "glm_audit_confirmed_unverified")
        )
        dispositions[status] += 1
        tasks.append(
            {
                **row,
                "task": fixed.model_dump(mode="json"),
                "task_id": fixed.id,
                "task_digest": digest(fixed),
                "v1_task_id": prior.id,
                "v1_task_digest": row["task_digest"],
                "audit_disposition": status,
            }
        )
        if fixed.id != prior.id:
            changes.append(
                {
                    "v1_task_id": prior.id,
                    "v1_task_digest": row["task_digest"],
                    "v2_task_id": fixed.id,
                    "v2_task_digest": digest(fixed),
                    "reason": "deterministic_reference_recomputation",
                }
            )
    if len({row["task_id"] for row in tasks}) != 305 or dispositions != {
        "glm_audit_confirmed_unverified": 245,
        "audit_disputed_unresolved": 58,
        "verified_reference_corrected": 2,
    }:
        raise ValueError(f"Unexpected v2 dispositions: {dispositions}")
    tasks.sort(key=lambda row: row["task_id"])
    changes.sort(key=lambda row: row["v1_task_id"])
    artifacts = {"taskspecs.jsonl": _artifact(tasks), "corrections.jsonl": _artifact(changes)}
    manifest = {
        "schema_version": "1",
        "campaign_id": "biotasks/combined/v2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "code_commit": code_commit,
        "release_status": "internal_unreviewed_do_not_publish",
        "warning": (
            "Only two reference corrections are independently verified; 58 disputed items remain unresolved. "
            "GLM confirmations are not independent verification. Do not use unresolved items in an official IRT panel."
        ),
        "summary": {"task_count": len(tasks), "dispositions": dict(sorted(dispositions.items()))},
        "inputs": {
            name: {"uri": uri, "sha256": hashlib.sha256(inputs[name]).hexdigest(), "bytes": len(inputs[name])}
            for name, uri in sources.items()
        },
        "artifacts": {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)} for name, data in artifacts.items()
        },
    }
    for name, data in artifacts.items():
        _write_new(output.rstrip("/") + "/" + name, data)
    _write_new(output.rstrip("/") + "/manifest.json", json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=V2_ROOT)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--v1-tasks", default=V1_TASKS)
    parser.add_argument("--first-audit", default=FIRST_AUDIT)
    parser.add_argument("--second-audit", default=SECOND_AUDIT)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze(
                output=args.output,
                code_commit=args.code_commit,
                v1_tasks=args.v1_tasks,
                first_audit=args.first_audit,
                second_audit=args.second_audit,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
