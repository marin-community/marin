# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small prompt views of workspace artifacts; full payloads remain in files."""

import json
from pathlib import PurePosixPath

from taskcompendium.grading_result import GradeResult
from taskcompendium.models import TaskSpec

from forge.model.review import ValidationResult
from forge.stages.rules import passed


def manifest(value: object, role: str = "") -> object:
    if isinstance(value, dict):
        source = value.get("source")
        if isinstance(source, dict) and source.get("kind") == "inline_file":
            encoded = source["content_base64"]
            return {
                "path": value["path"],
                "bytes": len(encoded) * 3 // 4 - len(encoded) + len(encoded.rstrip("=")),
                "role": role,
            }
        return {key: manifest(child, f"{role}.{key}".strip(".")) for key, child in value.items()}
    if isinstance(value, list):
        return [manifest(child, role) for child in value]
    return value


def task_manifest(task: TaskSpec) -> str:
    return json.dumps(manifest(task.model_dump(mode="json")), separators=(",", ":"))


def validation_path(workdir: str, index: int) -> PurePosixPath:
    return PurePosixPath(workdir, ".validation", f"{index}.json")


def excerpt(value: object) -> str:
    return "\n".join(value.splitlines()[:3])[:600] if isinstance(value, str) else ""


def grade_summary(grade: GradeResult | None) -> dict[str, object]:
    if grade is None:
        return {"status": "ungraded"}
    return {
        "status": grade.status,
        "passed": passed(grade),
        "reward": grade.reward,
        "error": excerpt(grade.error),
        "stdout": excerpt(grade.diagnostics.get("stdout")),
        "stderr": excerpt(grade.diagnostics.get("stderr")),
        **({"decision_model": read} if (read := (grade.detail or {}).get("decision_model")) else {}),
    }


def validation_summary(result: ValidationResult, path: PurePosixPath, limit: int) -> str:
    summary = json.dumps(
        {
            "task_digest": result.task_digest,
            "lowering_error": excerpt(result.lowering_error),
            "solver": [{"trial": t.trial, "cause": t.cause, **grade_summary(t.grade)} for t in result.solver],
            "controls": [
                {"index": c.index, "met": c.met, "cause": c.cause, **grade_summary(c.grade)} for c in result.controls
            ],
            "adversary": [
                {
                    "name": a.name,
                    "completed": a.completed,
                    "reasons": [excerpt(reason) for reason in a.reasons],
                    "unconfirmed": sum(not f.met for f in a.findings),
                    "confirmed": [
                        {"claim": excerpt(f.claim), "candidate_paths": [file.path for file in f.candidate.files]}
                        for f in a.findings
                        if f.met
                    ],
                }
                for a in result.adversary
            ],
        },
        separators=(",", ":"),
    )
    marker = f"Full validation JSON: {path}. Use read_file with offset/limit for details.\n"
    return marker + summary[: max(0, limit - len(marker))]
