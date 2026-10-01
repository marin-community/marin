# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run curation stages and publish a complete sample ledger."""

import hashlib
import inspect
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from typing import Any

import verifyit

from taskcompendium.importers.nemo_predicted_action import canonical_sha256
from taskcompendium.models import ResourceVisibility, Source, TaskSpec
from taskcompendium.pipeline.filtering import task_decision
from taskcompendium.pipeline.models import (
    CheckResult,
    DatasetRecipe,
    Decision,
    Disposition,
    FilterPolicy,
    ImportRejection,
    RawRow,
    ReviewRecord,
    ReviewStatus,
    TaskAudit,
)
from taskcompendium.pipeline.parquet import write_accepted_parquet, write_task_parquet
from taskcompendium.pipeline.records import read_jsonl, write_jsonl
from taskcompendium.pipeline.review import Reviewer
from taskcompendium.pipeline.verification import verify_task
from taskcompendium.runtime.models import RolloutRecord


def _code_digest(recipe: DatasetRecipe) -> str:
    """Fingerprint the semantic package, shared scorers, and recipe module."""
    roots = (Path(__file__).parents[1], Path(verifyit.__file__).parent)
    digest = hashlib.sha256()
    for root in roots:
        for file in sorted(root.rglob("*.py")):
            digest.update(str(file.relative_to(root)).encode())
            digest.update(file.read_bytes())
    module_path = inspect.getsourcefile(recipe.normalize)
    if module_path is None:
        raise ValueError("A recipe normalizer must be defined in a Python source file")
    digest.update(Path(module_path).read_bytes())
    return digest.hexdigest()


def _semantic_digest(task: TaskSpec, include_reference: bool) -> str:
    content = task.model_dump(mode="json", exclude={"id", "source"})
    # Control scripts are executable witnesses, not task semantics.
    content["resources"] = [
        resource for resource in content["resources"] if resource["visibility"] != ResourceVisibility.CONTROL
    ]
    if not include_reference:
        content.pop("verifier")
        content["resources"] = [
            resource for resource in content["resources"] if resource["visibility"] == ResourceVisibility.AGENT
        ]
    return canonical_sha256(content)


def run_pipeline(
    recipe: DatasetRecipe,
    rows: Iterable[Mapping[str, Any]],
    *,
    output_path: Path,
    limit: int,
    reviewer: Reviewer,
    policy: FilterPolicy = FilterPolicy(),
) -> dict[str, Any]:
    """Curate a bounded sample, preserving every input and stage observation.

    Resume reuses the saved raw rows and review batch. Changing only policy
    rewrites decisions without new source reads or inference. Different source,
    converter, rubric, model, budget, or code identities require a new directory.
    Output task specs and review files contain private reference data.
    """
    if limit <= 0:
        raise ValueError("A positive sample limit is required")
    output_path.mkdir(parents=True, exist_ok=True)
    identity = {
        "recipe": recipe.name,
        "recipe_version": recipe.version,
        "source": asdict(recipe.source),
        "rubric": asdict(recipe.rubric),
        "intended_use": recipe.intended_use.value,
        "limit": limit,
        "reviewer": reviewer.identity,
        "checks": (
            {
                "id": recipe.check_suite.id,
                "revision": recipe.check_suite.revision,
                "parameters": dict(recipe.check_suite.parameters),
            }
            if recipe.check_suite
            else {"id": "answer-controls", "revision": "1"}
        ),
        "code_sha256": _code_digest(recipe),
    }
    # JSON round-trip makes tuple-valued rubric fields match the persisted form.
    identity = json.loads(json.dumps(identity))
    identity_path = output_path / "run-config.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Output directory belongs to a different curation run")
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")

    raw_path = output_path / "raw.jsonl"
    if raw_path.exists():
        raw_rows = read_jsonl(raw_path)
    else:
        raw_rows: list[dict[str, Any]] = []
        for index, data in enumerate(islice(rows, limit)):
            source = Source(
                dataset=recipe.source.dataset,
                revision=recipe.source.revision,
                row=f"{recipe.source.config}:{recipe.source.split}:{index}",
                importer_revision=recipe.version,
            )
            task_id = f"{recipe.name}-{canonical_sha256(source.model_dump())}"
            raw_rows.append(
                {
                    "task_id": task_id,
                    "source": source.model_dump(),
                    "raw_sha256": canonical_sha256(dict(data)),
                    "data": dict(data),
                }
            )
        write_jsonl(raw_path, raw_rows)
    if len(raw_rows) != limit:
        raise ValueError(f"Source yielded {len(raw_rows)} rows; expected {limit}")

    normalized, decisions = [], []
    normalization_rejections = {}
    for raw in raw_rows:
        if canonical_sha256(raw["data"]) != raw["raw_sha256"]:
            raise ValueError(f"Saved source row digest changed: {raw['task_id']}")
        source = Source.model_validate(raw["source"])
        result = recipe.normalize(RawRow(raw["task_id"], source, raw["data"]))
        if isinstance(result, ImportRejection):
            normalization_rejections[raw["task_id"]] = result
            decisions.append(
                Decision(
                    task_id=raw["task_id"],
                    disposition=Disposition.REJECT,
                    reasons=[f"normalize:{result.reason}", result.detail],
                )
            )
            continue
        if result.id != raw["task_id"] or result.source != source:
            raise ValueError("A converter must retain its supplied task identity and source provenance")
        normalized.append(result)
    write_jsonl(output_path / "normalized.jsonl", (task.model_dump(mode="json") for task in normalized))

    prompt_groups: dict[str, set[str]] = {}
    for task in normalized:
        prompt_groups.setdefault(_semantic_digest(task, False), set()).add(_semantic_digest(task, True))
    conflicts = {key for key, values in prompt_groups.items() if len(values) > 1}
    seen: dict[str, str] = {}
    candidates, check_rows, rollouts = [], [], []
    checks_by_id = {}
    checks_path = output_path / "checks.jsonl"
    cached_checks = {row["task_id"]: row for row in read_jsonl(checks_path)} if checks_path.exists() else {}
    for task in normalized:
        if _semantic_digest(task, False) in conflicts:
            decisions.append(
                Decision(task_id=task.id, disposition=Disposition.REJECT, reasons=["conflicting_references"])
            )
            continue
        semantic_key = _semantic_digest(task, True)
        if semantic_key in seen:
            decisions.append(
                Decision(
                    task_id=task.id,
                    disposition=Disposition.REJECT,
                    reasons=["exact_semantic_duplicate"],
                    duplicate_of=seen[semantic_key],
                )
            )
            continue
        seen[semantic_key] = task.id
        task_digest = canonical_sha256(task.model_dump(mode="json", exclude={"id", "source"}))
        if checks_path.exists():
            cached = cached_checks[task.id]
            if cached["task_sha256"] != task_digest:
                raise ValueError(f"Saved verification belongs to different task content: {task.id}")
            checks = [CheckResult.model_validate(check) for check in cached["checks"]]
            task_rollouts = tuple(RolloutRecord.model_validate(row) for row in cached["rollouts"])
        else:
            if recipe.check_suite is None:
                checks, task_rollouts = verify_task(task), ()
            else:
                report = recipe.check_suite.run(task)
                checks, task_rollouts = report.checks, report.rollouts
        rollouts.extend(task_rollouts)
        check_rows.append(
            {
                "task_id": task.id,
                "task_sha256": task_digest,
                "checks": [check.model_dump(mode="json") for check in checks],
                "rollouts": [rollout.model_dump(mode="json") for rollout in task_rollouts],
            }
        )
        checks_by_id[task.id] = checks
        candidates.append(task)
    if cached_checks and cached_checks.keys() != checks_by_id.keys():
        raise ValueError("Saved verification records do not match normalized task membership")
    write_jsonl(checks_path, check_rows)
    write_jsonl(output_path / "rollouts.jsonl", (rollout.model_dump(mode="json") for rollout in rollouts))

    reviews_path = output_path / "reviews.jsonl"
    if reviews_path.exists():
        reviews = [ReviewRecord.model_validate_json(json.dumps(row)) for row in read_jsonl(reviews_path)]
    else:
        reviews = reviewer.review(candidates, recipe.rubric, output_path / "review")
        write_jsonl(reviews_path, (review.model_dump(mode="json") for review in reviews))
    if {review.task_id for review in reviews} != {task.id for task in candidates} or len(reviews) != len(candidates):
        raise ValueError("Review records do not match the eligible task membership")
    reviews_by_id = {review.task_id: review for review in reviews}
    for task_id, checks in checks_by_id.items():
        review = reviews_by_id.get(
            task_id,
            ReviewRecord(
                task_id=task_id, status=ReviewStatus.UNAVAILABLE, verdict=None, detail="No quality assessment available"
            ),
        )
        reviews_by_id[task_id] = review
        decisions.append(task_decision(task_id, checks, review, policy))
    decisions.sort(key=lambda decision: decision.task_id)
    if {decision.task_id for decision in decisions} != {raw["task_id"] for raw in raw_rows} or len(decisions) != len(
        raw_rows
    ):
        raise ValueError("Decision ledger does not account for every source row exactly once")
    write_jsonl(output_path / "decisions.jsonl", (decision.model_dump(mode="json") for decision in decisions))
    normalized_by_id = {task.id: task for task in normalized}
    decisions_by_id = {decision.task_id: decision for decision in decisions}
    audit_table = write_task_parquet(
        output_path / "audit.parquet",
        [
            TaskAudit(
                task_id=raw["task_id"],
                source=Source.model_validate(raw["source"]),
                raw=raw,
                normalized=normalized_by_id.get(raw["task_id"]),
                normalization_rejection=normalization_rejections.get(raw["task_id"]),
                checks=checks_by_id.get(raw["task_id"], []),
                review=reviews_by_id.get(raw["task_id"]),
                decision=decisions_by_id[raw["task_id"]],
            )
            for raw in raw_rows
        ],
    )
    write_accepted_parquet(output_path / "accepted.parquet", audit_table)
    manifest = {
        **identity,
        "policy": asdict(policy),
        "input_rows": len(raw_rows),
        "normalized_rows": len(normalized),
        "reviewed_rows": sum(review.status == ReviewStatus.REVIEWED for review in reviews),
        "dispositions": dict(Counter(decision.disposition.value for decision in decisions)),
        "reasons": dict(Counter(reason for decision in decisions for reason in decision.reasons)),
        "check_statuses": dict(Counter(check.status.value for checks in checks_by_id.values() for check in checks)),
        "raw_sample_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
