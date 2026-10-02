# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare generic/IF filtering and propose structured-output instruction repairs.

Input snapshots contain decoded instruction.md and tests/verifier_data.json from
the pinned TaskTrove revision. Sampling and HF range reads happen separately.
"""

import argparse
import json
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path

from marin.inference.openai_batch import OpenAIBatchClient
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.batches import batch_output
from taskcompendium.pipeline.datasets import instruction_following, structured_output
from taskcompendium.pipeline.filtering import task_decision
from taskcompendium.pipeline.models import CheckResult, Decision, FilterPolicy, ReviewRecord, ReviewRubric
from taskcompendium.pipeline.parquet import write_accepted_parquet
from taskcompendium.pipeline.review import BatchReviewer
from taskcompendium.pipeline.rewriting import (
    BatchRewriter,
    protected_text_checks,
    rewrite_candidate,
    write_rewrite_audit,
)
from taskcompendium.pipeline.runner import run_pipeline
from taskcompendium.pipeline.verification import verify_task, verify_witness

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_MODEL


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def save_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def solve_witnesses(client: OpenAIBatchClient, candidates: list[TaskSpec], path: Path) -> dict[str, str]:
    """Request candidate answers and retain complete text replies as grading witnesses."""
    if not candidates:
        return {}
    witnesses = {}
    requests = [
        {
            "custom_id": task.id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": GLM_MODEL,
                "messages": [{"role": "user", "content": task.context.events[0].content}],
                "chat_template_kwargs": {"reasoning_effort": "low"},
                "max_tokens": 8192,
            },
        }
        for task in candidates
    ]
    output = batch_output(client, requests, path / "solver", filename="rewrite-witnesses.jsonl", poll_seconds=5.0)
    for row in [json.loads(line) for line in output.split("\n") if line.strip()]:
        response = row.get("response")
        if response is None or response.get("status_code") != 200:
            continue
        choices = response["body"]["choices"]
        if (
            len(choices) == 1
            and choices[0]["finish_reason"] == "stop"
            and isinstance(choices[0]["message"].get("content"), str)
        ):
            witnesses[row["custom_id"]] = choices[0]["message"]["content"]
    return witnesses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--if-snapshot", type=Path, required=True)
    parser.add_argument("--structured-snapshot", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--filter-evidence", type=Path, help="Reuse completed filter outputs for a new rewrite experiment"
    )
    parser.add_argument("--stage", choices=("filter", "rewrite", "both"), default="both")
    parser.add_argument("--comparison", choices=("generic", "domain", "both"), default="domain")
    parser.add_argument("--if-rewrite-indices", nargs="+", type=int, help="Select exact snapshot scan_index values")
    parser.add_argument("--if-rewrite-limit", type=int, default=3, help="Select the first normalized IF rows")
    parser.add_argument("--structured-rewrite-limit", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    client = OpenAIBatchClient(args.base_url, os.environ[GLM_BULK_TOKEN_ENV])
    reviewer = BatchReviewer(client, GLM_MODEL, args.model_revision, max_tokens=4096, max_prompt_characters=64000)
    rewriter = BatchRewriter(client, GLM_MODEL, args.model_revision)
    generic = ReviewRubric("generic-quality", "1", ())
    manifests, normalized, raw_by_source = {}, {}, {}
    source_checks, source_reviews, source_decisions = {}, {}, {}
    filter_path = args.filter_evidence or args.output
    sources = (
        ("ifeval", instruction_following.recipe(args.if_snapshot), args.if_snapshot),
        ("structured", structured_output.recipe(args.structured_snapshot), args.structured_snapshot),
    )
    for name, recipe, snapshot in sources:
        rows = read_rows(snapshot)
        rubrics = {"generic": generic, "domain": recipe.rubric}
        selected = rubrics if args.comparison == "both" else {args.comparison: rubrics[args.comparison]}
        manifests[name] = {}
        for rubric_name, rubric in selected.items():
            path = filter_path / name / rubric_name
            if args.filter_evidence is None:
                manifest = run_pipeline(
                    replace(recipe, rubric=rubric), rows, output_path=path, limit=len(rows), reviewer=reviewer
                )
            else:
                manifest = json.loads((path / "manifest.json").read_text())
            manifests[name][rubric_name] = manifest
            print(
                json.dumps({"stage": name, "rubric": rubric_name, "dispositions": manifest["dispositions"]}), flush=True
            )
        raw = read_rows(path / "raw.jsonl")
        if [row["data"] for row in raw] != rows:
            raise ValueError("Filter evidence does not match the requested source snapshot")
        raw_by_source[name] = raw
        normalized[name] = [TaskSpec.model_validate(row) for row in read_rows(path / "normalized.jsonl")]
        source_checks[name] = {row["task_id"]: row["checks"] for row in read_rows(path / "checks.jsonl")}
        source_reviews[name] = {
            row["task_id"]: ReviewRecord.model_validate_json(json.dumps(row))
            for row in read_rows(path / "reviews.jsonl")
        }
        source_decisions[name] = {
            row["task_id"]: Decision.model_validate(row) for row in read_rows(path / "decisions.jsonl")
        }
    if args.stage == "filter":
        (args.output / "summary.json").write_text(json.dumps({"filter": manifests}, indent=2))
        return
    by_id = {task.id: task for task in normalized["ifeval"]}
    by_index = {
        row["data"]["scan_index"]: by_id[row["task_id"]] for row in raw_by_source["ifeval"] if row["task_id"] in by_id
    }
    structured = normalized["structured"][: args.structured_rewrite_limit]
    rewrite_rubric = ReviewRubric(
        "structured-instruction-repair",
        "1",
        (
            "Retain the original Evaluation contract and public JSON Schema byte-for-byte. "
            "Resolve stale extraction/grounding demands using that contract. Preserve all source facts. "
            "If the task is already unambiguous, mark it unchanged.",
            "If the original did not already permit choosing unstated values, do not introduce that permission; "
            "mark it unrepairable.",
        ),
    )
    groups = (
        ("structured", structured, rewrite_rubric, structured_output.RUBRIC),
        (
            "ifeval",
            (
                [by_index[i] for i in args.if_rewrite_indices]
                if args.if_rewrite_indices is not None
                else normalized["ifeval"][: args.if_rewrite_limit]
            ),
            instruction_following.RUBRIC,
            instruction_following.RUBRIC,
        ),
    )
    rewrite_summary = {}
    for name, originals, rubric, review_rubric in groups:
        path = args.output / f"rewrite-{name}"
        records = rewriter.rewrite(originals, rubric, path)
        parents = {task.id: task for task in originals}
        candidates, candidate_parents = [], {}
        for record in records:
            candidate = rewrite_candidate(parents[record.task_id], record)
            if candidate is not None:
                candidates.append(candidate)
                candidate_parents[candidate.id] = parents[record.task_id]
        after = reviewer.review(candidates, review_rubric, path / "review", originals=candidate_parents)
        witnesses = solve_witnesses(client, candidates, path) if name == "structured" else {}
        decisions, controls = [], []
        reviews_by_id = {review.task_id: review for review in after}
        for candidate in candidates:
            review = reviews_by_id[candidate.id]
            checks = (
                verify_witness(candidate, witnesses[candidate.id], "__invalid_submission__")
                if candidate.id in witnesses
                else verify_task(candidate)
            )
            if name == "structured":
                checks.extend(
                    protected_text_checks(candidate, structured_output.protected_spans(candidate_parents[candidate.id]))
                )
            decision = task_decision(candidate.id, checks, review, FilterPolicy())
            decisions.append(decision.model_dump(mode="json"))
            controls.append({"task_id": candidate.id, "checks": [check.model_dump(mode="json") for check in checks]})
        candidate_dispositions = dict(Counter(row["disposition"] for row in decisions))
        rewritten_parents = {parent.id for parent in candidate_parents.values()}
        for original in originals:
            if original.id in rewritten_parents:
                continue
            decisions.append(source_decisions[name][original.id].model_dump(mode="json"))
            controls.append({"task_id": original.id, "checks": source_checks[name].get(original.id, [])})
            if original.id in source_reviews[name]:
                after.append(source_reviews[name][original.id])
        save_rows(path / "reviews.jsonl", [record.model_dump(mode="json") for record in after])
        save_rows(path / "decisions.jsonl", decisions)
        save_rows(path / "checks.jsonl", controls)
        audit_table = write_rewrite_audit(
            path,
            checks={row["task_id"]: [CheckResult.model_validate(check) for check in row["checks"]] for row in controls},
            reviews=after,
            decisions=[Decision.model_validate(row) for row in decisions],
        )
        if any(status not in {"keep", "reject"} for status in audit_table["filter_status"].to_pylist()):
            raise ValueError("Final cleanup audit requires a binary decision for every input")
        write_accepted_parquet(path / "accepted.parquet", audit_table)
        rewrite_summary[name] = {
            "inputs": len(originals),
            "actions": dict(
                Counter(record.proposal.action.value if record.proposal else record.status.value for record in records)
            ),
            "candidate_dispositions": candidate_dispositions,
            "dispositions": dict(Counter(row["disposition"] for row in decisions)),
        }
        print(json.dumps({"stage": f"rewrite-{name}", **rewrite_summary[name]}), flush=True)
    (args.output / "summary.json").write_text(json.dumps({"filter": manifests, "rewrite": rewrite_summary}, indent=2))


if __name__ == "__main__":
    main()
