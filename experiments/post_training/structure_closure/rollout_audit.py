# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit generic rollout evidence against a fixed text-prefill task cohort."""

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter
from rolloutengine.harness.prefill import prefill_prompt
from rolloutengine.models import GradedRollout, ModelRequest
from taskcompendium.chat import chat_conversation
from taskcompendium.grading import grade_answer
from taskcompendium.grading_result import Outcome
from taskcompendium.models import GradingAttempt, TaskSpec, VerifyitGrader, verifyit_spec
from verifyit.spec import StructureClosureSpec

from experiments.post_training.structure_closure.eval_audit import AuditFailure, EvaluationAudit, case_result
from experiments.post_training.structure_closure.task_data import load_validation, task_digest


@dataclass(frozen=True)
class EvaluationRecord:
    task_id: str
    task_sha256: str
    request: ModelRequest | None
    result: GradedRollout


EVALUATION_RECORD = TypeAdapter(EvaluationRecord)


def task_group(task: TaskSpec, key: str) -> str:
    values = [tag.removeprefix(key + ":") for tag in task.tags if tag.startswith(key + ":")]
    if len(values) != 1:
        raise ValueError(f"A closure task requires one {key} tag")
    return values[0]


def evaluation_request(task: TaskSpec, sampling: dict[str, Any]) -> ModelRequest:
    """Use explicit prefill rendering with explicit experiment sampling and thinking settings."""
    prompt = prefill_prompt(task)
    if sampling.keys() & (prompt.options.keys() | {"messages", "model", "prompt", "stream", "n", "extra_body", "tools"}):
        raise ValueError("Sampling cannot override task conditioning or template controls")
    body = {
        **prompt.options.get("extra_body", {}),
        "chat_template_kwargs": {"enable_thinking": task_group(task, "family") == "reasoning_delimiter"},
    }
    return ModelRequest(prompt.messages, {**sampling, **prompt.options, "extra_body": body})


def audit_rollouts(
    tasks: Sequence[TaskSpec], records: Iterable[EvaluationRecord], *, sampling: dict[str, Any]
) -> EvaluationAudit:
    """Check pairing, public conditioning and actual grade, independently of tokenization."""
    expected = {task.id: task for task in tasks}
    if len(expected) != len(tasks):
        raise ValueError("The evaluation cohort contains duplicate task identities")
    seen: set[str] = set()
    results, failures = [], []
    observed = 0
    for record in records:
        observed += 1
        uid = record.task_id
        try:
            if uid not in expected or uid in seen:
                raise ValueError("Unknown or duplicate evaluation task")
            seen.add(uid)
            task = expected[uid]
            if record.task_sha256 != task_digest(task):
                raise ValueError("The evaluated task differs from the frozen cohort")
            request = record.request
            expected_request = evaluation_request(task, sampling)
            if request is None or request.messages != expected_request.messages:
                raise ValueError("The model did not receive the frozen public conditioning")
            if request.prompt_token_ids is not None or request.continuation is not None:
                raise ValueError("Text-prefill evaluation cannot substitute pinned token conditioning")
            if request.options != expected_request.options:
                raise ValueError("The model request changed sampling, thinking, prefill or tool settings")
            result = record.result
            rollout = result.rollout
            if rollout.failure is not None:
                raise ValueError("Interrupted rollout cannot count as scored evidence")
            if len(rollout.conversations) != 1 or len(rollout.conversations[0].steps) != 1:
                raise ValueError("A closure task requires exactly one captured model turn")
            conversation = rollout.conversations[0]
            turn = conversation.steps[0].turn
            if conversation.messages != (*request.messages, turn.message):
                raise ValueError("The graded conversation differs from the captured request and response")
            if turn.message.get("role") != "assistant" or not isinstance(turn.message.get("content"), str):
                raise ValueError("The model response is missing assistant text")
            if (
                turn.message.get("reasoning_content")
                or turn.message.get("reasoning")
                or turn.metadata.get("reasoning")
                or turn.message.get("tool_calls")
            ):
                raise ValueError("Parsed response channels cannot establish the raw structural continuation")
            attempt = GradingAttempt(chat_conversation(list(conversation.messages)), stop_reason=rollout.stop_reason)
            rescored = grade_answer(task, attempt)
            scored_outcomes = (Outcome.GRADED, Outcome.SUBMISSION_FAILURE)
            if result.grade.status not in scored_outcomes or rescored.status not in scored_outcomes:
                raise ValueError("The task has no scored verifier result")
            if result.grade != rescored:
                raise ValueError("Recorded grade disagrees with the captured response")
            if not isinstance(task.grader, VerifyitGrader):
                raise ValueError("The cohort contains a non-verifyit grader")
            spec = verifyit_spec(task.grader)
            if not isinstance(spec, StructureClosureSpec):
                raise ValueError("The cohort contains a non-closure grader")
            content = ""
            if rescored.status == Outcome.GRADED:
                content = turn.message["content"]
            results.append(
                case_result(
                    uid=uid,
                    case_id=uid.removeprefix("structure-closure:"),
                    family=task_group(task, "family"),
                    control=task_group(task, "control"),
                    prefix=spec.prefix,
                    expected=spec.expected,
                    structure_format=spec.format,
                    content=content,
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            failures.append(AuditFailure(uid, str(error)))
    return EvaluationAudit(
        len(tasks), observed, tuple(uid for uid in expected if uid not in seen), tuple(failures), tuple(results)
    )


def rollout_summary(tasks: Sequence[TaskSpec], audit: EvaluationAudit, *, sampling: dict[str, Any]) -> dict:
    """Keep family denominators fixed when evidence is missing or invalid."""
    summary = audit.summary()
    expected = Counter()
    for task in tasks:
        family, control = task_group(task, "family"), task_group(task, "control")
        expected.update((family, f"{family}/{control}"))
    for group, count in sorted(expected.items()):
        counts = summary["groups"].setdefault(group, {})
        counts["expected"] = count
        counts["unscored"] = count - counts.get("scored", 0)
        counts["success_rate"] = counts.get("success", 0) / count
    summary["cohort_sha256"] = hashlib.sha256("\n".join(task_digest(task) for task in tasks).encode()).hexdigest()
    summary["sampling"] = sampling
    return summary


def compare_evaluations(baseline: dict, candidate: dict) -> dict[str, dict[str, float]]:
    """Compare complete evaluations of the exact same ordered task cohort."""
    if not baseline["complete"] or not candidate["complete"]:
        raise ValueError("Incomplete evaluations cannot establish improvement")
    if baseline["cohort_sha256"] != candidate["cohort_sha256"]:
        raise ValueError("Evaluations used different task cohorts")
    if baseline["sampling"] != candidate["sampling"]:
        raise ValueError("Evaluations used different sampling settings")
    return {
        family: {
            "baseline": group["success_rate"],
            "candidate": candidate["groups"][family]["success_rate"],
            "change": candidate["groups"][family]["success_rate"] - group["success_rate"],
        }
        for family, group in baseline["groups"].items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--sampling", type=Path, required=True)
    args = parser.parse_args()
    tasks = load_validation(args.tasks)
    sampling = json.loads(args.sampling.read_text())
    records, malformed = [], []
    for number, line in enumerate(args.rollouts.read_bytes().splitlines(), 1):
        try:
            records.append(EVALUATION_RECORD.validate_json(line))
        except ValueError as error:
            malformed.append(AuditFailure(f"line:{number}", str(error)))
    audit = audit_rollouts(tasks, records, sampling=sampling)
    audit = EvaluationAudit(
        audit.expected, audit.observed + len(malformed), audit.missing_uids, (*audit.failures, *malformed), audit.results
    )
    report = rollout_summary(tasks, audit, sampling=sampling)
    comparable = True
    if args.baseline is not None:
        try:
            report["comparison"] = compare_evaluations(json.loads(args.baseline.read_text()), report)
        except ValueError as error:
            report["comparison_error"] = str(error)
            comparable = False
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({"complete": audit.complete, "scored": len(audit.results), "expected": audit.expected}))
    raise SystemExit(0 if audit.complete and comparable else 1)


if __name__ == "__main__":
    main()
