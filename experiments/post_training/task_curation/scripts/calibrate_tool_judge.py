# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure whether the tool judge accepts supported actions and rejects unsupported claims.

The labelled cases cover valid calls, alternative actions, clarification requests, unsupported
calls and invented results. Each case pins its normalized task hash to the original calibration
artifact identified in fixtures/tool_judge_provenance.json. The checked-in fixtures make the
challenge set reproducible without a campaign download.
"""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow.parquet as pq
from taskcompendium.grading import grade_remote_judge_answer
from taskcompendium.models import (
    AssistantToolCalls,
    ConversationToolCall,
    ConversationTrace,
    GradingAttempt,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
)
from verifyit.modes.grade_judge import JudgeConnection

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_MODEL, resolve_glm_base_url
from experiments.post_training.task_curation.datasets.nemotron_ultra.graders import (
    TOOL_ACTION_JUDGE_PROMPT,
    TOOL_ACTION_JUDGE_REASONING_EFFORT,
    TOOL_ACTION_JUDGE_SYSTEM,
)


@dataclass(frozen=True)
class CalibrationCase:
    """One human judgment of an assistant's next action."""

    name: str
    category: str
    source_row: int
    expected_reward: float
    rationale: str
    task_sha256: str
    text: str | None = None
    function: str | None = None
    arguments: dict | None = None


@dataclass(frozen=True)
class CalibrationResult:
    """One observed judge verdict for a calibration case."""

    case: str
    category: str
    repeat: int
    expected_reward: float
    reward: float | None
    status: str
    error: str | None
    correct: bool


FIXTURES = Path(__file__).parent / "fixtures"


def _cases(path: Path) -> tuple[CalibrationCase, ...]:
    return tuple(CalibrationCase(**json.loads(line)) for line in path.read_text().splitlines() if line.strip())


def _tasks(normalized: Path, cases: tuple[CalibrationCase, ...]) -> dict[int, TaskSpec]:
    wanted = {case.source_row: case.task_sha256 for case in cases}
    tasks = {}
    paths = normalized.glob("*.parquet") if normalized.is_dir() else (normalized,)
    for path in paths:
        if path.suffix == ".parquet":
            rows = pq.read_table(path, columns=["source_row", "task_json"]).to_pylist()
        else:
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            source_row = int(row["source_row"].rsplit(":", 1)[1])
            if source_row in wanted:
                if hashlib.sha256(row["task_json"].encode()).hexdigest() != wanted[source_row]:
                    raise ValueError(f"Calibration source row {source_row} differs from its labelled artifact")
                task = TaskSpec.model_validate_json(row["task_json"])
                assert isinstance(task.grader, VerifyitGrader)
                parameters = {
                    **task.grader.parameters,
                    "model": GLM_MODEL,
                    "system_prompt": TOOL_ACTION_JUDGE_SYSTEM,
                    "prompt_template": TOOL_ACTION_JUDGE_PROMPT,
                    "reasoning_effort": TOOL_ACTION_JUDGE_REASONING_EFFORT,
                }
                tasks[source_row] = task.model_copy(
                    update={"grader": task.grader.model_copy(update={"parameters": parameters})}
                )
    missing = wanted.keys() - tasks.keys()
    if missing:
        raise ValueError(f"Normalized input is missing source rows: {sorted(missing)}")
    return tasks


def _attempt(task: TaskSpec, case: CalibrationCase, repeat: int) -> GradingAttempt:
    if case.text is not None:
        action = TextMessage(role="assistant", content=case.text)
    else:
        assert case.function is not None and case.arguments is not None
        action = AssistantToolCalls(
            calls=(
                ConversationToolCall(
                    call_id=f"calibration-{case.name}-{repeat}", name=case.function, arguments=case.arguments
                ),
            )
        )
    return GradingAttempt(ConversationTrace(events=(*task.context.events, action)), {})


def _grade(
    case: CalibrationCase,
    repeat: int,
    tasks: dict[int, TaskSpec],
    connection: JudgeConnection,
) -> CalibrationResult:
    result = grade_remote_judge_answer(
        tasks[case.source_row], _attempt(tasks[case.source_row], case, repeat), connection
    )
    return CalibrationResult(
        case=case.name,
        category=case.category,
        repeat=repeat,
        expected_reward=case.expected_reward,
        reward=result.reward,
        status=result.status.value,
        error=result.error,
        correct=result.reward == case.expected_reward,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized", type=Path, default=FIXTURES / "tool_judge_tasks.jsonl")
    parser.add_argument("--cases", type=Path, default=FIXTURES / "tool_judge_cases.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relay-job", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")

    cases = _cases(args.cases)
    tasks = _tasks(args.normalized, cases)
    connection = JudgeConnection(resolve_glm_base_url(args.relay_job), os.environ[GLM_BULK_TOKEN_ENV])
    calls = [(case, repeat) for case in cases for repeat in range(args.repeats)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda call: _grade(*call, tasks, connection), calls))
    correct = sum(result.correct for result in results)
    category_summary = {
        category: {
            "correct": sum(result.correct for result in results if result.category == category),
            "total": sum(result.category == category for result in results),
        }
        for category in sorted({case.category for case in cases})
    }
    report = {
        "model": GLM_MODEL,
        "relay_job": args.relay_job,
        "repeats": args.repeats,
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(TOOL_ACTION_JUDGE_SYSTEM.encode()).hexdigest(),
        "prompt_template_sha256": hashlib.sha256(TOOL_ACTION_JUDGE_PROMPT.encode()).hexdigest(),
        "summary": {"correct": correct, "total": len(results), "accuracy": correct / len(results)},
        "categories": category_summary,
        "cases": [asdict(case) for case in cases],
        "results": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
