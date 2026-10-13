# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Calibrate the Nemotron tool-use judge against a small human-labeled challenge set."""

import argparse
import base64
import json
import os
import zlib
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


CASES = (
    CalibrationCase(
        "active-tickers",
        "valid",
        582,
        1.0,
        "Directly invokes the advertised market endpoint.",
        function="get_special_tickers",
        arguments={},
    ),
    CalibrationCase(
        "random-quote",
        "valid",
        6175,
        1.0,
        "Directly invokes the only relevant tool.",
        function="generate_random_quote",
        arguments={},
    ),
    CalibrationCase(
        "webhook",
        "valid",
        2020,
        1.0,
        "Directly invokes the advertised webhook generator.",
        function="generate_webhook",
        arguments={},
    ),
    CalibrationCase(
        "chess-advice",
        "valid",
        1587,
        1.0,
        "Searches the advertised advice tool using the user's topic.",
        function="search_advice",
        arguments={"query": "chess"},
    ),
    CalibrationCase(
        "pubmed-paraphrase",
        "alternative",
        1523,
        1.0,
        "A concise, semantically equivalent literature query.",
        function="pubmed_query_run",
        arguments={
            "query": (
                "recent aircraft aluminum corrosion effects on fatigue life and crack growth "
                "under cyclic loading humidity saltwater"
            )
        },
    ),
    CalibrationCase(
        "compound-interest-wolfram",
        "alternative",
        4893,
        1.0,
        "An advertised calculator can perform the same first computation.",
        function="wolfram_alpha_calculator",
        arguments={"input_query": "250000 * (1 + 0.045/4)^(4*15)"},
    ),
    CalibrationCase(
        "compound-interest-expression",
        "alternative",
        4893,
        1.0,
        "Equivalent exponent notation in the same calculator.",
        function="calculator",
        arguments={"input_query": "250000*(1+0.045/4)**60"},
    ),
    CalibrationCase(
        "second-crypto-first",
        "alternative",
        1592,
        1.0,
        "Either requested asset is a defensible first call in a multi-step request.",
        function="metadata",
        arguments={
            "asset_type": "spot",
            "market_venue": "BITFINEX",
            "symbol": "LTC",
            "base": "USD",
            "data_type": "metrics",
        },
    ),
    CalibrationCase(
        "ask-pinterest-user",
        "clarification",
        36817,
        1.0,
        "The user did not identify whose pins they want.",
        text="Which Pinterest username should I look up?",
    ),
    CalibrationCase(
        "ask-news-country",
        "clarification",
        66288,
        1.0,
        "The country is required to make the request specific.",
        text="Which country should I use for the headlines?",
    ),
    CalibrationCase(
        "ask-largest-count",
        "clarification",
        73287,
        1.0,
        "The requested number of largest values is missing.",
        text="How many of the largest numbers would you like?",
    ),
    CalibrationCase(
        "ask-weather-location",
        "clarification",
        2337,
        1.0,
        "The API requires coordinates not supplied in the conversation.",
        text="What latitude and longitude in Chicago should I use for the rooftop?",
    ),
    CalibrationCase(
        "flight-tool-unavailable",
        "unsupported",
        40329,
        1.0,
        "No advertised tool can book a flight.",
        text="I can't book a flight with the available recipe-search tool.",
    ),
    CalibrationCase(
        "flight-via-recipe",
        "unsupported",
        40329,
        0.0,
        "The recipe tool cannot book travel.",
        function="search_recipe",
        arguments={"ingredients": ["flight"]},
    ),
    CalibrationCase(
        "refuse-available-advice",
        "unsupported",
        1587,
        0.0,
        "A relevant advertised search tool is available.",
        text="I can't help with chess exercises because no suitable tool is available.",
    ),
    CalibrationCase(
        "invented-tool-name",
        "unsupported",
        582,
        0.0,
        "The function is not advertised.",
        function="get_most_active_tickers",
        arguments={},
    ),
    CalibrationCase(
        "invented-chicago-coordinates",
        "hallucinated",
        2337,
        0.0,
        "The exact coordinates are absent from the visible conversation.",
        function="get_hourly_forecast",
        arguments={"location": "41.8781,-87.6298", "hours": 24},
    ),
    CalibrationCase(
        "invented-news-country",
        "hallucinated",
        66288,
        0.0,
        "The user's country is absent from the visible conversation.",
        function="get_news_headlines",
        arguments={"country": "us"},
    ),
    CalibrationCase(
        "invented-pinterest-user",
        "hallucinated",
        36817,
        0.0,
        "The username is neither supplied nor a documented default.",
        function="get_user_pins",
        arguments={"username": "ben123"},
    ),
    CalibrationCase(
        "invented-booking-result",
        "hallucinated",
        40329,
        0.0,
        "Claims a tool result and booking that never occurred.",
        text="Your New York to London flight is booked for tomorrow at 9 AM.",
    ),
)


def _tasks(normalized: Path) -> dict[int, TaskSpec]:
    wanted = {case.source_row for case in CASES}
    tasks = {}
    paths = normalized.glob("*.parquet") if normalized.is_dir() else (normalized,)
    for path in paths:
        if path.suffix == ".parquet":
            rows = pq.read_table(path, columns=["source_row", "task_json"]).to_pylist()
        elif path.suffix == ".b64":
            rows = [json.loads(zlib.decompress(base64.b64decode(line))) for line in path.read_text().splitlines()]
        else:
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            source_row = int(row["source_row"].rsplit(":", 1)[1])
            if source_row in wanted:
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
    missing = wanted - tasks.keys()
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
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relay-job", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")

    tasks = _tasks(args.normalized)
    connection = JudgeConnection(resolve_glm_base_url(args.relay_job), os.environ[GLM_BULK_TOKEN_ENV])
    calls = [(case, repeat) for case in CASES for repeat in range(args.repeats)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda call: _grade(*call, tasks, connection), calls))
    correct = sum(result.correct for result in results)
    category_summary = {
        category: {
            "correct": sum(result.correct for result in results if result.category == category),
            "total": sum(result.category == category for result in results),
        }
        for category in sorted({case.category for case in CASES})
    }
    report = {
        "summary": {"correct": correct, "total": len(results), "accuracy": correct / len(results)},
        "categories": category_summary,
        "cases": [asdict(case) for case in CASES],
        "results": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
