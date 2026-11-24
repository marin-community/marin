# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-LCB: LiveCodeBench ``release_v6`` problems from 2024-07-01, answered in languages other than Python.

Prompts, test conversion and grading follow Multi-LCB at :data:`HARNESS_REVISION` with its defaults:
format ``v2`` with prompt enhancement. Every test runs on stdin and stdout; LeetCode's functional tests
are converted with Multi-LCB's vendored converter, and LeetCode prompts gain the problem's examples in
that converted form. Each language is its own benchmark, ``multi-lcb-<language>``, scored on the same
problems; C++ is the only one so far, graded on the worker image's g++. The vendored modules live in
``multi_lcb_scorers``; see its ``VENDORED.md``.
"""

import json
import pickle
import zlib
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

from taskcompendium.convert.answers import source_defect
from taskcompendium.convert.code import THREAD_ENVIRONMENT
from taskcompendium.convert.script_grader import grade_script, script_package, shipped_files
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation.spec import (
    CohortOrder,
    CohortPolicy,
    EvaluationProtocol,
    Metrics,
    OperationKind,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat, required_grader_environment
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, RawRow
from taskcompendium.runtime.resources import inline_resource
from verifyit.spec import DEFAULT_OUTPUT

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.benchmarks.environments import COMPILER_GRADER
from experiments.evaluation.benchmarks.livecodebench import (
    DATASET_REVISION,
    EVIDENCE_FIELDS,
    GRADER_STARTUP_ALLOWANCE,
    RELEASE_V6_FILES,
    livecodebench_item,
    private_test_cases,
)
from experiments.evaluation.benchmarks.multi_lcb_scorers.prompt_enhancer.example_parser import ExampleParser
from experiments.evaluation.benchmarks.multi_lcb_scorers.prompt_enhancer.unified_converter import ConverterFactory
from experiments.evaluation.preparation import EvaluationCorpusRecipe, prepare_evaluation
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

HARNESS_REVISION = "d80be9f24268e9e5c6e1463e26c9f05417fe06a0"
"""Multi-LCB/Multi-LCB commit whose prompt, test conversion and grading this benchmark follows."""
START_DATE = datetime(2024, 7, 1)
"""Multi-LCB's usual ``start_date``: problems from contests on or after it."""
POPULATION = 497
FORMAT_VERSION = "v2"
"""Multi-LCB's default ``format_version``, for converted tests and LeetCode prompt examples."""

SYSTEM_MESSAGE = (
    "You are an expert <param:Plang> programmer. You will be given a question (problem specification) and will "
    "generate a correct <param:Plang> program that matches the specification and passes all tests."
)
FORMAT_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the "
    "sample inputs). Enclose your code within delimiters as follows. Ensure that when the <param:plang> program "
    "runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n\n"
)
TWO_D_ARRAYS = "For 2D arrays, the first line indicates the number of rows, followed by newline-separated rows."
CODE_BLOCK = "```<param:plang>\n<param:comment> YOUR CODE HERE\n```\n\n"
ANSWER = "### Answer: (use the provided format with backticks)\n\n"

SCORER_DEADLINE = 480
"""Seconds the grade script gives one reply, across Multi-LCB's restarts. Curated runs stop grading at 600
seconds and record a timeout as unscored, so the script's own deadline fires first and scores 0."""

HERE = Path(__file__).parent
SCORERS = HERE / "multi_lcb_scorers"
GRADE = grade_script(
    HERE / "multi_lcb_grade.py",
    *shipped_files(SCORERS, "lcb_runner/evaluation/testing_plang.py", "lcb_runner/utils/extraction_utils.py"),
)
CONVERSION_REVISION = "1-" + canonical_sha256({resource.path: resource.source.content_base64 for resource in GRADE})[:16]
"""Names the prepared corpus. The hash follows the grade scripts; bump the leading counter whenever this
module's conversion output changes, such as the prompt, the grader config or the test conversion."""


@dataclass(frozen=True)
class Language:
    """A Multi-LCB language: its harness name and the names ``PLang`` substitutes into prompts."""

    harness: str
    display: str
    markdown: str
    comment: str
    slug: str

    @property
    def benchmark(self) -> str:
        return f"multi-lcb-{self.slug}"


CPP = Language("c++", "C++", "cpp", "//", "cpp")
LANGUAGES = (CPP,)


def is_scored_problem(row: dict[str, Any], context: ConversionContext) -> bool:
    return datetime.fromisoformat(row["contest_date"]) >= START_DATE


def raw_json_arguments(data: str) -> dict[str, Any]:
    """A functional test's JSON lines as named parameters, as ``Test.parse_raw_json`` reads them."""
    if not data.strip():
        return {"param_0": data}
    values = [json.loads(line) for line in data.split("\n") if line.strip()]
    if not values:
        return {"param_0": ""}
    return {f"param_{index}": value for index, value in enumerate(values)}


def stdin_tests(cases: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Each case's stdin and expected stdout, converting functional cases as ``CodeGenerationProblem`` does."""
    test_types = {case["testtype"] for case in cases}
    if test_types == {"stdin"}:
        return [case["input"] for case in cases], [case["output"] for case in cases]
    if test_types != {"functional"}:
        raise ValueError(f"A problem mixes test types: {sorted(test_types)}")
    converted = ConverterFactory.create_converter(FORMAT_VERSION).convert_all_tests(
        [
            {"input": raw_json_arguments(case["input"]), "output": raw_json_arguments(case["output"])["param_0"]}
            for case in cases
        ]
    )
    inputs = [test["stdin"] for test in converted]
    outputs = [test["stdout"] for test in converted]
    return [_newline_terminated(text) for text in inputs], [_newline_terminated(text) for text in outputs]


def _newline_terminated(text: str) -> str:
    return text if not text or text.endswith("\n") else text + "\n"


def leetcode_examples(question: str) -> str:
    """The problem's examples in converted stdin form, as ``EnhancedPromptEnhancer`` adds them, or ``""``."""
    examples = ExampleParser().extract_examples(question)
    if not examples:
        return ""
    converted = ConverterFactory.create_converter(FORMAT_VERSION).convert_all_tests(examples)
    blocks = "".join(
        f"Sample Input {example['example_id']}:\n\n{test['stdin']}\n\n"
        f"Sample Output {example['example_id']}:\n\n{test['stdout']}\n\n\n"
        for example, test in zip(examples, converted, strict=True)
    )
    return f"\n\n### Format: {FORMAT_WITHOUT_STARTER_CODE}\n {TWO_D_ARRAYS}\n\n{blocks}"


def substitute(template: str, language: Language) -> str:
    for parameter, value in (
        ("<param:Plang>", language.display),
        ("<param:plang>", language.markdown),
        ("<param:PLANG>", language.harness.upper()),
        ("<param:comment>", language.comment),
    ):
        template = template.replace(parameter, value)
    return template


def question_prompt(question: str, platform: str, language: Language) -> str:
    """Multi-LCB's user message: LeetCode problems carry converted examples, the rest a format block."""
    prompt = f"### Question:\n{question}\n\n"
    if platform == "leetcode":
        prompt += leetcode_examples(question) + CODE_BLOCK
    else:
        prompt += f"### Format: {FORMAT_WITHOUT_STARTER_CODE}\n" + CODE_BLOCK
    return substitute(prompt + ANSWER, language)


def convert_multi_lcb(row: RawRow, context: ConversionContext, language: Language) -> TaskSpec | ImportRejection:
    data = row.data
    try:
        inputs, outputs = stdin_tests(
            json.loads(data["public_test_cases"]) + private_test_cases(data["private_test_cases"])
        )
    except (zlib.error, pickle.UnpicklingError, ValueError) as error:
        return source_defect("invalid_test_cases", f"Tests do not decode: {error}")
    tests = inline_resource(
        "tests.json.zlib", zlib.compress(json.dumps({"inputs": inputs, "outputs": outputs}).encode(), 9)
    )
    package = script_package(
        (*GRADE, tests),
        {"language": language.harness, "deadline": SCORER_DEADLINE},
        environment=required_grader_environment(context),
        timeout=SCORER_DEADLINE + GRADER_STARTUP_ALLOWANCE,
        answer_path=DEFAULT_OUTPUT,
        env=THREAD_ENVIRONMENT,
    )
    events = (
        TextMessage(role="system", content=substitute(SYSTEM_MESSAGE, language)),
        TextMessage(role="user", content=question_prompt(data["question_content"], data["platform"], language)),
    )
    task = conversation_task(
        row, events=events, package=package, evidence={field: data[field] for field in EVIDENCE_FIELDS}
    )
    return task.model_copy(update={"tags": ("eval", "source:multi-lcb", f"benchmark:{language.benchmark}", "code")})


def source(language: Language) -> RlDataSource[EvaluationCorpusRecipe]:
    return RlDataSource(
        info=SourceInfo(
            id=f"Multi-LCB:{language.slug}",
            title=f"Multi-LCB ({language.display})",
            origin="Multi-LCB",
            family="competitive-programming",
            tags=("eval", "source:multi-lcb", "benchmark", "code", "single-turn"),
            count=POPULATION,
        ),
        config=EvaluationCorpusRecipe(
            recipe=CurationRecipe(
                name=language.benchmark,
                source=HfSource(
                    "livecodebench/code_generation_lite",
                    DATASET_REVISION,
                    RELEASE_V6_FILES,
                    SourceFormat.JSONL,
                    select=is_scored_problem,
                ),
                convert=partial(convert_multi_lcb, language=language),
                version=CONVERSION_REVISION,
                intended_use=IntendedUse.EVAL,
                grader=COMPILER_GRADER,
            ),
            protocol=EvaluationProtocol(
                benchmark=language.benchmark,
                version="2026.11.20",
                operation=OperationKind.CHAT_GENERATION,
                sampling=Sampling(
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    max_tokens=None,
                    stop=(),
                    chat_template_kwargs={"enable_thinking": True},
                ),
                trials=TrialPolicy(count=1, seed=None),
                metrics=(Metrics.ACCURACY,),
                primary_metric="accuracy",
                summary=SummaryKind.MEAN,
                expected_population=POPULATION,
                # A --limit run takes a fixed uniform sample, mixing AtCoder and LeetCode problems.
                cohort=CohortPolicy(order=CohortOrder.SHUFFLED),
            ),
            item_builder=livecodebench_item,
            protocol_name="official",
        ),
        pipeline=prepare_evaluation,
    )


def benchmarks() -> list[EvaluationBenchmark]:
    return fixed_benchmarks([source(language) for language in LANGUAGES])
