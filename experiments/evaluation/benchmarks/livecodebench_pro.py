# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LiveCodeBench Pro: Codeforces problems from July 2024 to June 2025, solved in C++ and judged on hidden tests.

Problem statements come from ``QAQAQAQAQ/LiveCodeBench-Pro`` (Apache-2.0, gated: accept its terms on
Hugging Face) and tests from ``QAQAQAQAQ/LiveCodeBench-Pro-Testcase`` (no stated license). The prompt and
answer extraction follow the official harness at :data:`HARNESS_REVISION`; judging follows its local judge,
LightCPVerifier, reimplemented in ``livecodebench_pro_grade.py`` rather than vendored (AGPL-3.0).

Each task ships its problem's test archive unchanged: ``config.yaml``, a testlib ``checker.cpp`` and
numbered ``.in``/``.ans`` files. Interactive problems are excluded: their archives replace the
interactor with an offline checker whose input holds the hidden answer, so correct solutions fail.
"""

import hashlib
import re
import zipfile
from io import BytesIO
from pathlib import Path

import yaml
from evalrecords.records import TaskEvidence
from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.convert.answers import source_evidence
from taskcompendium.convert.code import THREAD_ENVIRONMENT
from taskcompendium.convert.script_grader import grade_script, script_package
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import (
    EvaluationItem,
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
from taskcompendium.pipeline.models import IntendedUse, RawRow
from taskcompendium.runtime.resources import inline_resource
from verifyit.spec import DEFAULT_OUTPUT

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.benchmarks.environments import LIVECODEBENCH_PRO_GRADER
from experiments.evaluation.preparation import EvaluationCorpusRecipe, prepare_evaluation
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin

HARNESS_REVISION = "d614eef4f95b3654bf0758c64e1194b2575850b8"
"""GavinZhengOI/LiveCodeBench-Pro commit whose prompt, extraction and judge settings this benchmark follows."""
PROBLEMS_REVISION = "adebffce047dddb7768a86bace6aea4f7425e3bc"
PROBLEM_FILES = (
    "data/biannual_2024_7_12-00000-of-00001.parquet",
    "data/biannual_2025_1_6-00000-of-00001.parquet",
)
"""The two half-year releases that have public tests; the quarterly splits are their subsets."""
TESTS_REVISION = "5257736c0a4e30ba0949d41c56a257c323d9c600"
TESTS_INPUT = "tests"
HERE = Path(__file__).parent
TEST_FILES = tuple(f"{problem}.zip" for problem in (HERE / "livecodebench_pro_tests.txt").read_text().split())
"""Every test archive in the tests repository at :data:`TESTS_REVISION`."""
POPULATION = 678
"""The 720 problems less those :func:`judgeable` drops: 26 interactive, 10 without tests, and 6 whose tests
give one input two different answers (1988D, 1988E, 1998D, 2004F, 2023E, 2044H)."""
RESOURCE_BUDGET_BYTES = 400 << 20
"""Above the largest selected test archive, 203 MiB (problem 2053I1)."""

PROMPT = (
    "\n        You are a competitive programmer. You will be given a problem statement, please implement solution "
    "in C++. The execution time and memory limit are also stated in the statement so be aware of the "
    "complexity of the program. Please wrap the code in ```cpp and ``` so that it is properly formatted.\n        "
)
"""The official harness's prompt, indentation included; the statement follows it directly."""
INTERACTIVE = re.compile(r"\binteractive problem\b", re.IGNORECASE)
READS_ANSWER = re.compile(r"\bans\s*\.")
"""A testlib checker that reads the expected answer stream, so its answer files must agree."""

DEADLINE = 480
"""Most seconds one reply is judged. Curated runs stop grading at 600 seconds
(``marin.evaluation.dataset_evaluator``) and record a timeout as unscored, so judging must end first."""
GRADER_STARTUP_ALLOWANCE = 60
"""Seconds beyond the deadline for staging the task's files and starting the grade script. Unpacking the
tests and compiling count against the deadline, so judging ends within it whatever setup takes."""

GRADE = grade_script(HERE / "livecodebench_pro_grade.py")
CONVERSION_REVISION = "1-" + canonical_sha256({resource.path: resource.source.content_base64 for resource in GRADE})[:16]
"""Names the prepared corpus. The hash follows the grade script; bump the leading counter whenever this
module's conversion output changes, such as the prompt, the grader config or the test selection."""
EVIDENCE_FIELDS = ("problem_id", "problem_title", "difficulty", "platform", "link", "time_limit", "memory_limit")


def seconds(limit: str) -> float:
    if not limit.endswith("s"):
        raise ValueError(f"Unsupported time limit: {limit}")
    return float(limit[:-1])


def megabytes(limit: str) -> int:
    if not limit.endswith("m"):
        raise ValueError(f"Unsupported memory limit: {limit}")
    return int(limit[:-1])


def answers_agree(archive: zipfile.ZipFile, cases: list[list[str]]) -> bool:
    """Whether every two tests that share an input expect the same answer, so some program can pass all."""
    answers: dict[str, str] = {}
    for test_input, test_answer in cases:
        input_digest, answer_digest = (
            hashlib.file_digest(archive.open(name), "sha256").hexdigest() for name in (test_input, test_answer)
        )
        if answers.setdefault(input_digest, answer_digest) != answer_digest:
            return False
    return True


def judge_config(archive: zipfile.ZipFile) -> dict:
    """The cases and limits the official judge reads from the archive's ``config.yaml``."""
    config = yaml.safe_load(archive.read("config.yaml"))
    if config.get("type") != "default" or config.get("checker") != "checker.cpp":
        raise ValueError(f"Unsupported judge type {config.get('type')!r} or checker {config.get('checker')!r}")
    prefix_in, prefix_out = config.get("input_prefix", ""), config.get("output_prefix", "")
    suffix_in, suffix_out = config.get("input_suffix", ".in"), config.get("output_suffix", ".ans")
    count = sum(subtask["n_cases"] for subtask in config["subtasks"])
    cases = [
        [f"testdata/{prefix_in}{index}{suffix_in}", f"testdata/{prefix_out}{index}{suffix_out}"]
        for index in range(1, count + 1)
    ]
    missing = {name for case in cases for name in case} - set(archive.namelist())
    if missing:
        raise ValueError(f"The archive lacks {len(missing)} test files, such as {min(missing)}")
    return {
        "cases": cases,
        "time_limit": seconds(config["time_limit"]),
        "memory_mb": megabytes(config["memory_limit"]),
        "deadline": DEADLINE,
    }


def judgeable(data: dict, context: ConversionContext) -> bool:
    """Whether the offline judge can score the problem: it is not interactive, has tests, and they agree."""
    problem = data["problem_id"]
    if INTERACTIVE.search(data["problem_statement"]) or f"{problem}.zip" not in TEST_FILES:
        return False
    with zipfile.ZipFile(BytesIO((context.inputs[TESTS_INPUT] / f"{problem}.zip").read_bytes())) as archive:
        if not READS_ANSWER.search(archive.read("checker.cpp").decode()):
            return True
        return answers_agree(archive, judge_config(archive)["cases"])


def convert_livecodebench_pro(row: RawRow, context: ConversionContext) -> TaskSpec:
    data = row.data
    archive_bytes = (context.inputs[TESTS_INPUT] / f"{data['problem_id']}.zip").read_bytes()
    with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
        config = judge_config(archive)
    package = script_package(
        (*GRADE, inline_resource("tests.zip", archive_bytes)),
        config,
        environment=required_grader_environment(context),
        timeout=DEADLINE + GRADER_STARTUP_ALLOWANCE,
        answer_path=DEFAULT_OUTPUT,
        env=THREAD_ENVIRONMENT,
    )
    events = (TextMessage(role="user", content=PROMPT + data["problem_statement"]),)
    task = conversation_task(
        row, events=events, package=package, evidence={field: data[field] for field in EVIDENCE_FIELDS}
    )
    return task.model_copy(update={"tags": ("eval", "source:livecodebench-pro", "benchmark:livecodebench-pro", "code")})


def livecodebench_pro_item(task: TaskSpec) -> EvaluationItem:
    """A chat request scored on the final answer alone, grouped by difficulty."""
    item = chat_item(task, response_selection="final_only")
    return item.model_copy(update={"groups": {"difficulty": source_evidence(task)["difficulty"]}})


def source() -> RlDataSource[EvaluationCorpusRecipe]:
    return RlDataSource(
        info=SourceInfo(
            data_restriction=DataRestriction.OPEN,
            upstream_dataset="2024H2-2025H1",
            upstream_name="LiveCodeBench Pro",
            build_path="LiveCodeBench-Pro",
            origin=SourceOrigin.LIVECODEBENCH_PRO,
            family="competitive-programming",
            tags=("eval", "source:livecodebench-pro", "benchmark", "code", "single-turn"),
        ),
        config=EvaluationCorpusRecipe(
            recipe=CurationRecipe(
                name="livecodebench-pro",
                source=HfSource(
                    "QAQAQAQAQ/LiveCodeBench-Pro",
                    PROBLEMS_REVISION,
                    PROBLEM_FILES,
                    SourceFormat.PARQUET,
                    select=judgeable,
                ),
                convert=convert_livecodebench_pro,
                version=CONVERSION_REVISION,
                intended_use=IntendedUse.EVAL,
                # The converter reads each archive as bytes; an auxiliary input's format is never parsed.
                inputs={
                    TESTS_INPUT: HfSource(
                        "QAQAQAQAQ/LiveCodeBench-Pro-Testcase", TESTS_REVISION, TEST_FILES, SourceFormat.GENERATED
                    )
                },
                grader=LIVECODEBENCH_PRO_GRADER,
                resource_budget_bytes=RESOURCE_BUDGET_BYTES,
            ),
            protocol=EvaluationProtocol(
                benchmark="LiveCodeBench-Pro",
                version="2026.11.21",
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
            ),
            item_builder=livecodebench_pro_item,
            protocol_name="official",
            # Test archives reach 203 MiB per problem; an attempt keeps a corpus reference instead of a copy.
            task_evidence=TaskEvidence.REFERENCE,
        ),
        pipeline=prepare_evaluation,
    )


def benchmarks() -> list[EvaluationBenchmark]:
    return fixed_benchmarks([source()])
