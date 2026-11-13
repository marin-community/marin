# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LiveCodeBench ``release_v6`` code generation, prompted and graded as the official harness does.

Prompts are LiveCodeBench's OpenAI-chat template from ``lcb_runner/prompts/code_generation.py`` at
:data:`HARNESS_REVISION`. Each problem is graded on its public and private tests together, as
``CodeGenerationProblem.get_evaluation_sample`` does. The source stores private tests as base64
zlib-compressed pickles of JSON text; conversion decodes them once with an unpickler that refuses
every global, and ships the canonical cases zlib-compressed so the grader never unpickles.
"""

import base64
import io
import json
import pickle
import zlib
from pathlib import Path
from typing import Any

from taskcompendium.convert.answers import source_defect, source_evidence, unsupported
from taskcompendium.convert.code import THREAD_ENVIRONMENT
from taskcompendium.convert.script_grader import grade_script, script_package, shipped_files
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
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, RawRow
from taskcompendium.runtime.resources import inline_resource
from verifyit.spec import DEFAULT_OUTPUT

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.benchmarks.environments import LIVECODEBENCH_GRADER
from experiments.evaluation.preparation import EvaluationCorpusRecipe, prepare_evaluation
from experiments.post_training.task_curation.datasets.skyrl.code import SCORERS, lcb_test_cases
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

HARNESS_REVISION = "998c52d394b836f15fff3b9a29866191108ff81b"
"""LiveCodeBench/LiveCodeBench commit whose prompt, extraction and timeouts this benchmark follows."""
DATASET_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
RELEASE_V6_FILES = ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl")
"""The files ``code_generation_lite.py`` loads for ``version_tag="release_v6"``."""
RELEASE_V6_POPULATION = 1055

SYSTEM_MESSAGE = (
    "You are an expert Python programmer. You will be given a question (problem specification) and will "
    "generate a correct Python program that matches the specification and passes all tests."
)
FORMAT_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the problem and enclose your code "
    "within delimiters."
)
FORMAT_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on "
    "the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program "
    "runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
)

PER_TEST_TIMEOUT = 6
"""Seconds per test: LiveCodeBench's default ``--timeout``."""
SCORER_DEADLINE_CAP = 480
"""Most seconds the scorer runs one reply. Curated runs stop grading at 600 seconds
(``marin.evaluation.dataset_evaluator``) and record a timeout as unscored, so the scorer's own deadline must
fire first. A run past it fails, as at the official deadline; only replies whose tests together take
longer than this, on problems with more than 67 tests, score differently from the official harness."""
GRADER_STARTUP_ALLOWANCE = 60
"""Seconds beyond the scorer deadline for interpreter startup, imports and decompression."""

HERE = Path(__file__).parent
GRADE = grade_script(HERE / "livecodebench_grade.py", *shipped_files(SCORERS, "livecodebench.py"))
GRADER_REVISION = "1-" + canonical_sha256({resource.path: resource.source.content_base64 for resource in GRADE})[:16]
EVIDENCE_FIELDS = (
    "question_id",
    "question_title",
    "platform",
    "contest_id",
    "contest_date",
    "difficulty",
    "starter_code",
)


class _DataOnlyUnpickler(pickle.Unpickler):
    """Load pickled builtin data; refuse any global, so a payload cannot run code."""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"LiveCodeBench test payloads may not reference {module}.{name}")


def private_test_cases(encoded: str) -> list[Any]:
    """Decode ``private_test_cases`` as ``CodeGenerationProblem`` does: JSON, else a compressed pickle of JSON."""
    try:
        return json.loads(encoded)
    except json.JSONDecodeError:
        pass
    payload = _DataOnlyUnpickler(io.BytesIO(zlib.decompress(base64.b64decode(encoded)))).load()
    if not isinstance(payload, str):
        raise ValueError(f"Private test payload is a {type(payload).__name__}, not JSON text")
    return json.loads(payload)


def question_prompt(question: str, starter_code: str) -> str:
    prompt = f"### Question:\n{question}\n\n"
    if starter_code:
        prompt += f"### Format: {FORMAT_WITH_STARTER_CODE}\n```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {FORMAT_WITHOUT_STARTER_CODE}\n```python\n# YOUR CODE HERE\n```\n\n"
    return prompt + "### Answer: (use the provided format with backticks)\n\n"


def scorer_deadline(test_count: int) -> int:
    """The official process deadline for ``test_count`` tests, capped at :data:`SCORER_DEADLINE_CAP`."""
    return min((PER_TEST_TIMEOUT + 1) * test_count + 5, SCORER_DEADLINE_CAP)


def convert_livecodebench(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    data = row.data
    try:
        cases = json.loads(data["public_test_cases"]) + private_test_cases(data["private_test_cases"])
    except (json.JSONDecodeError, zlib.error, pickle.UnpicklingError, ValueError) as error:
        return source_defect("invalid_test_cases", f"Tests do not decode: {error}")
    function_name = json.loads(data["metadata"]).get("func_name")
    if function_name is not None:
        cases = [{**case, "metadata": {"func_name": function_name}} for case in cases]
    try:
        cases = lcb_test_cases(cases)
    except json.JSONDecodeError as error:
        return source_defect("invalid_test_cases", f"Functional test case is not JSON: {error}")
    except (ValueError, TypeError) as error:
        return unsupported("unsupported_test_cases", str(error))
    tests = inline_resource("tests.json.zlib", zlib.compress(json.dumps(cases, sort_keys=True).encode(), 9))
    package = script_package(
        (*GRADE, tests),
        {"per_test_timeout": PER_TEST_TIMEOUT, "deadline": scorer_deadline(len(cases))},
        environment=required_grader_environment(context),
        timeout=scorer_deadline(len(cases)) + GRADER_STARTUP_ALLOWANCE,
        answer_path=DEFAULT_OUTPUT,
        env=THREAD_ENVIRONMENT,
    )
    events = (
        TextMessage(role="system", content=SYSTEM_MESSAGE),
        TextMessage(role="user", content=question_prompt(data["question_content"], data["starter_code"])),
    )
    task = conversation_task(
        row, events=events, package=package, evidence={field: data[field] for field in EVIDENCE_FIELDS}
    )
    return task.model_copy(update={"tags": ("eval", "source:livecodebench", "benchmark:livecodebench", "code")})


def livecodebench_item(task: TaskSpec) -> EvaluationItem:
    """A chat request scored on the final answer, or on complete reasoning when there is no final answer."""
    item = chat_item(task, response_selection="final_or_complete_reasoning")
    evidence = source_evidence(task)
    return item.model_copy(update={"groups": {"difficulty": evidence["difficulty"], "platform": evidence["platform"]}})


def source() -> RlDataSource[EvaluationCorpusRecipe]:
    return RlDataSource(
        info=SourceInfo(
            id="LiveCodeBench:release_v6",
            title="LiveCodeBench",
            origin="LiveCodeBench",
            family="competitive-programming",
            tags=("eval", "source:livecodebench", "benchmark", "code", "single-turn"),
            count=RELEASE_V6_POPULATION,
        ),
        config=EvaluationCorpusRecipe(
            recipe=CurationRecipe(
                name="livecodebench-release-v6",
                source=HfSource(
                    "livecodebench/code_generation_lite", DATASET_REVISION, RELEASE_V6_FILES, SourceFormat.JSONL
                ),
                convert=convert_livecodebench,
                version=GRADER_REVISION,
                intended_use=IntendedUse.EVAL,
                grader=LIVECODEBENCH_GRADER,
            ),
            protocol=EvaluationProtocol(
                benchmark="LiveCodeBench",
                version="2026.11.13",
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
                expected_population=RELEASE_V6_POPULATION,
            ),
            item_builder=livecodebench_item,
            protocol_name="official",
        ),
        pipeline=prepare_evaluation,
    )


def benchmarks() -> list[EvaluationBenchmark]:
    return fixed_benchmarks([source()])
