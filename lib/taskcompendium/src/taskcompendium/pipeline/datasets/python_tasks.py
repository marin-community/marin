# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind converted Python unit-test tasks to the executable curation stages."""

import ast
import base64
import json
import re
import sys

from taskcompendium.models import ConversationInput, TextMessage, VerifierSpec
from taskcompendium.pipeline.datasets.executable_tasks import normalize, verification_report
from taskcompendium.pipeline.datasets.instruction_following import REVISION
from taskcompendium.pipeline.datasets.raw_conversion import RawConverter, with_raw_converter
from taskcompendium.pipeline.datasets.source_definitions import tasktrove_inputs, tasktrove_source
from taskcompendium.pipeline.models import (
    CheckSuite,
    DatasetRecipe,
    HFSource,
    ImportRejection,
    IntendedUse,
    NormalizationChange,
    NormalizedTask,
    RawRow,
    ReviewRubric,
)
from taskcompendium.verifiers.executable import TaskTroveExecutableVerifier

PUBLIC_FIXTURE_CRITERION = (
    "Oracle solutions and private tests must remain hidden; explicitly public setup tests are part of the contract."
)

PYTHON_FILE = re.compile(r"(?<![\w/])(?:/app/|app/)?[A-Za-z_]\w*(?:/[A-Za-z_]\w*)*\.py\b")
PACKAGE = re.compile(r"(?:package (?:at|under)|package[^\n]{0,30} at) /app/([A-Za-z_]\w*)")


def submission_paths(instruction: str) -> tuple[str, ...]:
    """Capture only Python filenames declared in the public task contract."""
    packages = set(PACKAGE.findall(instruction))
    paths = set()
    for match in PYTHON_FILE.finditer(instruction):
        filename = match.group().removeprefix("/app/").removeprefix("app/")
        if filename.startswith("test_") or filename.startswith("tests/"):
            continue
        if "/" not in filename and len(packages) == 1 and filename.removesuffix(".py") not in packages:
            filename = f"{next(iter(packages))}/{filename}"
        paths.add(f"/app/{filename}")
    return tuple(sorted(paths))


def normalize_python(row: RawRow, image: str, timeout: float, memory_mb: int) -> NormalizedTask | ImportRejection:
    task = normalize(row, image, timeout, memory_mb)
    if isinstance(task, ImportRejection):
        return task
    instruction = row.data["converted"]["instruction"]
    paths = submission_paths(instruction)
    changes = []
    if not paths:
        test_files = [
            base64.b64decode(encoded, validate=True).decode()
            for path, encoded in row.data["converted"]["data_files"].items()
            if path.startswith("tests/") and path.endswith(".py")
        ]
        imports = [
            node
            for text in test_files
            for node in ast.walk(ast.parse(text))
            if isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.split(".")[0] not in sys.stdlib_module_names | {"pytest", "numpy"}
        ]
        modules = {node.module for node in imports}
        if len(modules) == 1 and all(alias.name in instruction for node in imports for alias in node.names):
            module = next(iter(modules))
            assert module is not None
            paths = (f"/app/{module.replace('.', '/')}.py",)
            replacement = instruction + f"\n\nDelivery: write the requested implementation to `{paths[0]}`.\n"
            changes.append(
                NormalizationChange(
                    field="instruction",
                    reason="Make the grader's module filename explicit without changing the requested API",
                    original=instruction,
                    replacement=replacement,
                )
            )
            task = task.model_copy(
                update={"context": ConversationInput(events=(TextMessage(role="user", content=replacement),))}
            )
    if not paths:
        return ImportRejection(
            reason="unsupported_public_output_contract",
            detail="No explicit public Python filename and private imports require APIs absent from the request",
        )
    verifier = TaskTroveExecutableVerifier.model_validate_json(task.verifier.parameters_json)
    verifier = verifier.model_copy(update={"submission_paths": paths})
    task = task.model_copy(
        update={
            "output_paths": paths,
            "verifier": VerifierSpec(kind=task.verifier.kind, parameters_json=verifier.model_dump_json()),
        }
    )
    changes.append(
        NormalizationChange(
            field="output_paths",
            reason="Capture the Python filenames declared in the public instruction",
            original=json.dumps(["/app/solution.py", "/app/solution.cpp"]),
            replacement=json.dumps(paths),
        )
    )
    return NormalizedTask(task, tuple(changes))


def recipe(
    name: str,
    image: str,
    *,
    config: str,
    revision: str,
    rubric: ReviewRubric,
    converter: RawConverter,
    converter_revision: str,
    timeout: float,
    memory_mb: int,
) -> DatasetRecipe:
    """Bind a pinned source, rubric, and explicit grading limits."""

    def normalize_row(row: RawRow) -> NormalizedTask | ImportRejection:
        return normalize_python(row, image, timeout, memory_mb)

    source_recipe = DatasetRecipe(
        name=f"tasktrove-{name}",
        version=f"tasktrove-{name}-v1",
        source=HFSource("open-thoughts/TaskTrove", revision, config, "train"),
        normalize=normalize_row,
        rubric=rubric,
        intended_use=IntendedUse.TRAIN,
        inputs=tasktrove_inputs(config, revision),
        check_suite=CheckSuite(
            id="isolated-executable-controls",
            revision="1",
            parameters={"image": image, "timeout": timeout, "memory_mb": memory_mb},
            run=verification_report,
        ),
    )
    return with_raw_converter(source_recipe, converter, converter_revision)


PYMETHODS_COMMON_CRITERIA = (
    "Check parameter meanings and essential transition rules against the tests and oracle; an "
    "algorithmic constraint missing from the public request is a defect even when the oracle passes.",
    "Do not equate a quantity, rate, time limit, and lower bound; identify concrete parameter mismatches.",
    "Flag undefined behavior, missing fixtures, unavailable dependencies, and contradictory examples.",
    "Private tests and oracle solutions are review evidence and must remain hidden from the solving actor.",
    "A passing oracle shows test compatibility, not specification coverage; cite a concrete defect when rejecting.",
)


PYTHON_BASIC_CRITERIA = (
    "Check that the public Python API, output filenames, return values, and exceptions agree with private tests.",
    "Flag contradictory examples, unstated behavior, missing fixtures, and unavailable dependencies.",
    PUBLIC_FIXTURE_CRITERION,
    "A passing oracle shows compatibility with tests; assess whether those tests cover the public specification.",
)

SOURCES = {
    "curriculum_easy": tasktrove_source(
        config="DCAgent__exp_rpt_curriculum-easy",
        revision=REVISION,
        rubric=ReviewRubric(
            id="curriculum_easy-answerability",
            version="1",
            criteria=(
                *PYTHON_BASIC_CRITERIA,
                "Check the Python entry point and each stated beginner-level rule against test cases, including "
                "empty input and boundaries.",
            ),
        ),
    ),
    "curriculum_medium": tasktrove_source(
        config="DCAgent__exp_rpt_curriculum-medium-v2",
        revision=REVISION,
        rubric=ReviewRubric(
            id="curriculum_medium-answerability",
            version="3",
            criteria=(
                "For every boolean membership assertion in disclosed setup tests, derive the expected value from "
                "its literal group fixture and the public parsing rule before deciding quality. Including "
                "assertions in the contract does not excuse a contradiction with an explicit prose rule. Cite "
                "the fixture membership and contradictory assertion when one exists.",
                "Public setup tests may specify missing API details, but they do not override an explicit prose "
                "rule unless the task states a precedence rule. A membership fixture that marks a listed member "
                "false contradicts a rule that all listed members are true; cite the literal values.",
                "Check that the public Python API, output filenames, return values, and exceptions agree with "
                "private tests.",
                "The repair note explicitly exposes setup tests as API evidence; assess the request together with these "
                "fixtures and flag contradictions between them.",
                PUBLIC_FIXTURE_CRITERION,
                "A passing oracle shows compatibility with tests; assess whether those tests cover the public "
                "specification.",
                "Check every stated algorithmic rule, mutation requirement, and boundary against the private tests; "
                "difficulty alone is not a defect.",
            ),
        ),
    ),
    "e2egit": tasktrove_source(
        config="DCAgent__exp_rpt_e2egit-v2",
        revision=REVISION,
        rubric=ReviewRubric(
            id="e2egit-answerability",
            version="1",
            criteria=(
                *PYTHON_BASIC_CRITERIA,
                "Check calculator, banking, inventory, and library APIs against tests; inspect exact error messages "
                "and whether filename normalization leaves any unstated behavior.",
            ),
        ),
    ),
    "e2egit_large": tasktrove_source(
        config="DCAgent__exp_rpt_e2egit-large",
        revision=REVISION,
        rubric=ReviewRubric(
            id="e2egit_large-answerability",
            version="1",
            criteria=(
                *PYTHON_BASIC_CRITERIA,
                "Check Calculator arithmetic methods and exact zero-division error messages; repeated calculator "
                "tasks need duplicate review, and missing multiplication tests mean incomplete coverage.",
            ),
        ),
    ),
    "multifile": tasktrove_source(
        config="DCAgent__exp_rpt_multifile-v3",
        revision=REVISION,
        rubric=ReviewRubric(
            id="multifile-answerability",
            version="1",
            criteria=(
                "Check that the public Python API, output filenames, return values, and exceptions agree with "
                "private tests.",
                "The repair note explicitly exposes setup tests as API evidence; assess the request together with these "
                "fixtures and flag contradictions between them.",
                PUBLIC_FIXTURE_CRITERION,
                "A passing oracle shows compatibility with tests; assess whether those tests cover the public "
                "specification.",
                "Check that every required file and import is specified and captured by the grading contract; flag "
                "tests "
                "that require unavailable sibling modules.",
            ),
        ),
    ),
    "pymethods": tasktrove_source(
        config="DCAgent__exp_rpt_pymethods2test-v3",
        revision=REVISION,
        rubric=ReviewRubric(
            id="pymethods-answerability",
            version="2",
            criteria=(
                "For partitioning and scheduling problems, check whether contiguity, order, indivisibility, and "
                "coverage restrictions are explicitly supplied. Construct a better valid solution under the public "
                "rules before accepting a narrower private optimum.",
                "Check tests against the stated input domain, including zero values and allowed worker counts. "
                "Reject a contradiction in expected behavior; distinguish explicitly described edge cases from a "
                "merely abbreviated constraints list.",
                "Check method signatures, class context, return values, and exceptions against the private tests.",
                *PYMETHODS_COMMON_CRITERIA,
            ),
        ),
    ),
    "pymethods_large": tasktrove_source(
        config="DCAgent__exp_rpt_pymethods2test-large-v2",
        revision=REVISION,
        rubric=ReviewRubric(
            id="pymethods_large-answerability",
            version="2",
            criteria=(
                "Verify that every function or class name and signature required by private imports is present in "
                "the public request or public fixtures. A request to follow a provided signature is incomplete when "
                "no signature is supplied; a conventional name is not a public API contract.",
                "Check class context, method signatures, instance state, and dependency requirements against the "
                "private tests.",
                *PYMETHODS_COMMON_CRITERIA,
            ),
        ),
    ),
    "stack_pytest": tasktrove_source(
        config="DCAgent__exp_rpt_stack-pytest-v2",
        revision=REVISION,
        rubric=ReviewRubric(
            id="stack_pytest-answerability",
            version="1",
            criteria=(
                "Check that the adapted Stack Overflow request defines the tested API and supplies all relevant "
                "context.",
                "Check that the named modules and package files in the public request are captured by the runtime; "
                "a solution.py-only submission cannot implement a different named package.",
                "Missing oracle controls imply verification uncertainty, not an automatically bad problem.",
                "Flag undefined behavior, missing fixtures, unavailable dependencies, and contradictory examples.",
                "Private tests and oracle solutions are review evidence and must remain hidden from the solving actor.",
                "A passing oracle shows test compatibility, not specification coverage; cite a concrete defect when "
                "rejecting.",
            ),
        ),
    ),
    "unitsyn_large": tasktrove_source(
        config="DCAgent__exp_rpt_unitsyn-python-large-v2",
        revision=REVISION,
        rubric=ReviewRubric(
            id="unitsyn_large-answerability",
            version="1",
            criteria=(
                "Check that the public Python API, filenames, return values, and exception behavior "
                "agree with the private tests.",
                *PYMETHODS_COMMON_CRITERIA,
            ),
        ),
    ),
}


def recipe_for_source(
    name: str,
    image: str,
    *,
    converter: RawConverter,
    converter_revision: str,
    timeout: float,
    memory_mb: int,
) -> DatasetRecipe:
    source = SOURCES[name]
    return recipe(
        name,
        image,
        config=source.config,
        revision=source.revision,
        rubric=source.rubric,
        converter=converter,
        converter_revision=converter_revision,
        timeout=timeout,
        memory_mb=memory_mb,
    )
