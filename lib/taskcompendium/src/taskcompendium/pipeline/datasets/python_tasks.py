# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind converted Python unit-test tasks to the executable curation stages."""

import ast
import base64
import json
import re
import sys
from pathlib import Path

from taskcompendium.models import ConversationInput, TextMessage, VerifierSpec
from taskcompendium.pipeline.datasets.executable_tasks import normalize, verification_report
from taskcompendium.pipeline.models import (
    CheckSuite,
    DatasetRecipe,
    ImportRejection,
    IntendedUse,
    NormalizationChange,
    NormalizedTask,
    RawRow,
    ReviewRubric,
    SnapshotSource,
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
    snapshot: Path,
    image: str,
    *,
    config: str,
    revision: str,
    rubric: ReviewRubric,
    timeout: float,
    memory_mb: int,
) -> DatasetRecipe:
    """Bind a pinned source, rubric, and explicit grading limits."""

    def normalize_row(row: RawRow) -> NormalizedTask | ImportRejection:
        return normalize_python(row, image, timeout, memory_mb)

    return DatasetRecipe(
        name=f"tasktrove-{name}",
        version=f"tasktrove-{name}-v1",
        source=SnapshotSource("open-thoughts/TaskTrove", revision, config, "train", str(snapshot)),
        normalize=normalize_row,
        rubric=rubric,
        intended_use=IntendedUse.TRAIN,
        check_suite=CheckSuite(
            id="isolated-executable-controls",
            revision="1",
            parameters={"image": image, "timeout": timeout, "memory_mb": memory_mb},
            run=verification_report,
        ),
    )
