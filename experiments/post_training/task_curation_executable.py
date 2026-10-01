# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind existing TaskTrove cleanup converters to decoded snapshot rows."""

import base64
import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from tasktrove_verify.spec import PytestSpec, spec_to_table

from experiments.post_training.tasktrove.converters.codeforces import convert_codeforces
from experiments.post_training.tasktrove.converters.converted_task import ConvertedTask, Rejected
from experiments.post_training.tasktrove.converters.nl2bash import convert_nl2bash
from experiments.post_training.tasktrove.converters.python_unit_tests import convert as convert_unitsyn
from experiments.post_training.tasktrove.converters.taco import convert_taco
from experiments.post_training.tasktrove.taskbinary import TaskFiles

CONVERTERS: dict[str, Callable[[TaskFiles], ConvertedTask | Rejected]] = {
    "nl2bash": convert_nl2bash,
    "taco": convert_taco,
    "codeforces": convert_codeforces,
    "unitsyn": convert_unitsyn,
}

CHECKER_ENVIRONMENT_NAMES = {
    b"TASKTROVE_TESTS_DIR": b"VERIFYIT_TESTS_DIR",
    b"TASKTROVE_WORKSPACE": b"VERIFYIT_WORKSPACE",
    b"TASKTROVE_LOGS_DIR": b"VERIFYIT_LOGS_DIR",
}


def converted_row(
    data: Mapping[str, Any],
    name: str,
    *,
    converter: Callable[[TaskFiles], ConvertedTask | Rejected] | None = None,
) -> dict[str, Any]:
    """Retain raw input and add one converter's result or typed rejection."""
    files = {path: base64.b64decode(encoded, validate=True) for path, encoded in data["files"].items()}
    row = dict(data)
    try:
        convert = CONVERTERS[name] if converter is None else converter
        converted = convert(TaskFiles(files))
    except (ValueError, KeyError) as error:
        row["conversion_rejection"] = {"reason": "converter_error", "detail": str(error)}
        return row
    if isinstance(converted, Rejected):
        row["conversion_rejection"] = {"reason": converted.status.value, "detail": converted.detail}
        return row
    grader_spec = spec_to_table(converted.spec)
    if isinstance(converted.spec, PytestSpec):
        # The selected image owns its Python/report dependencies, rather than
        # assuming the source Dockerfile's private venv exists in this runtime.
        grader_spec["python"] = "python3"
    controls = converted.solution_files or {p: b for p, b in files.items() if p.startswith("solution/")}
    changes = []
    if converted.instruction != data["instruction"]:
        changes.append(
            {
                "field": "instruction",
                "reason": "Existing source converter corrected delivery boilerplate",
                "original": data["instruction"],
                "replacement": converted.instruction,
            }
        )
    data_files = {}
    for path, content in converted.data_files.items():
        replacement = content
        if path.startswith("tests/") and path.endswith((".py", ".sh")):
            for original_name, canonical_name in CHECKER_ENVIRONMENT_NAMES.items():
                replacement = replacement.replace(original_name, canonical_name)
        if replacement != content:
            changes.append(
                {
                    "field": f"data_files/{path}",
                    "reason": "Normalize standalone checker environment references to verifyit",
                    "original": base64.b64encode(content).decode(),
                    "replacement": base64.b64encode(replacement).decode(),
                }
            )
        data_files[path] = base64.b64encode(replacement).decode()
    row["converted"] = {
        "instruction": converted.instruction,
        "grader_spec": grader_spec,
        "data_files": data_files,
        "control_files": {p: base64.b64encode(b).decode() for p, b in controls.items()},
        "source_dockerfile_sha256": hashlib.sha256(converted.dockerfile.encode()).hexdigest(),
        "tags": list(converted.tags),
        "normalization_changes": changes,
    }
    return row


def convert_snapshot(
    snapshot: Path,
    output: Path,
    name: str,
    *,
    converter: Callable[[TaskFiles], ConvertedTask | Rejected] | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with snapshot.open() as source, output.open("w") as target:
        for line in source:
            target.write(
                json.dumps(converted_row(json.loads(line), name, converter=converter), ensure_ascii=False) + "\n"
            )
