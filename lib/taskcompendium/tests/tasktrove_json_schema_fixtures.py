# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic JSON-schema TaskTrove archives for importer contract tests."""

import json

from taskcompendium.importers.tasktrove.convert import read_archive
from taskcompendium.importers.tasktrove.models import TaskArchive

from .tasktrove_fixtures import FIXTURE_DATASET_URI, FIXTURE_REVISION, _tar_gz


def json_schema_archive(
    schema_format: str,
    converter: str,
    family: str,
    source: str,
    path: str,
) -> TaskArchive:
    """Build a synthetic structured-output task with a two-field schema."""
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
    )
    return read_archive(
        _tar_gz(
            {
                "task.toml": (
                    f"""version = "1.0"

[metadata]
tasktrove_source = "{source}"
tasktrove_path = "{path}"
family = "{family}"
converter = "{converter}"
mode = "json-schema"
tags = ["test", "structured-output", "{schema_format}"]
"""
                ),
                "instruction.md": (
                    "Provide a synthetic answer in the requested schema format.\n\n"
                    "Write your final answer to `/app/answer.txt`.\n"
                ),
                "tests/verifier.toml": (
                    f"""mode = "json-schema"
schema = "schema.json"
format = "{schema_format}"
output = "/app/answer.txt"
"""
                ),
                "tests/schema.json": schema,
            }
        ),
        source,
        path,
        FIXTURE_DATASET_URI,
        FIXTURE_REVISION,
    )
