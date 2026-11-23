# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import the published PDBThink coordinate panel without publishing prompts or gold."""

import gzip
import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from functools import cache
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath

from taskcompendium.convert.answers import source_evidence
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import EvaluationItem
from taskcompendium.importers.evalchemy.grader import source_package
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import RawRow

PDBTHINK_REVISION = "1bf06a464b2eab2f1e16c4dd16714314f225e196"
BUNDLE_PATH = f"Open-Athena/pdbthink/{PDBTHINK_REVISION}/evaluation.json.gz"
MANIFEST_PATH = Path(__file__).parent / "evalchemy/assets/pdbthink_manifest.json"


@cache
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def read_pdbthink(path: StoragePath, _context: ConversionContext) -> Iterator[dict]:
    """Validate even local source overrides before converting any benchmark row."""
    with path.open("rb") as stream:
        compressed = stream.read()
    if hashlib.sha256(compressed).hexdigest() != manifest()["bundle_sha256"]:
        raise ValueError("PDBThink bundle checksum does not match the frozen manifest")
    rows = json.loads(gzip.decompress(compressed))["renders"]
    counts = Counter(row["report_metadata"]["question_family"] for row in rows)
    for row in rows:
        yield {**row, "family_weight": 1 / counts[row["report_metadata"]["question_family"]]}


def convert_pdbthink(row: RawRow, context: ConversionContext) -> TaskSpec:
    data = dict(row.data)
    prompt_hash = hashlib.sha256(
        json.dumps(data["messages"], sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
    task = conversation_task(
        row,
        events=tuple(TextMessage(**message) for message in data["messages"]),
        package=source_package(
            "pdbthink",
            {
                "answer_schema": data["answer_schema"],
                "gold_answer": data["gold_answer"],
                "parameters": data["parameters"],
                "report_metadata": {**data["report_metadata"], "prompt_sha256": prompt_hash},
                "family_weight": data["family_weight"],
            },
            environment=required_grader_environment(context),
        ),
        evidence=data,
    )
    return task.model_copy(update={"tags": ("eval", "benchmark:pdbthink", "development")})


def pdbthink_item(task: TaskSpec) -> EvaluationItem:
    data = source_evidence(task)
    metadata = data["report_metadata"]
    return chat_item(task, response_selection="provider_response").model_copy(
        update={
            "groups": {
                "family": metadata["question_family"],
                "representation": metadata["representation"],
            }
        }
    )
