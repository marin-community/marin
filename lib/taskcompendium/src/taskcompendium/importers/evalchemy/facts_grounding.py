# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FACTS Grounding v2 public examples as single-turn tasks graded by a multi-step judge protocol."""

import pyarrow.csv as pacsv
from rigging.filesystem.storage_path import StoragePath
from verifyit.spec import JudgeSpec

from taskcompendium.convert.answers import source_evidence
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import EvaluationItem
from taskcompendium.importers.evalchemy.grader import source_package
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import RawRow

FACTS_GROUNDING_COLUMNS = (
    "system_instruction",
    "user_request",
    "context_document",
    "full_prompt",
    "domain",
    "type",
    "high_level_type",
)
# Each judge step may write a long baseline answer or one verdict per response sentence.
JUDGE_REPLY_TOKENS = 32768
JUDGE_REQUEST_TIMEOUT = 600
JUDGE_TRANSPORT_BUDGET = 1200
JUDGE_STEPS = 3


def read_facts_grounding(file: StoragePath, _context: ConversionContext):
    """Yield the CSV's rows; context documents exceed the standard library's CSV field limit."""
    with file.open("rb") as stream:
        table = pacsv.read_csv(
            stream,
            parse_options=pacsv.ParseOptions(newlines_in_values=True),
            convert_options=pacsv.ConvertOptions(
                column_types=dict.fromkeys(FACTS_GROUNDING_COLUMNS, "string"), strings_can_be_null=False
            ),
        )
    if tuple(table.column_names) != FACTS_GROUNDING_COLUMNS:
        raise ValueError(f"Unexpected FACTS Grounding columns: {table.column_names}")
    for index, row in enumerate(table.to_pylist()):
        yield {"source_index": index, **row}


def convert_facts_grounding(row: RawRow, context: ConversionContext) -> TaskSpec:
    """Send the official full prompt as one user message; the judge contract carries the request and document."""
    judge = JudgeSpec(
        # The scorer reads the user request and context document from the judge contract.
        question=row.data["user_request"],
        references=(row.data["context_document"],),
        exact_gate=False,
        max_completion_tokens=JUDGE_REPLY_TOKENS,
        request_timeout=JUDGE_REQUEST_TIMEOUT,
        transport_attempts=3,
        transport_budget=JUDGE_TRANSPORT_BUDGET,
    )
    environment = required_grader_environment(context)
    environment = environment.model_copy(update={"capabilities": (*environment.capabilities, "network")})
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=row.data["full_prompt"]),),
        package=source_package(
            "facts_grounding",
            {"full_prompt": row.data["full_prompt"]},
            environment=environment,
            judge=judge,
            timeout=60 + JUDGE_STEPS * JUDGE_TRANSPORT_BUDGET,
        ),
        evidence={key: row.data[key] for key in ("source_index", "domain", "type", "high_level_type")},
    )
    return task.model_copy(update={"tags": ("eval", "source:facts-grounding", "benchmark:facts-grounding-v2")})


def facts_grounding_item(task: TaskSpec) -> EvaluationItem:
    """Grade only the final answer, and report scores by document domain."""
    return chat_item(task, response_selection="final_only").model_copy(
        update={"groups": {"domain": source_evidence(task)["domain"]}}
    )
