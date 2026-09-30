# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Historical catalog reconstruction must preserve graders and accepted-row identity."""

import copy
import json

import pytest

from taskcompendium.catalog_release import EMPTY_FINAL_TOOLS, reconstruct_catalog_records, upgrade_catalog_task
from taskcompendium.models import (
    SCHEMA_VERSION,
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.verifiers.multiple_choice import multiple_choice_answer


@pytest.fixture
def joined_rows():
    pin = "2026.09.18.3#manifest-sha256=" + "a" * 64 + "#parquet-size=100#parquet-etag=etag#parquet-version-id=id"
    task = TaskSpec(
        id="accepted",
        context=ConversationInput(events=(TextMessage(role="user", content="Choose one option."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=multiple_choice_answer("B", 3),
        source=Source(dataset="source", revision=pin, row="subset:qa.jsonl", importer_revision="v1"),
        tags=("first", "second"),
    ).model_dump(mode="json")
    task.update(schema_version="0.13", final_tools=copy.deepcopy(EMPTY_FINAL_TOOLS))
    # Keep the original string byte-for-byte, including noncanonical whitespace.
    task["verifier"]["parameters_json"] = '{ "expected": "B", "options": 3 }'
    projected = {key: value for key, value in task.items() if key not in {"schema_version", "verifier"}}
    projected.update(source_category="qa", submission_instruction="Give a letter.")
    shared = dict(
        source="subset",
        path="qa.jsonl",
        route="qa",
        mode="mcq",
        family="qa",
        converter="mcqa",
        template_id="t",
        tags=task["tags"],
        archive_sha256="b" * 64,
    )
    catalog = dict(shared, id="accepted", specification_json=json.dumps(task), source_metadata_json="{}")
    ledger = dict(
        shared,
        imported_id="accepted",
        disposition="imported",
        input_split="tasks",
        input_file="source.parquet",
        input_object_pin=pin,
    )
    projection = dict(
        task=projected,
        source_proof=dict(
            source_row="subset:qa.jsonl",
            input_file="source.parquet",
            input_object_pin=pin,
            archive_path="qa.jsonl",
            archive_sha256="b" * 64,
        ),
    )
    return task, projection, catalog, ledger


def test_catalog_join_preserves_exact_verifier_and_explicitly_upgrades_empty_tools(joined_rows):
    payload, projection, catalog, ledger = joined_rows
    (record,) = reconstruct_catalog_records([projection], [catalog], [ledger])
    assert record.task.schema_version == SCHEMA_VERSION
    assert record.task.final_tools == ()
    assert record.task.verifier.model_dump(mode="json") == payload["verifier"]
    assert record.task.tags == ("first", "second")
    assert record.task.answer_type is AnswerType.TEXT
    assert record.source_proof.archive_sha256 == ledger["archive_sha256"]


@pytest.mark.parametrize("mutation", ["projection", "ledger", "missing", "duplicate"])
def test_catalog_join_rejects_source_or_projection_drift(joined_rows, mutation):
    _, projection, catalog, ledger = joined_rows
    ledgers = [ledger]
    if mutation == "projection":
        projection["task"]["context"]["events"][0]["content"] = "Changed question."
    elif mutation == "ledger":
        ledger["archive_sha256"] = "c" * 64
    elif mutation == "missing":
        ledgers = []
    else:
        ledgers.append(ledger)
    with pytest.raises(ValueError):
        reconstruct_catalog_records([projection], [catalog], ledgers)


@pytest.mark.parametrize("policy", ["functions", "tool_choice", "parallel_tool_calls"])
def test_catalog_upgrade_rejects_unmigrated_terminal_tool_policy(joined_rows, policy):
    payload, _, _, _ = joined_rows
    payload["final_tools"][policy] = (
        [{"name": "call"}] if policy == "functions" else "required" if policy == "tool_choice" else False
    )
    with pytest.raises(ValueError, match="no request policy"):
        upgrade_catalog_task(payload)
