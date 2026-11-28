# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The native conversion release's lexical population render: published bytes must
be their canonical serialization, and the record binds the release pin."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    ExecutionRequirements,
    PlainText,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.models import CommandSemantics as _CommandSemantics

from experiments.post_training.task_curation.native_population import published_rows


def _spec(task_id: str) -> TaskSpec:
    return TaskSpec(
        id=task_id,
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
        context=ConversationInput(events=(TextMessage(role="user", content="Fix the program."),)),
        environment_requirements=EnvironmentRequirements(
            command_semantics=_CommandSemantics.LINUX_PROCESS,
            docker_image="fixture@sha256:" + "a" * 64,
        ),
        execution_requirements=ExecutionRequirements(
            cpus=2,
            memory_mb=4096,
            storage_mb=10240,
            agent_timeout_sec=600,
            verifier_timeout_sec=60,
        ),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=ScriptGrader(
            argv=("bash", "/tests/test.sh"),
            environment=EnvironmentRequirements(
                command_semantics=_CommandSemantics.LINUX_PROCESS,
                docker_image="fixture@sha256:" + "a" * 64,
            ),
        ),
    )


def _write_tasks(path, rows: list[tuple[str, str | dict]]):
    table = pa.Table.from_pylist(
        [{"task_spec": spec, "runtime": runtime} for spec, runtime in rows],
        schema=pa.schema([("task_spec", pa.string()), ("runtime", pa.string())]),
    )
    pq.write_table(table, path)


def test_published_rows_accepts_canonical_bytes_and_decodes_runtime(tmp_path):
    spec = _spec("t-1").model_dump_json()
    _write_tasks(tmp_path / "tasks.parquet", [(spec, json.dumps({"cpus": 4}))])
    rows = published_rows(str(tmp_path / "tasks.parquet"))
    assert rows == [(json.loads(spec), {"cpus": 4})]


def test_published_rows_rejects_a_noncanonical_serialization(tmp_path):
    canonical = _spec("t-1").model_dump_json()
    rewritten = json.dumps(json.loads(canonical), sort_keys=True, indent=2)
    _write_tasks(tmp_path / "tasks.parquet", [(rewritten, "{}")])
    with pytest.raises(ValueError, match="canonical serialization"):
        published_rows(str(tmp_path / "tasks.parquet"))
