# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Consumer scope follows task contracts, independent of global schema alignment."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import ConsumerNeed
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    ExecutionRequirements,
    PlainText,
    ProviderRequirement,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from verifyit.spec import JudgeSpec

from experiments.post_training.task_curation.consumer_audit import RUNTIME_PATHS, audit_source
from experiments.post_training.task_curation.consumer_scope import ConsumerCapacity, consumer_needs

CAPACITY = ConsumerCapacity(cpus=4, memory_mb=8192, storage_mb=10240, agent_timeout=2400, verifier_timeout=300)


@pytest.fixture
def standard_task():
    return TaskSpec(
        id="native",
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
        context=ConversationInput(events=(TextMessage(role="user", content="Fix the program."),)),
        environment_requirements=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS,
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
                command_semantics=CommandSemantics.LINUX_PROCESS,
                docker_image="fixture@sha256:" + "a" * 64,
            ),
        ),
    )


def test_standard_harbor_task_with_satisfiable_nondefault_allocation_is_exempt(standard_task):
    assert consumer_needs(standard_task, CAPACITY) == frozenset()


def test_tools_and_service_binding_select_bespoke_scope(standard_task):
    task = standard_task.model_copy(
        update={
            "environment_requirements": standard_task.environment_requirements.model_copy(
                update={
                    "tool_providers": {
                        "retrieval": ProviderRequirement(
                            action_interface="openresearcher-bm25:v1",
                            network_access="service_only",
                            initial_state={},
                        )
                    }
                }
            )
        }
    )
    assert consumer_needs(task, CAPACITY) == {ConsumerNeed.TOOL_PROVIDER, ConsumerNeed.SERVICE}


def test_custom_judge_selects_bespoke_scope(standard_task):
    task = standard_task.model_copy(
        update={
            "grader": standard_task.grader.model_copy(
                update={"judge": JudgeSpec(references=("answer",), question="Question")}
            )
        }
    )
    assert consumer_needs(task, CAPACITY) == {ConsumerNeed.CUSTOM_JUDGE}


def test_distinct_supported_verifier_allocation_selects_scope(standard_task):
    task = standard_task.model_copy(
        update={
            "verifier_execution_requirements": standard_task.execution_requirements.model_copy(
                update={"memory_mb": 8192}
            )
        }
    )
    assert consumer_needs(task, CAPACITY) == {ConsumerNeed.VERIFIER_ALLOCATION}


@pytest.mark.parametrize(
    "update",
    [
        {"cpus": 5},
        {"memory_mb": 8193},
        {"storage_mb": 10241},
        {"agent_timeout_sec": 2401},
        {"verifier_timeout_sec": 301},
    ],
)
def test_allocation_selects_scope_only_when_consumer_cannot_satisfy_it(standard_task, update):
    task = standard_task.model_copy(
        update={"execution_requirements": standard_task.execution_requirements.model_copy(update=update)}
    )
    assert consumer_needs(task, CAPACITY) == {ConsumerNeed.EXECUTION}


@pytest.mark.parametrize("changed", [False, True])
def test_frozen_fingerprint_survives_storage_sort_but_rejects_changed_content(tmp_path, standard_task, changed):
    originals = [
        standard_task.model_copy(
            update={"id": identifier, "source": standard_task.source.model_copy(update={"row": f"tasks.parquet:{row}"})}
        ).model_dump_json()
        for identifier, row in (("z", 9), ("a", 10))
    ]
    expected = hashlib.sha256("".join(originals).encode()).hexdigest()
    stored = list(reversed(originals))
    if changed:
        stored[0] = stored[0].replace("Fix the program.", "Fix the changed program.")
    specs = tmp_path / "specs"
    specs.mkdir()
    pq.write_table(pa.table({"task_spec": stored}), specs / "part-000000.parquet")
    population = tmp_path / "population.json"
    population.write_text(json.dumps({"result": {"specs_output_dir": str(specs)}}))
    census = tmp_path / "census.json"
    census.write_text(json.dumps({"result": {"input_artifact": str(population)}}))
    row = {
        "source_id": "fixture:standard",
        "input": {
            "evidence_url": str(census),
            "measured_tasks": 2,
            "task_specs_sha256": expected,
            "dataset_id": "fixture",
            "dataset_revision": "1",
        },
    }
    config = {
        "regional_bucket": str(tmp_path),
        "capacity": CAPACITY.model_dump(),
        "runtime_commit": "a" * 40,
        "session": {},
        "judges": [],
        "default_judge": "",
    }
    runtime = Path(__file__).resolve().parents[4]
    if changed:
        with pytest.raises(ValueError, match="Frozen population fingerprint differs"):
            audit_source(row, runtime, config, str(tmp_path / "receipts"))
        return
    result = audit_source(row, runtime, config, str(tmp_path / "receipts"))
    assert result["task_specs_sha256"] == result["source_order_task_specs_sha256"] == expected
    assert result["scanned_task_specs_sha256"] != expected
    assert result["checked_tasks"] == 2 and result["failed_tasks"] == 0


@pytest.mark.parametrize("bound", [True, False])
def test_source_judge_uses_launch_binding_without_a_static_model_roster(standard_task, bound):
    task = standard_task.model_copy(
        update={
            "grader": standard_task.grader.model_copy(
                update={"judge": JudgeSpec(model="source-declared-model", references=("answer",), question="Question")}
            )
        }
    )
    config = {
        "session": {
            "task_session": "shellbox",
            "max_turns": None,
            "max_total_tokens": None,
            "model_turn_timeout": None,
            "tool_turn_timeout": None,
            "total_turn_timeout": None,
            "attempt_timeout": None,
            "verifier_timeout": None,
            "cleanup_timeout": 60,
        },
        "default_judge": "different-default-model",
        "judge_binding": {"base_url": "https://judge.invalid/v1", "api_key_env": "JUDGE_API_KEY"} if bound else None,
    }
    record = {"task": task.model_dump_json(), "needs": ["custom_judge"], "judge_model": "source-declared-model"}
    root = Path(__file__).resolve().parents[4]
    result = subprocess.run(
        [sys.executable, str(root / "experiments/post_training/task_curation/consumer_probe.py")],
        input="\n".join(json.dumps(value) for value in (config, record)) + "\n",
        text=True,
        capture_output=True,
        check=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(str(root / path) for path in RUNTIME_PATHS)},
    )
    receipt = json.loads(result.stdout)
    assert receipt["failed_tasks"] == (0 if bound else 1)
    assert receipt["failures"] == ({} if bound else {"CONSUMER_JUDGE_UNREGISTERED": 1})
    assert receipt["global_schema_rejected_tasks"] == 0
