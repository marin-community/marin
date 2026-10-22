# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Protect source session and private grading contracts during panel restoration."""

import json

import pytest
from rolloutengine.harness.tools import task_prompt
from rolloutengine.models import MachineRuntimeSpec, RolloutLimits
from taskcompendium.models import ConversationInput, EnvironmentRequirements, ScriptGrader, Source, TaskSpec, TextMessage
from taskcompendium.runtime.resources import inline_resource


@pytest.mark.manual
def test_restored_source_preserves_grader_machine_limits_and_model_request():
    from skyrl_gym.rollout_specs import shell_harness  # noqa: PLC0415 - optional frozen runtime dependency

    from experiments.post_training.snowball_inventory_validation import (  # noqa: PLC0415
        GRADER_IMAGE,
        panel_record,
        restored_script_rollout,
    )

    config = {"contract": {"dataset": "ultra_sft_step3200_calendar_v2", "exp_cal_state": {"private": "reference"}}}
    task = TaskSpec(
        id="heldout-calendar",
        context=ConversationInput(events=(TextMessage(role="user", content="Public question"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type="text",
        answer_format={"kind": "plain_text"},
        source=Source(dataset="nvidia/source", revision="immutable", row="validation:0", importer_revision="3"),
        grader=ScriptGrader(
            environment=EnvironmentRequirements(docker_image=GRADER_IMAGE, command_semantics="linux_process"),
            argv=("python3", "/tests/grade.py"),
        ),
        resources={"verifier": (inline_resource("config.json", json.dumps(config).encode()),)},
    )
    machine = MachineRuntimeSpec(
        backend="daytona",
        network="allow",
        cpus=2,
        memory_mb=4096,
        storage_mb=10240,
        gpus=0,
        user=None,
        startup_timeout=600,
        cleanup_timeout=60,
    )
    limits = RolloutLimits(
        max_turns=1,
        max_total_tokens=None,
        model_turn_timeout=300,
        tool_turn_timeout=90,
        total_turn_timeout=360,
        attempt_timeout=1200,
        verifier_timeout=300,
        cleanup_timeout=60,
    )
    old_session = {key: value for key, value in limits.model_dump(mode="json").items() if key != "max_total_tokens"}
    old_session.update(
        task_session="shellbox", shell_tool={"name": "shell", "command_parameter": "command"}, command_timeout=60
    )
    raw = json.dumps(
        {
            "task": {**task.model_dump(mode="json"), "schema_version": "historical"},
            "runtime": {"task_machine": None, "verifier_machine": machine.model_dump(mode="json")},
            "session": old_session,
        }
    )

    restored, family = restored_script_rollout(raw)

    prepared = panel_record({"lowered_task_spec": raw}, 0)
    record, audit = prepared.record, prepared.audit
    from rolloutengine.models import TaskRolloutSpec  # noqa: PLC0415 - optional runtime boundary

    exported = TaskRolloutSpec.model_validate_json(record["task_rollout_spec"])
    assert exported.task == task
    assert exported.verifier_machine == machine
    assert exported.limits == limits
    assert record["evaluation_family"] == family == "calendar"
    assert record["source_row_index"] == 0
    assert record["lowered_task_spec"] == raw
    assert audit["historical_equivalence_claimed"] is False
    assert shell_harness(exported.harness).command_timeout == 60
    assert family == "calendar"
    assert restored.task == task
    assert restored.task.grader.environment.docker_image == GRADER_IMAGE
    assert restored.verifier_machine == machine
    assert restored.task_machine is None
    assert restored.limits == limits
    harness = shell_harness(restored.harness)
    assert harness.command_timeout == 60
    request = task_prompt(restored.task, None, ())
    assert request.messages == (
        {"role": "user", "content": "Public question"},
        {"role": "user", "content": "Give your answer as plain text."},
    )
    assert request.options == {}
    assert "private" not in json.dumps(request.messages)


@pytest.mark.manual
def test_panel_native_prefill_retains_source_pairing_and_private_resources():
    from rolloutengine.models import TaskRolloutSpec  # noqa: PLC0415

    from experiments.post_training.snowball_curriculum.native import native_closure_task  # noqa: PLC0415
    from experiments.post_training.snowball_inventory_validation import panel_record  # noqa: PLC0415

    extras = {
        "prefix": '{"x":',
        "expected": '{"x":1}',
        "format": "json",
        "family": "object",
        "control": "valid",
        "prompt_token_ids": [123, 456],
    }
    task = TaskSpec(
        id="native-heldout",
        context=ConversationInput(events=(TextMessage(role="user", content="Complete JSON"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type="text",
        answer_format={"kind": "plain_text"},
        source=Source(dataset="structure-closure", revision="frozen", row="0", importer_revision="1"),
        grader={"kind": "none", "reason": "historical source fixture"},
        resources={"verifier": (inline_resource("config.json", json.dumps({"extras": extras}).encode()),)},
    )
    raw = json.dumps(
        {
            "task": {**task.model_dump(mode="json"), "grader": {"kind": "session"}},
            "runtime": {"task_machine": None, "verifier_machine": None},
            "session": {
                "task_session": "structure_closure",
                "max_turns": 1,
                "model_turn_timeout": 180,
                "tool_turn_timeout": 90,
                "total_turn_timeout": 240,
                "attempt_timeout": 300,
                "verifier_timeout": 900,
                "cleanup_timeout": 30,
            },
        }
    )
    source = native_closure_task(raw, "")
    prepared = panel_record({"lowered_task_spec": raw}, 17)
    record, audit = prepared.record, prepared.audit
    spec = TaskRolloutSpec.model_validate_json(record["task_rollout_spec"])

    assert record["task_id"] == source.task_id
    assert record["source_contract_id"] == source.contract_id
    assert record["source_public_content_id"] == source.public_content_id
    assert record["contract_id"] != source.contract_id
    assert spec.harness == "assistant_prefix"
    assert spec.task.context.events[-1] == TextMessage(role="assistant", content=extras["prefix"])
    assert spec.task.resources == task.resources
    assert "terminal_markers" not in spec.task.grader.parameters
    assert audit["task_semantic_changes"] == ["context", "grader"]
    assert audit["historical_equivalence_claimed"] is False
    assert record["lowered_task_spec"] == raw

    from taskcompendium.grading import grade_answer  # noqa: PLC0415 - optional verifier environment
    from taskcompendium.grading_result import Outcome  # noqa: PLC0415
    from taskcompendium.models import ConversationTrace, GradingAttempt  # noqa: PLC0415

    for answer, reward in (("1}", 1.0), ("invalid", 0.0)):
        grade = grade_answer(
            spec.task,
            GradingAttempt(
                ConversationTrace(events=(*spec.task.context.events, TextMessage(role="assistant", content=answer)))
            ),
        )
        assert grade.status is Outcome.GRADED
        assert grade.reward == reward
