# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Protect source session and private grading contracts during panel restoration."""

import json

from rolloutengine.harness.shell import task_prompt
from rolloutengine.models import MachineRuntimeSpec, RolloutLimits
from skyrl_gym.rollout_specs import shell_harness
from taskcompendium.models import ConversationInput, EnvironmentRequirements, ScriptGrader, Source, TaskSpec, TextMessage
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.snowball_inventory_validation import GRADER_IMAGE, restored_script_rollout


def test_restored_source_preserves_grader_machine_limits_and_model_request():
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
