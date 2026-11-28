# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The census engine measures declared machine envelopes through production identities."""

import base64

import pytest

pytest.importorskip("harbor", reason="Requires the isolated vendor/harbor runtime")

from shellbox.backends.daytona.snapshot import snapshot_request_for_spec
from shellbox.machine import MachineSpec, RegistryImage
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.census import TaskEnvelope

from experiments.post_training.task_curation.census_engine import DaytonaCensusEngine, native_shapes, native_task_toml

PINNED_ACTOR = "registry.example/actor@sha256:" + "a" * 64


def inline(path: str, content: bytes, mode: str | None = None) -> dict:
    resource = {
        "path": path,
        "source": {"kind": "inline_file", "content_base64": base64.b64encode(content).decode()},
    }
    if mode is not None:
        resource["mode"] = mode
    return resource


def task_document(task_id: str, *, environment: dict, grader: dict | None = None, tags: tuple = ()) -> dict:
    return {
        "id": task_id,
        "tags": tags,
        "environment_requirements": environment,
        "answer_type": "text",
        "answer_format": {"kind": "plain_text"},
        "grader": grader or {"kind": "none", "reason": "fixture"},
        "source": {"dataset": "fixture", "revision": "1", "row": "0", "importer_revision": "1"},
        "context": {"events": ({"type": "message", "role": "user", "content": "do the task"},)},
    }


def envelope(task: dict) -> TaskEnvelope:
    return TaskEnvelope(task["id"], task, TaskSpec.model_validate(task))


def lowered_task(task_id: str, docker_image: str, grader: dict | None = None) -> TaskEnvelope:
    return envelope(
        task_document(
            task_id,
            environment={"docker_image": docker_image, "command_semantics": "linux_process"},
            grader=grader,
        )
    )


def harbor_task_toml(cpus: int = 2) -> bytes:
    return (
        'version = "1.0"\n\n[environment]\n'
        f'docker_image = "{PINNED_ACTOR}"\n'
        f"cpus = {cpus}\nallow_internet = false\n\n[agent]\ntimeout_sec = 900.0\n"
    ).encode()


def native_task(dockerfile: str, cpus: int = 2) -> TaskEnvelope:
    task = task_document(
        "native-fixture",
        environment={"capabilities": (), "command_semantics": None},
        tags=("harbor",),
    )
    task["resources"] = {
        "verifier": [
            inline("task.toml", harbor_task_toml(cpus)),
            inline("environment/Dockerfile", dockerfile.encode()),
        ]
    }
    return envelope(task)


def test_native_tasks_measure_through_the_delivered_task_toml():
    measurement = DaytonaCensusEngine().measure([native_task("FROM ubuntu:24.04\nCOPY files/data.csv /app/data.csv\n")])
    assert measurement.converted == 1
    assert measurement.machine_slots == 1
    assert measurement.roles["agent"]["daytona"].machine_slots == 1
    assert measurement.snapshot_sizes == {
        snapshot_request_for_spec(MachineSpec(source=RegistryImage(PINNED_ACTOR), cpus=2)).name: 1
    }
    assert not measurement.execution_failures
    # The actor's environment source ledger reads the TaskSpec's declared
    # requirements; a native task's machine demand lives in its task.toml.
    assert measurement.actor_sources == {}
    shapes = native_shapes(harbor_task_toml())
    assert shapes["agent_machine"]["cpus"] == 2
    assert shapes["agent_machine"]["network"] == "deny"


def test_distinct_native_archives_split_identities():
    engine = DaytonaCensusEngine()
    measurement = engine.measure(
        [
            native_task("FROM ubuntu:24.04\nCOPY files/a.csv /app/a.csv\n", cpus=2),
            native_task("FROM ubuntu:24.04\nCOPY files/b.csv /app/b.csv\n", cpus=4),
        ]
    )
    assert measurement.machine_slots == 2
    assert len(measurement.snapshot_sizes) == 2
    assert sum(measurement.snapshot_sizes.values()) == 2


def test_lowered_declared_environments_count_agent_slots_and_snapshots():
    measurement = DaytonaCensusEngine().measure([lowered_task("a", PINNED_ACTOR), lowered_task("b", PINNED_ACTOR)])
    assert measurement.converted == 2
    assert measurement.machine_slots == 2
    assert measurement.roles["agent"]["daytona"].machine_slots == 2
    assert measurement.snapshot_sizes == {
        snapshot_request_for_spec(MachineSpec(source=RegistryImage(PINNED_ACTOR))).name: 2
    }
    assert measurement.actor_sources == {"docker_image": 2}


def test_package_lock_graders_demand_a_local_host_without_snapshots():
    grader = {
        "kind": "script",
        "argv": ("bash", "/tests/grade.sh"),
        "placement": {
            "kind": "grader_machine",
            "environment": {"packages_lock": "https://example.test/lock", "command_semantics": "linux_process"},
        },
    }
    measurement = DaytonaCensusEngine().measure([lowered_task("locked", PINNED_ACTOR, grader)])
    assert measurement.machine_slots == 1
    assert measurement.roles["grader"]["local"].machine_slots == 1
    assert "daytona" not in measurement.roles["grader"]
    assert set(measurement.snapshot_sizes) == {
        snapshot_request_for_spec(MachineSpec(source=RegistryImage(PINNED_ACTOR))).name
    }


def test_machine_free_tasks_do_not_double_count_declared_backends():
    """A backend absent from a later task merges an empty outcome, not the accumulated total."""
    declared = lowered_task("declared-fixture", PINNED_ACTOR)
    machine_free = envelope(task_document("empty-fixture", environment={}))
    measurement = DaytonaCensusEngine().measure([declared, machine_free])
    assert measurement.converted == 2
    assert measurement.machine_slots == 1
    assert measurement.roles["agent"]["daytona"].machine_slots == 1


def test_tasks_without_declared_environments_provision_nothing():
    empty = envelope(task_document("empty-fixture", environment={}))
    declared = lowered_task("declared-fixture", PINNED_ACTOR)
    measurement = DaytonaCensusEngine().measure([empty, declared])
    assert measurement.converted == 2
    assert measurement.machine_slots == 1
    assert measurement.roles["grader"] == {}


def test_native_task_toml_requires_the_delivered_configuration():
    with pytest.raises(ValueError, match=r"no delivered task\.toml"):
        native_task_toml({"resources": {"verifier": []}})
