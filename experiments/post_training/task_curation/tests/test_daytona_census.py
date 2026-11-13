# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Census behavior through the workspace conversion and execution declarations."""

import base64
from collections import Counter

import pytest
from taskcompendium.convert.answers import exact_answer_task
from taskcompendium.models import EnvironmentRequirements, ScriptGrader
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse, SourceRecipe
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.daytona_census import (
    converted_records,
    daytona_unsafe_counts,
    daytona_unsafe_reasons,
    measure_envelopes,
)
from experiments.post_training.task_curation.daytona_identity import (
    count_lowered_spec_identities,
    count_native_spec_identities,
)

pytest.importorskip("harbor")


def test_daytona_role_census_routes_lock_grader_locally_and_retains_its_quota_failure():

    task = {
        "id": "actor-with-lock-grader",
        "environment_requirements": {"docker_image": "actor@sha256:" + "a" * 64, "command_semantics": "linux_process"},
        "grader": {"environment": {"packages_lock": "private.lock", "command_semantics": "linux_process"}},
    }
    measured = measure_envelopes([(task, {"verifier_machine": {"storage_mb": 1024}})])
    assert measured.machine_slots == 1 and len(measured.snapshot_sizes) == 1
    assert measured.execution_failure_tasks == 0
    assert measured.roles["grader"]["local"]["machine_slots"] == 1
    assert sum(measured.roles["grader"]["local"]["failures"].values()) == 1
    task["environment_requirements"] = {}
    no_actor = measure_envelopes([(task, {})])
    assert no_actor.machine_slots == 0 and not no_actor.snapshot_sizes
    assert not no_actor.roles["grader"]["local"]["failures"]


def test_execution_cost_census_keeps_task_specs_rejected_by_the_grading_boundary():

    def convert(row, _context):
        task = exact_answer_task(row, prompt="Repair the repository", answers=("done",), ignore_case=False)
        environment = EnvironmentRequirements(
            command_semantics="linux_process", docker_image="actor@sha256:" + "a" * 64, capabilities=("git_repository",)
        )
        grader = ScriptGrader(
            argv=("python", "tests.py"),
            environment=EnvironmentRequirements(command_semantics="linux_process", packages_lock="grader.lock"),
        )
        return task.model_copy(update={"environment_requirements": environment, "grader": grader})

    recipe = SourceRecipe(
        name="workspace",
        version="1",
        source=SourceFiles("fixture", "pin", ("rows.jsonl",), SourceFormat.JSONL),
        convert=convert,
        intended_use=IntendedUse.TRAIN,
        rubric=None,
        controls=None,
    )
    rejected, admission = Counter(), Counter()
    census = measure_envelopes(converted_records([{"locator": "rows.jsonl:0", "data": {}}], recipe, rejected, admission))
    assert census.converted == census.machine_slots == 1 and len(census.snapshot_sizes) == 1
    assert census.execution_failure_tasks == 0
    assert admission == {"source_defect:grading_environment_mismatch": 1}
    assert not rejected


def test_actor_unsafe_reasons_distinguish_snapshot_budget_build_and_host_image():

    def actor(requirements):
        return {"id": "actor", "environment_requirements": requirements, "grader": {"environment": None}}

    budget = measure_envelopes(
        (actor({"command_semantics": "linux_process", "docker_image": f"image-{index}@sha256:" + "a" * 64}), {})
        for index in range(11)
    )
    assert set(daytona_unsafe_reasons(budget)) == {"too_many_snapshots"}
    assert daytona_unsafe_counts(budget) == {"too_many_snapshots": 11}
    build = measure_envelopes(
        [
            (
                actor(
                    {
                        "command_semantics": "linux_process",
                        "docker_build": {
                            "files": [
                                {
                                    "path": "Dockerfile",
                                    "source": {
                                        "kind": "inline_file",
                                        "content_base64": base64.b64encode(b"FROM ubuntu:24.04\nRUN true\n").decode(),
                                    },
                                }
                            ]
                        },
                    }
                ),
                {},
            )
        ]
    )
    assert set(daytona_unsafe_reasons(build)) == {"unbuilt_image"}
    assert daytona_unsafe_counts(build) == {"unbuilt_image": 1}
    locked = measure_envelopes([(actor({"command_semantics": "linux_process", "packages_lock": "actor.lock"}), {})])
    assert set(daytona_unsafe_reasons(locked)) == {"host_image_required"}
    assert daytona_unsafe_counts(locked) == {"host_image_required": 1}


def test_native_envelope_counts_original_toml_shape_without_unbuilt_lowering():

    dockerfile = "FROM ubuntu:24.04\nCOPY setup.sh /app/setup.sh\n"
    toml = (
        b"[environment]\ncpus=2\nmemory_mb=2048\nstorage_mb=4096\nallow_internet=false\n"
        b"[agent]\ntimeout_sec=900\n[verifier]\ntimeout_sec=600\n"
    )
    resources = [
        inline_resource("task.toml", toml),
        inline_resource("environment/Dockerfile", dockerfile.encode()),
        inline_resource("environment/setup.sh", b"#!/bin/sh\necho native\n").model_copy(update={"mode": "755"}),
    ]
    task = {
        "id": "native",
        "tags": ["harbor"],
        "resources": {"verifier": [r.model_dump(mode="json") for r in resources]},
        "environment_requirements": {
            "command_semantics": "linux_process",
            "docker_build": {
                "files": [
                    inline_resource("Dockerfile", dockerfile.encode()).model_dump(mode="json"),
                    inline_resource("setup.sh", b"#!/bin/sh\necho native\n")
                    .model_copy(update={"mode": "755"})
                    .model_dump(mode="json"),
                ]
            },
        },
        "grader": {"environment": None},
    }
    measured = measure_envelopes([(task, {"task_machine": {"cpus": 99}})])
    direct = count_native_spec_identities([("native", task)])
    assert measured.converted == measured.machine_slots == 1
    assert measured.snapshot_sizes == direct.sizes and len(direct.sizes) == 1
    assert not measured.execution_failures and not measured.mapping_failures
    # The generic unbuilt TaskCompendium route is distinct from native Harbor's build path.
    lowered = count_lowered_spec_identities([("native", {"task": task, "runtime": {"task_machine": {}}})])
    assert any("Unresolved Docker build context" in reason for _, reason in lowered.execution_failures)

    task["execution_requirements"] = {
        "cpus": 2,
        "memory_mb": 2048,
        "storage_mb": 4096,
        "agent_timeout_sec": 900,
        "verifier_timeout_sec": 600,
    }
    assert count_native_spec_identities([("native", task)]).sizes == direct.sizes
    task["execution_requirements"]["storage_mb"] = 8192
    with pytest.raises(ValueError, match="storage_mb conflicts"):
        count_native_spec_identities([("native", task)])
