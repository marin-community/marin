# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib

import pytest
from rolloutengine.harness.shell import DEFAULT_SHELL_TOOL, task_prompt
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.runtime.shell import resolve_shell_tool

from experiments.post_training.task_curation.datasets.facet.tasks import VERSION, convert_facet


@pytest.mark.parametrize("cpus,memory,deadline", [(1, 2048, 120), (2, 4096, 180)])
def test_pristine_image_replaces_the_build_without_changing_native_task_contract(cpus, memory, deadline):
    instruction = b"Write the requested result under /task_file/output.\n"
    verifier = b"#!/bin/bash\nsource /tests/setup.sh\npython /tests/test_state.py\n"
    solution = b"#!/bin/bash\nprintf correct > /task_file/output/result\n"
    config = (
        f"[metadata]\nworkdir='/task_file'\n[environment]\ncpus={cpus}\nmemory_mb={memory}\n"
        f"storage_mb=10240\nallow_internet=true\nbuild_timeout_sec=600\n[agent]\ntimeout_sec={deadline}\n"
        "[verifier]\ntimeout_sec=120\n"
    ).encode()
    files = {
        "instruction.md": instruction,
        "task.toml": config,
        "environment/Dockerfile": b"FROM mutable:latest\nRUN never-rebuild-this\n",
        "setup_files/context/input.txt": b"public fixture\n",
        "setup_files/setup.sh": b"#!/bin/bash\nexit 0\n",
        "tests/setup.sh": b"#!/bin/bash\nexport ORIGINAL=1\n",
        "tests/test.sh": verifier,
        "tests/test_state.py": b"assert open('/task_file/output/result').read() == 'correct'\n",
        "solution/solve.sh": solution,
    }
    image = "ghcr.io/example/facet@sha256:" + "a" * 64
    task = convert_facet(
        RawRow(
            id="facet-original",
            source=Source(
                dataset="FWeindel/validated-tasks", revision="b" * 40, row="task_1", importer_revision=VERSION
            ),
            data={
                "archive_links": {},
                "archive_sha256": hashlib.sha256(b"original native archive").hexdigest(),
                "files": {name: base64.b64encode(value).decode() for name, value in files.items()},
                "file_metadata": {
                    name: {"mode": "0755" if name.endswith(".sh") else "0644", "mtime_ns": 123456789000}
                    for name in files
                },
                "facet_image": {
                    "image": image,
                    "preserved_image": {
                        "sif_sha256": "c" * 64,
                        "provenance_uri": "s3://fixture/pristine/preservation.json",
                        "provenance_sha256": "d" * 64,
                    },
                },
            },
        ),
        ConversionContext(inputs={}, grader_environment=None),
    )
    assert isinstance(task, TaskSpec)
    assert task.context.events[0].content == instruction.decode()
    assert task.environment_requirements.docker_image == image
    assert task.environment_requirements.docker_build is None
    assert task.environment_requirements.working_directory == "/task_file"
    assert task.execution_requirements.model_dump() == {
        "cpus": cpus,
        "memory_mb": memory,
        "storage_mb": 10240,
        "agent_timeout_sec": deadline,
        "verifier_timeout_sec": 120,
    }
    assert task.verifier_execution_requirements == task.execution_requirements
    assert task.grader.environment.docker_image == image
    assert task.grader.timeout == 120
    assert {r.path: resource_bytes(r) for r in task.resources.worker} == {
        name: data for name, data in files.items() if name.startswith("setup_files/")
    }
    assert {r.path: resource_bytes(r) for r in task.resources.oracle} == {"solution/solve.sh": solution}
    private = {r.path: resource_bytes(r) for r in task.resources.verifier}
    assert private["source_archive/tests/test.sh"] == verifier
    assert private["source_archive/task.toml"] == config
    assert {r.name: r.sha256 for r in task.integrity_contract.verifier_files} == {
        name.removeprefix("tests/"): hashlib.sha256(value).hexdigest()
        for name, value in files.items()
        if name.startswith("tests/")
    }
    prompt = task_prompt(task, resolve_shell_tool(task, DEFAULT_SHELL_TOOL), ())
    assert prompt.options["tools"][0]["function"]["parameters"]["required"] == ["command"]
    # Revalidation checks that role limits and image/build exclusivity also survive serialization.
    assert TaskSpec.model_validate_json(task.model_dump_json()) == task
