# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run executable source graders in a fresh, privilege-separated Docker sandbox."""

import json
import subprocess
import uuid
from pathlib import Path

import msgspec
import tasktrove_verify

from taskcompendium.grading_paths import EXTERNAL_DIRECTORY
from taskcompendium.lowering import validate_workspace_submission
from taskcompendium.models import (
    ContainerRuntime,
    Embedded,
    GradingResult,
    ImageOverlay,
    Outcome,
    Rendering,
    ResourceRef,
    ResourceRole,
    TaskSpecification,
    verifier_runtime,
)
from taskcompendium.resources import resource_bytes
from taskcompendium.serialization import to_json


def grade_in_container(
    specification: TaskSpecification,
    protocol: Rendering,
    response: str | None,
    workspace: Path,
    transcript: tuple[dict, ...] = (),
    step_index: int = 0,
) -> GradingResult:
    """Grade a workspace snapshot without executing candidate code on the host."""
    validate_workspace_submission(specification, protocol, step_index)
    runtime = verifier_runtime(specification.steps[step_index].verifier)
    if not isinstance(runtime, ContainerRuntime):
        raise ValueError("Container grading requires ContainerRuntime")
    specification = msgspec.structs.replace(
        specification,
        steps=tuple(
            (
                msgspec.structs.replace(
                    step,
                    resources=tuple(
                        (
                            msgspec.structs.replace(resource, content=Embedded(resource_bytes(resource)))
                            if ResourceRole.VERIFIER in resource.roles and isinstance(resource.content, ResourceRef)
                            else resource
                        )
                        for resource in step.resources
                    ),
                )
                if index == step_index
                else step
            )
            for index, step in enumerate(specification.steps)
        ),
        resources=tuple(
            (
                msgspec.structs.replace(resource, content=Embedded(resource_bytes(resource)))
                if ResourceRole.VERIFIER in resource.roles and isinstance(resource.content, ResourceRef)
                else resource
            )
            for resource in specification.resources
        ),
    )
    workdir = specification.requirements.state.workdir
    package = Path(__file__).resolve().parents[1]
    source_verifier = Path(tasktrove_verify.__file__).resolve().parent
    name = f"taskcompendium-verifier-{uuid.uuid4().hex}"
    payload = json.dumps(
        {
            "specification": json.loads(to_json(specification)),
            "protocol": msgspec.to_builtins(protocol),
            "step_index": step_index,
            "attempt": {"response": response, "transcript": transcript},
        }
    )
    mounts = [
        (workspace.resolve(), "/snapshot"),
        (package, "/opt/runtime/taskcompendium"),
        (source_verifier, "/opt/runtime/tasktrove_verify"),
    ]
    if specification.requirements.state.additional_directories:
        for directory in specification.requirements.state.additional_directories:
            source = workspace / EXTERNAL_DIRECTORY / directory.lstrip("/")
            source.mkdir(parents=True, exist_ok=True)
            mounts.append((source.resolve(), directory))
    command = [
        "docker",
        "run",
        "--interactive",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--pids-limit",
        "128",
        "--memory",
        "1g",
        "--cpus",
        "2",
        "--cap-drop",
        "ALL",
        "--cap-add",
        "SETUID",
        "--cap-add",
        "SETGID",
        "--cap-add",
        "CHOWN",
        "--cap-add",
        "DAC_OVERRIDE",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        "/tmp:rw,exec,nosuid,size=268435456",
        "--user",
        "0:0",
        "--tmpfs",
        "/input:rw,noexec,nosuid,mode=0700,size=67108864",
        "--tmpfs",
        "/result:rw,noexec,nosuid,mode=0700,size=1048576",
        "--tmpfs",
        "/tests:rw,noexec,nosuid,mode=0700,size=67108864",
    ]
    if isinstance(runtime.workspace, ImageOverlay):
        # Docker populates a fresh anonymous volume from the pinned image. Keep
        # the image root read-only while retaining source-internal dependencies.
        command.extend(["--volume", workdir])
    else:
        command.extend(["--tmpfs", f"{workdir}:rw,exec,nosuid,size=536870912"])
    for source, target in mounts:
        command.extend(["--volume", f"{source}:{target}:ro"])
    command.extend(
        [runtime.image, runtime.supervisor_python, "-I", "/opt/runtime/taskcompendium/harbor/container_entry.py"]
    )
    try:
        completed = subprocess.run(command, input=payload, capture_output=True, text=True, timeout=runtime.timeout)
        if completed.returncode != 0:
            return GradingResult(
                Outcome.INFRA_ERROR,
                None,
                {
                    "error": "Container verifier did not produce a result",
                    "exit_code": completed.returncode,
                    "stderr": completed.stderr[-4000:],
                },
            )
        return msgspec.json.decode(completed.stdout, type=GradingResult)
    except subprocess.TimeoutExpired:
        return GradingResult(Outcome.INFRA_ERROR, None, {"error": "Container verifier timed out"})
    except msgspec.DecodeError as error:
        return GradingResult(Outcome.INFRA_ERROR, None, {"error": f"Invalid supervisor result: {error}"})
    finally:
        cleanup = subprocess.run(["docker", "rm", "--force", "--volumes", name], capture_output=True, text=True)
        if cleanup.returncode != 0 and "No such container" not in cleanup.stderr:
            raise RuntimeError(f"Could not remove verifier container {name}: {cleanup.stderr}")
