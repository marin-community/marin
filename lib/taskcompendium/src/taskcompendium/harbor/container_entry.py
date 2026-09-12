# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trusted root supervisor; this entry point must run only in the verifier image."""

import json
import os
import shlex
import shutil
import sys
from pathlib import Path

# -I prevents candidate cwd/PYTHONPATH from selecting supervisor imports. Only
# this read-only mount supplies the trusted TaskCompendium and source grader.
sys.path.insert(0, "/opt/runtime")

import msgspec
from tasktrove_verify.spec import Mode

from taskcompendium.grading import _grade_attempt
from taskcompendium.models import (
    ContainerRuntime,
    Embedded,
    ImageOverlay,
    NoEnvironment,
    Resource,
    ResourceRole,
    VerifierSpec,
)
from taskcompendium.resources import materialize
from taskcompendium.serialization import from_json, protocol_from_json

LAUNCHER = "/opt/runtime/taskcompendium/harbor/candidate_entry.py"


def _command(command: str) -> str:
    return shlex.join([sys.executable, "-I", LAUNCHER, *shlex.split(command)])


def main() -> None:
    if os.getuid() != 0 or not Path("/input").is_dir():
        raise RuntimeError("Container verifier entry point requires the isolated supervisor layout")
    payload = json.load(sys.stdin)
    Path("/input/specification.json").write_text(json.dumps(payload["specification"]))
    specification = from_json(json.dumps(payload["specification"]))
    protocol = protocol_from_json(json.dumps(payload["protocol"]))
    attempt = payload["attempt"]
    workspace = Path(
        "/app" if isinstance(specification.environment, NoEnvironment) else specification.environment.workdir
    )
    os.environ["TASKCOMPENDIUM_WORKSPACE"] = str(workspace)
    runtime = specification.verifier_runtime
    assert isinstance(runtime, ContainerRuntime)
    preserved = runtime.workspace.preserved_directories if isinstance(runtime.workspace, ImageOverlay) else ()
    if workspace.is_symlink():
        raise ValueError("Image workspace cannot be a symlink")
    for directory in preserved:
        path = workspace / directory
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"Preserved dependency must be a real image directory: {directory}")
        if (Path("/snapshot") / directory).exists() or (Path("/snapshot") / directory).is_symlink():
            raise ValueError("Submitted snapshots cannot contain preserved image dependencies")
    if preserved:
        for path in workspace.iterdir():
            if path.name not in preserved:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink()
    for path in Path("/snapshot").rglob("*"):
        if path.is_symlink():
            raise ValueError("Workspace snapshots cannot contain symlinks")
    shutil.copytree("/snapshot", workspace, dirs_exist_ok=True, symlinks=False)
    for path in [workspace, *Path("/snapshot").rglob("*")]:
        path = workspace / path.relative_to("/snapshot") if path != workspace else workspace
        if path.is_symlink():
            raise ValueError("Workspace snapshots cannot contain symlinks")
        os.chown(path, 65534, 65534)
    parameters = dict(specification.verifier.parameters)
    mode = specification.verifier.mode
    resources = specification.resources
    if mode == Mode.STDIO:
        if parameters.get("special_judge"):
            raise ValueError("Special judges require a dedicated isolated adapter")
        parameters["command"] = _command(parameters["command"])
        if parameters.get("build"):
            parameters["build"] = _command(shlex.join(["bash", "-lc", parameters["build"]]))
    elif mode == Mode.PYTEST:
        if any(path.startswith("/tests/") for path in parameters["paths"]):
            # Some retained source graders address /tests directly. This private
            # grading container exposes the original paths only after submission.
            tests = Path("/tests")
            materialize(specification, ResourceRole.VERIFIER, tests)
            for path in [tests, *tests.rglob("*")]:
                path.chmod(0o555 if path.is_dir() else 0o444)
        python = parameters.get("python", "python3")
        launcher = Path("/tmp/taskcompendium-python")
        launcher.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -I {LAUNCHER} {shlex.quote(python)} "$@"\n')
        launcher.chmod(0o755)
        parameters["python"] = str(launcher)
        if parameters.get("setup"):
            parameters["setup"] = _command(shlex.join(["bash", "-lc", parameters["setup"]]))
    elif mode == Mode.SCRIPT:
        original = parameters["path"]
        interpreter = "bash" if original.endswith(".sh") else "python3"
        wrapper = "__taskcompendium_script_wrapper.py"
        source = (
            "import os, pathlib\n"
            f"p = pathlib.Path(__file__).parent / {original!r}\n"
            f"os.execvp({sys.executable!r}, [{sys.executable!r}, '-I', {LAUNCHER!r}, {interpreter!r}, str(p), "
            f"*{parameters.get('args', [])!r}])\n"
        )
        resources = (*resources, Resource(wrapper, (ResourceRole.VERIFIER,), Embedded(source.encode())))
        parameters["path"] = wrapper
        parameters["args"] = []
    else:
        raise ValueError(f"Isolated executable mode is unsupported: {mode}")
    specification = msgspec.structs.replace(
        specification,
        resources=resources,
        verifier=VerifierSpec(mode, parameters, specification.verifier.judge),
    )
    result = _grade_attempt(specification, protocol, attempt["response"], workspace, tuple(attempt["transcript"]), None)
    result.detail["isolation"] = "docker-unprivileged-candidate"
    result.detail["source_harness_trust"] = "Inherited source pytest/script reports are not adversarially tamper-proof"
    encoded = msgspec.json.encode(result)
    Path("/result/result.json").write_bytes(encoded)
    sys.stdout.write(encoded.decode())


if __name__ == "__main__":
    main()
