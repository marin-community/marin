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

from taskcompendium.extraction import ExtractionError, extract
from taskcompendium.grading import _grade_attempt
from taskcompendium.models import (
    AssistantFinal,
    CodeAnswerVerifier,
    ContainerRuntime,
    Embedded,
    FinalState,
    GradingResult,
    ImageOverlay,
    Outcome,
    PlainText,
    Rendering,
    Resource,
    ResourceRole,
    TaskTroveVerifier,
    verifier_runtime,
)
from taskcompendium.resources import materialize
from taskcompendium.serialization import from_json, rendering_from_json

LAUNCHER = "/opt/runtime/taskcompendium/harbor/candidate_entry.py"


def _command(command: str) -> str:
    return shlex.join([sys.executable, "-I", LAUNCHER, *shlex.split(command)])


def _write_result(result: GradingResult) -> None:
    encoded = msgspec.json.encode(result)
    Path("/result/result.json").write_bytes(encoded)
    sys.stdout.write(encoded.decode())


def main() -> None:
    if os.getuid() != 0 or not Path("/input").is_dir():
        raise RuntimeError("Container verifier entry point requires the isolated supervisor layout")
    payload = json.load(sys.stdin)
    Path("/input/specification.json").write_text(json.dumps(payload["specification"]))
    specification = from_json(json.dumps(payload["specification"]))
    protocol = rendering_from_json(json.dumps(payload["protocol"]))
    attempt = payload["attempt"]
    step_index = payload["step_index"]
    workspace = Path(specification.requirements.state.workdir)
    os.environ["TASKCOMPENDIUM_WORKSPACE"] = str(workspace)
    source_contract = specification.steps[step_index].verifier
    runtime = verifier_runtime(source_contract)
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
    if isinstance(source_contract, CodeAnswerVerifier):
        if not isinstance(protocol.submission, AssistantFinal):
            raise ValueError("Code-answer verifier requires an assistant-final submission")
        raw_response = attempt["response"] or ""
        if isinstance(protocol.submission.extractor, PlainText) and not raw_response.strip():
            candidate = ""
        else:
            try:
                candidate = extract(raw_response, protocol.submission.extractor)
            except ExtractionError as error:
                _write_result(GradingResult(Outcome.EXTRACTION_ERROR, None, {"error": str(error)}))
                return
        output_path = source_contract.output_path
        target = workspace / output_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(candidate)
        os.chown(target, 65534, 65534)
        source_contract = source_contract.verifier
        attempt["response"] = None
        protocol = Rendering(protocol.id, FinalState((output_path,)))
    if not isinstance(source_contract, TaskTroveVerifier):
        raise ValueError("Container grading requires an executable TaskTrove verifier")
    parameters = dict(source_contract.parameters)
    mode = source_contract.mode
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
            materialize(specification, ResourceRole.VERIFIER, tests, step_index)
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
        steps=(
            *specification.steps[:step_index],
            msgspec.structs.replace(
                specification.steps[step_index],
                verifier=TaskTroveVerifier(
                    mode,
                    parameters,
                    source_contract.judge,
                    runtime=runtime,
                    implementation_revision=source_contract.implementation_revision,
                ),
            ),
            *specification.steps[step_index + 1 :],
        ),
    )
    result = _grade_attempt(
        specification, protocol, attempt["response"], workspace, tuple(attempt["transcript"]), None, step_index
    )
    result.detail["isolation"] = "docker-unprivileged-candidate"
    result.detail["source_harness_trust"] = "Inherited source pytest/script reports are not adversarially tamper-proof"
    _write_result(result)


if __name__ == "__main__":
    main()
