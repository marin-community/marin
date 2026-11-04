# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate staged SWE-Smith handoffs independently of the rollout harness."""

import base64
from pathlib import Path

from experiments.post_training.task_curation.swesmith_staged import (
    WORKSPACE,
    DockerPilot,
    is_test_path,
    pytest_result,
    sha256,
)


def repair_grade(contract: dict, outputs: dict[str, bytes], logs: Path) -> dict:
    baseline = contract["tracked_sha256"]
    allowed = set(contract["changed_source_paths"])
    if any(is_test_path(path) or path in contract["verifier"]["paths"] or path not in baseline for path in allowed):
        return {"passed": False, "error": "invalid_task_repair_scope_contains_tests_or_untracked_paths"}
    violations = [
        path
        for path, digest in baseline.items()
        if WORKSPACE + "/" + path not in outputs
        or (path not in allowed and sha256(outputs[WORKSPACE + "/" + path]) != digest)
    ]
    if violations:
        return {"passed": False, "integrity_violations": sorted(violations)}
    pilot = DockerPilot(logs)
    with pilot.container(contract["verifier_image"]) as container:
        pilot.upload(container, {path: outputs[WORKSPACE + "/" + path] for path in allowed})
        return pytest_result(pilot, container, contract["verifier"], "candidate")


def handoff_program(files: dict[str, str], allowed: list[str], root: Path) -> str:
    """Reject hidden-test/path escapes before materializing actor-visible source."""
    for name, content in files.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or name not in allowed or is_test_path(name):
            raise ValueError("Handoff overlay is outside permitted source")
        base64.b64decode(content, validate=True)
    return (
        "import base64,pathlib\nroot=pathlib.Path("
        + repr(str(root))
        + ")\n"
        + "files="
        + repr(files)
        + "\n"
        + "for name,content in files.items():\n"
        + " path=root/name\n"
        + " assert path.is_file() and not path.is_symlink()\n"
        + " assert path.resolve().is_relative_to(root)\n"
        + " path.write_bytes(base64.b64decode(content))\n"
    )
