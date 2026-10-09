# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trusted repository tests carried from TaskTrove into deferred build tasks."""

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from taskcompendium.convert.answers import unsupported
from taskcompendium.convert.tasktrove import TaskFiles
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection
from verifyit.spec import PytestSpec

from experiments.post_training.task_curation.datasets.tasktrove.repository_build import WORKSPACE

TRUSTED_PATHS = "trusted_test_paths.txt"
PYTEST_REPORT_PLUGIN = "pytest-json-report"
PYTEST_VERSION_CONSTRAINT = "pytest<9"
PYTEST_CONSTRAINT_PATH = "/opt/verifyit-pytest-constraints.txt"
# The source installer takes <repo> <trusted_commit> <manifest> [<patch>] [<fallback_commit>].
TRUSTED_INVOCATION = re.compile(
    r"install_trusted_test_paths\.sh\s*\\?\s*\n?\s*"
    r"\S+\s+(?P<trusted>[0-9a-f]{7,40})\s+\S+"
    r"(?:\s+(?:\"\"|\S+))?"
    r"(?:\s*\\?\s*\n?\s*(?P<fallback>[0-9a-f]{7,40}))?",
    re.MULTILINE,
)
SWESMITH_REPOSITORY = re.compile(r"https://github\.com/swesmith/(?P<repo>[^\s/]+)")
RESTORE_TESTS = """set -euo pipefail
ws="$VERIFYIT_WORKSPACE"
cd "$ws"
git -c safe.directory="$ws" cat-file -e TRUSTED_SHA^{commit}
restore_path() {
    path="$1"
    git -c safe.directory="$ws" clean -ffdx -- "$path" >/dev/null 2>&1 || true
    rm -rf -- "$path"
    if git -c safe.directory="$ws" cat-file -e TRUSTED_SHA:"$path" 2>/dev/null; then
        git -c safe.directory="$ws" archive --format=tar TRUSTED_SHA -- "$path" | tar -xf - -C "$ws"
    elif [ -n "FALLBACK_SHA" ] && git -c safe.directory="$ws" cat-file -e FALLBACK_SHA:"$path" 2>/dev/null; then
        git -c safe.directory="$ws" archive --format=tar FALLBACK_SHA -- "$path" | tar -xf - -C "$ws"
    fi
}
restore_manifest() {
    while IFS= read -r path || [ -n "$path" ]; do
        [ -z "$path" ] && continue
        case "$path" in
            ""|/*|.|..|../*|*/..|*/../*) exit 1 ;;
        esac
        restore_path "$path"
    done < "$1"
}
RESTORE_MANIFESTS
APPLY_PATCH
"""


def restore_setup(trusted: str, manifests: tuple[str, ...], fallback: str = "", patch: str = "") -> str:
    manifest_commands = "\n".join(
        f'restore_manifest "$VERIFYIT_TESTS_DIR/{PurePosixPath(path).name}"' for path in manifests
    )
    patch_command = (
        f'git -c safe.directory="$ws" apply --whitespace=nowarn "$VERIFYIT_TESTS_DIR/{PurePosixPath(patch).name}"'
        if patch
        else ""
    )
    return (
        RESTORE_TESTS.replace("TRUSTED_SHA", trusted)
        .replace("FALLBACK_SHA", fallback)
        .replace("RESTORE_MANIFESTS", manifest_commands)
        .replace("APPLY_PATCH", patch_command)
    )


def repository_test_ids(value: object) -> tuple[str, ...]:
    """Read the source's list or JSON-encoded list of pytest node ids."""
    if isinstance(value, str):
        decoded = json.loads(value) if value.strip() else []
    else:
        decoded = value if value is not None else []
    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
        raise ValueError("Repository test ids must be a list of strings")
    return tuple(decoded)


def uncollectable(node_id: str) -> bool:
    """Source doctest and truncated parametrized ids cannot run with cleared pytest addopts."""
    file, _, rest = node_id.partition("::")
    if not file.endswith(".py") or ("[" in rest and not rest.endswith("]")):
        return True
    name = rest.split("[", 1)[0]
    if "::" in name:
        return False
    path = PurePosixPath(file)
    module = path.parent.name if path.name == "__init__.py" else path.stem
    return "." in name or (name == module and not name.startswith("test"))


@dataclass(frozen=True)
class PytestSelection:
    must_pass: tuple[str, ...]
    must_not_break: tuple[str, ...]
    files: tuple[str, ...]


def pytest_selection(
    fail_to_pass: Sequence[str], pass_to_pass: Sequence[str], manifests: Iterable[str | None]
) -> PytestSelection | ImportRejection:
    """Select collectable tests only when trusted manifests cover every graded file."""
    foreign = [node_id for node_id in fail_to_pass if uncollectable(node_id)]
    if foreign:
        return unsupported("unsupported_variant", f"FAIL_TO_PASS ids the pytest mode cannot collect: {foreign[:3]}")
    retained = tuple(node_id for node_id in pass_to_pass if not uncollectable(node_id))
    files = {node_id.split("::", 1)[0] for node_id in (*fail_to_pass, *retained)}
    manifest = {line.strip() for text in manifests for line in (text or "").splitlines() if line.strip()}
    uncovered = sorted(files - manifest)
    if uncovered:
        return unsupported("unsupported_variant", f"graded test files missing from trusted manifest: {uncovered[:5]}")
    return PytestSelection(tuple(fail_to_pass), retained, tuple(sorted(files)))


def trusted_pytest(task: TaskFiles) -> PytestSpec | ImportRejection:
    """Recover the legacy SWE-smith pytest contract, including trusted-test restoration."""
    config = json.loads(task.text("tests/config.json"))
    fail_to_pass = repository_test_ids(config.get("FAIL_TO_PASS"))
    pass_to_pass = repository_test_ids(config.get("PASS_TO_PASS"))
    if not fail_to_pass and not pass_to_pass:
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="null_grader",
            detail="config.json has no FAIL_TO_PASS or PASS_TO_PASS tests",
        )
    match = TRUSTED_INVOCATION.search(task.text("tests/test.sh"))
    if match is None:
        return unsupported("unsupported_variant", "tests/test.sh does not call install_trusted_test_paths.sh")
    selection = pytest_selection(fail_to_pass, pass_to_pass, (task.text(path) for path in (f"tests/{TRUSTED_PATHS}",)))
    if isinstance(selection, ImportRejection):
        return selection
    return PytestSpec(
        paths=selection.files,
        must_pass=selection.must_pass,
        must_not_break=selection.must_not_break,
        setup=restore_setup(match["trusted"], (f"tests/{TRUSTED_PATHS}",), match["fallback"] or ""),
        protected_paths_files=(TRUSTED_PATHS,),
        workspace=WORKSPACE,
    )


def python_command(conda_lines: tuple[str, ...]) -> tuple[str, str]:
    """Return the grading interpreter and setup script for the source's Python environment."""
    if not conda_lines:
        return "python3", ""
    wrapper = "/tmp/tasktrove-python"
    body = "\n".join(conda_lines)
    heredoc = (
        f"cat > {wrapper} << 'TASKTROVE_PYTHON_EOF'\n"
        f'#!/bin/bash\n{body}\nexec python "$@"\n'
        f"TASKTROVE_PYTHON_EOF\n"
        f"chmod +x {wrapper}\n"
    )
    return wrapper, heredoc


def ensure_pytest_json_report(dockerfile: str, conda_lines: tuple[str, ...] = ()) -> str:
    """Install the report plugin in the repository's Python, including activated conda environments."""
    if PYTEST_REPORT_PLUGIN in dockerfile:
        return dockerfile
    lines = dockerfile.splitlines()
    for index, line in enumerate(lines):
        tokens = line.split()
        if tokens[:1] == ["RUN"] and "pip" in tokens and "install" in tokens and "pytest" in tokens:
            lines[index] = line + f" {PYTEST_REPORT_PLUGIN}"
            return "\n".join(lines) + "\n"
    if conda_lines:
        activate = " && ".join(conda_lines)
        install = f'RUN bash -lc "{activate} && pip install --no-cache-dir {PYTEST_REPORT_PLUGIN}"\n'
    else:
        install = (
            f"RUN (pip install --no-cache-dir {PYTEST_REPORT_PLUGIN}"
            f" || pip3 install --no-cache-dir {PYTEST_REPORT_PLUGIN})\n"
        )
    return dockerfile.rstrip("\n") + "\n" + install


def repository_dockerfile(dockerfile: str, instruction: str) -> str:
    """Carry the legacy pytest plugin and repository dependency fixes into the build recipe."""
    dockerfile = ensure_pytest_json_report(dockerfile)
    match = SWESMITH_REPOSITORY.search(instruction)
    if match is None:
        return dockerfile
    packages = f'"{PYTEST_VERSION_CONSTRAINT}"'
    additions = {
        "marshmallow-code__marshmallow.9716fc62": " simplejson",
        "conan-io__conan.86f29e13": " mock webtest PyJWT bottle parameterized",
        "oauthlib__oauthlib.1fd52536": " PyJWT cryptography blinker",
    }
    packages += additions.get(match["repo"], "")
    return (
        dockerfile.rstrip("\n")
        + f"\nRUN printf '{PYTEST_VERSION_CONSTRAINT}\\n' > {PYTEST_CONSTRAINT_PATH}\n"
        + f"ENV PIP_CONSTRAINT={PYTEST_CONSTRAINT_PATH}\n"
        + f"RUN python -m pip install --no-cache-dir {packages}\n"
    )
