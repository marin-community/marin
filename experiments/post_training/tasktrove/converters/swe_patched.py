# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE repositories graded by applying a hidden test patch on top of trusted test paths.

The old ``tests/test.sh`` restored ``trusted_test_paths.txt`` from the trusted commit, restored
``trusted_patch_paths.txt`` the same way and applied ``tests/test_patch.diff`` on top (adding or
rewriting the hidden ``FAIL_TO_PASS``/``PASS_TO_PASS`` tests), then ran the repo's own test
command and graded by test id. The ``pytest`` mode's ``setup`` hook reproduces the two restores
with ``git archive`` against the agent's own (full, non-shallow) clone — the same clone
``instruction.md`` tells the agent to make — followed by ``git apply``, instead of shipping the
old scripts; ``paths`` narrows the run to the files the graded node ids live in.
"""

import json
import re

from tasktrove_verify.spec import PytestSpec

from experiments.post_training.tasktrove.converters.converted_task import (
    ConvertedTask,
    Converter,
    ConverterKey,
    ConvertStatus,
    Rejected,
)
from experiments.post_training.tasktrove.converters.nemotron_data import metadata
from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, INSTRUCTION, SOLUTION_DIR, TaskFiles

CONFIG_JSON = "tests/config.json"
TEST_PATCH = "tests/test_patch.diff"
TRUSTED_TEST_PATHS = "tests/trusted_test_paths.txt"
TRUSTED_PATCH_PATHS = "tests/trusted_patch_paths.txt"
DEFAULT_WORKSPACE = "/testbed"

# The old ``tests/test.sh`` invokes ``install_trusted_test_patch.sh <repo> <patch> <trusted_commit>``;
# the commit is the only per-task value we need out of that call, and it only exists embedded in
# this shell text (``config.json``'s field name and casing vary across sources; this does not).
_PATCH_INVOCATION_RE = re.compile(
    r"install_trusted_test_patch\.sh\s*\\?\s*\n?\s*\S+\s+\S+\s+(?P<commit>[0-9a-f]{7,40})", re.MULTILINE
)
_REPO_DIR_RE = re.compile(r'REPO_DIR\s*=\s*"([^"]+)"')
_CONDA_LINE_RE = re.compile(r"^[ \t]*(?:source\s+\S*conda\S*\S*|conda activate\s+\S+)[ \t]*$", re.MULTILINE)

# Restores every entry of one manifest from the trusted commit via the agent's own clone, mirroring
# ``install_trusted_test_paths.sh`` without shipping its code, then applies the hidden test patch.
_SETUP_TEMPLATE = """set -euo pipefail
cd "$TASKTROVE_WORKSPACE"
git -c safe.directory="$TASKTROVE_WORKSPACE" cat-file -e TRUSTED_SHA^{commit}
restore_path() {
    path="$1"
    git -c safe.directory="$TASKTROVE_WORKSPACE" clean -ffdx -- "$path" >/dev/null 2>&1 || true
    rm -rf -- "$path"
    if git -c safe.directory="$TASKTROVE_WORKSPACE" cat-file -e TRUSTED_SHA:"$path" 2>/dev/null; then
        git -c safe.directory="$TASKTROVE_WORKSPACE" archive --format=tar TRUSTED_SHA -- "$path" \
            | tar -xf - -C "$TASKTROVE_WORKSPACE"
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
restore_manifest "$TASKTROVE_TESTS_DIR/trusted_test_paths.txt"
restore_manifest "$TASKTROVE_TESTS_DIR/trusted_patch_paths.txt"
git -c safe.directory="$TASKTROVE_WORKSPACE" apply --whitespace=nowarn "$TASKTROVE_TESTS_DIR/test_patch.diff"
"""


def _setup_script(trusted_commit: str) -> str:
    return _SETUP_TEMPLATE.replace("TRUSTED_SHA", trusted_commit)


def _as_str_list(value: object) -> list[str]:
    """``FAIL_TO_PASS``/``PASS_TO_PASS`` are sometimes JSON-encoded strings rather than lists."""
    if isinstance(value, str):
        decoded: object = json.loads(value) if value.strip() else []
    else:
        decoded = value if value is not None else []
    if not isinstance(decoded, list):
        raise ValueError(f"expected a list of test ids, got {type(decoded).__name__}")
    return [str(v) for v in decoded]


def _is_pytest_node_id(node_id: str) -> bool:
    """A pytest node id names a ``.py`` file before the first ``::``; other languages' ids don't."""
    file_part, _, rest = node_id.partition("::")
    return bool(rest) and file_part.endswith(".py")


def _fail_and_pass_to_pass(config: dict) -> tuple[list[str], list[str]]:
    """SWE-rebench-V2 and classic SWE-bench key these ``FAIL_TO_PASS``/``PASS_TO_PASS``; SWE-Gym
    keys them lowercase. Both are otherwise the same shape."""
    if "FAIL_TO_PASS" in config or "PASS_TO_PASS" in config:
        return _as_str_list(config.get("FAIL_TO_PASS")), _as_str_list(config.get("PASS_TO_PASS"))
    return _as_str_list(config.get("fail_to_pass")), _as_str_list(config.get("pass_to_pass"))


def _conda_activation(test_sh: str) -> tuple[str, ...]:
    """``source .../activate`` and ``conda activate <env>`` lines the old ``test.sh`` ran before
    testing, in order. Repos whose test command needs a conda env (rather than the image's system
    Python) carry these verbatim; the plain-pip repos carry none."""
    seen: list[str] = []
    for match in _CONDA_LINE_RE.finditer(test_sh):
        line = match.group(0).strip()
        if line not in seen:
            seen.append(line)
    return tuple(seen)


def _python_command(conda_lines: tuple[str, ...]) -> tuple[str, str]:
    """The interpreter to grade with, and the setup line that makes it resolve.

    A conda repo needs its env activated before ``python`` resolves to the right interpreter; that
    activation does not survive between the ``setup`` shell command and the separate ``pytest``
    subprocess, so a small wrapper script does the activation and execs ``python`` itself.
    """
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


def _ensure_pytest_json_report(dockerfile: str, conda_lines: tuple[str, ...]) -> str:
    """Add the ``pytest-json-report`` plugin the ``pytest`` mode needs into the repo's own Python.

    Installed at image build time (not in ``setup``) because the sampling harness and the real
    verifier both run with no network at grading time.
    """
    if "pytest-json-report" in dockerfile:
        return dockerfile
    if conda_lines:
        activate = " && ".join(conda_lines)
        install = f'RUN bash -lc "{activate} && pip install --no-cache-dir pytest-json-report"\n'
    else:
        install = (
            "RUN (pip install --no-cache-dir pytest-json-report" " || pip3 install --no-cache-dir pytest-json-report)\n"
        )
    return dockerfile.rstrip("\n") + "\n" + install


def convert_swe_patched(task: TaskFiles) -> ConvertedTask | Rejected:
    """SWE-bench-shaped repos whose hidden ``FAIL_TO_PASS``/``PASS_TO_PASS`` tests arrive as a
    patch applied on top of the trusted commit, rather than already present at it.

    Non-Python repos (Go, Rust, TypeScript, Elixir, Java, ...) are rejected: the old grader
    dispatched on a language-specific log parser, but the ``pytest`` mode can only run and score
    Python's own test node ids.
    """
    if CONFIG_JSON not in task.files:
        return Rejected(ConvertStatus.UNSUPPORTED_VARIANT, f"no {CONFIG_JSON}: not the FAIL_TO_PASS/PASS_TO_PASS shape")
    if any(path.startswith("setup_files/") for path in task.files):
        return Rejected(
            ConvertStatus.UNSUPPORTED_VARIANT,
            "environment setup lives in a top-level setup_files/ script instruction.md expects to "
            "run from the image root, which the pytest mode's sandbox never mounts",
        )

    config = json.loads(task.text(CONFIG_JSON))
    language = config.get("language")
    if language is not None and language != "python":
        return Rejected(ConvertStatus.UNSUPPORTED_VARIANT, f"language is {language!r}, not python")

    fail_to_pass, pass_to_pass = _fail_and_pass_to_pass(config)
    if not fail_to_pass:
        return Rejected(ConvertStatus.TOO_FEW_CASES, "config.json has no FAIL_TO_PASS tests")
    node_ids = [*fail_to_pass, *pass_to_pass]
    non_pytest = [node_id for node_id in node_ids if not _is_pytest_node_id(node_id)]
    if non_pytest:
        return Rejected(ConvertStatus.UNSUPPORTED_VARIANT, f"not pytest node ids: {non_pytest[:3]}")

    test_patch = task.get_text(TEST_PATCH) or ""
    if not test_patch.strip():
        return Rejected(ConvertStatus.NULL_GRADER, "tests/test_patch.diff is empty")

    test_sh = task.get_text("tests/test.sh") or ""
    match = _PATCH_INVOCATION_RE.search(test_sh)
    if match is None:
        return Rejected(ConvertStatus.UNSUPPORTED_VARIANT, "tests/test.sh does not call install_trusted_test_patch.sh")
    trusted_commit = match["commit"]

    graded_files = {node_id.split("::", 1)[0] for node_id in node_ids}
    manifest = {
        line.strip()
        for text in (task.get_text(TRUSTED_TEST_PATHS), task.get_text(TRUSTED_PATCH_PATHS))
        for line in (text or "").splitlines()
        if line.strip()
    }
    uncovered = sorted(graded_files - manifest)
    if uncovered:
        return Rejected(
            ConvertStatus.UNSUPPORTED_VARIANT, f"graded test files missing from trusted manifest: {uncovered[:5]}"
        )

    workspace_match = _REPO_DIR_RE.search(test_sh)
    workspace = workspace_match.group(1) if workspace_match else DEFAULT_WORKSPACE
    conda_lines = _conda_activation(test_sh)
    python, python_setup = _python_command(conda_lines)

    spec = PytestSpec(
        paths=tuple(sorted(graded_files)),
        must_pass=tuple(fail_to_pass),
        must_not_break=tuple(pass_to_pass),
        setup=python_setup + _setup_script(trusted_commit),
        python=python,
        workspace=workspace,
    )
    data_files = {
        TEST_PATCH: test_patch.encode(),
        TRUSTED_TEST_PATHS: task.files.get(TRUSTED_TEST_PATHS, b""),
        TRUSTED_PATCH_PATHS: task.files.get(TRUSTED_PATCH_PATHS, b""),
    }
    return ConvertedTask(
        instruction=task.text(INSTRUCTION),
        spec=spec,
        dockerfile=_ensure_pytest_json_report(task.text(DOCKERFILE), conda_lines),
        tags=("code", "swe", "swe-repo", "python", "patched"),
        language="python",
        data_files=data_files,
        solution_files=task.under(SOLUTION_DIR),
        metadata=metadata(task),
    )


CONVERTER = Converter(
    name="swe_patched",
    keys=(
        ConverterKey(
            "swe-repo",
            frozenset({"tests/test.sh", "tests/install_trusted_test_patch.sh", "tests/install_trusted_test_paths.sh"}),
        ),
    ),
    convert=convert_swe_patched,
)
