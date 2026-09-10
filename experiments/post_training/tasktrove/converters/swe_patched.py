# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE repositories graded by applying a hidden test patch on top of trusted test paths.

The old ``tests/test.sh`` restored ``trusted_test_paths.txt`` from the trusted commit, restored
``trusted_patch_paths.txt`` the same way and applied ``tests/test_patch.diff`` on top (adding or
rewriting the hidden ``FAIL_TO_PASS``/``PASS_TO_PASS`` tests), then ran the repo's own test
command and graded by test id. Python tasks use the normalized ``pytest`` mode. Retained
non-Python languages use the source's self-contained parsers behind the ``script`` fallback, with
its fail-open exit-code path removed and its grader dependencies installed in the image.
"""

import json
import re

from tasktrove_verify.spec import PytestSpec, ScriptSpec

from experiments.post_training.tasktrove.contract import UV_IMAGE
from experiments.post_training.tasktrove.converters.converted_task import (
    ConvertedTask,
    Converter,
    ConverterKey,
    ConvertStatus,
    Rejected,
)
from experiments.post_training.tasktrove.converters.swe_repo import (
    CONFIG_JSON,
    TESTBED,
    TRUSTED_TEST_PATHS,
    ensure_pytest_json_report,
    test_file,
    test_ids,
    uncollectable,
    uncovered_files,
)
from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, INSTRUCTION, SOLUTION_DIR, TEST_SH, TaskFiles

TEST_PATCH = "tests/test_patch.diff"
TRUSTED_PATCH_PATHS = "tests/trusted_patch_paths.txt"
TEST_STATE = "tests/test_state.py"
LEGACY_TEST_SH = "tests/legacy_test.sh"
REJECTED_SCRIPT_LANGUAGES = frozenset({"js", "ts"})
# Their shipped goldens failed too often to retain without a row-level oracle gate.

_LEGACY_GRADER_BLOCK = """cd /tests
uv init --python 3.12 --no-progress >/dev/null 2>&1 || true
uv add --no-progress pytest==8.4.1 pytest-json-ctrf==0.3.5 >/dev/null 2>&1
uv run --no-progress pytest --ctrf /logs/verifier/ctrf.json test_state.py -rA
"""
_NORMALIZED_GRADER_BLOCK = """\
/opt/tasktrove-legacy-grader/bin/python -m pytest -p no:cacheprovider \\
    --ctrf /logs/verifier/ctrf.json "$TASKTROVE_TESTS_DIR/test_state.py" -rA
"""
_FAIL_OPEN_BLOCK = """    # Fallback: if parser found *no* tests at all (statuses empty) AND
    # the test command exited 0 AND there were no FAIL_TO_PASS/PASS_TO_PASS
    # markers we could match, trust the exit code. This unblocks tasks
    # whose test output format the parser doesn't recognize but which
    # really did pass (e.g. cmake-built C++ runners with non-gtest output,
    # custom shell-driven test harnesses, etc.).
    if not resolved and not statuses:
        exit_code = _read_exit_code()
        if exit_code == 0 and (f2p_total > 0 or p2p_total > 0):
            # Trust the exit code: all named tests assumed passed.
            report["FAIL_TO_PASS"]["success"] = list(fail_to_pass)
            report["FAIL_TO_PASS"]["failure"] = []
            report["PASS_TO_PASS"]["success"] = list(pass_to_pass)
            report["PASS_TO_PASS"]["failure"] = []
            resolved = True
            report["fallback_exit_code"] = True

"""
_EXIT_CODE_READER = """def _read_exit_code(path="/logs/test_exit_code.txt"):
    try:
        return int(Path(path).read_text().strip())
    except Exception:
        return None


"""
_TEST_STATE_CALL = 'report = evaluate_test_results("/logs/test_output.log")'
_NORMALIZED_TEST_STATE_CALL = 'report = evaluate_test_results("/logs/verifier/test_output.log")'
_SOURCE_REWARD_PATH = "/logs/verifier/reward.txt"
_NORMALIZED_REWARD_PATH = '"$TASKTROVE_LOGS_DIR/reward.txt"'
_SOURCE_TEST_OUTPUT_PATH = "/logs/test_output.log"
_NORMALIZED_TEST_OUTPUT_PATH = "/logs/verifier/test_output.log"
_EXIT_CODE_COMMENT = """# We capture the exit status — it's used by test_state.py as a fallback
# signal when the parser can't identify per-test names.
"""
_LEGACY_GRADER_DOCKERFILE = f"""\

# Isolated runtime for the source's self-contained test-output parser.
COPY --from={UV_IMAGE} /uv /usr/local/bin/uv
RUN uv venv --python 3.12 /opt/tasktrove-legacy-grader \\
    && uv pip install --python /opt/tasktrove-legacy-grader/bin/python \\
        pytest==8.4.1 pytest-json-ctrf==0.3.5
"""

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


def _is_pytest_node_id(node_id: str) -> bool:
    """A pytest node id names a ``.py`` file before the first ``::``; other languages' ids don't."""
    file_part, _, rest = node_id.partition("::")
    return bool(rest) and file_part.endswith(".py")


def _fail_and_pass_to_pass(config: dict) -> tuple[list[str], list[str]]:
    """SWE-rebench-V2 and classic SWE-bench key these ``FAIL_TO_PASS``/``PASS_TO_PASS``; SWE-Gym
    keys them lowercase. Both are otherwise the same shape."""
    if "FAIL_TO_PASS" in config or "PASS_TO_PASS" in config:
        return test_ids(config.get("FAIL_TO_PASS")), test_ids(config.get("PASS_TO_PASS"))
    return test_ids(config.get("fail_to_pass")), test_ids(config.get("pass_to_pass"))


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


def _legacy_script_task(
    task: TaskFiles, language: str, fail_to_pass: list[str], test_sh: str
) -> ConvertedTask | Rejected:
    """Keep a non-Python source grader behind script mode after removing its fail-open path."""
    required = {
        TEST_STATE,
        TEST_PATCH,
        TRUSTED_TEST_PATHS,
        TRUSTED_PATCH_PATHS,
        "tests/install_trusted_test_patch.sh",
        "tests/install_trusted_test_paths.sh",
        f"{SOLUTION_DIR}solve.sh",
    }
    missing = sorted(required - task.files.keys())
    if missing:
        return Rejected(ConvertStatus.NULL_GRADER, f"legacy script grader is missing {missing}")
    if not fail_to_pass:
        return Rejected(ConvertStatus.TOO_FEW_CASES, "config.json has no FAIL_TO_PASS tests")
    if not (task.get_text(TEST_PATCH) or "").strip():
        return Rejected(ConvertStatus.NULL_GRADER, "tests/test_patch.diff is empty")
    if _LEGACY_GRADER_BLOCK not in test_sh:
        return Rejected(ConvertStatus.UNSUPPORTED_VARIANT, "tests/test.sh has an unknown grader bootstrap")

    test_state = task.text(TEST_STATE)
    expected_blocks = (_FAIL_OPEN_BLOCK, _EXIT_CODE_READER, _TEST_STATE_CALL)
    if not all(block in test_state for block in expected_blocks):
        return Rejected(ConvertStatus.UNSUPPORTED_VARIANT, "tests/test_state.py has an unknown fail-open implementation")

    normalized_test_sh = test_sh.replace(_LEGACY_GRADER_BLOCK, _NORMALIZED_GRADER_BLOCK)
    normalized_test_sh = normalized_test_sh.replace(_SOURCE_REWARD_PATH, _NORMALIZED_REWARD_PATH)
    normalized_test_sh = normalized_test_sh.replace(_SOURCE_TEST_OUTPUT_PATH, _NORMALIZED_TEST_OUTPUT_PATH)
    normalized_test_sh = normalized_test_sh.replace("echo $? > /logs/test_exit_code.txt\n", "")
    normalized_test_sh = normalized_test_sh.replace(_EXIT_CODE_COMMENT, "")
    normalized_test_state = (
        test_state.replace(_FAIL_OPEN_BLOCK, "")
        .replace(_EXIT_CODE_READER, "")
        .replace(_TEST_STATE_CALL, _NORMALIZED_TEST_STATE_CALL)
    )

    data_files = task.under("tests/")
    data_files.pop(TEST_SH)
    data_files[LEGACY_TEST_SH] = normalized_test_sh.encode()
    data_files[TEST_STATE] = normalized_test_state.encode()
    return ConvertedTask(
        instruction=task.text(INSTRUCTION),
        spec=ScriptSpec(path=LEGACY_TEST_SH.removeprefix("tests/"), workspace=TESTBED),
        dockerfile=task.text(DOCKERFILE) + _LEGACY_GRADER_DOCKERFILE,
        tags=("code", "swe", "swe-repo", language, "patched", "script-fallback"),
        language=language,
        data_files=data_files,
        solution_files=task.under(SOLUTION_DIR),
    )


def convert_swe_patched(task: TaskFiles) -> ConvertedTask | Rejected:
    """SWE-bench-shaped repos whose hidden ``FAIL_TO_PASS``/``PASS_TO_PASS`` tests arrive as a
    patch applied on top of the trusted commit, rather than already present at it.

    Retained non-Python repos keep their self-contained source parser through script mode. The
    parser must resolve every named test; an unrecognized log can no longer pass merely because
    the test command exited zero.
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
    language = config.get("language") or "python"
    if not isinstance(language, str):
        return Rejected(ConvertStatus.UNSUPPORTED_VARIANT, f"language is not a string: {type(language).__name__}")
    fail_to_pass, pass_to_pass = _fail_and_pass_to_pass(config)

    test_sh = task.get_text(TEST_SH) or ""
    if language != "python":
        if language in REJECTED_SCRIPT_LANGUAGES:
            return Rejected(
                ConvertStatus.UNSUPPORTED_VARIANT,
                f"{language} script graders failed the language-level golden sample",
            )
        return _legacy_script_task(task, language, fail_to_pass, test_sh)

    if not fail_to_pass:
        return Rejected(ConvertStatus.TOO_FEW_CASES, "config.json has no FAIL_TO_PASS tests")
    foreign = [node_id for node_id in fail_to_pass if uncollectable(node_id)]
    if foreign:
        return Rejected(
            ConvertStatus.UNSUPPORTED_VARIANT, f"FAIL_TO_PASS ids the pytest mode cannot collect: {foreign[:3]}"
        )
    pass_to_pass = [node_id for node_id in pass_to_pass if not uncollectable(node_id)]
    node_ids = [*fail_to_pass, *pass_to_pass]
    non_pytest = [node_id for node_id in node_ids if not _is_pytest_node_id(node_id)]
    if non_pytest:
        return Rejected(ConvertStatus.UNSUPPORTED_VARIANT, f"not pytest node ids: {non_pytest[:3]}")

    test_patch = task.get_text(TEST_PATCH) or ""
    if not test_patch.strip():
        return Rejected(ConvertStatus.NULL_GRADER, "tests/test_patch.diff is empty")

    match = _PATCH_INVOCATION_RE.search(test_sh)
    if match is None:
        return Rejected(ConvertStatus.UNSUPPORTED_VARIANT, "tests/test.sh does not call install_trusted_test_patch.sh")
    trusted_commit = match["commit"]

    graded_files = {test_file(node_id) for node_id in node_ids}
    uncovered = uncovered_files(graded_files, [task.get_text(TRUSTED_TEST_PATHS), task.get_text(TRUSTED_PATCH_PATHS)])
    if uncovered:
        return Rejected(
            ConvertStatus.UNSUPPORTED_VARIANT, f"graded test files missing from trusted manifest: {uncovered[:5]}"
        )

    workspace_match = _REPO_DIR_RE.search(test_sh)
    workspace = workspace_match.group(1) if workspace_match else TESTBED
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
        dockerfile=ensure_pytest_json_report(task.text(DOCKERFILE), conda_lines),
        tags=("code", "swe", "swe-repo", "python", "patched"),
        language="python",
        data_files=data_files,
        solution_files=task.under(SOLUTION_DIR),
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
