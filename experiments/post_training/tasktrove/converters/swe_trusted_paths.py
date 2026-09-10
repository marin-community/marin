# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE repositories graded with trusted test paths."""

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

# The old ``tests/test.sh`` invokes ``install_trusted_test_paths.sh <repo> <trusted_commit>
# <manifest> [<patch_path>] [<fallback_commit>]``; the commits are the only per-task values we need
# out of that call, and they only exist embedded in this shell text.
_INVOCATION_RE = re.compile(
    r"install_trusted_test_paths\.sh\s*\\?\s*\n?\s*"
    r"\S+\s+(?P<trusted>[0-9a-f]{7,40})\s+\S+"
    r"(?:\s+(?:\"\"|\S+))?"
    r"(?:\s*\\?\s*\n?\s*(?P<fallback>[0-9a-f]{7,40}))?",
    re.MULTILINE,
)

# Restores each manifest path from the git history the agent's own environment-setup clone already
# carries (a full, non-shallow clone per ``instruction.md``), mirroring the old
# ``install_trusted_test_paths.sh`` for the explicit-manifest case without shipping its code.
_RESTORE_SETUP = """set -euo pipefail
ws="$TASKTROVE_WORKSPACE"
cd "$ws"
git -c safe.directory="$ws" cat-file -e TRUSTED_SHA^{commit}
while IFS= read -r path || [ -n "$path" ]; do
    [ -z "$path" ] && continue
    case "$path" in
        ""|/*|.|..|../*|*/..|*/../*) exit 1 ;;
    esac
    git -c safe.directory="$ws" clean -ffdx -- "$path" >/dev/null 2>&1 || true
    rm -rf -- "$path"
    if git -c safe.directory="$ws" cat-file -e TRUSTED_SHA:"$path" 2>/dev/null; then
        git -c safe.directory="$ws" archive --format=tar TRUSTED_SHA -- "$path" | tar -xf - -C "$ws"
    elif [ -n "FALLBACK_SHA" ] && git -c safe.directory="$ws" cat-file -e FALLBACK_SHA:"$path" 2>/dev/null; then
        git -c safe.directory="$ws" archive --format=tar FALLBACK_SHA -- "$path" | tar -xf - -C "$ws"
    fi
done < "$TASKTROVE_TESTS_DIR/trusted_test_paths.txt"
"""


def _restore_setup(trusted: str, fallback: str) -> str:
    return _RESTORE_SETUP.replace("TRUSTED_SHA", trusted).replace("FALLBACK_SHA", fallback)


def convert_swe_trusted_paths(task: TaskFiles) -> ConvertedTask | Rejected:
    """SWE-bench-shaped repos with ``config.json``'s ``FAIL_TO_PASS``/``PASS_TO_PASS`` node ids.

    ``install_trusted_test_paths.sh`` restored an explicit manifest of test files from a trusted
    commit before running pytest, so an agent editing a test file could not force a pass. The
    ``pytest`` mode's ``setup`` hook reproduces that restore with ``git archive`` against the
    agent's own (full, non-shallow) clone instead of shipping the old script; ``paths`` narrows
    the run to the files the graded node ids live in, matching the old ``test.sh``.

    A sibling template (``laion__r2egym-patched-full-oracle-v3``) ships one flattened test file
    graded by dotted, file-less names compared on partial overlap — a shape ``pytest`` mode's
    exact ``must_pass``/``must_not_break`` node-id matching cannot reproduce without guessing at a
    node-id mapping, so it is rejected rather than converted.
    """
    if CONFIG_JSON not in task.files:
        return Rejected(ConvertStatus.UNSUPPORTED_VARIANT, f"no {CONFIG_JSON}: not the FAIL_TO_PASS/PASS_TO_PASS shape")

    config = json.loads(task.text(CONFIG_JSON))
    fail_to_pass = test_ids(config.get("FAIL_TO_PASS", []))
    pass_to_pass = test_ids(config.get("PASS_TO_PASS", []))
    if not fail_to_pass and not pass_to_pass:
        return Rejected(ConvertStatus.NULL_GRADER, "config.json has no FAIL_TO_PASS or PASS_TO_PASS tests")

    match = _INVOCATION_RE.search(task.get_text(TEST_SH) or "")
    if match is None:
        return Rejected(ConvertStatus.UNSUPPORTED_VARIANT, "tests/test.sh does not call install_trusted_test_paths.sh")
    trusted, fallback = match["trusted"], match["fallback"] or ""

    foreign = [node_id for node_id in fail_to_pass if uncollectable(node_id)]
    if foreign:
        return Rejected(
            ConvertStatus.UNSUPPORTED_VARIANT, f"FAIL_TO_PASS ids the pytest mode cannot collect: {foreign[:3]}"
        )
    pass_to_pass = [node_id for node_id in pass_to_pass if not uncollectable(node_id)]

    graded_files = {test_file(node_id) for node_id in [*fail_to_pass, *pass_to_pass]}
    uncovered = uncovered_files(graded_files, [task.get_text(TRUSTED_TEST_PATHS)])
    if uncovered:
        detail = f"graded test files missing from trusted manifest: {uncovered[:5]}"
        return Rejected(ConvertStatus.UNSUPPORTED_VARIANT, detail)

    spec = PytestSpec(
        paths=tuple(sorted(graded_files)),
        must_pass=tuple(fail_to_pass),
        must_not_break=tuple(pass_to_pass),
        setup=_restore_setup(trusted, fallback),
        workspace=TESTBED,
    )
    return ConvertedTask(
        instruction=task.text(INSTRUCTION),
        spec=spec,
        dockerfile=ensure_pytest_json_report(task.text(DOCKERFILE)),
        tags=("code", "swe", "swe-repo", "trusted-test-paths"),
        language="python",
        data_files={TRUSTED_TEST_PATHS: task.files[TRUSTED_TEST_PATHS]},
        solution_files=task.under(SOLUTION_DIR),
    )


CONVERTER = Converter(
    name="swe_trusted_paths",
    keys=(ConverterKey("swe-repo", frozenset({"tests/test.sh", "tests/install_trusted_test_paths.sh"})),),
    convert=convert_swe_trusted_paths,
)
