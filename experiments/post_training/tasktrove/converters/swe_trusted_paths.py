# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE repositories graded with trusted test paths."""

import json

from taskcompendium.convert.tasktrove import DOCKERFILE, INSTRUCTION, SOLUTION_DIR, TEST_SH, TaskFiles
from taskcompendium.convert.tasktrove_converted_task import (
    ConvertedTask,
    Converter,
    ConverterKey,
    ConvertStatus,
    Rejected,
)
from taskcompendium.pipeline.models import ImportRejection
from verifyit.spec import PytestSpec

from experiments.post_training.task_curation.datasets.tasktrove.repository_pytest import (
    TRUSTED_INVOCATION,
    pytest_selection,
    repository_dockerfile,
)
from experiments.post_training.tasktrove.converters.swe_repo import (
    CONFIG_JSON,
    TESTBED,
    TRUSTED_TEST_PATHS,
    restore_setup,
    test_ids,
)


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

    match = TRUSTED_INVOCATION.search(task.get_text(TEST_SH) or "")
    if match is None:
        return Rejected(ConvertStatus.UNSUPPORTED_VARIANT, "tests/test.sh does not call install_trusted_test_paths.sh")
    trusted, fallback = match["trusted"], match["fallback"] or ""

    selection = pytest_selection(fail_to_pass, pass_to_pass, [task.get_text(TRUSTED_TEST_PATHS)])
    if isinstance(selection, ImportRejection):
        return Rejected(ConvertStatus.UNSUPPORTED_VARIANT, selection.detail)

    spec = PytestSpec(
        paths=selection.files,
        must_pass=selection.must_pass,
        must_not_break=selection.must_not_break,
        setup=restore_setup(trusted, (TRUSTED_TEST_PATHS,), fallback),
        protected_paths_files=(TRUSTED_TEST_PATHS.removeprefix("tests/"),),
        workspace=TESTBED,
    )
    return ConvertedTask(
        instruction=task.text(INSTRUCTION),
        spec=spec,
        dockerfile=repository_dockerfile(task.text(DOCKERFILE), task.text(INSTRUCTION)),
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
