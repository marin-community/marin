# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from infra import linter


@pytest.mark.parametrize(
    ("agent_command", "expected"),
    [
        (["codex", "exec"], ["codex", "exec", "--ephemeral", "--sandbox", "read-only"]),
        (
            ["codex", "exec", "--ephemeral", "--sandbox", "workspace-write"],
            ["codex", "exec", "--ephemeral", "--sandbox", "read-only"],
        ),
        (
            ["codex", "e", "--yolo", "--sandbox=danger-full-access"],
            ["codex", "e", "--sandbox=read-only", "--ephemeral"],
        ),
    ],
)
def test_codex_review_command_is_read_only_and_ephemeral(agent_command: list[str], expected: list[str]):
    command = linter._with_readonly_access(agent_command)

    assert command == expected


def test_lint_review_environment_removes_parent_agent_session(monkeypatch: pytest.MonkeyPatch):
    parent_session = {
        "CODEX_THREAD_ID": "codex-parent",
        "LOOM_SESSION_ID": "loom-parent",
        "LOOM_TOKEN": "loom-token",
        "WEAVER_BRANCH": "loom-branch",
    }
    for name, value in parent_session.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("LINT_REVIEW_TEST_MARKER", "kept")

    environment = linter._lint_review_env()

    assert environment["LINT_REVIEW_TEST_MARKER"] == "kept"
    assert parent_session.keys().isdisjoint(environment)
