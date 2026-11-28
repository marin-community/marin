# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path

import pytest
import rolloutengine
from marin.evaluation.task_worker import EPISODE_HARNESS_REVISIONS

ROLLOUTENGINE = Path(rolloutengine.__file__).parent
AGENT_HARNESS = ("harness/core.py", "harness/external.py", "agents/environment.py")
HARNESS_SOURCES = {
    "shell": ("harness/core.py", "harness/shell.py"),
    "terminus-2": (*AGENT_HARNESS, "agents/terminus"),
    "pi": (*AGENT_HARNESS, "agents/model_relay.py", "agents/pi"),
    "mini-swe-agent": (*AGENT_HARNESS, "agents/model_relay.py", "agents/mini_swe"),
}
# The revision each harness's sources were last reviewed at, and their digest. An edit to these sources changes
# what an episode is graded on; bump the harness's revision in EPISODE_HARNESS_REVISIONS and update its pin.
PINNED = {
    "shell": ("2026.11.23.1", "1592f8f482ca434f304dfed73654a4e0c5c18e6e9d86095d0f7da3713eab0013"),
    "terminus-2": ("2026.11.23.1", "9b910b78a31812d8109d189b6c10819da551d095d33aa7371a9cc5281a4c1ef4"),
    "pi": ("2026.11.24.1", "bfe2cff8b7df0725a3b6ce281371dd5543e104b3fcbcc8338c60f6eff5c2ac67"),
    "mini-swe-agent": ("2026.11.23.1", "d27cb0002e477e075c86208f8c0ef45799a46f69788949cf0fa1e9040adb622a"),
}


def sources_digest(paths: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    files = sorted(
        file
        for path in paths
        for file in ((ROLLOUTENGINE / path).rglob("*") if (ROLLOUTENGINE / path).is_dir() else (ROLLOUTENGINE / path,))
        if file.is_file() and "__pycache__" not in file.parts
    )
    for file in files:
        digest.update(file.relative_to(ROLLOUTENGINE).as_posix().encode() + b"\0" + file.read_bytes() + b"\0")
    return digest.hexdigest()


@pytest.mark.parametrize("kind", sorted(HARNESS_SOURCES))
def test_a_harness_whose_sources_change_gets_a_new_revision(kind):
    assert HARNESS_SOURCES.keys() == EPISODE_HARNESS_REVISIONS.keys()
    revision, digest = PINNED[kind]
    assert (EPISODE_HARNESS_REVISIONS[kind], sources_digest(HARNESS_SOURCES[kind])) == (
        revision,
        digest,
    ), f"The {kind} harness changed: bump its EPISODE_HARNESS_REVISIONS entry and update PINNED"
