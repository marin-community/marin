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
    "shell": ("2026.11.21.1", "0d4b84d771b227db3b8059a81a0d344a33f5bc21872001b46e3fa5dd2cb34d06"),
    "terminus-2": ("2026.11.21.1", "ba4f7e65f97d4164a13dfe8abfed45b74f8a7f2b57c79b7bb656e6434eacbdd7"),
    "pi": ("2026.11.21.1", "d7d5ee760a8561f40ae0abdec87040e0c42bd15f34ebd5b6f68a09ef059f2ad9"),
    "mini-swe-agent": ("2026.11.21.1", "3ef08ee8b6d4ea04fbaaa806a0513bea16ab8a626a9183559cd861f3423420ba"),
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
