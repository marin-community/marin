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
    "shell": ("2026.11.23.1", "1b77535d3f21ef3ec4fc91d6a2d7c161acb990b6b5ada1a0a23b1f02c8bce2e9"),
    "terminus-2": ("2026.11.23.1", "8c4a6367cd3443f5e1df83aa456af04d64e9b4350cc0c9465672d7eb8edb5da8"),
    "pi": ("2026.11.27.1", "ccfa63c5d6d62bf7520e511b552f66cf682f07db9fd4e9ef77f68402988db561"),
    "mini-swe-agent": ("2026.11.27.1", "92dc905587d689d2d4a9290bb4d3ac68a83a13b4880961365687eae803801053"),
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
