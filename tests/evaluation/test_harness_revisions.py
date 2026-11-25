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
    "shell": ("2026.11.23.1", "ec23b95b046fc769bc7a046bbca2933c93f50fed9a295d03df938b018b5a50a3"),
    "terminus-2": ("2026.11.23.1", "10dbf350893d45606ddfbf67dfab9fc9033d6cb6c2f6e4097cae02cb54761767"),
    "pi": ("2026.11.24.1", "e31aa43d1af93f32c23dbc809d17b04d40c7b6035727d653053ddf288685af8d"),
    "mini-swe-agent": ("2026.11.23.1", "7bb0bd4921e06639cd39b21cec00505937136ed602bd9e456e4b2b03f7a2f30c"),
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
