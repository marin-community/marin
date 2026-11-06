# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Tell a waiting gdoc session that a person changed its doc.

Loom does not see a Google Doc, so the `gdoc-changes` watch (`infra/loom/watches/gdoc_changes.py`) polls the doc of
each waiting session and sends the session a message when the doc or its comments change. The agent's own edits land
while it works, so the watch compares the doc with how it was when the session began to wait, not with how it was
when the agent last read it. A change by a person between the agent's last read and the watch's first look, which
comes when Loom reports the session idle, is not reported.

The watch reads a doc every `FAST_CHECK` seconds after its session begins to wait or the doc changes, and half as
often for each `DECAY_STEP` without a change, down to every `SLOW_CHECK` seconds. It sends the message once the doc
has stayed the same for `QUIET` seconds, so a person who is still typing gets one message when they stop.
"""

import hashlib
import json
import re
from typing import NamedTuple

from infra.loom.gdoc.cli import doc_comments
from infra.loom.gdoc.google_api import DOCS_API, GoogleApi

DOC_URL = re.compile(r"https://docs\.google\.com/document/d/([A-Za-z0-9_-]+)")
CHANGED_MESSAGE = "The doc changed: a person changed {what}. Read the doc and its open comments, and act on the change."
FAST_CHECK = 5
SLOW_CHECK = 30
DECAY_STEP = 60
QUIET = 10


class DocVersion(NamedTuple):
    # The Docs API revision, which changes when the doc's content changes. It stays the same for one reader, and the
    # watch always reads as the same account.
    revision: str
    # A digest of each comment's id and modified time, which changes when a comment or a reply is added or changed.
    comments: str


class Watched(NamedTuple):
    """What the watch remembers about one session between rounds."""

    doc_id: str
    # The doc when the session began to wait, or None while the session works.
    waiting_version: DocVersion | None = None
    # The doc at the previous look, to tell a change that holds from one still in progress.
    last_version: DocVersion | None = None
    # When, in seconds since the epoch, the session began to wait or the doc last changed between two looks.
    changed_at: float | None = None

    def to_json(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "waiting": self.waiting_version,
            "last": self.last_version,
            "changed_at": self.changed_at,
        }

    @classmethod
    def from_json(cls, value: dict) -> "Watched":
        def version(raw: list | None) -> DocVersion | None:
            return DocVersion(*raw) if raw else None

        return cls(value["doc_id"], version(value["waiting"]), version(value["last"]), value["changed_at"])


def goal_doc_id(goal: str) -> str | None:
    """The id of the first Google Doc that the goal links, or None."""
    match = DOC_URL.search(goal)
    return match.group(1) if match else None


def doc_version(api: GoogleApi, doc_id: str) -> DocVersion:
    revision = api.get(f"{DOCS_API}/documents/{doc_id}", params={"fields": "revisionId"})["revisionId"]
    comments = sorted((comment["id"], comment["modifiedTime"]) for comment in doc_comments(api, doc_id))
    return DocVersion(revision, hashlib.sha256(json.dumps(comments).encode()).hexdigest()[:16])


def changed_message(before: DocVersion, after: DocVersion) -> str:
    parts = [
        name
        for name, changed in (
            ("the text", before.revision != after.revision),
            ("the comments", before.comments != after.comments),
        )
        if changed
    ]
    return CHANGED_MESSAGE.format(what=" and ".join(parts))


def next_round(watched: Watched, version: DocVersion | None, now: float) -> tuple[Watched, str | None]:
    """The session's state after a look at its doc at time `now`, and the message to send it now, if any.

    version is the doc now while the session waits, or None while it works.
    """
    if version is None:
        return Watched(watched.doc_id), None
    if watched.waiting_version is None:
        return Watched(watched.doc_id, version, version, now), None
    if version != watched.last_version:
        return watched._replace(last_version=version, changed_at=now), None
    if version == watched.waiting_version or now - watched.changed_at < QUIET:
        return watched, None
    # The change held. Wait for the next change from here, in case the session does not start a turn.
    return watched._replace(waiting_version=version), changed_message(watched.waiting_version, version)


def next_check(watched: Watched, now: float) -> float | None:
    """Seconds until the watch should look at a waiting session's doc again, or None while the session works."""
    if watched.waiting_version is None:
        return None
    if watched.last_version != watched.waiting_version:
        # A change waits to hold for QUIET seconds.
        return FAST_CHECK
    steps = int((now - watched.changed_at) // DECAY_STEP)
    return min(SLOW_CHECK, FAST_CHECK * 2**steps)
