# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Tell a `doc` session that a person saved an edit to its `doc` artifact.

Loom delivers a submitted review to the agent itself, but a person's edit to an artifact's source only adds a
revision. The `doc-edits` watch (`infra/loom/watches/doc_edits.py`) reads each `doc` session's artifact and sends the
session one message for the person's revisions since the last one it looked at. Loom records whether the agent or a
person wrote each revision, so the agent's own writes never count.
"""

DOC_ARTIFACT = "doc"
EDITED_MESSAGE = (
    "A person edited the doc: revision {rev} of the artifact `doc`. Read it with artifact_get, treat the change as a "
    "request, and act on it."
)
# Loom's author of a revision that a person saved in the artifact editor.
PERSON_AUTHOR = "user"


def edit_message(versions: list[dict], seen_rev: int) -> tuple[int, str | None]:
    """The newest revision, and the message for a person's revisions after `seen_rev`, if any.

    versions are the artifact's revisions, each with its `rev` and `author`.
    """
    latest = max((version["rev"] for version in versions), default=seen_rev)
    edited = [version["rev"] for version in versions if version["author"] == PERSON_AUTHOR and version["rev"] > seen_rev]
    if not edited:
        return latest, None
    return latest, EDITED_MESSAGE.format(rev=max(edited))
