# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The capability idea that task generators take as input, and the JSON record that their prompts show."""

import re

from pydantic import BaseModel, ConfigDict

FILE_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
"""The form of a capability id or a task id that can name a file."""


class CapabilityIdea(BaseModel):
    """A catalog capability with the learning-progression edges leading into it.

    ``capability`` and ``prerequisite_edges`` hold the catalog's JSON objects as read, because the
    prompt shows them to the model verbatim. ``capability_hash`` is the ``taskforge.content_hash.digest``
    of ``capability``.
    """

    model_config = ConfigDict(frozen=True)

    capability_id: str
    subject_id: str
    subject_name: str
    catalog_version: str
    capability: dict[str, object]
    prerequisite_edges: tuple[dict[str, object], ...]
    capability_hash: str


def capability_idea_record(idea: CapabilityIdea) -> dict[str, object]:
    """Build the JSON object the prompt shows for ``idea``: its catalog identifiers and capability record.

    The record gets the prerequisite edges under ``learning_progression``.
    """
    return {
        "capability_id": idea.capability_id,
        "subject_id": idea.subject_id,
        "subject_name": idea.subject_name,
        "catalog_version": idea.catalog_version,
        "capability_hash": idea.capability_hash,
        "record": {
            **idea.capability,
            "learning_progression": {"catalog_version": idea.catalog_version, "edges": list(idea.prerequisite_edges)},
        },
    }
