# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The capability catalog: one ``CapabilityIdea`` per catalog capability, and the records a run keeps of it.

A catalog file holds ``catalog_version``, ``curricula`` (each a ``curriculum`` with its ``subject_id``,
``subject_name`` and ``sections``, of which the ``capability`` sections are ideas) and an optional
``learning_progression`` of prerequisite edges.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from taskforge.content_hash import digest


@dataclass(frozen=True)
class CapabilityIdea:
    """One catalog capability with the learning-progression edges that point at it.

    ``capability`` and ``prerequisite_edges`` are source records shown to the model verbatim, so they
    stay as the catalog's JSON objects. ``capability_hash`` is the ``taskforge.content_hash.digest``
    of the capability record.
    """

    capability_id: str
    subject_id: str
    subject_name: str
    catalog_version: str
    capability: Mapping[str, object]
    prerequisite_edges: tuple[Mapping[str, object], ...]
    capability_hash: str


def load_capability_ideas(path: Path) -> dict[str, CapabilityIdea]:
    """Read a capability catalog (``catalog_version``, ``curricula``, ``learning_progression``) into ideas by id.

    Each idea carries the learning-progression edges whose ``dependent_id`` is that capability.
    """
    document = json.loads(path.read_text())
    version = document["catalog_version"]
    progression = document.get("learning_progression")
    edges: dict[str, list[Mapping[str, object]]] = {}
    if progression is not None:
        if progression["catalog_version"] != version:
            raise ValueError(f"{path}: learning_progression is for {progression['catalog_version']}, not {version}")
        for edge in progression["edges"]:
            edges.setdefault(edge["dependent_id"], []).append(edge)
    ideas: dict[str, CapabilityIdea] = {}
    for wrapper in document["curricula"]:
        curriculum = wrapper["curriculum"]
        for section in curriculum["sections"]:
            if section["kind"] != "capability":
                continue
            capability_id = section["id"]
            if capability_id in ideas:
                raise ValueError(f"{path}: duplicate capability id {capability_id}")
            ideas[capability_id] = CapabilityIdea(
                capability_id=capability_id,
                subject_id=curriculum["subject_id"],
                subject_name=curriculum["subject_name"],
                catalog_version=version,
                capability=section,
                prerequisite_edges=tuple(edges.get(capability_id, ())),
                capability_hash=digest(section),
            )
    return ideas


def capability_prompt_record(idea: CapabilityIdea) -> dict[str, object]:
    """The capability record plus its incoming learning-progression edges, as shown to the model."""
    return {
        **idea.capability,
        "learning_progression": {"catalog_version": idea.catalog_version, "edges": list(idea.prerequisite_edges)},
    }


def capability_idea_record(idea: CapabilityIdea) -> dict[str, object]:
    """A capability idea's catalog identifiers and the capability record its prompts show the models."""
    return {
        "capability_id": idea.capability_id,
        "subject_id": idea.subject_id,
        "subject_name": idea.subject_name,
        "catalog_version": idea.catalog_version,
        "capability_hash": idea.capability_hash,
        "record": capability_prompt_record(idea),
    }
