# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load the capability catalog as a ``CapabilityIdea`` per capability, and build each idea's prompt record.

A catalog file holds a ``catalog_version``, a list of ``curricula`` and an optional
``learning_progression`` of prerequisite edges. Each entry in ``curricula`` wraps a ``curriculum``
with a ``subject_id``, a ``subject_name`` and ``sections``. The loader turns every section of kind
``capability`` into an idea and attaches the prerequisite edges leading into it.
"""

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from taskforge.content_hash import digest

CAPABILITY_KIND = "capability"
EXAMPLE_CATALOG_FILE = "example_catalog.json"
FILE_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class Section(BaseModel):
    """A curriculum section; fields other than ``kind`` and ``id`` pass through verbatim to the prompt."""

    model_config = ConfigDict(extra="allow", frozen=True)

    kind: str
    id: str


class Curriculum(BaseModel):
    model_config = ConfigDict(frozen=True)

    subject_id: str
    subject_name: str
    sections: tuple[Section, ...]


class CurriculumEntry(BaseModel):
    """An element of ``curricula``, wrapping a curriculum in a ``curriculum`` key."""

    model_config = ConfigDict(frozen=True)

    curriculum: Curriculum


class ProgressionEdge(BaseModel):
    """A prerequisite edge into ``dependent_id``; its other fields pass through verbatim to the prompt."""

    model_config = ConfigDict(extra="allow", frozen=True)

    dependent_id: str


class Progression(BaseModel):
    model_config = ConfigDict(frozen=True)

    catalog_version: str
    edges: tuple[ProgressionEdge, ...]


class CatalogDocument(BaseModel):
    """A capability catalog file."""

    model_config = ConfigDict(frozen=True)

    catalog_version: str
    curricula: tuple[CurriculumEntry, ...]
    learning_progression: Progression | None = None


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


def example_catalog_path() -> Path:
    """The path of this package's example catalog, ``data/example_catalog.json``."""
    return Path(__file__).resolve().parent / "data" / EXAMPLE_CATALOG_FILE


def load_capability_ideas(path: Path) -> dict[str, CapabilityIdea]:
    """Read a capability catalog into ideas by id.

    Each idea carries the learning-progression edges whose ``dependent_id`` is that capability. A
    duplicate capability id, or a learning progression from another catalog version, raises
    ``ValueError``.
    """
    document = CatalogDocument.model_validate_json(path.read_bytes())
    version = document.catalog_version
    progression = document.learning_progression
    edges: dict[str, list[dict[str, object]]] = {}
    if progression is not None:
        if progression.catalog_version != version:
            raise ValueError(f"{path}: learning_progression is for {progression.catalog_version}, not {version}")
        for edge in progression.edges:
            edges.setdefault(edge.dependent_id, []).append(edge.model_dump())
    ideas: dict[str, CapabilityIdea] = {}
    for entry in document.curricula:
        curriculum = entry.curriculum
        for section in curriculum.sections:
            if section.kind != CAPABILITY_KIND:
                continue
            if section.id in ideas:
                raise ValueError(f"{path}: duplicate capability id {section.id}")
            record = section.model_dump()
            ideas[section.id] = CapabilityIdea(
                capability_id=section.id,
                subject_id=curriculum.subject_id,
                subject_name=curriculum.subject_name,
                catalog_version=version,
                capability=record,
                prerequisite_edges=tuple(edges.get(section.id, ())),
                capability_hash=digest(record),
            )
    return ideas


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
