# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The capability catalog: one ``CapabilityIdea`` per catalog capability, and the records a run keeps of it.

A catalog file holds ``catalog_version``, ``curricula`` (each a ``curriculum`` with its ``subject_id``,
``subject_name`` and ``sections``, of which the ``capability`` sections are ideas) and an optional
``learning_progression`` of prerequisite edges.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict
from taskforge.content_hash import digest

CAPABILITY_KIND = "capability"
EXAMPLE_CATALOG_FILE = "example_catalog.json"


class Section(BaseModel):
    """A curriculum section; every field beyond ``kind`` and ``id`` is kept verbatim for the prompt."""

    model_config = ConfigDict(extra="allow", frozen=True)

    kind: str
    id: str


class Curriculum(BaseModel):
    model_config = ConfigDict(frozen=True)

    subject_id: str
    subject_name: str
    sections: tuple[Section, ...]


class CurriculumEntry(BaseModel):
    """One element of ``curricula``: the file wraps each curriculum in a ``curriculum`` key."""

    model_config = ConfigDict(frozen=True)

    curriculum: Curriculum


class ProgressionEdge(BaseModel):
    """A prerequisite edge into ``dependent_id``; its other fields are kept verbatim for the prompt."""

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
    """One catalog capability with the learning-progression edges that point at it.

    ``capability`` and ``prerequisite_edges`` are source records shown to the model verbatim, so they
    stay as the catalog's JSON objects. ``capability_hash`` is the ``taskforge.content_hash.digest``
    of the capability record.
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
    """The catalog that ships with this package, ``data/example_catalog.json``."""
    return Path(__file__).resolve().parent / "data" / EXAMPLE_CATALOG_FILE


def load_capability_ideas(path: Path) -> dict[str, CapabilityIdea]:
    """Read a capability catalog into ideas by id.

    Each idea carries the learning-progression edges whose ``dependent_id`` is that capability.
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
    """A capability idea's catalog identifiers and the capability record its prompts show the models."""
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
