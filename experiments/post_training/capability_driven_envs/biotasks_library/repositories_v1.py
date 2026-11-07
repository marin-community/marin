# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hydrate repository evidence for the BioTasks repositories/v1 authoring campaign."""

import hashlib

import fsspec
from pydantic import BaseModel, ConfigDict

from experiments.post_training.capability_driven_envs.biotasks_library.notebooks_v1 import (
    GeneratedPortfolio,
    SourceRecord,
    extract_source_text,
    generation_prompt,
    hydrate_ideas,
)
from experiments.post_training.capability_driven_envs.biotasks_library.pilot_first3 import Assessment, BioTaskIdea

MAX_REPOSITORY_TEXT_CHARACTERS = 80_000


class RepositoryCatalogSource(BaseModel):
    """One immutable repository evidence packet from the pinned issue-5 audit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repository: str
    revision: str
    source_group_id: str
    source_key: str
    source_sha256: str
    packet_object_uri: str
    primary_domain: str
    manual_topic: str
    source_type: str
    notebook_detection_count: int
    selected_document_count: int
    acquisition_status: str
    review: str
    release_status: str


class PacketDocument(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    format: str
    git_blob_sha1: str
    source_sha256: str
    source_bytes: int
    object_uri: str


class RepositoryPacket(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    repository: str
    revision: str
    source_id: str
    primary_domain: str
    manual_topic: str
    source_type: str
    notebook_detection_count: int
    notebook_count_complete: bool
    readme_sha256: str | None
    readme_bytes: int | None
    readme_object_uri: str | None
    documents: tuple[PacketDocument, ...]
    acquisition_errors: tuple[str, ...]


def read_bytes(uri: str) -> bytes:
    with fsspec.open(uri, "rb") as stream:
        return stream.read()


def load_packet(source: RepositoryCatalogSource) -> RepositoryPacket:
    """Load and verify one repository packet object."""
    raw = read_bytes(source.packet_object_uri)
    if hashlib.sha256(raw).hexdigest() != source.source_sha256:
        raise ValueError(f"repository packet digest mismatch for {source.repository}")
    return RepositoryPacket.model_validate_json(raw)


def _source_record(source: RepositoryCatalogSource, packet: RepositoryPacket) -> SourceRecord:
    return SourceRecord(
        canonical_document_key=source.source_key,
        document_key=source.source_key,
        fields=(source.primary_domain,),
        format="repository-evidence-packet",
        modalities=(),
        object_uri=source.packet_object_uri,
        operations=(),
        path=source.repository,
        repo=source.repository,
        source_bytes=(packet.readme_bytes or 0) + sum(document.source_bytes for document in packet.documents),
        source_ids=(source.source_group_id,),
        source_role="repository_evidence_lead",
        source_sha256=source.source_sha256,
        title=source.repository,
    )


def repository_source_text(source: RepositoryCatalogSource, packet: RepositoryPacket) -> str:
    """Render bounded README and scientific-document source without notebook outputs."""
    sections = [
        "# Repository evidence scope",
        f"Repository: {source.repository}",
        f"Pinned revision: {source.revision}",
        f"Provisional primary domain: {source.primary_domain}",
        f"Provisional topic: {source.manual_topic}",
        (
            "The packet contains a bounded README and at most three of the smallest detected notebook/literate "
            "documents. Absence from this packet does not establish absence from the repository."
        ),
    ]
    if packet.readme_object_uri and packet.readme_sha256:
        raw = read_bytes(packet.readme_object_uri)
        if hashlib.sha256(raw).hexdigest() != packet.readme_sha256:
            raise ValueError(f"README digest mismatch for {source.repository}")
        sections.extend(("\n# README", raw.decode("utf-8", errors="replace")))
    for document in packet.documents:
        raw = read_bytes(document.object_uri)
        record = SourceRecord(
            canonical_document_key=f"{source.source_key}:{document.path}",
            document_key=f"{source.source_key}:{document.path}",
            fields=(source.primary_domain,),
            format=document.format,
            modalities=(),
            object_uri=document.object_uri,
            operations=(),
            path=document.path,
            repo=source.repository,
            source_bytes=document.source_bytes,
            source_ids=(source.source_group_id,),
            source_role="repository_evidence_document",
            source_sha256=document.source_sha256,
            title=document.path,
        )
        sections.extend((f"\n# Evidence document: {document.path}", extract_source_text(record, raw)))
    text = "\n".join(sections)
    if len(text) <= MAX_REPOSITORY_TEXT_CHARACTERS:
        return text
    head = MAX_REPOSITORY_TEXT_CHARACTERS * 3 // 4
    tail = MAX_REPOSITORY_TEXT_CHARACTERS - head
    return f"{text[:head]}\n\n[... deterministic repository packet truncation ...]\n\n{text[-tail:]}"


def repository_generation_prompt(source: RepositoryCatalogSource, packet: RepositoryPacket, source_text: str) -> str:
    """Render the generic objective-authoring prompt with repository-specific scope limits."""
    record = _source_record(source, packet)
    prompt = generation_prompt(record, source_text)
    return (
        prompt + "\n\nRepository-specific rule: treat this packet as a discovery lead, not proof that all repository "
        "behavior was inspected. Ground every idea in the supplied README or evidence documents. Do not invent "
        "package outputs, "
        "datasets, APIs, biological claims, or runtime behavior absent from the packet. Return no_objective_task when "
        "the evidence only demonstrates infrastructure, visualization mechanics, or generic software usage."
    )


def hydrate_repository_ideas(
    source: RepositoryCatalogSource,
    packet: RepositoryPacket,
    portfolio: GeneratedPortfolio,
) -> list[BioTaskIdea]:
    """Assign stable identities using the repository packet digest and project split group."""
    return hydrate_ideas(_source_record(source, packet), portfolio)


def unavailable_portfolio(source: RepositoryCatalogSource) -> GeneratedPortfolio:
    """Return an explicit null disposition when no source bytes were recoverable."""
    return GeneratedPortfolio(
        source_summary=f"No bounded source evidence was recoverable for {source.repository} at its pinned revision.",
        disposition="no_objective_task",
        null_reason="The repository packet contains no README or scientific-document bytes to ground a task.",
        null_assessment=Assessment(
            information_sufficiency=1,
            biological_substance=1,
            scientific_validity=2,
            objective_verifiability=1,
            hack_resistance=2,
            practical_usefulness=1,
            blocking_issues=("No source content was recoverable.",),
            recommended_action="reject",
            rationale="Generating a task without source evidence would require unsupported invention.",
            confidence=0.99,
        ),
        ideas=(),
    )
