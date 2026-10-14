# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the reviewed curriculum into semantic anchors and navigation metadata."""

from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.curriculum_alignment import AnchorKind, CurriculumAnchor, CurriculumFacet

from experiments.post_training.task_curriculum.models import CurriculumCatalog


def curriculum_anchors(catalog: CurriculumCatalog) -> list[CurriculumAnchor]:
    anchors = []
    for entry in catalog.curricula:
        curriculum = entry.curriculum
        facet = CurriculumFacet(entry.routing_facet.value)
        label = "Subject domain" if facet == CurriculumFacet.SUBJECT_DOMAIN else "Task mechanic"
        for root in curriculum.sections:
            if root.parent_id is not None:
                continue
            text = (
                f"{label}: {curriculum.subject_name}. {root.name}. {root.scope_text()}. "
                f"Includes: {'; '.join(root.includes)}."
            )
            value = {
                "kind": AnchorKind.SUBJECT,
                "subject_id": curriculum.subject_id,
                "capability_id": None,
                "facet": facet,
                "text": text,
            }
            anchors.append(CurriculumAnchor(anchor_id=canonical_sha256(value), **value))
        for capability in curriculum.capability_sections():
            for included in capability.includes:
                text = f"Task: {capability.name}. {capability.outcome}. Required operations: {included}."
                value = {
                    "kind": AnchorKind.CAPABILITY,
                    "subject_id": curriculum.subject_id,
                    "capability_id": capability.id,
                    "facet": facet,
                    "text": text,
                }
                anchors.append(CurriculumAnchor(anchor_id=canonical_sha256(value), **value))
    return list({anchor.anchor_id: anchor for anchor in anchors}.values())


def curriculum_nodes(catalog: CurriculumCatalog) -> list[dict]:
    nodes = []
    for entry in catalog.curricula:
        curriculum = entry.curriculum
        subject_node = f"subject:{curriculum.subject_id}"
        nodes.append(
            {
                "node_id": subject_node,
                "subject_id": curriculum.subject_id,
                "parent_id": None,
                "kind": "subject",
                "name": curriculum.subject_name,
                "description": "",
            }
        )
        for section in curriculum.sections:
            nodes.append(
                {
                    "node_id": section.id,
                    "subject_id": curriculum.subject_id,
                    "parent_id": section.parent_id or subject_node,
                    "kind": section.kind.value,
                    "name": section.name,
                    "description": section.scope_text(),
                }
            )
    return nodes
