# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project source declarations into the curation catalog inventory."""

from finestore.rl.curation import PipelineReference, SourceDeclaration

from experiments.post_training.task_curation.source import RlDataSource


def source_declaration(source: RlDataSource) -> SourceDeclaration:
    """Project a source declaration into catalog metadata without executing it."""
    info, pipeline = source.info, source.pipeline
    dataset = source.dataset
    invocation = None
    if pipeline is not None:
        invocation = PipelineReference(
            name=source.name,
            version=source.version,
            source=dataset.name if dataset else None,
            revision=dataset.revision if dataset else None,
            files=list(source.files),
        )
    tags = info.tags
    row = {
        "id": info.id,
        "name": source.name,
        "display_name": info.title,
        "origin": info.origin,
        "family": info.family,
        "tags": list(tags),
        "task_count": info.count,
        "notes": info.notes,
        "dataset_id": dataset.name if dataset else None,
        "dataset_revision": dataset.revision if dataset else None,
        "url": dataset.url if dataset else None,
        "type": next(
            (
                label
                for tag, label in (("rlvr", "RLVR"), ("agentic", "Agentic"), ("alignment", "Alignment"))
                if tag in tags
            ),
            "",
        ),
        "turns": next(
            (label for tag, label in (("single-turn", "Single-turn"), ("multi-turn", "Multi-turn")) if tag in tags), ""
        ),
        "is_benchmark": "benchmark" in tags,
        "kind": "Generator" if "generator" in tags else "Dataset",
        "status": "Excluded" if "excluded" in tags else "Available",
        "pipeline": invocation,
    }
    return SourceDeclaration.model_validate(row)
