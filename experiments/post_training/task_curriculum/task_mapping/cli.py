# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Map reviewed semantic task keys to a curriculum."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import click
import numpy as np
from openai import OpenAI
from zephyr.readers import load_jsonl
from zephyr.writers import write_jsonl_file

from experiments.post_training.task_curriculum.models import CurriculumCatalog, RoutingFacet
from experiments.post_training.task_curriculum.task_mapping.cache import EmbeddingCache, cached_embeddings
from experiments.post_training.task_curriculum.task_mapping.embedding import (
    MappingInputs,
    graph_anchors,
    map_task_vectors,
    section_anchors,
)
from experiments.post_training.task_curriculum.task_mapping.models import AssignmentAnchor, TaskAnnotation


def _openai_embedding_function(model: str, batch_size: int) -> Callable[[Sequence[str]], np.ndarray]:
    def embed(texts: Sequence[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        with OpenAI(timeout=300.0) as client:
            for start in range(0, len(texts), batch_size):
                response = client.embeddings.create(model=model, input=list(texts[start : start + batch_size]))
                vectors.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
        return np.asarray(vectors, dtype=np.float32)

    return embed


@click.command()
@click.option("--annotations", type=click.Path(path_type=Path, exists=True), required=True, multiple=True)
@click.option("--catalog", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--assignment-anchors", type=click.Path(path_type=Path, exists=True), multiple=True)
@click.option("--cache", type=click.Path(path_type=Path), required=True)
@click.option("--embedding-model", required=True)
@click.option("--embedding-batch-size", type=click.IntRange(min=1), default=128, show_default=True)
@click.option("--mapping-batch-size", type=click.IntRange(min=1), default=4096, show_default=True)
@click.option("--top-k", type=click.IntRange(min=1), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def main(
    annotations: tuple[Path, ...],
    catalog: Path,
    assignment_anchors: tuple[Path, ...],
    cache: Path,
    embedding_model: str,
    embedding_batch_size: int,
    mapping_batch_size: int,
    top_k: int,
    output: Path,
) -> None:
    """Rank annotated tasks within each graph in the canonical curriculum catalog."""
    task_rows = [TaskAnnotation.model_validate(row) for path in annotations for row in load_jsonl(str(path))]
    anchor_rows = [AssignmentAnchor.model_validate(row) for path in assignment_anchors for row in load_jsonl(str(path))]
    catalog_row = CurriculumCatalog.model_validate_json(catalog.read_bytes())
    graph_anchor_rows = graph_anchors(catalog_row, anchor_rows)
    section_anchor_rows = section_anchors(catalog_row, anchor_rows)
    embed = _openai_embedding_function(embedding_model, embedding_batch_size)
    with EmbeddingCache(cache) as local_cache:
        membership_vectors = {
            facet: cached_embeddings(
                local_cache,
                [row.key.membership_text(facet) for row in task_rows],
                embedding_model,
                embed,
            )
            for facet in RoutingFacet
        }
        operation_vectors = cached_embeddings(
            local_cache,
            [row.key.operation_text() for row in task_rows],
            embedding_model,
            embed,
        )
        graph_vectors = cached_embeddings(
            local_cache,
            [anchor.text for anchor in graph_anchor_rows],
            embedding_model,
            embed,
        )
        section_vectors = cached_embeddings(
            local_cache,
            [anchor.text for anchor in section_anchor_rows],
            embedding_model,
            embed,
        )
    mappings = map_task_vectors(
        MappingInputs(
            annotations=task_rows,
            membership_vectors=membership_vectors,
            operation_vectors=operation_vectors,
            catalog=catalog_row,
            graph_anchor_rows=graph_anchor_rows,
            graph_anchor_vectors=graph_vectors,
            section_anchor_rows=section_anchor_rows,
            section_anchor_vectors=section_vectors,
            embedding_model=embedding_model,
            top_k=top_k,
            row_batch_size=mapping_batch_size,
        )
    )
    write_jsonl_file((mapping.model_dump(mode="json") for mapping in mappings), str(output))


if __name__ == "__main__":
    main()
