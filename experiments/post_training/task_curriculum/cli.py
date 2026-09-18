# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Map reviewed semantic task keys to a curriculum."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import click
import numpy as np
from openai import OpenAI
from pydantic import BaseModel

from experiments.post_training.task_curriculum.cache import EmbeddingCache, cached_embeddings
from experiments.post_training.task_curriculum.mapping import curriculum_anchors, map_task_vectors
from experiments.post_training.task_curriculum.models import Curriculum, TaskAnnotation


def _read_jsonl[ModelT: BaseModel](path: Path, model: type[ModelT]) -> list[ModelT]:
    return [model.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, values: Sequence[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(value.model_dump_json() + "\n" for value in values))


def _openai_embeddings(model: str, batch_size: int) -> Callable[[Sequence[str]], np.ndarray]:
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
@click.option("--curriculum", type=click.Path(path_type=Path, exists=True), required=True, multiple=True)
@click.option("--cache", type=click.Path(path_type=Path), required=True)
@click.option("--embedding-model", required=True)
@click.option("--embedding-batch-size", type=click.IntRange(min=1), default=128, show_default=True)
@click.option("--mapping-batch-size", type=click.IntRange(min=1), default=4096, show_default=True)
@click.option("--top-k", type=click.IntRange(min=1), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def main(
    annotations: tuple[Path, ...],
    curriculum: tuple[Path, ...],
    cache: Path,
    embedding_model: str,
    embedding_batch_size: int,
    mapping_batch_size: int,
    top_k: int,
    output: Path,
) -> None:
    """Rank annotated tasks against one or more reviewed curricula."""
    task_rows = [row for path in annotations for row in _read_jsonl(path, TaskAnnotation)]
    curriculum_rows = [Curriculum.model_validate_json(path.read_text()) for path in curriculum]
    anchor_ids, anchor_texts = curriculum_anchors(curriculum_rows)
    task_texts = [row.key.embedding_text() for row in task_rows]
    embed = _openai_embeddings(embedding_model, embedding_batch_size)
    with EmbeddingCache(cache) as local_cache:
        task_vectors = cached_embeddings(local_cache, task_texts, embedding_model, embed)
        anchor_vectors = cached_embeddings(local_cache, anchor_texts, embedding_model, embed)
    mappings = map_task_vectors(
        task_rows,
        task_vectors,
        curriculum_rows,
        anchor_ids,
        anchor_vectors,
        embedding_model,
        top_k,
        mapping_batch_size,
    )
    _write_jsonl(output, mappings)


if __name__ == "__main__":
    main()
