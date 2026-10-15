# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical task and curriculum embedding graph over existing curation outputs."""

import json
import os
from dataclasses import dataclass, replace
from functools import partial

import numpy as np
from finestore.schema import arrow_schema
from marin.execution.artifact import Artifact
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.inference.openai_chat import OpenAIChatClient
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.chat_requests import chat_output
from taskcompendium.pipeline.curriculum_alignment import (
    CurriculumAnchor,
    CurriculumVectors,
)
from taskcompendium.pipeline.semantics import (
    DESCRIPTION_PROMPT_VERSION,
    DescriptionModel,
    TaskInputFile,
    describe_tasks,
    prepare_semantic_inputs,
)
from taskcompendium.pipeline.task_alignment import align_tasks
from taskcompendium.pipeline.task_embeddings import (
    EmbeddingSpec,
    TextEmbedder,
    embed_texts,
    task_text_records,
    text_record,
)
from zephyr.dataset import Dataset
from zephyr.readers import load_parquet

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_REQUEST_TIMEOUT, resolve_glm_base_url
from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.curriculum import curriculum_anchors, curriculum_nodes
from experiments.post_training.task_curriculum.models import CurriculumCatalog

INPUTS_VERSION = "2026.10.14.1"
DESCRIPTIONS_VERSION = "2026.10.14.5"
EMBEDDINGS_VERSION = "2026.10.14.4"
ALIGNMENT_VERSION = "2026.10.14.3"


class SemanticArtifact(Artifact):
    paths: list[str]


@dataclass(frozen=True)
class CurationEmbeddingStages:
    inputs: ArtifactStep[SemanticArtifact]
    descriptions: ArtifactStep[SemanticArtifact]
    task_embeddings: ArtifactStep[SemanticArtifact]
    curriculum_embeddings: ArtifactStep[SemanticArtifact]
    assignments: ArtifactStep[SemanticArtifact]


@dataclass(frozen=True)
class DescriptionEndpoint:
    base_url: str | None
    relay_job: str
    concurrency: int = 16


def _prepare(
    runtime: CampaignRuntime,
    *,
    curated_roots: dict[str, str],
    population: str,
    sample_size: int | None,
    seed: int,
    output_root: str,
) -> SemanticArtifact:
    sources = []
    for source_id, root in curated_roots.items():
        manifest = json.loads((StoragePath(root) / "manifest.json").read_text())
        view = "final" if population == "admitted" else "normalize"
        paths = sorted(str(path) for path in (StoragePath(root) / f"{view}/*.parquet").glob())
        if not paths:
            raise ValueError(f"Curation source {source_id} has no {view} parquet files")
        sources.extend(
            TaskInputFile(source_id, canonical_sha256(manifest), path, manifest["intended_use"], population)
            for path in paths
        )
    paths = prepare_semantic_inputs(
        runtime.context, sources=sources, output_root=output_root, sample_size=sample_size, seed=seed
    )
    return SemanticArtifact(path=output_root, paths=paths)


def _describe(
    runtime: CampaignRuntime,
    endpoint: DescriptionEndpoint,
    *,
    inputs_root: str,
    model: DescriptionModel,
    prompt_version: str,
    cache_root: str,
    output_root: str,
) -> SemanticArtifact:
    assert prompt_version == DESCRIPTION_PROMPT_VERSION
    base_url = endpoint.base_url or resolve_glm_base_url(endpoint.relay_job)
    submit = partial(
        chat_output,
        OpenAIChatClient(base_url, os.environ[GLM_BULK_TOKEN_ENV], timeout=GLM_REQUEST_TIMEOUT),
        max_concurrent=endpoint.concurrency,
    )
    paths = describe_tasks(
        runtime.context,
        input_paths=SemanticArtifact.raw_load(inputs_root).paths,
        model=model,
        submit=submit,
        cache_root=cache_root,
        output_root=output_root,
    )
    return SemanticArtifact(path=output_root, paths=paths)


def _embed_tasks(
    runtime: CampaignRuntime,
    embedder: TextEmbedder,
    *,
    descriptions_root: str,
    spec: EmbeddingSpec,
    cache_root: str,
    output_root: str,
) -> SemanticArtifact:
    texts = (
        Dataset.from_list(SemanticArtifact.raw_load(descriptions_root).paths)
        .load_parquet()
        .flat_map(partial(task_text_records, spec=spec))
    )
    paths = embed_texts(
        runtime.context, texts=texts, spec=spec, embedder=embedder, cache_root=cache_root, output_root=output_root
    )
    return SemanticArtifact(path=output_root, paths=paths)


def _embed_curriculum(
    runtime: CampaignRuntime,
    embedder: TextEmbedder,
    *,
    catalog_root: str,
    spec: EmbeddingSpec,
    cache_root: str,
    output_root: str,
) -> SemanticArtifact:
    catalog = CurriculumCatalog.model_validate_json((StoragePath(catalog_root) / "curriculum.json").read_text())
    anchors = curriculum_anchors(catalog)
    anchor_result = runtime.context.execute(
        Dataset.from_list([anchors])
        .flat_map(lambda batch: [anchor.model_dump(mode="json") for anchor in batch])
        .write_parquet(
            str(StoragePath(output_root) / "anchors/part-{shard:05d}.parquet"), schema=arrow_schema(CurriculumAnchor)
        )
    )
    # The catalog fits in memory. One input shard keeps inference batches full;
    # a shard per anchor would force singleton model and cache requests.
    texts = Dataset.from_list([anchors]).flat_map(lambda batch: [text_record(anchor.text, spec) for anchor in batch])
    paths = embed_texts(
        runtime.context, texts=texts, spec=spec, embedder=embedder, cache_root=cache_root, output_root=output_root
    )
    (StoragePath(output_root) / "curriculum.json").write_text(catalog.model_dump_json())
    (StoragePath(output_root) / "nodes.json").write_text(json.dumps(curriculum_nodes(catalog), ensure_ascii=False))
    (StoragePath(output_root) / "anchor_paths.json").write_text(json.dumps(anchor_result.results))
    return SemanticArtifact(path=output_root, paths=paths)


def _align(
    runtime: CampaignRuntime,
    *,
    inputs_root: str,
    descriptions_root: str,
    task_vectors_root: str,
    curriculum_vectors_root: str,
    spec: EmbeddingSpec,
    cache_root: str,
    output_root: str,
) -> SemanticArtifact:
    task_spec = EmbeddingSpec.model_validate_json((StoragePath(task_vectors_root) / "embedding_spec.json").read_text())
    curriculum_spec = EmbeddingSpec.model_validate_json(
        (StoragePath(curriculum_vectors_root) / "embedding_spec.json").read_text()
    )
    if task_spec != spec or curriculum_spec != spec:
        raise ValueError("Task and curriculum embedding specifications differ")
    anchor_paths = json.loads((StoragePath(curriculum_vectors_root) / "anchor_paths.json").read_text())
    anchors = tuple(CurriculumAnchor.model_validate(row) for path in anchor_paths for row in load_parquet(path))
    vectors = {
        row["embedding_key"]: row["embedding"]
        for path in SemanticArtifact.raw_load(curriculum_vectors_root).paths
        for row in load_parquet(path)
    }
    curriculum = CurriculumVectors(
        anchors,
        np.asarray(
            [vectors[text_record(anchor.text, spec)["embedding_key"]] for anchor in anchors],
            dtype=np.float32,
        ),
    )
    paths = align_tasks(
        runtime.context,
        input_paths=SemanticArtifact.raw_load(inputs_root).paths,
        description_paths=SemanticArtifact.raw_load(descriptions_root).paths,
        curriculum=curriculum,
        curriculum_key=canonical_sha256(
            {"anchors": [anchor.anchor_id for anchor in anchors], "spec": spec.model_dump()}
        ),
        cache_root=cache_root,
        spec=spec,
        output_root=output_root,
    )
    (StoragePath(output_root) / "alignment.json").write_text(
        json.dumps(
            {
                "inputs_root": inputs_root,
                "descriptions_root": descriptions_root,
                "task_vectors_root": task_vectors_root,
                "curriculum_vectors_root": curriculum_vectors_root,
                "spec": spec.model_dump(mode="json"),
                "paths": paths,
            },
            ensure_ascii=False,
        )
    )
    return SemanticArtifact(path=output_root, paths=paths)


def _named(step: ArtifactStep[SemanticArtifact]) -> ArtifactStep[SemanticArtifact]:
    return replace(step, name=f"{step.name}-{step.fingerprint()[:16]}")


def curation_embedding_pipeline(
    *,
    curated: dict[str, ArtifactStep],
    curriculum: ArtifactStep,
    runtime: CampaignRuntime,
    description_model: DescriptionModel,
    endpoint: DescriptionEndpoint,
    spec: EmbeddingSpec,
    embedder: TextEmbedder,
    cache_root: str,
    population: str,
    sample_size: int | None,
    seed: int,
) -> CurationEmbeddingStages:
    """Bind independent task and curriculum embeddings before the assignment pass."""
    inputs = _named(
        apply(
            "rl-semantics/inputs",
            partial(_prepare, runtime),
            version=INPUTS_VERSION,
            artifact_type=SemanticArtifact,
            curated_roots=curated,
            population=population,
            sample_size=sample_size,
            seed=seed,
            output_root=OUT,
        )
    )
    descriptions = _named(
        apply(
            "rl-semantics/descriptions",
            partial(_describe, runtime, endpoint),
            version=DESCRIPTIONS_VERSION,
            artifact_type=SemanticArtifact,
            inputs_root=inputs,
            model=description_model,
            prompt_version=DESCRIPTION_PROMPT_VERSION,
            cache_root=str(StoragePath(cache_root) / "descriptions"),
            output_root=OUT,
        )
    )
    vectors = _named(
        apply(
            "rl-semantics/task-embeddings",
            partial(_embed_tasks, runtime, embedder),
            version=EMBEDDINGS_VERSION,
            artifact_type=SemanticArtifact,
            descriptions_root=descriptions,
            spec=spec,
            cache_root=str(StoragePath(cache_root) / "vectors"),
            output_root=OUT,
        )
    )
    anchors = _named(
        apply(
            "rl-semantics/curriculum-embeddings",
            partial(_embed_curriculum, runtime, embedder),
            version=EMBEDDINGS_VERSION,
            artifact_type=SemanticArtifact,
            catalog_root=curriculum,
            spec=spec,
            cache_root=str(StoragePath(cache_root) / "vectors"),
            output_root=OUT,
        )
    )
    assignments = _named(
        apply(
            "rl-semantics/assignments",
            partial(_align, runtime),
            version=ALIGNMENT_VERSION,
            artifact_type=SemanticArtifact,
            inputs_root=inputs,
            descriptions_root=descriptions,
            task_vectors_root=vectors,
            curriculum_vectors_root=anchors,
            spec=spec,
            cache_root=str(StoragePath(cache_root) / "vectors"),
            output_root=OUT,
        )
    )
    return CurationEmbeddingStages(inputs, descriptions, vectors, anchors, assignments)
