# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical task and curriculum embedding graph over existing curation outputs."""

import json
import os
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import partial

import numpy as np
from finestore.rl.curation import SourceDeclaration
from finestore.rl.curriculum import AlignmentRelease, AlignmentSource, EmbeddingSpec
from finestore.schema import arrow_schema
from marin.execution.artifact import Artifact
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.inference.openai_chat import OpenAIChatClient
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.alignment_publication import publish_alignments
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
    prepare_instruction_texts,
    prepare_semantic_inputs,
)
from taskcompendium.pipeline.task_alignment import align_tasks
from taskcompendium.pipeline.task_embeddings import (
    TextEmbedder,
    embed_texts,
    task_text_records,
    text_record,
)
from zephyr.dataset import Dataset
from zephyr.readers import load_parquet

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_REQUEST_TIMEOUT, resolve_glm_base_url
from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curriculum.models import CurriculumCatalog

INPUTS_VERSION = "2026.10.14.1"
DESCRIPTIONS_VERSION = "2026.10.15.1"
INSTRUCTION_TEXTS_VERSION = "2026.10.15.1"
EMBEDDINGS_VERSION = "2026.10.15.2"
CURRICULUM_EMBEDDINGS_VERSION = "2026.10.15.2"
ALIGNMENT_VERSION = "2026.10.15.1"
PUBLICATION_VERSION = "2026.10.14.1"


class SemanticArtifact(Artifact):
    paths: list[str]


@dataclass(frozen=True)
class CurationEmbeddingStages:
    inputs: ArtifactStep[SemanticArtifact]
    descriptions: ArtifactStep[SemanticArtifact]
    task_embeddings: ArtifactStep[SemanticArtifact]
    curriculum_embeddings: ArtifactStep[SemanticArtifact]
    assignments: ArtifactStep[SemanticArtifact]
    publication: ArtifactStep[SemanticArtifact]


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
    # Resolve credentials during execution so they never enter the artifact identity.
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


def _instructions(runtime: CampaignRuntime, *, inputs_root: str, output_root: str) -> SemanticArtifact:
    paths = prepare_instruction_texts(
        runtime.context, input_paths=SemanticArtifact.raw_load(inputs_root).paths, output_root=output_root
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
    anchors = catalog.embedding_anchors()
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
    (StoragePath(output_root) / "nodes.json").write_text(
        json.dumps([node.model_dump(mode="json") for node in catalog.atlas_nodes()], ensure_ascii=False)
    )
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
    curriculum_key = canonical_sha256({"anchors": [anchor.anchor_id for anchor in anchors], "spec": spec.model_dump()})
    paths = align_tasks(
        runtime.context,
        input_paths=SemanticArtifact.raw_load(inputs_root).paths,
        description_paths=SemanticArtifact.raw_load(descriptions_root).paths,
        vector_paths=SemanticArtifact.raw_load(task_vectors_root).paths,
        curriculum=curriculum,
        curriculum_key=curriculum_key,
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
                "curriculum_key": curriculum_key,
                "spec": spec.model_dump(mode="json"),
                "paths": paths,
            },
            ensure_ascii=False,
        )
    )
    return SemanticArtifact(path=output_root, paths=paths)


def _named(step: ArtifactStep[SemanticArtifact]) -> ArtifactStep[SemanticArtifact]:
    return replace(step, name=f"{step.name}-{step.fingerprint()[:16]}")


def _publish(
    runtime: CampaignRuntime,
    *,
    alignment_root: str,
    curated_roots: dict[str, str],
    inventory: list[SourceDeclaration],
    population: str,
    sample_size: int | None,
    output_root: str,
) -> SemanticArtifact:
    alignment = json.loads((StoragePath(alignment_root) / "alignment.json").read_text())
    data = publish_alignments(
        runtime.context,
        input_paths=SemanticArtifact.raw_load(alignment["inputs_root"]).paths,
        alignment_paths=SemanticArtifact.raw_load(alignment_root).paths,
        output_root=output_root,
    )
    counts = {row.source_id: row for row in data.sources}
    if set(counts) - curated_roots.keys():
        raise ValueError("Alignment contains sources outside the selected corpus")
    if curated_roots.keys() - {row.id for row in inventory}:
        raise ValueError("Selected sources are absent from the published inventory")
    sources = []
    for source_id, root in curated_roots.items():
        manifest = json.loads((StoragePath(root) / "manifest.json").read_text())
        source_key = canonical_sha256(manifest)
        row = counts.get(source_id) or AlignmentSource(
            source_id=source_id,
            source_key=source_key,
            population=population,
            intended_use=manifest["intended_use"],
            task_count=0,
            ready_count=0,
            insufficient_context_count=0,
            unique_content_count=0,
        )
        if (row.source_key, row.population, row.intended_use) != (source_key, population, manifest["intended_use"]):
            raise ValueError(f"Published source lineage differs from the selected manifest: {source_id}")
        if population == "converted" and sample_size is None and row.task_count != manifest["converted_rows"]:
            raise ValueError(f"Alignment does not cover the complete converted source: {source_id}")
        sources.append(
            row.model_copy(
                update={
                    "curation_root": root,
                    "input_rows": manifest.get("input_rows"),
                    "converted_rows": manifest.get("converted_rows"),
                    "rejections": manifest.get("rejections", {}),
                }
            )
        )
    curriculum_root = alignment["curriculum_vectors_root"]
    lineage = {
        key: alignment[key]
        for key in ("inputs_root", "descriptions_root", "task_vectors_root", "curriculum_vectors_root")
    }
    lineage["alignment_root"] = alignment_root
    release = AlignmentRelease(
        release_id=canonical_sha256(
            {
                "lineage": lineage,
                "sources": [row.model_dump() for row in sources],
                "inventory": [row.model_dump() for row in inventory],
            }
        ),
        created_at=datetime.now(UTC).isoformat(),
        curriculum_key=alignment["curriculum_key"],
        population=population,
        sample_size=sample_size,
        task_count=sum(row.task_count for row in sources),
        ready_count=sum(row.ready_count for row in sources),
        insufficient_context_count=sum(row.insufficient_context_count for row in sources),
        unique_content_count=data.unique_content_count,
        unique_embedding_count=data.unique_embedding_count,
        sources=sources,
        inventory=inventory,
        nodes=json.loads((StoragePath(curriculum_root) / "nodes.json").read_text()),
        paths=data.paths,
        lineage=lineage,
        embedding_spec=alignment["spec"],
    )
    (StoragePath(output_root) / "release.json").write_text(release.model_dump_json(indent=2))
    return SemanticArtifact(path=output_root, paths=data.paths)


def curation_embedding_pipeline(
    *,
    curated: dict[str, ArtifactStep],
    curriculum: ArtifactStep,
    runtime: CampaignRuntime,
    description_model: DescriptionModel | None,
    endpoint: DescriptionEndpoint,
    spec: EmbeddingSpec,
    embedder: TextEmbedder,
    cache_root: str,
    population: str,
    sample_size: int | None,
    seed: int,
    inventory: list[SourceDeclaration],
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
        if description_model is not None
        else apply(
            "rl-semantics/instruction-texts",
            partial(_instructions, runtime),
            version=INSTRUCTION_TEXTS_VERSION,
            artifact_type=SemanticArtifact,
            inputs_root=inputs,
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
            version=CURRICULUM_EMBEDDINGS_VERSION,
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
            output_root=OUT,
        )
    )
    publication = _named(
        apply(
            "rl-semantics/publication",
            partial(_publish, runtime),
            version=PUBLICATION_VERSION,
            artifact_type=SemanticArtifact,
            alignment_root=assignments,
            curated_roots=curated,
            inventory=inventory,
            population=population,
            sample_size=sample_size,
            output_root=OUT,
        )
    )
    return CurationEmbeddingStages(inputs, descriptions, vectors, anchors, assignments, publication)
