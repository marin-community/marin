# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Describe, embed and align curated RL tasks with the reviewed curriculum."""

import json
from contextlib import ExitStack
from pathlib import Path

import click
from finestore.rl.curation import SourceDeclaration
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from marin.execution.lazy import ArtifactStep, run
from marin.processing.embeddings.harrier import HARRIER_DIMENSIONS, HARRIER_MODEL, HARRIER_REVISION, HarrierEmbedder
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.semantics import DescriptionModel
from taskcompendium.pipeline.task_embeddings import EmbeddingSpec
from zephyr.context import ZephyrContext

from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB, GLM_MODEL
from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.embedding import (
    INPUTS_VERSION,
    DescriptionEndpoint,
    curation_embedding_pipeline,
)
from experiments.post_training.task_curation.local import local_artifact_prefix
from experiments.post_training.task_curation.source_metadata import source_declaration
from experiments.post_training.task_curation.sources import all_sources
from experiments.post_training.task_curriculum.catalog_artifact import TASK_CURRICULUM


@click.command(help=__doc__)
@click.option("--curated", multiple=True, nargs=2, metavar="SOURCE_ID ROOT")
@click.option("--corpus", help="Complete corpus.json produced by convert_corpus; alternative to --curated.")
@click.option("--catalog-root", help="Immutable curriculum artifact directory; defaults to TASK_CURRICULUM.")
@click.option("--cache-root", required=True, help="Persistent, region-local artifact and inference cache.")
@click.option("--population", type=click.Choice(["admitted", "converted"]), required=True)
@click.option("--sample-size", type=click.IntRange(min=1), help="Seeded sample per source; omit for all tasks.")
@click.option("--seed", type=int, default=0)
@click.option("--model", default=GLM_MODEL, show_default=True)
@click.option("--model-revision", required=True)
@click.option("--base-url")
@click.option("--relay-job", default=DEFAULT_GLM_RELAY_JOB)
@click.option("--review-concurrency", type=click.IntRange(1, 64), default=16)
@click.option("--embedding-backend", type=click.Choice(["cpu", "gpu"]), required=True)
@click.option("--compute-dtype", type=click.Choice(["float32", "bfloat16", "float16"]), required=True)
@click.option(
    "--max-tokens",
    type=click.IntRange(min=1),
    default=512,
    show_default=True,
    help="Embedding input limit, including the instruction prefix. Full task instructions go to the describer.",
)
@click.option("--max-workers", type=click.IntRange(min=1), default=2)
@click.option("--worker-cpu", type=click.FloatRange(min=0, min_open=True), default=4, show_default=True)
@click.option("--worker-memory", default="8g")
@click.option(
    "--stage",
    type=click.Choice(
        ["inputs", "descriptions", "task_embeddings", "curriculum_embeddings", "assignments", "publication"]
    ),
    default="publication",
    show_default=True,
    help="Run this artifact stage and its dependencies.",
)
@click.option("--run", "do_run", is_flag=True, help="Execute the graph; otherwise print its plan.")
def main(
    curated,
    corpus,
    catalog_root,
    cache_root,
    population,
    sample_size,
    seed,
    model,
    model_revision,
    base_url,
    relay_job,
    review_concurrency,
    embedding_backend,
    compute_dtype,
    max_tokens,
    max_workers,
    worker_cpu,
    worker_memory,
    stage,
    do_run,
):
    if bool(curated) == bool(corpus):
        raise click.UsageError("Supply either --curated or --corpus")
    inventory = [source_declaration(source) for source in all_sources().values()]
    if corpus:
        selected = json.loads(StoragePath(corpus).read_text())
        inventory = [SourceDeclaration.model_validate(row) for row in selected["inventory"]]
        if population != selected["population"]:
            raise click.UsageError("Requested population differs from the corpus manifest")
        curated = []
        for row in selected["sources"]:
            manifest = json.loads((StoragePath(row["root"]) / "manifest.json").read_text())
            if canonical_sha256(manifest) != row["source_key"]:
                raise ValueError(f"Corpus source manifest changed: {row['source_id']}")
            curated.append((row["source_id"], row["root"]))
    if len({name for name, _ in curated}) != len(curated):
        raise click.UsageError("Each curated source ID must occur once")
    spec = EmbeddingSpec(
        model=HARRIER_MODEL,
        revision=HARRIER_REVISION,
        tokenizer_revision=HARRIER_REVISION,
        dimensions=HARRIER_DIMENSIONS,
        max_tokens=max_tokens,
        compute_dtype=compute_dtype,
    )
    embedder = HarrierEmbedder(
        model=spec.model,
        revision=spec.revision,
        tokenizer_revision=spec.tokenizer_revision,
        dimensions=spec.dimensions,
        max_tokens=spec.max_tokens,
        compute_dtype=spec.compute_dtype,
        device="cpu" if embedding_backend == "cpu" else "cuda",
        cpu_threads=max(1, int(worker_cpu)),
    )
    runtime = CampaignRuntime()
    with ExitStack() as scope:
        if not StoragePath(cache_root).is_remote:
            cache_root = str(Path(cache_root).expanduser().resolve())
            scope.enter_context(local_artifact_prefix(cache_root))
        adopted = {}
        for source_id, root in curated:
            manifest = json.loads((StoragePath(root) / "manifest.json").read_text())
            digest = canonical_sha256(manifest)
            adopted[source_id] = ArtifactStep.adopt(f"rl-semantics/source-{digest[:16]}", INPUTS_VERSION, source=root)
        curriculum = TASK_CURRICULUM
        if catalog_root:
            catalog = json.loads((StoragePath(catalog_root) / "curriculum.json").read_text())
            digest = canonical_sha256(catalog)
            curriculum = ArtifactStep.adopt(
                f"rl-semantics/curriculum-{digest[:16]}", INPUTS_VERSION, source=catalog_root
            )
        stages = curation_embedding_pipeline(
            curated=adopted,
            curriculum=curriculum,
            runtime=runtime,
            description_model=DescriptionModel(model, model_revision),
            endpoint=DescriptionEndpoint(base_url, relay_job, review_concurrency),
            spec=spec,
            embedder=embedder,
            cache_root=cache_root,
            population=population,
            sample_size=sample_size,
            seed=seed,
            inventory=inventory,
        )
        plan = {
            name: getattr(stages, name).path()
            for name in (
                "inputs",
                "descriptions",
                "task_embeddings",
                "curriculum_embeddings",
                "assignments",
                "publication",
            )
        }
        click.echo(
            json.dumps(
                {
                    "stages": plan,
                    "spec": spec.model_dump(mode="json"),
                    "sample_size": sample_size,
                    "population": population,
                    "stage": stage,
                },
                indent=2,
            )
        )
        if not do_run:
            return
        client = None
        if not StoragePath(cache_root).is_remote or embedding_backend == "gpu":
            client = LocalClient()
            scope.callback(client.shutdown)
            scope.enter_context(set_current_client(client))
        context = scope.enter_context(
            ZephyrContext(
                client=client,
                max_workers=max_workers,
                resources=ResourceConfig(cpu=worker_cpu, ram=worker_memory),
                chunk_storage_prefix=str(StoragePath(cache_root) / ".zephyr/embeddings"),
                name="rl-task-embeddings",
            )
        )
        scope.enter_context(runtime.activate(context))
        result = run(getattr(stages, stage), max_concurrent=1)[0]
        click.echo(json.dumps({"stage": stage, "root": result.path, "paths": result.paths}, indent=2))


if __name__ == "__main__":
    main()
