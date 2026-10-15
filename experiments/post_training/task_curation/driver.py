# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan RL curation, then run source pipelines locally or on Iris."""

import json
import os
from contextlib import ExitStack
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import cast

import click
from finestore.rl.curation import SourceDeclaration
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from iris.cluster.client.job_info import get_job_info
from marin.execution.lazy import ArtifactStep, run
from marin.processing.embeddings.harrier import HARRIER_DIM, HARRIER_REPO, HARRIER_REVISION, HarrierConfig, load_harrier
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.machine import Backend, MachineFactory
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.assessment import AssessmentSampling
from taskcompendium.pipeline.assessment_review import ASSESSMENT_RUBRIC, AssessmentModel
from taskcompendium.pipeline.chat_requests import MAX_DIRECT_CONCURRENT_REQUESTS
from taskcompendium.pipeline.curriculum.models import EmbeddingSpec
from taskcompendium.pipeline.query_cache import compact_query_cache
from taskcompendium.pipeline.semantics import DescriptionModel
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from taskcompendium.pipeline.source_quality import MAX_SOURCE_QUALITY_SAMPLE_SIZE, SOURCE_QUALITY_SAMPLE_SIZE
from taskcompendium.pipeline.stages import AuditExecution, ReviewConfig, ReviewMode
from zephyr.context import ZephyrContext

from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB, GLM_BULK_TOKEN_ENV, GLM_MODEL
from experiments.post_training.task_curation.assessment import (
    AssessmentSettings,
    assessment_paths,
    assessment_pipeline,
)
from experiments.post_training.task_curation.campaign import (
    CampaignArtifact,
    CampaignFailed,
    CampaignPool,
    CampaignRuntime,
    campaign_plan,
    run_campaign,
)
from experiments.post_training.task_curation.catalog_snapshot import publish_catalog
from experiments.post_training.task_curation.config import (
    InputOverrides,
    PipelineOptions,
    RecipeSettings,
)
from experiments.post_training.task_curation.curation_catalog import CATALOG_DIRECTORY
from experiments.post_training.task_curation.embedding import (
    INPUTS_VERSION,
    DescriptionEndpoint,
    curation_embedding_pipeline,
)
from experiments.post_training.task_curation.local import artifact_prefix, run_local_steps
from experiments.post_training.task_curation.pipeline import CampaignMachines
from experiments.post_training.task_curation.source import CurationPipeline, RlDataSource
from experiments.post_training.task_curation.sources import all_sources, runnable_sources
from experiments.post_training.task_curriculum.catalog_artifact import TASK_CURRICULUM

LOCAL_MAX_WORKERS = 4
LOCAL_MEMORY = "4g"
LOCAL_NORMALIZED_SHARDS = 4
WORKER_CONTAINER_PROFILE = "CONTAINER_PROFILE_PRIVILEGED"
IRIS_JOB_TTL = 1800
CORPUS_CONVERSION_VERSION = "2026.10.14.1"
CANONICAL_ARTIFACT_ROOT = "s3://marin-us-east-02a/marin"
EMBEDDING_STAGES = (
    "inputs",
    "descriptions",
    "task_embeddings",
    "curriculum_embeddings",
    "assignments",
    "publication",
    "coverage",
)


def _selected_sources(sources: tuple[str, ...]) -> dict[str, RlDataSource]:
    """The named catalog sources in catalog order, or the whole catalog when none is named."""
    catalog = runnable_sources()
    unknown = set(sources) - catalog.keys()
    if unknown:
        raise click.UsageError(f"Unknown source: {', '.join(sorted(unknown))}")
    return {name: pipeline for name, pipeline in catalog.items() if not sources or name in sources}


def _image_factory(backend: Backend | None, controller_url: str | None) -> MachineFactory | None:
    if backend is None:
        return None
    if backend == Backend.GVISOR:
        if controller_url is None:
            job = get_job_info()
            controller_url = job.controller_address if job is not None else None
        if controller_url is None:
            raise click.UsageError("--image-backend gvisor requires --controller-url outside an Iris job")
        return IrisMachineFactory(controller_url=controller_url, job_ttl=IRIS_JOB_TTL, secret_env=None)
    if backend != Backend.DAYTONA:
        raise ValueError(f"Unsupported image backend: {backend}")
    # Daytona is an optional Shellbox dependency; local and Iris campaigns do not import its SDK.
    try:
        from shellbox.backends.daytona.machine import DaytonaMachineFactory  # noqa: PLC0415
    except ImportError as error:
        raise click.UsageError("--image-backend daytona requires the Shellbox daytona extra") from error
    return DaytonaMachineFactory()


def _publish_campaign(report_path: str, selected: dict[str, RlDataSource]) -> None:
    path = StoragePath(report_path)
    report = json.loads(path.read_text())
    # Selected recipes may supply declarations absent from the default inventory.
    sources = {**all_sources(), **{source.info.id: source for source in selected.values()}}
    publish_catalog(str(path.parent / CATALOG_DIRECTORY), report["run_id"], sources.values())


def _conversion_steps(
    sources: dict[str, RlDataSource], options: PipelineOptions
) -> dict[str, ArtifactStep[CampaignArtifact]]:
    steps = {}
    for source in sources.values():
        step = cast(CurationPipeline, source.pipeline)(source, options)
        declaration = source.declaration
        assert declaration.pipeline is not None
        identity = canonical_sha256({"pipeline": declaration.pipeline.model_dump(), "dependencies": step.fingerprint()})
        steps[source.name] = replace(
            step, name=f"rl-semantics/converted/{source.name}-{identity[:16]}", version=CORPUS_CONVERSION_VERSION
        )
    return steps


def _quick_conversion(
    steps: dict[str, ArtifactStep[CampaignArtifact]],
    selected: dict[str, RlDataSource],
    *,
    options: PipelineOptions,
    output_root: str,
    cache_dir: Path | None,
    pool: CampaignPool,
    stage: str | None,
    do_run: bool,
) -> dict[str, str] | None:
    local = not StoragePath(output_root).is_remote
    report_path = str(StoragePath(output_root) / "campaign.json")
    if not do_run:
        plan = {
            "mode": options.mode,
            "sources": [
                {"name": name, "fingerprint": step.fingerprint(), "dependencies": [dep.name for dep in step.deps]}
                for name, step in steps.items()
            ],
            "input_root": options.inputs.root,
            "input_files": {name: str(file) for name, file in options.inputs.files.items()},
            "inputs": dict(options.inputs.auxiliary),
            "output_root": str(output_root),
            "cache_dir": str(cache_dir),
            "max_workers": pool.max_workers,
            "embedding_stage": stage,
        }
        if not local:
            plan.update(campaign_plan(list(steps.values()), pool))
        click.echo(json.dumps(plan, indent=2))
        return None
    if local:
        assert cache_dir is not None
        outcomes = run_local_steps(
            steps,
            Path(output_root),
            runtime=options.runtime,
            max_workers=pool.max_workers,
            worker_resources=pool.worker_resources,
            cache_dir=cache_dir,
            source_ids={name: source.info.id for name, source in selected.items()},
        )
    else:
        outcomes = run_campaign(
            list(steps.values()),
            pool=pool,
            runtime=options.runtime,
            report_path=report_path,
            source_ids={steps[name].name: source.info.id for name, source in selected.items()},
            mode=options.mode,
        )
    identifiers = {name if local else steps[name].name: source.info.id for name, source in selected.items()}
    roots = {identifiers[outcome.name]: outcome.path for outcome in outcomes}
    return roots


def _write_corpus(roots: dict[str, str], inventory: list[SourceDeclaration], report_path: str, population: str) -> None:
    sources = []
    for identifier, root in roots.items():
        manifest = json.loads((StoragePath(root) / "manifest.json").read_text())
        row = {
            "source_id": identifier,
            "root": root,
            "source_key": canonical_sha256(manifest),
            "intended_use": manifest["intended_use"],
        }
        if population == "converted":
            if manifest["input_rows"] != manifest["converted_rows"] + sum(manifest["rejections"].values()):
                raise ValueError(f"Conversion does not account for all input rows: {identifier}")
            row.update({key: manifest[key] for key in ("input_rows", "converted_rows", "rejections")})
        sources.append(row)
    result = {
        "version": CORPUS_CONVERSION_VERSION,
        "population": population,
        "sources": sources,
        "inventory": [source.model_dump(mode="json") for source in inventory],
        "campaign": report_path,
    }
    (StoragePath(report_path).parent / "corpus.json").write_text(json.dumps(result, indent=2))


@dataclass(frozen=True)
class _CorpusSources:
    roots: dict[str, str]
    inventory: list[SourceDeclaration]
    population: str


def _corpus_sources(corpus: str) -> _CorpusSources:
    document = json.loads(StoragePath(corpus).read_text())
    inventory = [SourceDeclaration.model_validate(row) for row in document["inventory"]]
    roots = {}
    for row in document["sources"]:
        manifest = json.loads((StoragePath(row["root"]) / "manifest.json").read_text())
        if canonical_sha256(manifest) != row["source_key"]:
            raise ValueError(f"Corpus source manifest changed: {row['source_id']}")
        if row["source_id"] in roots:
            raise ValueError(f"Duplicate corpus source: {row['source_id']}")
        roots[row["source_id"]] = row["root"]
    return _CorpusSources(roots, inventory, document["population"])


def _embedding_pass(
    roots: dict[str, str],
    *,
    inventory: list[SourceDeclaration],
    population: str,
    catalog_root: str | None,
    runtime: CampaignRuntime,
    model: DescriptionModel | None,
    endpoint: DescriptionEndpoint,
    cache_root: str,
    cache_dir: Path | None,
    sample_size: int | None,
    seed: int,
    pool: CampaignPool,
    stage: str,
    do_run: bool,
) -> None:
    curated = {}
    for source_id, root in roots.items():
        manifest = json.loads((StoragePath(root) / "manifest.json").read_text())
        digest = canonical_sha256(manifest)
        curated[source_id] = ArtifactStep.adopt(f"rl-semantics/source-{digest[:16]}", INPUTS_VERSION, source=root)
    curriculum = TASK_CURRICULUM
    if catalog_root is not None:
        catalog = json.loads((StoragePath(catalog_root) / "curriculum.json").read_text())
        curriculum = ArtifactStep.adopt(
            f"rl-semantics/curriculum-{canonical_sha256(catalog)[:16]}", INPUTS_VERSION, source=catalog_root
        )
    spec = EmbeddingSpec(
        model=HARRIER_REPO,
        revision=HARRIER_REVISION,
        tokenizer_revision=HARRIER_REVISION,
        dimensions=HARRIER_DIM,
        max_tokens=16384 if model is None else 512,
        compute_dtype="bfloat16",
    )
    embedder = HarrierConfig(
        model=spec.model,
        revision=spec.revision,
        tokenizer_revision=spec.tokenizer_revision,
        dimensions=spec.dimensions,
        max_tokens=spec.max_tokens,
        compute_dtype=spec.compute_dtype,
        device="cpu",
        cpu_threads=max(1, int(pool.worker_resources.cpu)),
    )
    stages = curation_embedding_pipeline(
        curated=curated,
        curriculum=curriculum,
        runtime=runtime,
        description_model=model,
        endpoint=endpoint,
        spec=spec,
        embedder_factory=partial(load_harrier, embedder),
        cache_root=cache_root,
        population=population,
        sample_size=sample_size,
        seed=seed,
        inventory=inventory,
    )
    with ExitStack() as scope:
        if cache_dir is not None:
            scope.enter_context(artifact_prefix(str(cache_dir)))
        click.echo(json.dumps({"stages": {name: getattr(stages, name).path() for name in EMBEDDING_STAGES}}, indent=2))
        if not do_run:
            return
        if cache_dir is not None:
            client = LocalClient()
            scope.callback(client.shutdown)
            scope.enter_context(set_current_client(client))
        context = scope.enter_context(
            ZephyrContext(
                max_workers=pool.max_workers,
                resources=pool.worker_resources,
                coordinator_resources=pool.coordinator_resources,
                chunk_storage_prefix=pool.chunk_storage_prefix,
                name="rl-task-embeddings",
            )
        )
        scope.enter_context(runtime.activate(context))
        result = run(getattr(stages, stage), max_concurrent=1)[0]
        click.echo(json.dumps({"stage": stage, "root": result.path, "paths": result.paths}, indent=2))


@click.command(help=__doc__)
@click.option("--model", default=GLM_MODEL, show_default=True)
@click.option("--model-revision", help="Required when a recipe uses model review.")
@click.option("--base-url", help="OpenAI-compatible review endpoint; defaults to the relay job's endpoint.")
@click.option("--relay-job", default=DEFAULT_GLM_RELAY_JOB, show_default=True, help="Iris GLM relay job to resolve.")
@click.option("--review-cache", help="Required when a recipe uses model review.")
@click.option("--review-mode", type=click.Choice(["batch", "chat"]), help="Required when a recipe uses model review.")
@click.option(
    "--review-concurrency",
    type=click.IntRange(min=1, max=MAX_DIRECT_CONCURRENT_REQUESTS),
    default=MAX_DIRECT_CONCURRENT_REQUESTS,
    show_default=True,
)
@click.option(
    "--mode", type=click.Choice([mode.value for mode in SourceProcessingMode]), help="Required without --corpus."
)
@click.option("--corpus", help="Resume embedding from a completed corpus.json without rerunning curation.")
@click.option("--embed", type=click.Choice(EMBEDDING_STAGES), help="Run this embedding stage after curation.")
@click.option("--representation", type=click.Choice(["direct", "rewrite"]), default="direct", show_default=True)
@click.option("--embedding-sample-size", type=click.IntRange(min=1), help="Seeded embedding sample per source.")
@click.option("--catalog-root", help="Immutable directory containing curriculum.json; defaults to TASK_CURRICULUM.")
@click.option("--max-workers", type=click.IntRange(min=1), help="Local default: 4; required for Iris execution.")
@click.option("--worker-cpu", type=click.FloatRange(min=0, min_open=True), help="CPU cores per Zephyr worker.")
@click.option("--worker-memory", help="RAM per Zephyr worker; defaults to 8g for embeddings.")
@click.option("--coordinator-memory", help="Iris coordinator RAM, e.g. 16g; required for Iris execution.")
@click.option("--normalized-shards", type=click.IntRange(min=1), help="Local default: 4; required for Iris recipes.")
@click.option("--concurrent-sources", type=click.IntRange(min=1), default=10, show_default=True)
@click.option("--worker-image", help="Zephyr worker image; required for Iris execution.")
@click.option(
    "--container-profile",
    default=None,
    help="Iris worker profile; defaults to CONTAINER_PROFILE_PRIVILEGED for in-pod sandbox graders.",
)
@click.option(
    "--image-backend",
    type=click.Choice([Backend.GVISOR.value, Backend.DAYTONA.value]),
    help="Image controls: gvisor via Iris or daytona. Omit for local and simulator controls only.",
)
@click.option(
    "--controller-url",
    help="Controller for --image-backend gvisor; defaults to the current Iris job's controller.",
)
@click.option(
    "--simulator-catalog",
    help="Portable software catalog URL or path; use an immutable snapshot for cached campaign results.",
)
@click.option("--seed", type=int, default=0)
@click.option(
    "--quality-sample-size",
    type=click.IntRange(min=1, max=MAX_SOURCE_QUALITY_SAMPLE_SIZE),
    default=SOURCE_QUALITY_SAMPLE_SIZE,
    show_default=True,
)
@click.option("--verification-sample-size", type=click.IntRange(min=1), default=20)
@click.option(
    "--curation-attempt",
    type=click.IntRange(min=0),
    default=0,
    help="New curation key for retrying incomplete controls; successful review requests remain cached.",
)
@click.option("--assess/--skip-assessment", default=None, help="Automatic for FULL; opt in for SAMPLE. QUICK skips it.")
@click.option(
    "--assessment-sample-size", type=click.IntRange(min=1), default=AssessmentSampling.admitted, show_default=True
)
@click.option(
    "--excluded-sample-size", type=click.IntRange(min=0), default=AssessmentSampling.excluded, show_default=True
)
@click.option(
    "--assessment-rubric", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Extra rubric criteria."
)
@click.option("--assessment-model-revision", help="Grading model revision; defaults to the curation review revision.")
@click.option("--report-path", help="SAMPLE/FULL report; defaults under --cache-dir. QUICK writes under --output-root.")
@click.option("--source", "sources", multiple=True, help="Catalog source name to execute; repeat to select multiple.")
@click.option(
    "--input-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Local runs: staged primary inputs instead of downloading the pinned source.",
)
@click.option(
    "--input-file",
    "local_files",
    type=(str, click.Path(exists=True, dir_okay=False, path_type=Path)),
    multiple=True,
    help="Local runs: primary file under its declared logical filename; repeat as needed.",
)
@click.option(
    "--input",
    "auxiliary",
    type=(str, click.Path(exists=True, file_okay=False)),
    multiple=True,
    help="Local runs: override a named auxiliary input with a local directory.",
)
@click.option(
    "--output-root",
    help="Required for QUICK outputs; a local directory or region-local storage URL.",
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Local artifact cache. Enables local SAMPLE/FULL execution; QUICK defaults to ~/.cache/marin.",
)
@click.option(
    "--artifact-root",
    default=CANONICAL_ARTIFACT_ROOT,
    show_default=True,
    help="Remote artifact root; overrides MARIN_PREFIX. Local --cache-dir takes precedence.",
)
@click.option("--run", "do_run", is_flag=True)
def main(
    model: str,
    model_revision: str | None,
    base_url: str | None,
    relay_job: str,
    review_cache: str | None,
    review_mode: str | None,
    review_concurrency: int,
    mode: str | None,
    corpus: str | None,
    embed: str | None,
    representation: str,
    embedding_sample_size: int | None,
    catalog_root: str | None,
    max_workers: int | None,
    worker_cpu: float | None,
    worker_memory: str | None,
    coordinator_memory: str | None,
    normalized_shards: int | None,
    concurrent_sources: int,
    worker_image: str | None,
    container_profile: str | None,
    image_backend: str | None,
    controller_url: str | None,
    simulator_catalog: str | None,
    seed: int,
    quality_sample_size: int,
    verification_sample_size: int,
    curation_attempt: int,
    assess: bool | None,
    assessment_sample_size: int,
    excluded_sample_size: int,
    assessment_rubric: Path | None,
    assessment_model_revision: str | None,
    report_path: str | None,
    sources: tuple[str, ...],
    input_root: Path | None,
    local_files: tuple[tuple[str, Path], ...],
    auxiliary: tuple[tuple[str, str], ...],
    output_root: str | None,
    cache_dir: Path | None,
    artifact_root: str,
    do_run: bool,
) -> None:
    if corpus is not None and (mode is not None or sources or output_root is not None):
        raise click.UsageError(
            "--corpus selects its own sources and population; omit --mode, --source and --output-root"
        )
    if mode is None and corpus is None:
        raise click.UsageError("Supply --mode or --corpus")
    if corpus is not None and embed is None:
        raise click.UsageError("--corpus requires --embed")
    if corpus is not None and (input_root is not None or local_files or auxiliary):
        raise click.UsageError("--corpus uses completed conversions; omit local input overrides")
    configure_coreweave_s3()
    click.get_current_context().with_resource(artifact_prefix(artifact_root))
    processing_mode = SourceProcessingMode(mode) if mode else SourceProcessingMode.QUICK
    selected = _selected_sources(sources)
    runtime = CampaignRuntime()
    assess = assess if assess is not None else processing_mode == SourceProcessingMode.FULL
    if curation_attempt and processing_mode == SourceProcessingMode.QUICK:
        raise click.UsageError("--curation-attempt requires a SAMPLE/FULL curation run")
    if cache_dir is not None:
        cache_dir = cache_dir.expanduser().resolve()
    if output_root is not None and not StoragePath(output_root).is_remote and cache_dir is None:
        cache_dir = Path.home() / ".cache/marin"
    if cache_dir is None and (input_root is not None or local_files or auxiliary):
        raise click.UsageError("Local input options require local execution")
    if embed is not None:
        review_cache = review_cache or (str(cache_dir / "review-cache") if cache_dir else None)
        if review_cache is None:
            raise click.UsageError("Embedding requires --review-cache (or --cache-dir)")
        if representation == "rewrite" and model_revision is None:
            raise click.UsageError("Rewrites require --model-revision")
    description_model = (
        DescriptionModel(model, model_revision) if representation == "rewrite" and model_revision else None
    )
    worker_resources = ResourceConfig(
        cpu=worker_cpu if worker_cpu is not None else (4 if embed else (1 if cache_dir else 2)),
        ram=worker_memory or ("8g" if embed or cache_dir is None else LOCAL_MEMORY),
        image=worker_image,
        container_profile=(
            (
                container_profile
                or (
                    WORKER_CONTAINER_PROFILE
                    if processing_mode != SourceProcessingMode.QUICK and corpus is None
                    else None
                )
            )
            if cache_dir is None
            else None
        ),
    )
    max_workers = max_workers if max_workers is not None else (LOCAL_MAX_WORKERS if cache_dir else None)
    if max_workers is None:
        raise click.UsageError("Iris execution requires --max-workers")
    pool = CampaignPool(
        max_workers,
        concurrent_sources,
        worker_resources=worker_resources,
        coordinator_resources=ResourceConfig(cpu=1, ram=coordinator_memory or LOCAL_MEMORY, preemptible=False),
        chunk_storage_prefix=(
            str(cache_dir / ".zephyr")
            if cache_dir is not None
            else str(StoragePath(marin_prefix()) / ".zephyr/rl-data-campaign")
        ),
    )
    embedding = None
    if embed is not None:
        assert review_cache is not None
        embedding = partial(
            _embedding_pass,
            catalog_root=catalog_root,
            runtime=runtime,
            model=description_model,
            endpoint=DescriptionEndpoint(base_url, relay_job, review_concurrency),
            cache_root=review_cache,
            cache_dir=cache_dir,
            sample_size=embedding_sample_size,
            seed=seed,
            pool=pool,
            stage=embed,
        )
    inventory = [source.declaration for source in {**all_sources(), **selected}.values()]
    if corpus is not None:
        restored = _corpus_sources(corpus)
        assert embedding is not None
        embedding(restored.roots, inventory=restored.inventory, population=restored.population, do_run=do_run)
        return
    if input_root is not None and local_files:
        raise click.UsageError("Choose either --input-root or --input-file")
    inputs = InputOverrides(
        str(input_root.resolve()) if input_root is not None else None,
        {name: file.resolve() for name, file in local_files},
        {name: str(Path(path).resolve()) for name, path in auxiliary},
    )
    if processing_mode == SourceProcessingMode.QUICK:
        if assess:
            raise click.UsageError("Assessment requires SAMPLE or FULL")
        if output_root is None:
            raise click.UsageError("QUICK requires --output-root")
        local = not StoragePath(output_root).is_remote
        if not local and cache_dir is not None:
            raise click.UsageError("Remote QUICK outputs do not accept --cache-dir")
        if local and sources:
            selected = {name: selected[name] for name in dict.fromkeys(sources)}
        options = PipelineOptions(processing_mode, runtime, inputs=inputs)
        steps = (
            {name: cast(CurationPipeline, source.pipeline)(source, options) for name, source in selected.items()}
            if local
            else _conversion_steps(selected, options)
        )
        roots = _quick_conversion(
            steps,
            selected,
            options=options,
            output_root=output_root,
            cache_dir=cache_dir,
            pool=pool,
            stage=embed,
            do_run=do_run,
        )
        if roots is None:
            return
        report_path = str(StoragePath(output_root) / "campaign.json")
        if not local or embed is not None:
            _write_corpus(roots, inventory, report_path, "converted")
        if embedding is not None:
            embedding(roots, inventory=inventory, population="converted", do_run=True)
        return
    if output_root is not None:
        raise click.UsageError("--output-root requires --mode quick; use --report-path for SAMPLE/FULL")
    if cache_dir is None and (input_root is not None or local_files or auxiliary):
        raise click.UsageError("Local input options require --cache-dir for SAMPLE/FULL")
    if cache_dir is not None and (
        worker_image is not None or container_profile is not None or coordinator_memory is not None
    ):
        raise click.UsageError(
            "Local execution does not accept --worker-image, --container-profile or --coordinator-memory"
        )
    if coordinator_memory is None and cache_dir is None:
        raise click.UsageError("SAMPLE/FULL requires --coordinator-memory")
    if worker_image is None and cache_dir is None:
        raise click.UsageError("SAMPLE/FULL requires --worker-image")
    if report_path is None and cache_dir is None:
        raise click.UsageError("SAMPLE/FULL requires --report-path")
    review = (
        ReviewConfig(model=model, model_revision=model_revision, mode=ReviewMode(review_mode))
        if model_revision is not None and review_mode is not None
        else None
    )
    recipe_settings = RecipeSettings(
        review=review,
        review_cache=review_cache or (str(cache_dir / "review-cache") if cache_dir else None),
        base_url=base_url,
        api_key=os.environ.get(GLM_BULK_TOKEN_ENV),
        relay_job=relay_job,
        review_concurrency=review_concurrency,
        normalized_shards=(
            normalized_shards if normalized_shards is not None else (LOCAL_NORMALIZED_SHARDS if cache_dir else None)
        ),
        machines=CampaignMachines(
            image_factory=_image_factory(Backend(image_backend) if image_backend is not None else None, controller_url),
            simulator_catalog=simulator_catalog,
        ),
        seed=seed,
        quality_sample_size=quality_sample_size,
        verification_sample_size=verification_sample_size,
        curation_attempt=curation_attempt,
        execution=AuditExecution(max_workers=max_workers, worker_resources=worker_resources),
    )
    options = PipelineOptions(processing_mode, runtime, inputs=inputs, recipe_settings=recipe_settings)
    assessment = None
    if assess:
        if review is None or recipe_settings.review_cache is None:
            raise click.UsageError(
                "Assessment requires --model-revision, --review-mode and a review cache; use --skip-assessment to skip"
            )
        rubric = ASSESSMENT_RUBRIC
        if assessment_rubric is not None:
            rubric += "\nAdditional criteria:\n" + assessment_rubric.read_text()
        assessment = AssessmentSettings(
            AssessmentSampling(assessment_sample_size, excluded_sample_size, seed),
            AssessmentModel(review.model, assessment_model_revision or review.model_revision),
            rubric,
        )
    with ExitStack() as scope:
        if cache_dir is not None:
            scope.enter_context(artifact_prefix(str(cache_dir)))
            if report_path is None:
                report_path = str(cache_dir / "campaigns/campaign.json")
        steps = [cast(CurationPipeline, source.pipeline)(source, options) for source in selected.values()]
        curation_steps = steps
        assessments = {}
        if assessment is not None:
            stages = [
                assessment_pipeline(
                    step,
                    source=source.name,
                    settings=assessment,
                    runtime=runtime,
                    recipe_settings=recipe_settings,
                )
                for step, source in zip(steps, selected.values(), strict=True)
            ]
            steps = [stage.judgments for stage in stages]
            assessments = {stage.judgments.name: assessment_paths(stage) for stage in stages}
        if not do_run:
            plan = campaign_plan(steps, pool)
            plan["cache_dir"] = str(cache_dir) if cache_dir else None
            plan["report_path"] = report_path
            plan["quality_policy"] = {"sample_size": quality_sample_size, "seed": seed}
            plan["embedding_stage"] = embed
            click.echo(json.dumps(plan, indent=2))
            return
        if cache_dir is not None:
            client = LocalClient()
            scope.callback(client.shutdown)
            scope.enter_context(set_current_client(client))
        assert report_path is not None
        if assessment is not None:
            assert recipe_settings.review_cache is not None
            compact_query_cache(recipe_settings.review_cache)
        try:
            run_campaign(
                steps,
                runtime=runtime,
                pool=pool,
                report_path=report_path,
                mode=processing_mode,
                source_ids={step.name: source.info.id for step, source in zip(steps, selected.values(), strict=True)},
                assessments=assessments,
            )
        except CampaignFailed:
            _publish_campaign(report_path, selected)
            raise
        else:
            _publish_campaign(report_path, selected)
        if embedding is not None:
            roots = {source.info.id: step.path() for source, step in zip(selected.values(), curation_steps, strict=True)}
            _write_corpus(roots, inventory, report_path, "admitted")
            embedding(roots, inventory=inventory, population="admitted", do_run=True)


if __name__ == "__main__":
    main()
