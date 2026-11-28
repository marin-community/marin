# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind sealed upstream files to maintained TaskSpec conversion and its drop evidence."""

from uuid import uuid4

from marin.execution.artifact import RECORD_FILENAME
from marin.execution.lazy import run as run_artifacts
from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.models import EnvironmentRequirements
from taskcompendium.pipeline.inputs import SourceFileOverride, SourceFiles
from taskcompendium.pipeline.models import SourceRecipe
from taskcompendium.pipeline.source_processing import ConversionResult, SourceProcessingMode, run_source_pipeline
from taskcompendium.runtime.local import ENVIRONMENT_MANIFEST, FILES_ARCHIVE, LOCK_FILE
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.atlas_deployment import region_deployment
from experiments.post_training.task_curation.backend_census_manifest import seal_manifest
from experiments.post_training.task_curation.images.build import environment_artifact
from experiments.post_training.task_curation.ingest_stages.converter_binding import RowConverterBinding, load_converter
from experiments.post_training.task_curation.ingest_stages.graph import StageJob, parent_receipt
from experiments.post_training.task_curation.ingest_stages.ingest_config import IngestMode
from experiments.post_training.task_curation.ingest_stages.placement import regional_worker
from experiments.post_training.task_curation.ingest_stages.receipts import SHA256, IngestStage, verify_evidence
from experiments.post_training.task_curation.ingest_stages.upstream import StagedUpstream, UpstreamInventory
from experiments.post_training.task_curation.local import artifact_prefix, local_client
from experiments.post_training.task_curation.pipeline import environment_requirements
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref


class ConvertedPopulation(BaseModel):
    """Only completed, SHA-bound QUICK outputs can enter the population writer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_sha256: str = Field(pattern=SHA256)
    manifest: ArtifactRef
    normalized: dict[str, ArtifactRef] = Field(min_length=1)
    drops: dict[str, ArtifactRef]
    grader: dict[str, ArtifactRef]

    def artifacts(self) -> dict[str, ArtifactRef]:
        return {"manifest": self.manifest, **self.normalized, **self.drops, **self.grader}


def staged_conversion_inputs(job: StageJob) -> dict[str, SourceFileOverride]:
    """Use precisely the dependency's sealed file map, without staging-directory globbing."""
    parent = parent_receipt(
        job.execution, IngestStage.STAGE_UPSTREAM, job.inputs.dependencies[IngestStage.STAGE_UPSTREAM]
    )
    reference = job.inputs.artifacts["stage-upstream/staged"]
    if parent.outputs["staged"] != reference:
        raise ValueError("conversion input differs from the staged upstream dependency")
    staged = StagedUpstream.model_validate(read_ref(reference))
    inventory = UpstreamInventory.model_validate(read_ref(staged.inventory))
    if (
        inventory.input_sha256 != parent.inputs.digest()
        or inventory.source_id != job.execution.policy.source_id
        or inventory.upstream != job.execution.policy.upstream
        or set(staged.files) != {file.path for file in inventory.files}
        or any(parent.outputs[f"raw/{name}"] != ref for name, ref in staged.files.items())
    ):
        raise ValueError("staged file map differs from its source pin or receipt")
    return {name: SourceFileOverride(path=ref.uri, sha256=ref.sha256) for name, ref in staged.files.items()}


def resolved_grader(
    job: StageJob, binding: RowConverterBinding
) -> tuple[EnvironmentRequirements | None, dict[str, ArtifactRef]]:
    """Build or reuse the maintained content-addressed grader environment in-region."""
    environment = binding.grader
    if environment is None:
        return None, {}
    if environment.image is not None:
        return environment_requirements(environment), {}
    prefix = region_deployment(job.execution.policy.cluster).region_prefix
    with artifact_prefix(prefix):
        built = run_artifacts(environment_artifact(environment), max_concurrent=1)[0]
    references = {}
    names = [LOCK_FILE]
    if environment.files:
        names.append(FILES_ARCHIVE)
    if environment.files or environment.env:
        names.append(ENVIRONMENT_MANIFEST)
    for name in names:
        path = StoragePath(built.path) / name
        references[f"grader/{name}"] = ArtifactRef(uri=str(path), sha256=file_sha256(path))
    record = StoragePath(built.path) / RECORD_FILENAME
    references["grader/record"] = ArtifactRef(uri=str(record), sha256=file_sha256(record))
    return environment_requirements(environment, built), references


def converted_outputs(job: StageJob, result: ConversionResult, grader: dict[str, ArtifactRef]) -> ConvertedPopulation:
    manifest = StoragePath(result.manifest_path)
    normalized = {
        f"normalized/{path.name}": ArtifactRef(uri=str(path), sha256=file_sha256(path))
        for path in sorted((StoragePath(result.normalized_path) / "*.parquet").glob(), key=str)
    }
    drops = {
        f"drops/{path.name}": ArtifactRef(uri=str(path), sha256=file_sha256(path))
        for path in sorted(StoragePath(result.within_source_dedup["drop_manifest_uri"]).glob(), key=str)
    }
    if result.converted_rows == 0 or any(not reason.startswith("source_defect:") for reason in result.rejections):
        raise ValueError("conversion contains no tasks or unresolved unsupported/converter failures")
    return ConvertedPopulation(
        input_sha256=job.inputs.digest(),
        manifest=ArtifactRef(uri=str(manifest), sha256=file_sha256(manifest)),
        normalized=normalized,
        drops=drops,
        grader=grader,
    )


def convert_stage(job: StageJob) -> dict[str, ArtifactRef]:
    """Convert staged bytes through QUICK; unsealed attempts cannot mix partial shards."""
    regional_worker(job.execution.policy.cluster, job.execution.artifact_root, job.output_root)
    if job.execution.policy.mode is not IngestMode.INGEST:
        raise ValueError("population repairs require their existing-declaration adapter")
    output = prefix_join(job.output_root, "converted.json")
    target = StoragePath(output)
    if target.exists():
        converted = ConvertedPopulation.model_validate_json(target.read_bytes())
        if converted.input_sha256 != job.inputs.digest():
            raise ValueError("cached conversion belongs to another ingest execution")
        verify_evidence(list(converted.artifacts().values()))
    else:
        policy = job.execution.policy
        inputs = staged_conversion_inputs(job)
        binding = load_converter(policy.converter_module)
        grader, references = resolved_grader(job, binding)
        recipe = SourceRecipe(
            name=policy.source_id,
            version=job.execution.source_revision,
            source=SourceFiles(
                policy.upstream.repository, policy.upstream.revision, tuple(sorted(inputs)), binding.source_format
            ),
            convert=binding.convert,
            rubric=None,
            controls=None,
            # QUICK uses this scalar only as metadata; the stage receipt binds all policy uses.
            intended_use=policy.intended_use[0],
            grader_environment=grader,
        )
        attempt = prefix_join(job.output_root, f"attempts/{uuid4().hex}")
        with local_client() as client:
            with ZephyrContext(
                client=client, max_workers=1, chunk_storage_prefix=prefix_join(attempt, "chunks")
            ) as context:
                result = run_source_pipeline(
                    recipe,
                    context,
                    attempt,
                    attempt,
                    mode=SourceProcessingMode.QUICK,
                    canonical_source=policy.source_id,
                    declared_task_format=binding.task_format,
                    source_overrides=inputs,
                )
        if not isinstance(result, ConversionResult):
            raise ValueError("QUICK did not return completed conversion evidence")
        converted = converted_outputs(job, result, references)
    reference = seal_manifest(converted.model_dump(mode="json"), output)
    return {"converted": reference, **converted.artifacts()}
