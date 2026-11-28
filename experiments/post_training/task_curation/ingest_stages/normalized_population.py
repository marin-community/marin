# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the standard deduplicated TaskSpec population and measure its actor/integrity census."""

from uuid import uuid4

from marin.execution.artifact import RECORD_FILENAME, ArtifactRecord, result_type_name, write_record
from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.evaluation.corpus import file_sha256

from experiments.post_training.task_curation.backend_census_manifest import seal_manifest
from experiments.post_training.task_curation.ingest_stages.conversion import ConvertedPopulation
from experiments.post_training.task_curation.ingest_stages.graph import StageJob, parent_receipt
from experiments.post_training.task_curation.ingest_stages.placement import regional_worker
from experiments.post_training.task_curation.ingest_stages.receipts import SHA256, IngestStage, verify_evidence
from experiments.post_training.task_curation.intake_population import QuickPopulationConfig, normalize_quick
from experiments.post_training.task_curation.lexical_artifacts import ActorCensus, SourceTextPopulation
from experiments.post_training.task_curation.published_census import CensusConfig, measure_population
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref


class PopulationStage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input_sha256: str = Field(pattern=SHA256)
    population: ArtifactRef
    files: dict[str, ArtifactRef]

    def artifacts(self) -> dict[str, ArtifactRef]:
        return {"population": self.population, **self.files}


def conversion_population(job: StageJob) -> tuple[ConvertedPopulation, dict]:
    """Check the completed conversion's policy and evidence before preparing any population."""
    parent = parent_receipt(job.execution, IngestStage.CONVERT, job.inputs.dependencies[IngestStage.CONVERT])
    reference = job.inputs.artifacts["convert/converted"]
    if reference != parent.outputs["converted"]:
        raise ValueError("dedup input differs from its conversion dependency")
    converted = ConvertedPopulation.model_validate(read_ref(reference))
    if converted.input_sha256 != parent.inputs.digest():
        raise ValueError("converted population belongs to another stage execution")
    verify_evidence(list(converted.artifacts().values()))
    manifest = read_ref(converted.manifest)
    policy = job.execution.policy
    for field, expected in (
        ("source", policy.source_id),
        ("source_dataset", policy.upstream.repository),
        ("source_revision", policy.upstream.revision),
        ("recipe_revision", job.execution.source_revision),
        ("mode", "quick"),
    ):
        if manifest[field] != expected:
            raise ValueError("converted manifest differs from the ingest policy or code pin")
    return converted, manifest


def dedup_stage(job: StageJob) -> dict[str, ArtifactRef]:
    """Preserve QUICK's complete drop proof while exporting the standard population shape."""
    regional_worker(job.execution.policy.cluster, job.execution.artifact_root, job.output_root)
    path = prefix_join(job.output_root, "population-stage.json")
    target = StoragePath(path)
    if target.exists():
        result = PopulationStage.model_validate_json(target.read_bytes())
        if result.input_sha256 != job.inputs.digest():
            raise ValueError("cached population belongs to another dedup execution")
        verify_evidence(list(result.artifacts().values()))
    else:
        converted, manifest = conversion_population(job)
        policy = job.execution.policy
        root = prefix_join(job.output_root, f"attempts/{uuid4().hex}")
        population = normalize_quick(
            QuickPopulationConfig(
                source_id=policy.source_id,
                dataset_id=policy.upstream.repository,
                revision=policy.upstream.revision,
                version=job.execution.source_revision,
                converter_revision=job.execution.source_revision,
                manifest_path=converted.manifest.uri,
                manifest_sha256=converted.manifest.sha256,
                inputs=tuple((ref.uri, ref.sha256) for _, ref in sorted(converted.normalized.items())),
                converted_rows=manifest["converted_rows"],
                rejections=manifest["rejections"],
                output_path=root,
            )
        )
        evidence = manifest["within_source_dedup"]
        assert population.within_source_dedup is not None
        if (
            population.task_count != evidence["unique_count"]
            or evidence["raw_count"] != evidence["unique_count"] + evidence["within_source_duplicates"]
            or population.within_source_dedup["within_source_duplicates"]
        ):
            raise ValueError("QUICK and population deduplication evidence disagree")
        population = population.model_copy(update={"within_source_dedup": evidence})
        write_record(
            ArtifactRecord(
                output_path=root,
                result_type=result_type_name(SourceTextPopulation),
                result=population.result_payload(),
            )
        )
        files = {}
        for name, pattern in (
            ("specs", prefix_join(population.specs_output_dir, "*.parquet")),
            ("text", prefix_join(population.main_output_dir, "*.parquet")),
            ("drops", prefix_join(root, "within-source-dedup/*.parquet")),
        ):
            for file in sorted(StoragePath(pattern).glob(), key=str):
                files[f"{name}/{file.name}"] = ArtifactRef(uri=str(file), sha256=file_sha256(file))
        record = StoragePath(root) / RECORD_FILENAME
        files["population-record"] = ArtifactRef(uri=str(record), sha256=file_sha256(record))
        result = PopulationStage(
            input_sha256=job.inputs.digest(),
            population=seal_manifest(population.model_dump(mode="json"), prefix_join(root, "population.json")),
            files=files,
        )
    reference = seal_manifest(result.model_dump(mode="json"), path)
    return {"population-stage": reference, **result.artifacts()}


def census_stage(job: StageJob) -> dict[str, ArtifactRef]:
    """Measure the exact population record sealed by dedup, with no provider calls."""
    regional_worker(job.execution.policy.cluster, job.execution.artifact_root, job.output_root)
    parent = parent_receipt(job.execution, IngestStage.DEDUP, job.inputs.dependencies[IngestStage.DEDUP])
    reference = job.inputs.artifacts["dedup/population"]
    if reference != parent.outputs["population"]:
        raise ValueError("census population differs from the dedup dependency")
    population = SourceTextPopulation.model_validate(read_ref(reference))
    if (
        population.source_id != job.execution.policy.source_id
        or SourceTextPopulation.raw_load(population.path) != population
    ):
        raise ValueError("population's framework record differs from its sealed census input")
    measured_root = prefix_join(job.output_root, "measurement")
    result = measure_population(CensusConfig(population_path=population.path, output_path=measured_root))
    if result.task_count != population.task_count or result.task_specs_sha256 != population.task_specs_sha256:
        raise ValueError("actor/integrity census differs from the deduplicated population")
    reference = seal_manifest(result.model_dump(mode="json"), prefix_join(job.output_root, "census.json"))
    write_record(
        ArtifactRecord(
            output_path=measured_root,
            result_type=result_type_name(ActorCensus),
            result=result.result_payload(),
        )
    )
    record = StoragePath(measured_root) / RECORD_FILENAME
    outputs = {"actor-integrity": reference, "actor-record": ArtifactRef(uri=str(record), sha256=file_sha256(record))}
    if result.integrity_contracts.get("evidence_url") is not None:
        path = StoragePath(result.integrity_contracts["evidence_url"])
        outputs["integrity-contracts"] = ArtifactRef(uri=str(path), sha256=file_sha256(path))
    return outputs
