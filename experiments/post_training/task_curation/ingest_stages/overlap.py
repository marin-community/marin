# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fingerprint the sealed census population and run the production overlap checks."""

from datetime import UTC, datetime
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from uuid import uuid4

from finestore.rl.curation import ActorCensusInput, PipelineReference
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.source_fingerprints import DuplicateLabel, SourceIdentitySurvey

from experiments.post_training.task_curation.backend_census_manifest import seal_manifest
from experiments.post_training.task_curation.catalog_identity import canonical_identity_survey
from experiments.post_training.task_curation.ingest_stages.graph import StageJob, parent_receipt
from experiments.post_training.task_curation.ingest_stages.placement import regional_worker
from experiments.post_training.task_curation.ingest_stages.population import (
    PopulationFingerprint,
    fingerprint_population,
)
from experiments.post_training.task_curation.ingest_stages.receipts import IngestStage
from experiments.post_training.task_curation.ingest_stages.registry_update import RegistryPointers, read_fingerprints
from experiments.post_training.task_curation.lexical_artifacts import ActorCensus
from experiments.post_training.task_curation.registry import read_registry_document
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref
from experiments.post_training.task_curation.source_duplicate_ci import fingerprint_input
from experiments.post_training.task_curation.source_duplicates import load_acknowledgements, scan, scan_pairs
from experiments.post_training.task_curation.source_population_cache import (
    PopulationLocations,
    PreparedPopulation,
    cache_population,
    population_preflight,
)

TOP_K = 3
SHINGLE_CACHE_BYTES = 4 * 1024**3
PACKAGE = Path(__file__).resolve().parents[1]


def measured_declaration(job: StageJob, census: ActorCensus) -> dict:
    """Derive the task-producing declaration fields from the measured population."""
    policy = job.execution.policy
    binding = ActorCensusInput(
        dataset_id=census.dataset_id,
        dataset_revision=census.dataset_revision,
        converter_version=census.converter_version,
        converter_revision=census.converter_revision,
        measured_tasks=census.task_count,
        task_specs_sha256=census.task_specs_sha256,
        evidence_url=job.inputs.artifacts["census/actor-record"].uri,
    )
    pipeline = PipelineReference(
        name=policy.source_id,
        version=job.execution.source_revision,
        source=policy.upstream.repository,
        revision=policy.upstream.revision,
        files=[policy.upstream.path],
    )
    return {
        "id": policy.source_id,
        "dataset_id": census.dataset_id,
        "dataset_revision": census.dataset_revision,
        "pipeline": pipeline.model_dump(mode="json"),
        "actor_census_input": binding.model_dump(mode="json"),
    }


def baseline_survey(job: StageJob, declarations: dict) -> tuple[SourceIdentitySurvey, PopulationLocations]:
    """Check the frozen registry against every declaration using CI's input projection."""
    pointers = RegistryPointers.model_validate(read_ref(job.inputs.artifacts["registry-pointers"]))
    fingerprints = dict.fromkeys(declarations)
    documents = {}
    for source_id, reference in read_fingerprints(pointers.fingerprints).items():
        if source_id not in declarations:
            continue
        expected = canonical_sha256(fingerprint_input(declarations[source_id]).model_dump(mode="json"))
        if reference.input_sha256 != expected:
            raise ValueError(f"fingerprint task input is stale: {source_id}")
        artifact = ArtifactRef(uri=reference.uri, sha256=reference.sha256)
        key = (artifact.uri, artifact.sha256)
        if key not in documents:
            documents[key] = canonical_identity_survey(read_ref(artifact))
        fingerprint = documents[key].fingerprints[source_id]
        if fingerprint is None:
            raise ValueError(f"published fingerprint is unassessed: {source_id}")
        fingerprints[source_id] = fingerprint
    survey = SourceIdentitySurvey(
        catalog_root=job.execution.artifact_root,
        snapshot_id=canonical_sha256(job.execution.identity()),
        generated_at=datetime.now(UTC).isoformat(),
        fingerprints=fingerprints,
        reviews=[],
        groups=[],
        judge=None,
    )
    return survey, PopulationLocations.model_validate(read_registry_document(pointers.locators))


def fingerprint_overlap_stage(job: StageJob) -> dict[str, ArtifactRef]:
    """Refuse raw-population mistakes and unresolved overlaps before admission."""
    regional_worker(job.execution.policy.cluster, job.execution.artifact_root, job.output_root)
    parent = parent_receipt(job.execution, IngestStage.CENSUS, job.inputs.dependencies[IngestStage.CENSUS])
    census_ref = job.inputs.artifacts["census/actor-integrity"]
    if census_ref != parent.outputs["actor-integrity"]:
        raise ValueError("overlap census differs from its dependency")
    census = ActorCensus.model_validate(read_ref(census_ref))
    declaration = measured_declaration(job, census)
    binding = ActorCensusInput.model_validate(declaration["actor_census_input"])
    measured = fingerprint_population(
        job.execution.policy.source_id,
        binding,
        job.inputs.artifacts["dedup/population"],
        region_prefix=job.execution.artifact_root,
    )
    declarations = read_ref(job.inputs.artifacts["catalog-declarations"])
    survey, locations = baseline_survey(job, declarations)
    outputs = review_population(job, measured, survey, locations, job.output_root)
    outputs["declaration-input"] = seal_manifest(declaration, prefix_join(job.output_root, "declaration-input.json"))
    return outputs


def review_population(
    job: StageJob,
    measured: PopulationFingerprint,
    survey: SourceIdentitySurvey,
    locations: PopulationLocations,
    output_root: str,
) -> dict[str, ArtifactRef]:
    """Apply the same overlap gates to original or retained measured populations."""
    source_id = measured.population.source_id
    survey = survey.model_copy(update={"fingerprints": {**survey.fingerprints, source_id: measured.fingerprint}})
    if sum(value is not None for value in survey.fingerprints.values()) < 2:
        raise ValueError("duplicate review needs published reference-source fingerprints")
    # A repaired source must never retain its old prepared cache at a new pin.
    locations = locations.model_copy(
        update={
            "files": {**locations.files, **measured.locations.files},
            "prepared": {key: value for key, value in locations.prepared.items() if key != source_id},
        }
    )
    bucket = StoragePath(job.execution.artifact_root).bucket
    population_preflight(survey, locations, bucket)
    root = prefix_join(output_root, f"attempts/{uuid4().hex}")
    acknowledgements = load_acknowledgements(PACKAGE / "source_duplicate_acknowledgements.json")
    with TemporaryDirectory(prefix="atlas-overlap-") as temporary:
        scratch = Path(temporary)
        prepared = cache_population(source_id, measured.fingerprint, scratch, locations, {}, source_bucket=bucket)
        prepared_path = StoragePath(root) / "prepared.parquet"
        with prepared.open("rb") as incoming, prepared_path.open("wb", auto_mkdir=True) as outgoing:
            copyfileobj(incoming, outgoing)
        prepared_ref = PreparedPopulation(
            uri=str(prepared_path),
            sha256=file_sha256(prepared_path),
            inputs_sha256=canonical_sha256(measured.fingerprint.inputs),
            task_count=measured.fingerprint.task_count,
        )
        new_locations = measured.locations.model_copy(update={"prepared": {source_id: prepared_ref}})
        needed = {key for pair in scan_pairs(survey, {source_id}, TOP_K, []) for key in (pair.left, pair.right)}
        paths = {}
        for key in sorted(needed):
            fingerprint = survey.fingerprints[key]
            assert fingerprint is not None
            paths[key] = cache_population(key, fingerprint, scratch, locations, {}, source_bucket=bucket)
        result = scan(
            survey,
            {source_id},
            TOP_K,
            paths,
            acknowledgements,
            [],
            scratch / "pairs.jsonl",
            shingle_cache_bytes=SHINGLE_CACHE_BYTES,
            response_receipts=StoragePath(prefix_join(root, "neighbors")),
        )
    if any(row.verdict.label != DuplicateLabel.DISTINCT and not row.acknowledgement for row in result.reviews):
        seal_manifest(result.model_dump(mode="json"), prefix_join(root, "unresolved.json"))
        raise ValueError("resolve or acknowledge measured source-overlap flags before admission")
    return {
        "survey": seal_manifest(result.model_dump(mode="json"), prefix_join(root, "survey.json")),
        "locators": seal_manifest(new_locations.model_dump(mode="json"), prefix_join(root, "locators.json")),
    }
