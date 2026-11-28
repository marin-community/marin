# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish through the maintained retention path and remeasure the private pin."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from finestore.rl.curation import (
    ConvertedExamples,
    DeconReceiptStatus,
    LexicalContaminationEvidence,
    LexicalReceipt,
    LexicalUse,
    SourceOrigin,
    WithinSourceDedupEvidence,
)
from finestore.rl.status_taxonomy import DataRestriction
from pydantic import BaseModel, ConfigDict
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256

from experiments.post_training.task_curation.backend_census_manifest import seal_manifest
from experiments.post_training.task_curation.catalog_identity import canonical_identity_survey
from experiments.post_training.task_curation.census_binding import (
    bind_actor_census,
    integrity_summary_for_actor,
    receipt_from_census,
)
from experiments.post_training.task_curation.ingest_registration import IngestAdmission, qualified_source
from experiments.post_training.task_curation.ingest_stages.converter_binding import load_converter
from experiments.post_training.task_curation.ingest_stages.graph import StageJob, parent_receipt
from experiments.post_training.task_curation.ingest_stages.licensing import LicenseAssessment
from experiments.post_training.task_curation.ingest_stages.overlap import baseline_survey, review_population
from experiments.post_training.task_curation.ingest_stages.placement import regional_worker
from experiments.post_training.task_curation.ingest_stages.population import fingerprint_population
from experiments.post_training.task_curation.ingest_stages.qc import QCOutcome
from experiments.post_training.task_curation.ingest_stages.raw_archive import raw_archive
from experiments.post_training.task_curation.ingest_stages.receipts import IngestStage, verify_evidence
from experiments.post_training.task_curation.ingest_stages.upstream import UpstreamInventory
from experiments.post_training.task_curation.lexical_artifacts import (
    ActorCensus,
    LexicalMeasurements,
    SourceTextPopulation,
)
from experiments.post_training.task_curation.lexical_binding import LexicalMeasurementReceipt
from experiments.post_training.task_curation.oversight_binding import bind_derived_oversight
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref
from experiments.post_training.task_curation.remediation_binding import PublicationReceipt, RetainedReleaseReceipt
from experiments.post_training.task_curation.remediation_publish import ReleasePlan, publish_plan
from experiments.post_training.task_curation.retained_census import census_release
from experiments.post_training.task_curation.retained_grading import RetainedGradingReceipt, grading_receipt
from experiments.post_training.task_curation.sample_qc_binding import SampleQCReceipt
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo
from experiments.post_training.task_curation.source_duplicate_ci import FingerprintReference, fingerprint_input
from experiments.post_training.task_curation.source_population_cache import PopulationLocations
from experiments.post_training.task_curation.within_source_census import census_source


class DeclarationOutputs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input_sha256: str
    outputs: dict[str, ArtifactRef]


def source_restriction(job: StageJob) -> DataRestriction:
    declarations = read_ref(job.inputs.artifacts["starting-declarations"])
    source_id = job.execution.policy.source_id
    if source_id in declarations:
        return DataRestriction(declarations[source_id]["data_restriction"])
    inventory = UpstreamInventory.model_validate(read_ref(job.inputs.artifacts["stage-upstream/inventory"]))
    return DataRestriction.PRIVATE_ALWAYS if inventory.private else DataRestriction.OPEN


def source_conversion(job: StageJob) -> RlDataSource[CurationRecipe]:
    """Build the original measured source entirely from sealed stage outputs."""
    policy = job.execution.policy
    binding = load_converter(policy.converter_module)
    population = SourceTextPopulation.model_validate(read_ref(job.inputs.artifacts["dedup/population"]))
    census = ActorCensus.model_validate(read_ref(job.inputs.artifacts["census/actor-integrity"]))
    record = census.model_dump(mode="json")
    actor = receipt_from_census(
        record, integrity_summary_for_actor(record, {**record, "summary": census.integrity_contracts})
    )
    converted = read_ref(job.inputs.artifacts["convert/manifest"])
    assessment = LicenseAssessment.model_validate(read_ref(job.inputs.artifacts["license/assessment"]))
    if assessment.receipt is None:
        raise ValueError("Publication requires the original permitted ingest license record")
    recipe = CurationRecipe(
        name=policy.source_id,
        source=HfSource(
            policy.upstream.repository, policy.upstream.revision, (policy.upstream.path,), binding.source_format
        ),
        convert=binding.convert,
        version=job.execution.source_revision,
        intended_use=policy.intended_use[0],
        controls=binding.controls,
        grader=binding.grader,
    )
    dedup = population.within_source_dedup
    assert dedup is not None
    starting = read_ref(job.inputs.artifacts["starting-declarations"]).get(policy.source_id)
    info = SourceInfo(
        id=policy.source_id,
        origin=SourceOrigin(policy.source_id.split(":", 1)[0]),
        upstream_dataset=policy.source_id.split(":", 1)[1],
        upstream_name=policy.source_id.split(":", 1)[1] if starting is None else starting["upstream_name"],
        build_path="atlas-ingest" if starting is None else starting["build_path"],
        data_restriction=source_restriction(job),
        family=policy.upstream.repository.split("/", 1)[1] if starting is None else starting["family"],
        tags=() if starting is None else tuple(starting["tags"]),
        notes="" if starting is None else starting["notes"],
        task_format=binding.task_format,
        count=converted["input_rows"],
        license_receipt=assessment.receipt,
        within_source_dedup=WithinSourceDedupEvidence(
            **{
                name: dedup[name]
                for name in ("raw_count", "unique_count", "within_source_duplicates", "drop_manifest_uri")
            },
            source_dataset_id=policy.upstream.repository,
            source_revision=policy.upstream.revision,
            converter_revision=job.execution.source_revision,
            converter_version=job.execution.source_revision,
            evidence_url=job.inputs.artifacts["dedup/population-stage"].uri,
        ),
        converted_examples=ConvertedExamples(
            count=population.task_count,
            input_rows=converted["input_rows"],
            rejections=converted["rejections"],
            admission_rejections={},
            source_dataset_id=policy.upstream.repository,
            source_revision=policy.upstream.revision,
            converter_revision=job.execution.source_revision,
            converter_version=job.execution.source_revision,
            evidence_url=job.inputs.artifacts["convert/manifest"].uri,
        ),
    )
    return bind_derived_oversight(
        bind_actor_census(RlDataSource(info=info, config=recipe, pipeline=process_rows), actor), None, None
    )


def retention_plan(
    job: StageJob, population: SourceTextPopulation, root: str
) -> tuple[ReleasePlan, LexicalMeasurementReceipt]:
    report = LexicalMeasurements.model_validate(read_ref(job.inputs.artifacts["lexical/report"]))
    measured = LexicalContaminationEvidence.model_validate(report.measurements[population.source_id])
    use = LexicalUse(job.execution.policy.intended_use[0].value)
    if use is LexicalUse.EVAL:
        measured = measured.model_copy(
            update={
                "receipt": LexicalReceipt(
                    use=use,
                    status=DeconReceiptStatus.BENCHMARK_MEASURED,
                    retained_tasks=population.task_count,
                    dropped_tasks=0,
                    release=None,
                )
            }
        )
    path = StoragePath(root) / "plan.json"
    if path.exists():
        plan = ReleasePlan.model_validate_json(path.read_bytes())
        if plan.task_specs_sha256 != population.task_specs_sha256 or plan.source_id != population.source_id:
            raise ValueError("Retention selection belongs to another population")
    else:
        selection = census_source(
            population, measured.model_dump(mode="json"), prefix_join(root, "selection"), lexical_use=use
        )
        plan = ReleasePlan.model_validate(
            {
                **selection,
                "multiplicities": {str(key): value for key, value in selection["multiplicities"].items()},
                "census_evidence_url": job.inputs.artifacts["census/actor-record"].uri,
                "data_restriction": source_restriction(job),
            }
        )
        seal_manifest(plan.model_dump(mode="json"), str(path))
    original = source_conversion(job)
    assert original.info.actor_census_input is not None
    return plan, LexicalMeasurementReceipt(
        source_id=population.source_id,
        input=original.info.actor_census_input,
        use=use,
        measurement=measured,
    )


def declare_stage(job: StageJob) -> dict[str, ArtifactRef]:
    """Publish, remeasure and seal a qualification for host-side Git landing."""
    regional_worker(job.execution.policy.cluster, job.execution.artifact_root, job.output_root)
    parent_receipt(job.execution, IngestStage.SAMPLE_QC, job.inputs.dependencies[IngestStage.SAMPLE_QC])
    complete = StoragePath(job.output_root) / "completed.json"
    if complete.exists():
        previous = DeclarationOutputs.model_validate_json(complete.read_bytes())
        if previous.input_sha256 != job.inputs.digest():
            raise ValueError("Completed publication belongs to another stage input")
        verify_evidence(list(previous.outputs.values()))
        return previous.outputs
    attempt_path = StoragePath(job.output_root) / "attempt.json"
    if attempt_path.exists():
        attempt = json.loads(attempt_path.read_bytes())
        if attempt["input_sha256"] != job.inputs.digest():
            raise ValueError("Publication attempt belongs to another stage input")
        root = attempt["root"]
        StoragePath(root).relative_to(StoragePath(job.output_root))
    else:
        root = prefix_join(job.output_root, f"attempts/{uuid4().hex}")
        seal_manifest({"input_sha256": job.inputs.digest(), "root": root}, str(attempt_path))
    original = source_conversion(job)
    population = SourceTextPopulation.model_validate(read_ref(job.inputs.artifacts["dedup/population"]))
    plan, measurement = retention_plan(job, population, root)
    archive = None
    if plan.lexical_use is LexicalUse.TRAIN and plan.raw_lexical_count:
        archive = raw_archive(job, original, root)
    assert original.info.license_receipt is not None
    with TemporaryDirectory(prefix="atlas-publish-license-") as temporary:
        index = Path(temporary) / "license.json"
        index.write_text(
            json.dumps({job.execution.policy.source_id: original.info.license_receipt.model_dump(mode="json")})
        )
        publication = PublicationReceipt.model_validate(
            publish_plan(
                plan,
                job.execution.source_revision,
                prefix_join(root, "publication"),
                original.info.license_receipt,
                archive,
                measurement_receipt=measurement,
                index_path=index,
            )
        )
    census_path = StoragePath(root) / "retained-census.json"
    if census_path.exists():
        census = json.loads(census_path.read_bytes())
    else:
        census = census_release(
            publication, prefix_join(root, f"census-attempts/{uuid4().hex}"), job.execution.source_revision
        )
        seal_manifest(census, str(census_path))
    grading_path = StoragePath(root) / "retained-grading.json"
    if grading_path.exists():
        grading = RetainedGradingReceipt.model_validate_json(grading_path.read_bytes())
    else:
        grading = grading_receipt(census, prefix_join(root, f"grading-attempts/{uuid4().hex}"))
        seal_manifest(grading.model_dump(mode="json"), str(grading_path))
    retained = RetainedReleaseReceipt(
        source_id=job.execution.policy.source_id,
        publication=publication,
        actor=census["actor"],
        grading=grading,
        original_pipeline=original.pipeline_reference,
        original_info=original.info,
        retained_dedup=census["dedup"],
    )
    outcome = QCOutcome.model_validate(read_ref(job.inputs.artifacts["sample-qc/outcome"]))
    sample = None
    if outcome.sample_passed:
        assert outcome.review is not None
        sample = SampleQCReceipt.model_validate(read_ref(outcome.review))
    admission = IngestAdmission(
        source_id=job.execution.policy.source_id, policy=job.execution.policy, retained=retained, sample=sample
    )
    source = qualified_source(admission)
    declaration = source.declaration.model_dump(mode="json")
    starting = read_ref(job.inputs.artifacts["starting-declarations"]).get(admission.source_id)
    changes = {
        "starting": starting,
        "result": declaration,
        "changed_fields": sorted(
            key for key in set(starting or {}) | set(declaration) if (starting or {}).get(key) != declaration.get(key)
        ),
    }
    population_ref = seal_manifest(census["population"], prefix_join(root, "population.json"))
    measured = fingerprint_population(
        admission.source_id, retained.actor.input, population_ref, region_prefix=job.execution.artifact_root
    )
    if measured.population.task_specs_sha256 == population.task_specs_sha256:
        previous = canonical_identity_survey(read_ref(job.inputs.artifacts["fingerprint-overlap/survey"]))
        survey = previous.model_copy(
            update={"fingerprints": {**previous.fingerprints, admission.source_id: measured.fingerprint}}
        )
        survey_ref = seal_manifest(survey.model_dump(mode="json"), prefix_join(root, "survey.json"))
        locations = PopulationLocations.model_validate(read_ref(job.inputs.artifacts["fingerprint-overlap/locators"]))
        prepared = locations.prepared[admission.source_id].model_copy(
            update={"inputs_sha256": canonical_sha256(measured.fingerprint.inputs)}
        )
        locations = measured.locations.model_copy(update={"prepared": {admission.source_id: prepared}})
    else:
        baseline, locations = baseline_survey(job, read_ref(job.inputs.artifacts["catalog-declarations"]))
        reviewed = review_population(job, measured, baseline, locations, prefix_join(root, "retained-overlap"))
        survey_ref = reviewed["survey"]
        locations = PopulationLocations.model_validate(read_ref(reviewed["locators"]))
    fingerprint_ref = FingerprintReference(
        uri=survey_ref.uri,
        sha256=survey_ref.sha256,
        input_sha256=canonical_sha256(fingerprint_input(declaration).model_dump(mode="json")),
    )
    outputs = {
        "selection": ArtifactRef(
            uri=plan.combined_drop_manifest_uri, sha256=file_sha256(StoragePath(plan.combined_drop_manifest_uri))
        ),
        "changes": seal_manifest(changes, prefix_join(root, "changes.json")),
        "admission": seal_manifest(admission.model_dump(mode="json"), prefix_join(root, "admission.json")),
        "declaration": seal_manifest(declaration, prefix_join(root, "declaration.json")),
        "population": population_ref,
        "fingerprint-reference": seal_manifest(
            fingerprint_ref.model_dump(mode="json"), prefix_join(root, "fingerprint-reference.json")
        ),
        "locators": seal_manifest(locations.model_dump(mode="json"), prefix_join(root, "locators.json")),
        "private-tasks": ArtifactRef(
            uri=str(StoragePath(publication.specs_output_dir) / "part-000000.parquet"),
            sha256=publication.tasks_parquet_sha256,
        ),
    }
    if archive is not None:
        outputs["raw-archive"] = seal_manifest(
            archive.model_dump(mode="json"), prefix_join(root, "raw-source-archive.json")
        )
    verify_evidence(list(outputs.values()))
    seal_manifest(
        DeclarationOutputs(input_sha256=job.inputs.digest(), outputs=outputs).model_dump(mode="json"), str(complete)
    )
    return outputs
