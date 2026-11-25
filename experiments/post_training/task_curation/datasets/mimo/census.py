# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure the repaired MiMo release: actor census, grading, derived contract, retention.

One regional job through the maintained modules, without provider calls. The actor
census measures the declared Daytona demands of the published release's converted
population and joins its integrity-contract summary; the grading census retains each
grading-environment assessment; the derived-contract census binds the population's
shape summary; static retention retains the complete population under the approved
no-oracle policy; and the backend survey measures the converted rows against the
production checker matrix. Receipts land beside the release under the census version.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

from finestore.rl.curation import (
    ActorCensusInput,
    ContentEvidenceReference,
    DatasetRelease,
    NoOracleExecutabilityPolicy,
    SourceUse,
)
from huggingface_hub import hf_hub_download
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from shellbox.survey import production_checkers
from taskcompendium.evaluation.corpus import file_sha256

from experiments.post_training.task_curation.backend_capabilities import build_survey
from experiments.post_training.task_curation.census_binding import (
    check_receipt_evidence,
    integrity_summary_for_actor,
    receipt_from_census,
    write_receipt_shards,
)
from experiments.post_training.task_curation.derived_contract_census import (
    DerivedContractCensusInput,
)
from experiments.post_training.task_curation.derived_contract_census import (
    census_source as derived_census_source,
)
from experiments.post_training.task_curation.grading_census import (
    GradingCensusInput,
)
from experiments.post_training.task_curation.grading_census import (
    census_source as grading_census_source,
)
from experiments.post_training.task_curation.lexical_artifacts import ActorCensus, SourceTextPopulation
from experiments.post_training.task_curation.published_census import CensusConfig, measure_population
from experiments.post_training.task_curation.static_retention import NoOracleSourceContext, collect_static_retention
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id

logger = logging.getLogger(__name__)

REGIONAL_PREFIX = "s3://marin-us-east-02a/marin"
SOURCE_ID = "FWeindel:validated-tasks/mimo_v1"
RELEASE_REPO = "open-athena/fweindel-validated-tasks-mimo-v1-pinned-images-20261124-a-v1"
RELEASE_REVISION = "729925160824efba40d672c10ee4311169dc1a8b"
CONVERTER_VERSION = "2026.11.23.pinned-images.1"
CONVERTER_REVISION = "831fb168588948a3a9c28ab1d6063d255555f0f4"
EXPECTED_TASKS = 2501
EXPECTED_TASKS_PARQUET_SHA256 = "e320474a633072e760e171227c928fd2de6cadde1813771d95ceeeed7d80c05e"
POPULATION_RECORD = (
    REGIONAL_PREFIX + "/rl-data/atlas/ingest-b/fweindel-mimo-v1/2026.11.23.1/measurements-2026.11.25.3/population-v1"
)
NO_ORACLE_POLICY = NoOracleExecutabilityPolicy(
    decision=ContentEvidenceReference(
        uri="s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/policy/no-oracle-executability-policy-20261114.md",
        sha256="028c7dfbc7800350eb6bf1ccb2be61b1810447f3d7fb76596318764c0e2f3953",
    ),
    executability_proof_required_for_no_oracle=False,
    version=1,
)
MANIFEST_BOUND_CATALOG_ROOT = "manifest-bound:mimo-repaired-census"
"""The census survey binds no catalog snapshot: the release pins its own manifest scope."""
SURVEY_SNAPSHOT_ID = EXPECTED_TASKS_PARQUET_SHA256
"""The measured release rows are the survey's bound snapshot identity."""
SURVEY_RUNTIME_DEFAULTS = {
    "agent": {
        "cleanup_timeout": 120.0,
        "cpus": None,
        "gpus": 0,
        "memory_mb": None,
        "network": "allow",
        "startup_timeout": 900.0,
        "storage_mb": None,
        "user": None,
    },
    "grader": {
        "cleanup_timeout": 120.0,
        "cpus": None,
        "gpus": 0,
        "memory_mb": None,
        "network": "deny",
        "startup_timeout": 900.0,
        "storage_mb": None,
        "user": None,
    },
}
STATIC_RETENTION_FILES = (
    "oracle-filter.json",
    "static-exclusions.json",
    "static-retention.json",
    "static-retention-evidence.json",
    "tasks.parquet",
)


def download_release(workdir: Path) -> Path:
    """Fetch the published release rows and verify them against the conversion summary."""
    path = Path(hf_hub_download(RELEASE_REPO, "tasks.parquet", repo_type="dataset", revision=RELEASE_REVISION))
    digest = file_sha256(StoragePath(str(path)))
    if digest != EXPECTED_TASKS_PARQUET_SHA256:
        raise ValueError(f"The release rows differ from the conversion summary: {digest}")
    local = workdir / "tasks.parquet"
    shutil.copyfile(path, local)
    return local


def actor_census_input(census: ActorCensus) -> ActorCensusInput:
    """The census input binding the measured population and its artifact record."""
    return ActorCensusInput(
        dataset_id=census.dataset_id,
        dataset_revision=census.dataset_revision,
        converter_version=census.converter_version,
        converter_revision=census.converter_revision,
        measured_tasks=census.task_count,
        task_specs_sha256=census.task_specs_sha256,
        evidence_url=str(StoragePath(census.path) / ".artifact.json"),
    )


def upload_directory(source: Path, root: StoragePath, name: str) -> None:
    """Copy a local receipt directory into the census root."""
    for path in sorted(source.iterdir()):
        (root / name / path.name).write_bytes(path.read_bytes())


def run_census(output_root: str, workdir: Path, survey_revision: str) -> dict:
    """Measure the release population through the maintained censuses and survey."""
    population = SourceTextPopulation.raw_load(POPULATION_RECORD)
    if (population.dataset_id, population.dataset_revision, population.source_version, population.task_count) != (
        RELEASE_REPO,
        RELEASE_REVISION,
        CONVERTER_VERSION,
        EXPECTED_TASKS,
    ):
        raise ValueError("The frozen population belongs to another repaired pin")
    root = StoragePath(output_root)

    census = measure_population(CensusConfig(population.path, str(root / "actor")))
    write_record(
        ArtifactRecord(
            output_path=census.path, result_type=result_type_name(ActorCensus), result=census.result_payload()
        )
    )
    record = census.model_dump(mode="json")
    integrity = integrity_summary_for_actor(record, {**record, "summary": census.integrity_contracts})
    receipt = receipt_from_census(record, integrity)
    check_receipt_evidence(receipt)
    receipt_shards = workdir / "actor-receipts"
    receipt_shards.mkdir(parents=True, exist_ok=True)
    write_receipt_shards(receipt_shards, {str(receipt.source_id): receipt.model_dump(mode="json")})
    upload_directory(receipt_shards, root, "actor-receipts")
    input_binding = actor_census_input(census)

    grading = grading_census_source(
        GradingCensusInput(population=population, input=input_binding), str(root / "grading")
    )
    derived = derived_census_source(
        DerivedContractCensusInput(population=population, input=input_binding), str(root / "derived")
    )
    if derived is None:
        raise ValueError("The derived contract census must write a receipt on a fresh output root")

    tasks_local = download_release(workdir)
    static_output = workdir / "static-retention"
    context = NoOracleSourceContext(
        source_id=canonical_source_id(SOURCE_ID),
        source_release=DatasetRelease(
            repo=RELEASE_REPO,
            revision=RELEASE_REVISION,
            url=f"https://huggingface.co/datasets/{RELEASE_REPO}/tree/{RELEASE_REVISION}",
        ),
        input_cohort_sha256=EXPECTED_TASKS_PARQUET_SHA256,
        current_input=input_binding,
        intended_use=SourceUse.TRAIN,
        policy=NO_ORACLE_POLICY,
    )
    cohort_evidence = ContentEvidenceReference(
        uri=f"https://huggingface.co/datasets/{RELEASE_REPO}/resolve/{RELEASE_REVISION}/tasks.parquet",
        sha256=EXPECTED_TASKS_PARQUET_SHA256,
    )
    collect_static_retention(tasks_local, context, cohort_evidence, static_output)
    upload_directory(static_output, root, "static-retention")
    summary_evidence = json.loads((static_output / "static-retention-evidence.json").read_text())
    if summary_evidence["static_retained_count"] != EXPECTED_TASKS or summary_evidence["exclusion_counts"]:
        raise ValueError(f"Static retention did not retain the complete population: {summary_evidence}")

    manifest = {
        "catalog_root": MANIFEST_BOUND_CATALOG_ROOT,
        "snapshot_id": SURVEY_SNAPSHOT_ID,
        "runtime_defaults": SURVEY_RUNTIME_DEFAULTS,
        "source_ids": [SOURCE_ID],
        "inputs": [
            {
                "source_id": SOURCE_ID,
                "paths": [str(tasks_local)],
                "column": "task_spec",
                "runtime_column": "runtime",
                "expected_tasks": EXPECTED_TASKS,
            }
        ],
        "grading_environments": {SOURCE_ID: grading.evidence.model_dump(mode="json")},
        "integrity_contracts": {SOURCE_ID: integrity.model_dump(mode="json")},
    }
    survey = build_survey(manifest, production_checkers(), survey_revision)
    (root / "survey" / "backend-map.json").write_text(survey.model_dump_json(indent=2) + "\n")

    summary = {
        "source_id": SOURCE_ID,
        "release": {"repo": RELEASE_REPO, "revision": RELEASE_REVISION},
        "converter": {"version": CONVERTER_VERSION, "revision": CONVERTER_REVISION},
        "tasks": EXPECTED_TASKS,
        "actor": {
            "path": census.path,
            "daytona_safety": census.daytona_safety,
            "daytona_snapshot_count": census.daytona_snapshot_count,
            "daytona_machine_slots": census.daytona_machine_slots,
            "daytona_unsafe_reasons": census.daytona_unsafe_reasons,
        },
        "grading": {
            "evidence": grading.evidence.model_dump(mode="json"),
            "task_evidence_uri": grading.task_evidence_uri,
            "unassessable_reasons": grading.unassessable_reasons,
        },
        "derived": {"summary": derived.summary.model_dump(mode="json")},
        "static_retention": {
            "retained_count": summary_evidence["retained_count"],
            "static_retained_count": summary_evidence["static_retained_count"],
            "exclusion_counts": summary_evidence["exclusion_counts"],
            "evidence_url": str(root / "static-retention" / "static-retention-evidence.json"),
        },
        "survey": {
            "path": str(root / "survey" / "backend-map.json"),
            "sources": len(survey.sources),
            "rows": [row.model_dump(mode="json") for row in survey.sources],
        },
    }
    (root / "census-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info(
        "Census complete: %d tasks; actor %s; grading %s; survey %d sources",
        EXPECTED_TASKS,
        census.daytona_safety,
        grading.evidence.status_counts,
        len(survey.sources),
    )
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--survey-revision", required=True, help="The landed revision the survey records.")
    parser.add_argument("--workdir", default="/workspace/mimo-census")
    args = parser.parse_args()
    configure_coreweave_s3()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    run_census(args.output_root, workdir, args.survey_revision)


if __name__ == "__main__":
    main()
