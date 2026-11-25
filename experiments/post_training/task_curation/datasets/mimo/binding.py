# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the repaired MiMo qualification from its measured receipts.

The qualification document carries the exact-pin ``SourceInfo`` and ``RetainedCohort``
the release pipeline reads, plus {uri, sha256} pointers for every receipt family:
the repaired publication's identity proof, the census receipts, the measurements, the
frozen SAMPLE receipt and the stage-g NoP controls. Every pointer is written after a
hash readback, and the document must round-trip through the maintained source models
before it publishes.
"""

import argparse
import dataclasses
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
from finestore.rl.curation import (
    ContentEvidenceReference,
    DatasetRelease,
    DeconReceiptStatus,
    DecontaminatedRelease,
    LexicalContaminationEvidence,
    LexicalReceipt,
    LexicalUse,
    NoOracleExecutabilityPolicy,
    PipelineReference,
    WithinSourceDedupEvidence,
)
from finestore.rl.licenses import LicenseEvidence, LicenseReceipt
from finestore.rl.status_taxonomy import DataRestriction
from huggingface_hub import HfApi, hf_hub_download
from pydantic import TypeAdapter
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.models import IntendedUse
from taskcompendium.source_fingerprints import SourceIdentitySurvey

from experiments.post_training.task_curation.catalog_identity import identity_survey_payload
from experiments.post_training.task_curation.census_binding import ActorCensusReceipt, bind_actor_census
from experiments.post_training.task_curation.datasets.mimo.repair import REPAIR_VERSION as CONVERTER_VERSION
from experiments.post_training.task_curation.derived_contract_census import DerivedContractReceipt
from experiments.post_training.task_curation.derived_contract_tags import bind_derived_contract_tags
from experiments.post_training.task_curation.grading_binding import GradingCensusReceipt
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_binding import LexicalMeasurementReceipt
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.retained_release import RetainedCohort, import_retained
from experiments.post_training.task_curation.sample_qc_binding import SampleQCReceipt, bind_sample_result
from experiments.post_training.task_curation.source import (
    RlDataSource,
    SourceInfo,
    SourceOrigin,
    SourceReference,
    TaskFormat,
)
from experiments.post_training.task_curation.source_duplicate_ci import FingerprintInput
from experiments.post_training.task_curation.source_fingerprints import fingerprint_source
from experiments.post_training.task_curation.static_retention import NoOracleStaticRetentionEvidence

logger = logging.getLogger(__name__)

LANE = "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/fweindel-mimo-v1/2026.11.23.1"
CENSUS_ROOT = f"{LANE}/census-repaired-2026.11.25.2"
MEASUREMENTS = f"{LANE}/measurements-2026.11.25.3"
SAMPLE_PREFIX = f"{LANE}/sample-2026.11.25.1"
CONTROLS_ROOT = f"{LANE}/controls-2026.11.25.6"
REPAIRED_PREFIX = f"{LANE}/repaired-v1"
SOURCE_HASH = "0387d8af2772e0f0e405"
SOURCE_ID = "FWeindel:validated-tasks/mimo_v1"
RELEASE_REPO = "open-athena/fweindel-validated-tasks-mimo-v1-pinned-images-20261124-a-v1"
RELEASE_REVISION = "729925160824efba40d672c10ee4311169dc1a8b"
CONVERTER_REVISION = "831fb168588948a3a9c28ab1d6063d255555f0f4"
RELEASE_URL = f"https://huggingface.co/datasets/{RELEASE_REPO}/tree/{RELEASE_REVISION}"
TASKS_PARQUET_SHA256 = "e320474a633072e760e171227c928fd2de6cadde1813771d95ceeeed7d80c05e"
EXPECTED_TASKS = 2501
UPSTREAM_DATASET = SourceReference(
    "FWeindel/validated-tasks",
    "317659ac88c8b706d318020c5fc3813a931fb6ec",
    "https://huggingface.co/datasets/FWeindel/validated-tasks/tree/317659ac88c8b706d318020c5fc3813a931fb6ec/mimo_v1",
)
LICENSE_SOURCE = {
    "dataset_id": "XiaomiMiMo/MiMo-V2.6-RL-oss",
    "revision": "639865fd3374018d6cb29b9fb82dd531406fcf5f",
    "url": "https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss/tree/639865fd3374018d6cb29b9fb82dd531406fcf5f",
}
REGISTRY_REPO = "open-athena/marin-rl-data-atlas-registry"
REGISTRY_PATH = "experiments/post_training/task_curation/datasets/mimo/qualification.json"
NO_ORACLE_POLICY = NoOracleExecutabilityPolicy(
    decision=ContentEvidenceReference(
        uri="s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/policy/no-oracle-executability-policy-20261114.md",
        sha256="028c7dfbc7800350eb6bf1ccb2be61b1810447f3d7fb76596318764c0e2f3953",
    ),
    executability_proof_required_for_no_oracle=False,
    version=1,
)
NOTES = (
    "Backend-repaired pin: every task's agent environment is its digest-pinned prepared image "
    "(built in-region from the task's own Dockerfile recipe) and the grader is declared from the "
    "source's own verifier package (setup.sh, test.sh, reward fail-closed) with the shared "
    "root-delivery contract. The eight 16-GiB LOCAL-routed tasks are excluded in the release "
    "ledger (backend_contract_unsupported; Daytona sandboxes cap at 8 GiB); 2,501 tasks are "
    "served. Approved no-oracle static retention applies; executability unmeasured under the "
    "policy. Stage g NoP controls on the LOCAL trial path prove the declared grader fail-closed "
    "over the frozen SAMPLE panel's own tasks. The original release rows stay immutable."
)


def verified_reference(uri: str) -> dict:
    """A {uri, sha256} pointer whose bytes hash to the recorded digest."""
    return {"uri": uri, "sha256": file_sha256(StoragePath(uri))}


def load_license_receipt(card_path: Path) -> LicenseReceipt:
    """The repaired release card's license evidence at the immutable publication pin."""
    lines = card_path.read_text().splitlines()
    license_lines = [f"L{index}: {line}" for index, line in enumerate(lines[:12], start=1) if "license" in line.lower()]
    if not license_lines:
        raise ValueError("The repaired release card declares no license line")
    card_sha256 = file_sha256(StoragePath(str(card_path)))
    evidence = LicenseEvidence(
        url=f"{RELEASE_URL}/resolve/{RELEASE_REVISION}/README.md",
        kind="dataset_card",
        sha256=card_sha256,
        license_ids=["apache-2.0"],
        quoted_lines=license_lines,
    )
    wrapper_declaration = LicenseEvidence(
        url=f"{RELEASE_URL}/resolve/{RELEASE_REVISION}/README.md",
        kind="wrapper_source",
        sha256=card_sha256,
        license_ids=[],
        quoted_lines=[
            f"L{index}: {line}"
            for index, line in enumerate(lines[:12], start=1)
            if "license_source" in line or "XiaomiMiMo" in line
        ],
    )
    return LicenseReceipt(
        source_id=SOURCE_ID,
        dataset_id=RELEASE_REPO,
        revision=RELEASE_REVISION,
        resolved_at=datetime.now(UTC).isoformat(),
        resolver_version="binding-2026.11.25.1",
        evidence=[evidence],
        bundled_third_party=False,
        model_generated_outputs=False,
        resolved_source={
            "dataset_id": LICENSE_SOURCE["dataset_id"],
            "revision": LICENSE_SOURCE["revision"],
            "url": LICENSE_SOURCE["url"],
            "declaration": wrapper_declaration.model_dump(mode="json"),
        },
    )


def binding_source(actor: ActorCensusReceipt, license_receipt: LicenseReceipt) -> RlDataSource[RetainedCohort]:
    """Assemble the repaired qualification's source from its verified receipts."""
    population = SourceTextPopulation.raw_load(f"{MEASUREMENTS}/population-v1")
    if (population.dataset_id, population.dataset_revision, population.task_count) != (
        RELEASE_REPO,
        RELEASE_REVISION,
        actor.input.measured_tasks,
    ):
        raise ValueError("The population belongs to another repaired pin")
    measurement = json.loads(StoragePath(f"{MEASUREMENTS}/within-source-measurement.json").read_text())
    dedup = WithinSourceDedupEvidence.model_validate(
        {name: measurement[name] for name in WithinSourceDedupEvidence.model_fields if name in measurement}
    )
    lexical = json.loads(StoragePath(f"{MEASUREMENTS}/lexical-result.json").read_text())["measurements"][SOURCE_ID]
    # The qualification's lexical evidence certifies the clean repaired release; the
    # registry's measurement receipt stays raw (raw training measurements never certify).
    lexical["receipt"] = LexicalReceipt(
        use=LexicalUse.TRAIN,
        status=DeconReceiptStatus.CLEAN,
        retained_tasks=EXPECTED_TASKS,
        dropped_tasks=0,
        duplicate_dropped_tasks=0,
        release=DecontaminatedRelease(
            private=True,
            format="taskcompendium",
            dataset_id=RELEASE_REPO,
            dataset_revision=RELEASE_REVISION,
            drop_manifest_uri=lexical["task_flags_uri"].rsplit("/task_flags", 1)[0] + "/drop_manifest.parquet",
            original_private_dataset_id=None,
            original_private_dataset_revision=None,
            original_dataset_id=RELEASE_REPO,
            original_dataset_revision=RELEASE_REVISION,
        ),
    ).model_dump(mode="json")
    grading = GradingCensusReceipt.model_validate(
        json.loads(StoragePath(f"{CENSUS_ROOT}/grading/{SOURCE_HASH}/receipt.json").read_text())
    )
    derived = DerivedContractReceipt.model_validate(
        json.loads(StoragePath(f"{CENSUS_ROOT}/derived/{SOURCE_HASH}/receipt.json").read_text())
    )
    static = NoOracleStaticRetentionEvidence.model_validate(
        json.loads(StoragePath(f"{CENSUS_ROOT}/static-retention/static-retention-evidence.json").read_text())
    )
    if static.current_input != actor.input:
        raise ValueError("Static retention census input differs from the actor census")
    census = actor.measurement.model_dump(mode="json")
    info = SourceInfo(
        data_restriction=DataRestriction.OPEN,
        origin=SourceOrigin.FWEINDEL,
        upstream_dataset="validated-tasks",
        upstream_name="MiMo code tasks",
        subset="mimo_v1",
        upstream_url=UPSTREAM_DATASET.url,
        count=actor.input.measured_tasks,
        family="terminal",
        build_path=(
            "Native Harbor files with unchanged budgets; digest-pinned prepared images; " "declared root-delivery grader"
        ),
        task_format=TaskFormat.TASKSPEC,
        current_release=DatasetRelease(repo=RELEASE_REPO, revision=RELEASE_REVISION, url=RELEASE_URL),
        within_source_dedup=dedup,
        grading_environment=grading.evidence,
        integrity_contracts=actor.integrity_contracts,
        license_receipt=license_receipt,
        no_oracle_executability_policy=NO_ORACLE_POLICY,
        no_oracle_static_retention=static,
        lexical_contamination=LexicalContaminationEvidence.model_validate(lexical),
        notes=NOTES,
        daytona_snapshot_count=census["daytona_snapshot_count"],
        daytona_count_basis=census["daytona_count_basis"],
        daytona_execution_failures=census["daytona_execution_failures"],
        daytona_machine_slots=census["daytona_machine_slots"],
        daytona_roles=census["daytona_roles"],
        daytona_unsafe_reasons=census["daytona_unsafe_reasons"],
        daytona_unsafe_details=census["daytona_unsafe_details"],
        daytona_unsafe_counts=census["daytona_unsafe_counts"],
    )
    cohort = RetainedCohort(
        name="fweindel-mimo-v1-pinned-images",
        version=CONVERTER_VERSION,
        release=DatasetRelease(repo=RELEASE_REPO, revision=RELEASE_REVISION, url=RELEASE_URL),
        original_dataset=UPSTREAM_DATASET,
        expected_rows=actor.input.measured_tasks,
        tasks_parquet_sha256=TASKS_PARQUET_SHA256,
        intended_use=IntendedUse.TRAIN,
        evidence_url=f"{REPAIRED_PREFIX}/identity-proof.json",
    )
    source = bind_actor_census(RlDataSource(info=info, config=cohort, pipeline=import_retained), actor)
    source = bind_derived_contract_tags(source, derived)
    receipt = SampleQCReceipt.model_validate(json.loads(StoragePath(f"{SAMPLE_PREFIX}/sample-receipt.json").read_text()))
    return bind_sample_result(source, receipt)


def jsonable(value):
    """Serialize dataclass/pydantic values the registry readers validate back.

    Pydantic models keep their explicit None fields (the models require them);
    dataclass fields drop None so optional contract fields stay absent.
    """
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            name: jsonable(getattr(value, name))
            for name in value.__dataclass_fields__
            if getattr(value, name) is not None
        }
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    return value


def qualification_document(source: RlDataSource[RetainedCohort]) -> dict:
    """The registry qualification document with hash-verified receipt pointers."""
    document = {
        "source_id": SOURCE_ID,
        "info": jsonable(source.info),
        "cohort": jsonable(source.config),
        "identity_proof": verified_reference(f"{REPAIRED_PREFIX}/identity-proof.json"),
        "sample_receipt": verified_reference(f"{SAMPLE_PREFIX}/sample-receipt.json"),
        "sample_review_artifact": verified_reference(f"{SAMPLE_PREFIX}/result.json"),
        "lexical_readback": verified_reference(f"{MEASUREMENTS}/lexical-result.json"),
        "controls_receipt": verified_reference(f"{CONTROLS_ROOT}/controls-summary.json"),
        "finalization_result": verified_reference(f"{SAMPLE_PREFIX}/result.json"),
    }
    # The registry consumers must parse the doc back into the exact source contract.
    TypeAdapter(SourceInfo).validate_python(document["info"])
    TypeAdapter(RetainedCohort).validate_python(document["cohort"])
    return document


def publish(document: dict) -> dict:
    """Commit the qualification document to the private registry repository."""
    staging = Path("/tmp/mimo-qualification-publish.json")
    staging.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    commit = HfApi().upload_file(
        path_or_fileobj=str(staging),
        path_in_repo=REGISTRY_PATH,
        repo_id=REGISTRY_REPO,
        repo_type="dataset",
        commit_message=f"Bind the repaired MiMo qualification at {RELEASE_REVISION[:12]}",
    )
    reference = {
        "repo": REGISTRY_REPO,
        "revision": getattr(commit, "oid", None) or str(commit),
        "path": REGISTRY_PATH,
        "sha256": file_sha256(StoragePath(str(staging))),
    }
    return reference


FINGERPRINT_INDEX_POINTER = Path(__file__).parents[2] / "source_fingerprint_index.json"
SOURCE_IDENTITY_PREFIX = (
    "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/fweindel-mimo-v1/2026.11.23.1/source-identity"
)


def fingerprint_input_sha256() -> str:
    """The task-input binding the duplicate review compares against the declaration."""
    pipeline = {
        "name": "fweindel-mimo-v1-pinned-images",
        "version": CONVERTER_VERSION,
        "source": RELEASE_REPO,
        "revision": RELEASE_REVISION,
        "files": ["tasks.parquet"],
    }
    binding = FingerprintInput(
        dataset_id=RELEASE_REPO,
        dataset_revision=RELEASE_REVISION,
        pipeline=PipelineReference.model_validate(pipeline),
        converter_revision=CONVERTER_REVISION,
        task_specs_sha256="c0f2f8ea38ad306f658e6022f66460c74598c10087ca4c5401cf6012b4f29a89",
    )
    return canonical_sha256(binding.model_dump(mode="json"))


def fingerprint_survey(workdir: Path) -> SourceIdentitySurvey:
    """Fingerprint the repaired population through the maintained identity sketch."""
    tasks_local = workdir / "tasks.parquet"
    if not tasks_local.exists():
        tasks_local = Path(
            hf_hub_download(RELEASE_REPO, "tasks.parquet", repo_type="dataset", revision=RELEASE_REVISION)
        )
    digest = file_sha256(StoragePath(str(tasks_local)))
    if digest != TASKS_PARQUET_SHA256:
        raise ValueError(f"The release rows differ from the conversion summary: {digest}")

    def documents():
        for batch in pq.ParquetFile(tasks_local).iter_batches(batch_size=16):
            for row in batch.to_pylist():
                runtime = json.loads(row["runtime"]) if row.get("runtime") else {}
                yield {"task": json.loads(row["task_spec"]), "runtime": runtime}

    fingerprint = fingerprint_source(documents(), {"tasks.parquet": TASKS_PARQUET_SHA256})
    return SourceIdentitySurvey(
        catalog_root="manifest-bound:mimo-repaired-census",
        snapshot_id=TASKS_PARQUET_SHA256,
        generated_at=datetime.now(UTC).isoformat(),
        fingerprints={SOURCE_ID: fingerprint},
        reviews=[],
        groups=[],
        judge=None,
    )


def publish_fingerprint(survey: SourceIdentitySurvey) -> dict:
    """Publish the survey artifact, update the fingerprint index and commit the registry."""
    payload = identity_survey_payload(survey)
    document = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    artifact_uri = f"{SOURCE_IDENTITY_PREFIX}/source-identities-{SOURCE_HASH}.json"
    StoragePath(artifact_uri).write_text(document)
    reference = {
        "uri": artifact_uri,
        "sha256": hashlib.sha256(document.encode()).hexdigest(),
        "input_sha256": fingerprint_input_sha256(),
    }
    current = read_registry_json(FINGERPRINT_INDEX_POINTER)
    current[SOURCE_ID] = reference
    staged = Path("/tmp/mimo-fingerprint-index.json")
    staged.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
    commit = HfApi().upload_file(
        path_or_fileobj=str(staged),
        path_in_repo="experiments/post_training/task_curation/source_fingerprint_index_mimo_repair_20261125_v1.json",
        repo_id=REGISTRY_REPO,
        repo_type="dataset",
        commit_message=f"Refresh the MiMo fingerprint binding at {RELEASE_REVISION[:12]}",
    )
    pointer = {
        "repo": REGISTRY_REPO,
        "revision": getattr(commit, "oid", None) or str(commit),
        "path": "experiments/post_training/task_curation/source_fingerprint_index_mimo_repair_20261125_v1.json",
        "sha256": file_sha256(StoragePath(str(staged))),
    }
    return {"pointer": pointer, "artifact": reference}


DEDUP_RECEIPTS_POINTER = Path(__file__).parents[2] / "datasets" / "within_source_dedup.json"
DEDUP_RECEIPTS_PATH = "experiments/post_training/task_curation/datasets/within_source_dedup_mimo_repair_20261125_v1.json"


def publish_dedup_receipt() -> dict:
    """Register the repaired pin's dedup receipt beside the current receipt shards."""
    measurement = json.loads(StoragePath(f"{MEASUREMENTS}/within-source-measurement.json").read_text())
    evidence = {
        name: measurement[name]
        for name in (
            "raw_count",
            "unique_count",
            "within_source_duplicates",
            "source_dataset_id",
            "source_revision",
            "converter_revision",
            "converter_version",
            "drop_manifest_uri",
            "evidence_url",
        )
    }
    evidence["carry_forward"] = None
    current = read_registry_json(DEDUP_RECEIPTS_POINTER)
    current[SOURCE_ID] = evidence
    staged = Path("/tmp/mimo-dedup-receipts.json")
    staged.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
    commit = HfApi().upload_file(
        path_or_fileobj=str(staged),
        path_in_repo=DEDUP_RECEIPTS_PATH,
        repo_id=REGISTRY_REPO,
        repo_type="dataset",
        commit_message=f"Register the MiMo repaired-pin dedup receipt at {RELEASE_REVISION[:12]}",
    )
    pointer = {
        "repo": REGISTRY_REPO,
        "revision": getattr(commit, "oid", None) or str(commit),
        "path": DEDUP_RECEIPTS_PATH,
        "sha256": file_sha256(StoragePath(str(staged))),
    }
    return {"pointer": pointer, "evidence": evidence}


ACTOR_RECEIPTS_DIR = Path(__file__).parents[2] / "datasets" / "daytona_census_receipts"


def actor_receipt_part(source_id: str) -> Path:
    """The receipt shard pointer that carries this source's actor census receipt."""
    for part in sorted(ACTOR_RECEIPTS_DIR.glob("part-*.json")):
        if source_id in read_registry_json(part):
            return part
    raise ValueError(f"No census receipt shard carries {source_id}")


def publish_actor_receipt(actor: ActorCensusReceipt) -> dict:
    """Register the repaired pin's actor census receipt in its census receipt shard."""
    pointer = actor_receipt_part(SOURCE_ID)
    current = read_registry_json(pointer)
    current[SOURCE_ID] = actor.model_dump(mode="json")
    staged = Path("/tmp/mimo-census-receipt-shard.json")
    staged.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
    name = f"daytona_census_receipts_mimo_repair_20261125_v1/{pointer.name}"
    commit = HfApi().upload_file(
        path_or_fileobj=str(staged),
        path_in_repo=f"experiments/post_training/task_curation/datasets/{name}",
        repo_id=REGISTRY_REPO,
        repo_type="dataset",
        commit_message=f"Register the MiMo repaired-pin census receipt at {RELEASE_REVISION[:12]}",
    )
    return {
        "pointer": {
            "repo": REGISTRY_REPO,
            "revision": getattr(commit, "oid", None) or str(commit),
            "path": f"experiments/post_training/task_curation/datasets/{name}",
            "sha256": file_sha256(StoragePath(str(staged))),
        },
        "replaced": str(pointer),
    }


SAMPLE_RECEIPTS_DIR = Path(__file__).parents[2] / "datasets" / "sample_qc_receipts"
GRADING_RECEIPTS_DIR = Path(__file__).parents[2] / "datasets" / "grading_census_receipts"
LEXICAL_RECEIPTS_DIR = Path(__file__).parents[2] / "datasets" / "lexical_measurement_receipts"


def _receipt_shard_with(source_id: str, directory: Path, kind: str) -> Path:
    """The receipt shard pointer that carries this source's receipt in a shard family."""
    for part in sorted(directory.glob("part-*.json")):
        if source_id in read_registry_json(part):
            return part
    raise ValueError(f"No {kind} receipt shard carries {source_id}")


def _publish_receipt_shard(receipts: dict, pointer: Path, name: str, message: str) -> dict:
    """Replace one source's receipt in its shard and commit the updated shard document."""
    current = read_registry_json(pointer)
    current.update(receipts)
    staged = Path(f"/tmp/mimo-{name}.json")
    staged.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
    path = f"experiments/post_training/task_curation/datasets/{name}"
    if not path.endswith(".json"):
        path += ".json"
    commit = HfApi().upload_file(
        path_or_fileobj=str(staged),
        path_in_repo=path,
        repo_id=REGISTRY_REPO,
        repo_type="dataset",
        commit_message=message,
    )
    return {
        "pointer": {
            "repo": REGISTRY_REPO,
            "revision": getattr(commit, "oid", None) or str(commit),
            "path": path,
            "sha256": file_sha256(StoragePath(str(staged))),
        },
        "replaced": str(pointer),
    }


def sample_receipt_part(source_id: str) -> Path:
    """The sampled-Result shard pointer that carries this source's SAMPLE receipt."""
    return _receipt_shard_with(source_id, SAMPLE_RECEIPTS_DIR, "sampled content Result")


def publish_sample_receipt(receipt: SampleQCReceipt) -> dict:
    """Register the repaired pin's SAMPLE receipt in its sampled-Result shard."""
    pointer = sample_receipt_part(SOURCE_ID)
    return _publish_receipt_shard(
        {SOURCE_ID: receipt.model_dump(mode="json")},
        pointer,
        "sample_qc_receipts_mimo_repair_20261125_v1",
        f"Register the MiMo repaired-pin SAMPLE receipt at {RELEASE_REVISION[:12]}",
    )


def publish_grading_receipt(grading: GradingCensusReceipt) -> dict:
    """Register the repaired pin's grading census receipt in its shard."""
    pointer = _receipt_shard_with(SOURCE_ID, GRADING_RECEIPTS_DIR, "grading census")
    return _publish_receipt_shard(
        {SOURCE_ID: grading.model_dump(mode="json")},
        pointer,
        "grading_census_receipts_mimo_repair_20261125_v1",
        f"Register the MiMo repaired-pin grading census at {RELEASE_REVISION[:12]}",
    )


def publish_lexical_receipt(actor: ActorCensusReceipt) -> dict:
    """Register the repaired pin's lexical measurement receipt in its shard."""
    pointer = _receipt_shard_with(SOURCE_ID, LEXICAL_RECEIPTS_DIR, "lexical")
    lexical = dict(json.loads(StoragePath(f"{MEASUREMENTS}/lexical-result.json").read_text())["measurements"][SOURCE_ID])
    lexical["receipt"] = LexicalReceipt(
        use=LexicalUse.TRAIN,
        status=DeconReceiptStatus.CLEAN,
        retained_tasks=EXPECTED_TASKS,
        dropped_tasks=0,
        duplicate_dropped_tasks=0,
        release=DecontaminatedRelease(
            private=True,
            format="taskcompendium",
            dataset_id=RELEASE_REPO,
            dataset_revision=RELEASE_REVISION,
            drop_manifest_uri=lexical["task_flags_uri"].rsplit("/task_flags", 1)[0] + "/drop_manifest.parquet",
            original_private_dataset_id=None,
            original_private_dataset_revision=None,
            original_dataset_id=RELEASE_REPO,
            original_dataset_revision=RELEASE_REVISION,
        ),
    ).model_dump(mode="json")
    receipt = LexicalMeasurementReceipt(
        source_id=SOURCE_ID,
        input=actor.input,
        use=LexicalUse.TRAIN,
        measurement=LexicalContaminationEvidence.model_validate(lexical),
    )
    return _publish_receipt_shard(
        {SOURCE_ID: receipt.model_dump(mode="json")},
        pointer,
        "lexical_measurement_receipts_mimo_repair_20261125_v1",
        f"Register the MiMo repaired-pin lexical measurement at {RELEASE_REVISION[:12]}",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the qualification document here.")
    parser.add_argument("--workdir", default="/workspace/mimo-binding")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--skip-fingerprint", action="store_true", help="Skip the fingerprint index refresh.")
    args = parser.parse_args()
    configure_coreweave_s3()
    card = Path(hf_hub_download(RELEASE_REPO, "README.md", repo_type="dataset", revision=RELEASE_REVISION))
    receipts = json.loads(StoragePath(f"{CENSUS_ROOT}/actor-receipts/part-000.json").read_text())
    actor = ActorCensusReceipt.model_validate(receipts[SOURCE_ID])
    source = binding_source(actor, load_license_receipt(card))
    document = qualification_document(source)
    if args.output is not None:
        args.output.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
        logger.info("Wrote the qualification document: %s", args.output)
    if args.publish:
        reference = publish(document)
        logger.info("Published qualification: %s", json.dumps(reference))
        print(json.dumps(reference, indent=1))
        if not args.skip_fingerprint:
            workdir = Path(args.workdir)
            workdir.mkdir(parents=True, exist_ok=True)
            survey = fingerprint_survey(workdir)
            refreshed = publish_fingerprint(survey)
            logger.info("Refreshed the fingerprint binding: %s", json.dumps(refreshed["pointer"]))
            print(json.dumps(refreshed, indent=1))


if __name__ == "__main__":
    main()
