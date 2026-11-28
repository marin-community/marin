# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The license stage carries ingest provenance and stops acquisition on doubt."""

import json
from pathlib import Path

import pytest
from finestore.rl.licenses import LicenseDecision, LicenseSourcePin
from requests import ConnectionError as RequestsConnectionError
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation import license_resolution
from experiments.post_training.task_curation.backend_census_manifest import seal_manifest
from experiments.post_training.task_curation.ingest_stages.graph import (
    IngestExecution,
    StageJob,
    execute_stage,
    ingest_graph,
    run_ingest_graph,
)
from experiments.post_training.task_curation.ingest_stages.ingest_config import load_ingest_config
from experiments.post_training.task_curation.ingest_stages.licensing import (
    LicenseAssessment,
    LicenseBlocked,
    LicenseDisposition,
    capture_license_context,
    license_stage,
)
from experiments.post_training.task_curation.ingest_stages.receipts import IngestStage, StageInputs, load_receipt
from experiments.post_training.task_curation.release_stages.manifest import read_ref
from experiments.post_training.task_curation.tests.license_fixtures import fixture_license
from infra.marina.apps.rl_data_catalog.server import licensing

CONFIGS = Path(__file__).parents[1] / "ingest_stages/configs"


def license_job(tmp_path, policy, records):
    index = tmp_path / "registry.json"
    index.write_text(json.dumps({key: receipt.model_dump(mode="json") for key, receipt in records.items()}))
    context = capture_license_context(policy, index)
    reference = seal_manifest(context.model_dump(mode="json"), str(tmp_path / "license-context.json"))
    execution = IngestExecution(
        policy=policy,
        source_revision="a" * 40,
        artifact_version="2026.11.28",
        artifact_root=str(tmp_path / "graph"),
        worker_argv=("python", "-m", "worker"),
        initial_artifacts={"license-context": reference},
    )
    inputs = StageInputs(
        stage=IngestStage.LICENSE,
        source_id=policy.source_id,
        config_sha256=policy.digest(),
        source_revision=execution.source_revision,
        worker_argv_sha256=canonical_sha256({"argv": list(execution.worker_argv)}),
        dependencies={},
        artifacts=execution.initial_artifacts,
    )
    return StageJob(execution=execution, inputs=inputs, output_root=str(tmp_path / "license"))


def record_external_work(job):
    """A persisted effect at the acquisition/later external-work boundary."""
    path = Path(job.output_root) / "external-work.json"
    return {"effect": seal_manifest({"stage": job.inputs.stage.value}, str(path))}


def folder_record(policy, source_id=None):
    receipt = fixture_license(source_id or policy.source_id, "publication/repaired", "b" * 40)
    upstream = LicenseSourcePin(
        dataset_id="original/tasks",
        revision="c" * 40,
        url="https://huggingface.co/datasets/original/tasks/tree/" + "c" * 40,
        declaration=receipt.evidence[0],
    )
    return receipt.model_copy(update={"resolved_source": upstream})


def test_new_source_seals_exact_pin_and_reuses_evidence_after_an_interrupted_attempt(tmp_path, pinned_dataset_metadata):
    repositories, _ = pinned_dataset_metadata
    policy = load_ingest_config(CONFIGS / "litmus-train.json")
    repositories[policy.upstream.repository, policy.upstream.revision] = {
        "README.md": b"---\nlicense: apache-2.0\n---\nSynthetic benchmark tasks.\n",
    }
    job = license_job(tmp_path, policy, {})

    original = license_stage(job)
    frozen = {path: path.read_bytes() for path in Path(job.output_root).glob("*.json")}
    repositories.clear()
    artifact = execute_stage(license_stage, job)
    receipt = load_receipt(artifact.receipt, job.inputs)
    assessment = LicenseAssessment.model_validate(read_ref(receipt.outputs["assessment"]))

    assert receipt.outputs == original
    assert assessment.disposition is LicenseDisposition.PERMIT
    assert (assessment.dataset_id, assessment.revision) == (policy.upstream.repository, policy.upstream.revision)
    assert assessment.receipt.license_ids == ("apache-2.0",)
    assert {path: path.read_bytes() for path in frozen} == frozen


@pytest.mark.parametrize(
    "card",
    [
        b"# No grant\n",
        b"---\nlicense: other\n---\n",
        b"---\nlicense: cc-by-nc-4.0\n---\n",
        b"---\nlicense: mit\n---\nAggregated tasks from third-party projects.\n",
    ],
    ids=["missing", "custom", "restricted", "aggregate-without-folder-lineage"],
)
def test_license_doubt_seals_a_dataset_ban_and_prevents_graph_acquisition(tmp_path, pinned_dataset_metadata, card):
    repositories, _ = pinned_dataset_metadata
    policy = load_ingest_config(CONFIGS / "litmus-train.json")
    repositories[policy.upstream.repository, policy.upstream.revision] = {"README.md": card}
    job = license_job(tmp_path, policy, {})
    handlers = dict.fromkeys(IngestStage, record_external_work)
    handlers[IngestStage.LICENSE] = license_stage
    graph = ingest_graph(job.execution, handlers)

    with pytest.raises(RuntimeError, match="step\\(s\\) failed"):
        run_ingest_graph(graph)

    path = Path(graph.nodes[1].path(job.execution.artifact_root)) / "assessment.json"
    assessment = LicenseAssessment.model_validate_json(path.read_bytes())
    assert assessment.disposition is LicenseDisposition.BAN
    assert (assessment.dataset_id, assessment.revision) == (policy.upstream.repository, policy.upstream.revision)
    assert assessment.refusals
    assert not (Path(graph.nodes[1].path(job.execution.artifact_root)) / ".artifact.json").exists()
    assert not Path(graph.nodes[2].path(job.execution.artifact_root)).exists()


def test_repair_carries_original_folder_license_without_resolving_its_new_pin(tmp_path, pinned_dataset_metadata):
    policy = load_ingest_config(CONFIGS / "mimo-repair.json")
    original = folder_record(policy)
    policy = policy.model_copy(update={"upstream": policy.upstream.model_copy(update={"revision": "d" * 40})})
    job = license_job(tmp_path, policy, {policy.source_id: original})

    artifact = execute_stage(license_stage, job)
    assessment = LicenseAssessment.model_validate(
        read_ref(load_receipt(artifact.receipt, job.inputs).outputs["assessment"])
    )

    assert assessment.inherited
    assert assessment.disposition is LicenseDisposition.PERMIT
    assert assessment.receipt == original
    assert assessment.revision == original.revision
    assert assessment.receipt.resolved_source.revision != policy.upstream.revision


def test_known_repackager_cannot_borrow_a_sibling_folders_receipt(tmp_path, pinned_dataset_metadata):
    policy = load_ingest_config(CONFIGS / "mimo-repair.json")
    sibling_id = "FWeindel:validated-tasks/another_v1"
    job = license_job(tmp_path, policy, {sibling_id: folder_record(policy, sibling_id)})

    with pytest.raises(LicenseBlocked) as blocked:
        execute_stage(license_stage, job)

    assessment = LicenseAssessment.model_validate(read_ref(blocked.value.assessment))
    assert assessment.receipt is None
    assert assessment.disposition is LicenseDisposition.BAN
    assert not (Path(job.output_root) / "resolution.json").exists()


def test_unreachable_primary_evidence_seals_a_retryable_refusal_not_permission(
    tmp_path, pinned_dataset_metadata, monkeypatch
):
    repositories, _ = pinned_dataset_metadata
    policy = load_ingest_config(CONFIGS / "litmus-train.json")
    repositories[policy.upstream.repository, policy.upstream.revision] = {"README.md": b"unreachable"}
    job = license_job(tmp_path, policy, {})

    def unavailable(*args, **kwargs):
        raise RequestsConnectionError("metadata endpoint unavailable")

    monkeypatch.setattr(license_resolution, "_fetch", unavailable)
    with pytest.raises(LicenseBlocked) as blocked:
        execute_stage(license_stage, job)
    reference = blocked.value.assessment
    repositories.clear()

    with pytest.raises(LicenseBlocked) as repeated:
        execute_stage(license_stage, job)

    assessment = LicenseAssessment.model_validate(read_ref(reference))
    assert repeated.value.assessment == reference
    assert assessment.disposition is LicenseDisposition.BAN
    assert assessment.receipt is None
    assert assessment.checks["resolution"].rule_id.value == "E4"


def test_both_purposes_require_permission_for_every_use(tmp_path, pinned_dataset_metadata):
    policy = load_ingest_config(CONFIGS / "mimo-repair.json").model_copy(
        update={"intended_use": (IntendedUse.TRAIN, IntendedUse.EVAL)}
    )
    receipt = folder_record(policy)
    decision = LicenseDecision(
        kind="ban",
        approved=True,
        who="user",
        when="2026-11-28T00:00:00+00:00",
        source_id=receipt.source_id,
        dataset_id=receipt.dataset_id,
        revision=receipt.revision,
        scope=["eval"],
        rationale="Use-scoped recorded decision",
        evidence_urls=["https://example.test/ruling"],
        issue_url="https://github.com/marin-community/marin/issues/11497",
    )
    receipt = receipt.model_copy(update={"decisions": [decision]})
    job = license_job(tmp_path, policy, {policy.source_id: receipt})

    with pytest.raises(LicenseBlocked) as blocked:
        execute_stage(license_stage, job)

    assessment = LicenseAssessment.model_validate(read_ref(blocked.value.assessment))
    assert set(assessment.checks) == {"train", "eval"}
    assert assessment.disposition is LicenseDisposition.BAN
    assert assessment.receipt.decisions == [decision]


def test_dataset_level_ban_overrides_a_passing_inherited_folder_license(tmp_path, pinned_dataset_metadata, monkeypatch):
    policy = load_ingest_config(CONFIGS / "mimo-repair.json")
    original = folder_record(policy)
    job = license_job(tmp_path, policy, {policy.source_id: original})
    monkeypatch.setattr(
        licensing,
        "LICENSE_BANS",
        json.dumps(
            {
                "bans": [
                    {
                        "source_id": "Other:tasks",
                        "upstream_id": original.resolved_source.dataset_id,
                        "upstream_revision": original.resolved_source.revision,
                        "classification": "unknown",
                        "license_ids": ["other"],
                        "reason": "Recorded dataset-level doubt",
                        "evidence_urls": ["https://example.test/ruling"],
                        "decided_at": "2026-11-28T00:00:00+00:00",
                        "decision_by": "user",
                        "issue_url": "https://github.com/marin-community/marin/issues/11497",
                    }
                ]
            }
        ),
    )
    licensing.license_bans.cache_clear()

    with pytest.raises(LicenseBlocked) as blocked:
        execute_stage(license_stage, job)

    assessment = LicenseAssessment.model_validate(read_ref(blocked.value.assessment))
    assert assessment.disposition is LicenseDisposition.BAN
    assert assessment.checks["train"].rule_id.value == "F3"
    assert assessment.receipt == original


@pytest.mark.parametrize("identifier", ["mit", "openmdw-1.1"])
def test_new_first_party_source_requires_its_openmdw_grant(tmp_path, pinned_dataset_metadata, identifier):
    repositories, _ = pinned_dataset_metadata
    policy = load_ingest_config(CONFIGS / "litmus-train.json").model_copy(update={"source_id": "Marin:synthetic"})
    repositories[policy.upstream.repository, policy.upstream.revision] = {
        "README.md": f"---\nlicense: {identifier}\n---\nSynthetic examples.\n".encode(),
    }
    job = license_job(tmp_path, policy, {})

    if identifier == "mit":
        with pytest.raises(LicenseBlocked) as blocked:
            execute_stage(license_stage, job)
        assessment = LicenseAssessment.model_validate(read_ref(blocked.value.assessment))
        assert assessment.disposition is LicenseDisposition.BAN
    else:
        artifact = execute_stage(license_stage, job)
        assessment = LicenseAssessment.model_validate(
            read_ref(load_receipt(artifact.receipt, job.inputs).outputs["assessment"])
        )
        assert assessment.disposition is LicenseDisposition.PERMIT
        assert assessment.receipt.license_ids == (identifier,)
