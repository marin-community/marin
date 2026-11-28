# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned payload acquisition stays regional and refuses same-size corrupt objects."""

import hashlib
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import fsspec
import pytest
from fsspec.implementations.memory import MemoryFileSystem
from huggingface_hub import HfApi
from huggingface_hub.hf_api import BlobLfsInfo, RepoSibling
from iris.cluster.client.job_info import JobInfo, get_job_info, set_job_info
from iris.cluster.types import JobName
from taskcompendium.identity import canonical_sha256

from experiments.post_training.task_curation.atlas_deployment import region_deployment
from experiments.post_training.task_curation.backend_census_manifest import seal_manifest
from experiments.post_training.task_curation.ingest_stages.graph import (
    IngestExecution,
    StageJob,
    execute_stage,
    ingest_graph,
    verify_built_stages,
)
from experiments.post_training.task_curation.ingest_stages.ingest_config import load_ingest_config
from experiments.post_training.task_curation.ingest_stages.licensing import LicenseAssessment
from experiments.post_training.task_curation.ingest_stages.receipts import (
    IngestStage,
    StageInputs,
    StageReceipt,
    load_receipt,
    seal_receipt,
)
from experiments.post_training.task_curation.ingest_stages.upstream import StagedUpstream, stage_upstream
from experiments.post_training.task_curation.license_check import CHECKER_VERSION
from experiments.post_training.task_curation.release_stages.manifest import read_ref
from experiments.post_training.task_curation.tests.license_fixtures import fixture_license

CONFIGS = Path(__file__).parents[1] / "ingest_stages/configs"


class MemoryS3(MemoryFileSystem):
    """S3 protocol transport with actual fsspec operations against memory."""

    protocol = "s3"

    @classmethod
    def _strip_protocol(cls, path):
        return MemoryFileSystem._strip_protocol(str(path).removeprefix("s3://"))


class MemoryHf(MemoryFileSystem):
    """HF file listing returns revision-qualified paths like HfFileSystem."""

    protocol = "hf"

    @classmethod
    def _strip_protocol(cls, path):
        return MemoryFileSystem._strip_protocol(str(path).removeprefix("hf://"))

    def glob(self, path, **kwargs):
        parts = self._strip_protocol(path).strip("/").split("/")
        parts[2] += "@" + kwargs.pop("revision")
        rows = super().glob("/" + "/".join(parts), **kwargs)
        return {name.lstrip("/"): {**info, "name": name.lstrip("/")} for name, info in rows.items()}


@pytest.fixture
def remote(monkeypatch):
    original = {name: fsspec.get_filesystem_class(name) for name in ("s3", "hf")}
    before = set(MemoryFileSystem.store)
    records = {}
    fsspec.register_implementation("s3", MemoryS3, clobber=True)
    fsspec.register_implementation("hf", MemoryHf, clobber=True)

    def info(_client, repository, *, revision, files_metadata):
        assert files_metadata
        return SimpleNamespace(sha=revision, siblings=records[repository, revision])

    monkeypatch.setattr(HfApi, "dataset_info", info)
    prior = get_job_info()
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    set_job_info(JobInfo(task_id=JobName.from_wire("/worker/atlas-ingest-test").task(0), worker_region="US-EAST-02A"))
    yield records
    set_job_info(prior)
    for name, implementation in original.items():
        fsspec.register_implementation(name, implementation, clobber=True)
    for name in set(MemoryFileSystem.store) - before:
        MemoryFileSystem.store.pop(name)


def populate_source(records, policy, files):
    siblings = []
    filesystem = MemoryHf()
    for index, (name, content) in enumerate(files.items()):
        path = f"datasets/{policy.upstream.repository}@{policy.upstream.revision}/{name}"
        filesystem.pipe_file(path, content)
        sha256 = hashlib.sha256(content).hexdigest()
        # An LFS sibling's Git blob identifies its pointer, not its payload.
        blob_content = f"oid sha256:{sha256}\nsize {len(content)}\n".encode() if index == 0 else content
        blob = subprocess.check_output(["git", "hash-object", "--stdin"], input=blob_content).decode().strip()
        siblings.append(
            RepoSibling(
                rfilename=name,
                size=len(content),
                blob_id=blob,
                lfs=BlobLfsInfo(size=len(content), sha256=sha256, pointer_size=127) if index == 0 else None,
            )
        )
    records[policy.upstream.repository, policy.upstream.revision] = siblings


def upstream_job(tmp_path, policy):
    deployment = region_deployment(policy.cluster)
    execution = IngestExecution(
        policy=policy,
        source_revision="a" * 40,
        artifact_version="2026.11.28",
        artifact_root=deployment.region_prefix + f"tests/{tmp_path.name}",
        worker_argv=("python", "-m", "worker"),
        initial_artifacts={},
    )
    common = dict(
        source_id=policy.source_id,
        config_sha256=policy.digest(),
        source_revision=execution.source_revision,
        worker_argv_sha256=canonical_sha256({"argv": list(execution.worker_argv)}),
    )
    parent_inputs = StageInputs(stage=IngestStage.LICENSE, dependencies={}, artifacts={}, **common)
    assessment = LicenseAssessment(
        input_sha256=parent_inputs.digest(),
        checker_version=CHECKER_VERSION,
        source_id=policy.source_id,
        dataset_id=policy.upstream.repository,
        revision=policy.upstream.revision,
        inherited=False,
        disposition="permit",
        receipt=fixture_license(policy.source_id, policy.upstream.repository, policy.upstream.revision),
        checks={},
        refusals=(),
    )
    reference = seal_manifest(assessment.model_dump(mode="json"), str(tmp_path / "assessment.json"))
    parent = seal_receipt(
        str(tmp_path / "license"), StageReceipt(inputs=parent_inputs, outputs={"assessment": reference})
    )
    inputs = StageInputs(
        stage=IngestStage.STAGE_UPSTREAM,
        dependencies={IngestStage.LICENSE: parent},
        artifacts={"license/assessment": reference},
        **common,
    )
    return StageJob(execution=execution, inputs=inputs, output_root=execution.artifact_root + "/stage-upstream")


@pytest.mark.parametrize("config", ["litmus-train", "mimo-repair"])
def test_regional_acquisition_seals_only_selected_files_with_upstream_hashes(tmp_path, remote, config):
    policy = load_ingest_config(CONFIGS / f"{config}.json")
    if config == "litmus-train":
        files = {"train.jsonl": b'{"task":"one"}\n', "validation.jsonl": b'{"task":"held-out"}\n'}
        selected = {"train.jsonl"}
    else:
        files = {"mimo_v1/one.tar": b"one", "mimo_v1/two.tar": b"two", "other_v1/one.tar": b"other"}
        selected = {"mimo_v1/one.tar", "mimo_v1/two.tar"}
    populate_source(remote, policy, files)
    job = upstream_job(tmp_path, policy)

    artifact = execute_stage(stage_upstream, job)
    receipt = load_receipt(artifact.receipt, job.inputs)
    staged = StagedUpstream.model_validate(read_ref(receipt.outputs["staged"]))

    assert set(staged.files) == selected
    filesystem = MemoryS3()
    for name, reference in staged.files.items():
        assert filesystem.cat_file(reference.uri) == files[name]
        assert reference.sha256 == hashlib.sha256(files[name]).hexdigest()
    assert not filesystem.exists(job.output_root + "/raw/other_v1")
    assert not filesystem.exists(job.output_root + "/raw/validation.jsonl")
    damaged = staged.files[sorted(selected)[0]]
    filesystem.pipe_file(damaged.uri, b"damaged after sealing")
    with pytest.raises(ValueError, match="evidence digest mismatch"):
        load_receipt(artifact.receipt, job.inputs)


def test_acquisition_refuses_same_size_corruption_in_a_reused_destination(tmp_path, remote):
    policy = load_ingest_config(CONFIGS / "litmus-train.json")
    populate_source(remote, policy, {"train.jsonl": b"correct"})
    job = upstream_job(tmp_path, policy)
    filesystem = MemoryS3()
    path = job.output_root + "/raw/train.jsonl"
    filesystem.pipe_file(path, b"corrupt")

    with pytest.raises(ValueError, match="pinned upstream content digest"):
        execute_stage(stage_upstream, job)

    assert filesystem.cat_file(path) == b"corrupt"
    assert not filesystem.exists(job.output_root + "/staged.json")


@pytest.mark.parametrize("region", [None, "US-EAST-08A"])
def test_unidentified_or_wrong_region_worker_cannot_create_dataset_objects(tmp_path, remote, region):
    policy = load_ingest_config(CONFIGS / "litmus-train.json")
    job = upstream_job(tmp_path, policy)
    set_job_info(
        None
        if region is None
        else JobInfo(
            task_id=JobName.from_wire("/worker/atlas-ingest-test").task(0),
            worker_region=region,
        )
    )

    with pytest.raises(ValueError, match="region"):
        execute_stage(stage_upstream, job)

    assert not MemoryS3().exists(job.output_root)
    assert remote == {}


def test_cross_region_output_root_is_refused_before_listing_or_transfer(tmp_path, remote):
    policy = load_ingest_config(CONFIGS / "litmus-train.json")
    job = upstream_job(tmp_path, policy)
    root = "s3://marin-us-west-04a/ingest-test"
    job = replace(job, execution=job.execution.model_copy(update={"artifact_root": root}), output_root=root + "/raw")

    with pytest.raises(ValueError):
        execute_stage(stage_upstream, job)

    assert not MemoryS3().exists(root)
    assert remote == {}


def test_interrupted_stage_reuses_valid_file_bytes_and_its_frozen_inventory(tmp_path, remote):
    policy = load_ingest_config(CONFIGS / "litmus-train.json")
    populate_source(remote, policy, {"train.jsonl": b"correct"})
    job = upstream_job(tmp_path, policy)
    original = stage_upstream(job)
    reference = original["raw/train.jsonl"]
    filesystem = MemoryS3()
    before = filesystem.info(reference.uri)["created"]
    remote.clear()  # The original metadata request is unavailable on this retry.

    artifact = execute_stage(stage_upstream, job)
    receipt = load_receipt(artifact.receipt, job.inputs)

    assert receipt.outputs == original
    assert filesystem.info(reference.uri)["created"] == before
    assert filesystem.cat_file(reference.uri) == b"correct"


def test_mutable_transfer_listing_is_refused_even_with_matching_file_size(tmp_path, remote, monkeypatch):
    policy = load_ingest_config(CONFIGS / "litmus-train.json")
    populate_source(remote, policy, {"train.jsonl": b"correct"})
    MemoryHf().pipe_file(f"datasets/{policy.upstream.repository}@main/train.jsonl", b"correct")
    job = upstream_job(tmp_path, policy)
    original_glob = MemoryHf.glob

    def unpinned_listing(filesystem, path, **kwargs):
        return original_glob(filesystem, path, **{**kwargs, "revision": "main"})

    monkeypatch.setattr(MemoryHf, "glob", unpinned_listing)
    with pytest.raises(ValueError, match="download plan differs"):
        execute_stage(stage_upstream, job)

    assert not MemoryS3().exists(job.output_root + "/raw/train.jsonl")
    assert not MemoryS3().exists(job.output_root + "/staged.json")


def test_remote_cache_verification_refuses_local_reads_before_inspecting_objects(tmp_path, remote, monkeypatch):
    policy = load_ingest_config(CONFIGS / "litmus-train.json")
    job = upstream_job(tmp_path, policy)
    graph = ingest_graph(job.execution, dict.fromkeys(IngestStage, stage_upstream))
    set_job_info(None)

    def forbidden_inspection(*args, **kwargs):
        raise AssertionError("remote cache inspected before worker placement")

    monkeypatch.setattr(MemoryS3, "info", forbidden_inspection)
    with pytest.raises(ValueError, match="region-identified Iris worker"):
        verify_built_stages(graph)
