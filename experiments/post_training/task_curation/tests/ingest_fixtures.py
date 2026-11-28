# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Real QUICK conversion feeds dedup and census from the sealed upstream map."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import CommandSemantics, EnvironmentRequirements

from experiments.post_training.task_curation.backend_census_manifest import seal_manifest
from experiments.post_training.task_curation.ingest_stages import conversion, normalized_population
from experiments.post_training.task_curation.ingest_stages.graph import IngestExecution, StageJob, execute_stage
from experiments.post_training.task_curation.ingest_stages.ingest_config import load_ingest_config
from experiments.post_training.task_curation.ingest_stages.receipts import (
    IngestStage,
    StageInputs,
    StageReceipt,
    load_receipt,
    seal_receipt,
)
from experiments.post_training.task_curation.ingest_stages.upstream import (
    SourceDigest,
    StagedUpstream,
    UpstreamFile,
    UpstreamInventory,
)
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef

CONFIGS = Path(__file__).parents[1] / "ingest_stages/configs"


def next_job(job, handler, stage):
    artifact = execute_stage(handler, job)
    receipt = load_receipt(artifact.receipt, job.inputs)
    inputs = job.inputs.model_copy(
        update={
            "stage": stage,
            "dependencies": {job.inputs.stage: artifact.receipt},
            "artifacts": {
                **job.inputs.artifacts,
                **{f"{job.inputs.stage.value}/{name}": ref for name, ref in receipt.outputs.items()},
            },
        }
    )
    return replace(job, inputs=inputs, output_root=str(StoragePath(job.output_root).parent / stage.value)), receipt


@pytest.fixture
def staged_job(tmp_path, monkeypatch):
    # Placement and environment provisioning are the external boundaries. The
    # converter, Zephyr execution, dedup, parquet writers and census are real.
    monkeypatch.setattr(conversion, "regional_worker", lambda *args: "fixture")
    monkeypatch.setattr(normalized_population, "regional_worker", lambda *args: "fixture")
    environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS, packages_lock="/fixture/lock"
    )
    monkeypatch.setattr(conversion, "resolved_grader", lambda *args: (environment, {}))
    policy = load_ingest_config(CONFIGS / "litmus-train.json")
    execution = IngestExecution(
        policy=policy,
        source_revision="a" * 40,
        artifact_version="2026.11.28",
        artifact_root=str(tmp_path),
        worker_argv=("python", "-m", "worker"),
        initial_artifacts={},
    )
    common = dict(
        source_id=policy.source_id,
        config_sha256=policy.digest(),
        source_revision=execution.source_revision,
        worker_argv_sha256=canonical_sha256({"argv": list(execution.worker_argv)}),
    )
    parent_inputs = StageInputs(stage=IngestStage.STAGE_UPSTREAM, dependencies={}, artifacts={}, **common)
    row = {
        "agent_ref": {"name": "litmus_agent_agent"},
        "responses_create_params": {"input": [{"role": "user", "content": "Report the value as [[value]]."}]},
        "question_license": "OpenMDW-1.1",
        "output_regex": r"\[\[([-\d.]+)\]\]",
        "expected_answer": 5.0,
        "answer_type": "float",
        "match": {"rule": "abs_window", "abs_tol": 0.2},
    }
    payload = (json.dumps(row) + "\n") * 2
    raw = tmp_path / "train.jsonl"
    raw.write_text(payload)
    raw_ref = ArtifactRef(uri=str(raw), sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
    inventory = UpstreamInventory(
        input_sha256=parent_inputs.digest(),
        source_id=policy.source_id,
        upstream=policy.upstream,
        private=False,
        files=(
            UpstreamFile(
                path="train.jsonl", size=len(raw.read_bytes()), algorithm=SourceDigest.LFS_SHA256, digest=raw_ref.sha256
            ),
        ),
    )
    inventory_ref = seal_manifest(inventory.model_dump(mode="json"), str(tmp_path / "inventory.json"))
    staged = StagedUpstream(inventory=inventory_ref, worker_region="fixture", files={"train.jsonl": raw_ref})
    outputs = {
        "inventory": inventory_ref,
        "staged": seal_manifest(staged.model_dump(mode="json"), str(tmp_path / "staged.json")),
        "raw/train.jsonl": raw_ref,
    }
    parent = seal_receipt(str(tmp_path / "upstream"), StageReceipt(inputs=parent_inputs, outputs=outputs))
    inputs = StageInputs(
        stage=IngestStage.CONVERT,
        dependencies={IngestStage.STAGE_UPSTREAM: parent},
        artifacts={f"stage-upstream/{name}": ref for name, ref in outputs.items()},
        **common,
    )
    return StageJob(execution, inputs, str(tmp_path / "convert"))
