# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The exact argv shared by local preflight and regional ingest workers."""

import argparse
import json
import os
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import cast

import requests
from marin.execution.lazy import materialized_config
from pydantic import BaseModel, ConfigDict
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.identity import canonical_sha256

from experiments.post_training.task_curation.backend_census_manifest import seal_manifest
from experiments.post_training.task_curation.ingest_stages.backend import backend_oracle_stage
from experiments.post_training.task_curation.ingest_stages.conversion import convert_stage
from experiments.post_training.task_curation.ingest_stages.converter_binding import load_converter
from experiments.post_training.task_curation.ingest_stages.eligibility import require_starting_state
from experiments.post_training.task_curation.ingest_stages.graph import (
    IngestExecution,
    StageHandler,
    StageJob,
    ingest_graph,
    run_ingest_graph,
)
from experiments.post_training.task_curation.ingest_stages.guard import (
    SourceWitness,
    assert_bundle_allowed,
    verify_source_witness,
)
from experiments.post_training.task_curation.ingest_stages.ingest_config import IngestMode
from experiments.post_training.task_curation.ingest_stages.lexical import lexical_stage
from experiments.post_training.task_curation.ingest_stages.licensing import license_stage
from experiments.post_training.task_curation.ingest_stages.normalized_population import census_stage, dedup_stage
from experiments.post_training.task_curation.ingest_stages.overlap import fingerprint_overlap_stage
from experiments.post_training.task_curation.ingest_stages.placement import regional_worker
from experiments.post_training.task_curation.ingest_stages.publication import declare_stage
from experiments.post_training.task_curation.ingest_stages.qc import (
    prepare_sample,
    require_sample_capacity,
    sample_qc_stage,
)
from experiments.post_training.task_curation.ingest_stages.receipts import IngestStage
from experiments.post_training.task_curation.ingest_stages.requester import Requester, claim_stage
from experiments.post_training.task_curation.ingest_stages.upstream import stage_upstream
from experiments.post_training.task_curation.local import artifact_prefix
from experiments.post_training.task_curation.release_stages.manifest import read_ref

MODULE = "experiments.post_training.task_curation.ingest_stages.worker"
REQUEST_FILENAME = "atlas-ingest-request.json"
REPORT_FILENAME = "atlas-ingest-report.json"
LANDING_FILENAME = "atlas-ingest-landings.jsonl"


class WorkerPhase(StrEnum):
    PREFLIGHT = "preflight"
    PREPARE = "prepare"
    COMPLETE = "complete"


class WorkerRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    phase: WorkerPhase
    execution: IngestExecution
    witness: SourceWitness


def worker_argv(mode: IngestMode) -> tuple[str, ...]:
    return ("uv", "run", "--no-sync", "python", "-m", MODULE, "--workflow", mode.value, "--request", REQUEST_FILENAME)


def production_handlers(requester: Requester) -> dict[IngestStage, StageHandler]:
    return {
        IngestStage.CLAIM: partial(claim_stage, requester=requester),
        IngestStage.LICENSE: license_stage,
        IngestStage.STAGE_UPSTREAM: stage_upstream,
        IngestStage.CONVERT: convert_stage,
        IngestStage.DEDUP: dedup_stage,
        IngestStage.CENSUS: census_stage,
        IngestStage.FINGERPRINT_OVERLAP: fingerprint_overlap_stage,
        IngestStage.BACKEND_ORACLE: backend_oracle_stage,
        IngestStage.LEXICAL: lexical_stage,
        IngestStage.SAMPLE_QC: sample_qc_stage,
        IngestStage.DECLARE: declare_stage,
    }


def execute_request(request: WorkerRequest, workflow: IngestMode, repo_root: Path) -> dict:
    execution = request.execution
    if execution.policy.mode is not workflow or execution.worker_argv != worker_argv(workflow):
        raise ValueError("worker argv or workflow differs from the reviewed source policy")
    if len(execution.policy.intended_use) != 1:
        raise ValueError("The canonical Atlas declaration supports one intended use per source")
    verify_source_witness(repo_root, execution.policy, request.witness)
    if execution.source_revision != request.witness.source_revision:
        raise ValueError("worker execution differs from its source witness")
    if request.phase is WorkerPhase.PREFLIGHT:
        if any(StoragePath(ref.uri).scheme not in (None, "file") for ref in execution.initial_artifacts.values()):
            raise ValueError("local preflight requires local metadata stubs before any object-store read")
    else:
        configure_coreweave_s3()
        regional_worker(execution.policy.cluster, execution.artifact_root, execution.artifact_root)
    declarations = read_ref(execution.initial_artifacts["starting-declarations"])
    require_starting_state(execution.policy, declarations)
    load_converter(execution.policy.converter_module)
    require_sample_capacity(repo_root)
    with requests.Session() as session:
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            session.headers["Authorization"] = f"Bearer {token}"
        graph = ingest_graph(execution, production_handlers(Requester(session)))
        plan = [
            {"stage": stage.value, "fingerprint": node.fingerprint()}
            for stage, node in zip(IngestStage, graph.nodes, strict=True)
        ]
        if request.phase is WorkerPhase.PREFLIGHT:
            return {
                "phase": request.phase.value,
                "execution_sha256": canonical_sha256(execution.identity()),
                "stages": plan,
            }
        assert_bundle_allowed(repo_root, execution.policy, request.witness, repo_root / LANDING_FILENAME)
        if request.phase is WorkerPhase.PREPARE:
            run_ingest_graph(graph, IngestStage.LEXICAL)
            with artifact_prefix(execution.artifact_root):
                step = graph.nodes[list(IngestStage).index(IngestStage.SAMPLE_QC)]
                job = cast(StageJob, materialized_config(step, execution.artifact_root))
                sample = prepare_sample(job)
            result = {
                "phase": request.phase.value,
                "sample_request": sample.model_dump(mode="json"),
                "qc_outcome_uri": prefix_join(job.output_root, "outcome.json"),
            }
        else:
            terminal = run_ingest_graph(graph)
            result = {
                "phase": request.phase.value,
                "outputs": {name: ref.model_dump(mode="json") for name, ref in terminal.outputs.items()},
            }
    result["execution_sha256"] = canonical_sha256(execution.identity())
    reference = seal_manifest(result, prefix_join(execution.artifact_root, f"{request.phase.value}-report.json"))
    return {**result, "report": reference.model_dump(mode="json")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=IngestMode, choices=list(IngestMode), required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    request = WorkerRequest.model_validate_json(args.request.read_bytes())
    result = execute_request(request, args.workflow, Path(__file__).resolve().parents[4])
    Path(REPORT_FILENAME).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
