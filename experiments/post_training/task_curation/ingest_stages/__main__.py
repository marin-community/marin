# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-source Atlas ingest and repair with independent starting-state gates."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import requests
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.backend_census_manifest import seal_manifest
from experiments.post_training.task_curation.ingest_stages.graph import IngestExecution
from experiments.post_training.task_curation.ingest_stages.guard import assert_checkout_allowed, checkout_witness
from experiments.post_training.task_curation.ingest_stages.host_inputs import metadata_inputs
from experiments.post_training.task_curation.ingest_stages.ingest_config import IngestMode, load_ingest_config
from experiments.post_training.task_curation.ingest_stages.planning import derive_execution
from experiments.post_training.task_curation.ingest_stages.preflight import worker_preflight
from experiments.post_training.task_curation.ingest_stages.qualification_run import land_qualification
from experiments.post_training.task_curation.ingest_stages.queue import QueueClaim, claim_queue
from experiments.post_training.task_curation.ingest_stages.requester import Requester
from experiments.post_training.task_curation.ingest_stages.submission import (
    job_name,
    review_sample,
    sample_job_name,
    submit_worker,
    upload_metadata,
)
from experiments.post_training.task_curation.ingest_stages.worker import WorkerPhase, worker_argv
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref

REPO_ROOT = Path(__file__).resolve().parents[4]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=IngestMode, choices=list(IngestMode))
    parser.add_argument("mode", choices=("preflight", "dry-run", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--queue", type=Path, default=Path.home() / "Documents/experiments/active/new-rl-data-ingest/CANDIDATES.md"
    )
    parser.add_argument(
        "--watchdog-feed", type=Path, default=Path.home() / "Documents/experiments/active/job-watchdog/alerts.jsonl"
    )
    parser.add_argument("--landing-log", type=Path, default=Path.home() / ".local/share/atlas-push/landings.jsonl")
    return parser


def alert(feed: Path, kind: str, source_id: str, evidence: dict) -> None:
    feed.parent.mkdir(parents=True, exist_ok=True)
    with feed.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(
            json.dumps({"at": datetime.now(UTC).isoformat(), "kind": kind, "source_id": source_id, **evidence}) + "\n"
        )


def execute(args: argparse.Namespace) -> dict:
    policy = load_ingest_config(args.config)
    if policy.mode is not args.workflow:
        raise ValueError("The config mode does not match this command's workflow")
    if len(policy.intended_use) != 1:
        raise ValueError("The canonical Atlas declaration supports one intended use per source")
    configure_coreweave_s3()
    with TemporaryDirectory(prefix="atlas-ingest-host-") as temporary:
        output = Path(temporary)
        references = metadata_inputs(policy, REPO_ROOT, args.queue, output)
        execution = derive_execution(args.config, REPO_ROOT, worker_argv(policy.mode), references)
        witness = checkout_witness(REPO_ROOT, policy)
        preflight = worker_preflight(execution, witness, REPO_ROOT)
        if args.mode != "run":
            return {
                "workflow": policy.mode.value,
                "mode": args.mode,
                "source_id": policy.source_id,
                "preflight": preflight,
                "planned_jobs": [
                    job_name(execution, WorkerPhase.PREPARE),
                    sample_job_name(execution),
                    job_name(execution, WorkerPhase.COMPLETE),
                ],
                "worker_argv": list(execution.worker_argv),
            }
        assert_checkout_allowed(REPO_ROOT, policy, witness, args.landing_log)
        lock_root = Path.home() / ".local/share/atlas-ingest/locks"
        lock_root.mkdir(parents=True, exist_ok=True)
        with (lock_root / policy.digest()).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            frozen = StoragePath(execution.artifact_root) / "execution.json"
            if frozen.exists():
                previous = IngestExecution.model_validate_json(frozen.read_bytes())
                if previous.policy != policy or previous.source_revision != execution.source_revision:
                    raise ValueError("Saved execution belongs to another source policy or code revision")
                execution = previous
            else:
                execution = upload_metadata(execution)
                seal_manifest(execution.model_dump(mode="json"), str(frozen))
            captured = QueueClaim.model_validate(read_ref(execution.initial_artifacts["queue-claim"]))
            if policy.mode is IngestMode.INGEST and claim_queue(args.queue, policy, captured.selection) != captured:
                raise ValueError("Live queue claim differs from the frozen preflight selection")
            if not os.environ.get("GITHUB_TOKEN"):
                os.environ["GITHUB_TOKEN"] = subprocess.check_output(["gh", "auth", "token"], text=True).strip()
            alert(args.watchdog_feed, "atlas-ingest-start", policy.source_id, {"revision": execution.source_revision})
            prepared = submit_worker(execution, witness, WorkerPhase.PREPARE, REPO_ROOT, args.landing_log)
            review_sample(prepared, execution, REPO_ROOT)
            completed = submit_worker(execution, witness, WorkerPhase.COMPLETE, REPO_ROOT, args.landing_log)
            revision = land_qualification(execution, completed, REPO_ROOT, args.landing_log)
            outputs = read_ref(completed)["outputs"]
            declaration_ref = ArtifactRef.model_validate(outputs["declaration"])
            declaration = read_ref(declaration_ref)
            claim = QueueClaim.model_validate(read_ref(execution.initial_artifacts["queue-claim"]))
            with requests.Session() as session:
                session.headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
                terminal = Requester(session).terminal(claim, declaration["status"], declaration_ref)
            result = {
                "source_id": policy.source_id,
                "status": declaration["status"],
                "landing_revision": revision,
                "receipt": completed.model_dump(mode="json"),
                "terminal": terminal.model_dump(mode="json"),
            }
            alert(args.watchdog_feed, "atlas-ingest-complete", policy.source_id, result)
            return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(execute(args), indent=2, sort_keys=True))
    return 0


def ingest_main() -> int:
    return main([IngestMode.INGEST.value, *sys.argv[1:]])


def repair_main() -> int:
    return main([IngestMode.REPAIR.value, *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
