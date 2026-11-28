# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit source-region workers and a separate GB200 SAMPLE root from the host."""

import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256

from experiments.post_training.task_curation.backend_census_manifest import seal_manifest
from experiments.post_training.task_curation.ingest_stages.graph import IngestExecution
from experiments.post_training.task_curation.ingest_stages.guard import (
    SOURCE_SUFFIXES,
    SourceWitness,
    assert_checkout_allowed,
    converter_directory,
)
from experiments.post_training.task_curation.ingest_stages.qc import QC_CLUSTER, QCOutcome, QCRequest, sample_outcome
from experiments.post_training.task_curation.ingest_stages.worker import (
    LANDING_FILENAME,
    REQUEST_FILENAME,
    WorkerPhase,
    WorkerRequest,
)
from experiments.post_training.task_curation.judges import require_review_serving_root
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref

JOB_PREFIX = "atlas-ingest"
JOB_OWNER = "benfeuer"
SOURCE_CPU = 16
SOURCE_MEMORY = "64GB"
SOURCE_DISK = "128GB"
JOB_TIMEOUT = 259200


def job_name(execution: IngestExecution, phase: WorkerPhase) -> str:
    return f"{JOB_PREFIX}-{execution.policy.digest()[:12]}-{execution.source_revision[:12]}-{phase.value}"


def sample_job_name(execution: IngestExecution) -> str:
    return f"{JOB_PREFIX}-{execution.policy.digest()[:12]}-{execution.source_revision[:12]}-sample"


def upload_metadata(execution: IngestExecution) -> IngestExecution:
    references = {}
    for name, reference in execution.initial_artifacts.items():
        references[name] = seal_manifest(
            read_ref(reference), prefix_join(execution.artifact_root, f"inputs/{reference.sha256}.json")
        )
    return execution.model_copy(update={"initial_artifacts": references})


def submit_worker(
    execution: IngestExecution, witness: SourceWitness, phase: WorkerPhase, repo_root: Path, landing_log: Path
) -> ArtifactRef:
    """Bundle a clean landed revision; runtime documents never enter the checkout."""
    assert_checkout_allowed(repo_root, execution.policy, witness, landing_log)
    path = StoragePath(prefix_join(execution.artifact_root, f"{phase.value}-report.json"))
    if path.exists():
        if json.loads(path.read_bytes())["execution_sha256"] != canonical_sha256(execution.identity()):
            raise ValueError("Worker report belongs to another frozen execution")
        return ArtifactRef(uri=str(path), sha256=file_sha256(path))
    request = WorkerRequest(phase=phase, execution=execution, witness=witness)
    with TemporaryDirectory(prefix=".atlas-ingest-bundle-", dir=repo_root.parent) as temporary:
        bundle = Path(temporary) / "checkout"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(bundle), execution.source_revision], cwd=repo_root, check=True
        )
        try:
            (bundle / REQUEST_FILENAME).write_text(request.model_dump_json(indent=2) + "\n")
            (bundle / LANDING_FILENAME).write_bytes(landing_log.read_bytes())
            # Reuse the host's environment for the submission client. Iris excludes
            # this symlink; its normal setup builds the worker's own environment.
            (bundle / ".venv").symlink_to(repo_root / ".venv", target_is_directory=True)
            command = [
                "uv",
                "run",
                "--no-sync",
                "iris",
                "--cluster=marin",
                "job",
                "run",
                "--target-cluster",
                execution.policy.cluster,
                "--job-name",
                job_name(execution, phase),
                "--user",
                JOB_OWNER,
                "--cpu",
                str(SOURCE_CPU),
                "--memory",
                SOURCE_MEMORY,
                "--disk",
                SOURCE_DISK,
                "--enable-extra-resources",
                "--extra",
                "cpu",
                "--extra",
                "dedup",
                "--extra",
                "daytona",
                "--sync-package",
                "marin-root",
                "--sync-package",
                "marin-shellbox",
                "--priority",
                "interactive",
                "--timeout",
                str(JOB_TIMEOUT),
                "--max-retries",
                "0",
                "--no-cancel-on-exit",
                "--include",
                REQUEST_FILENAME,
                "--include",
                LANDING_FILENAME,
            ]
            for suffix in sorted(SOURCE_SUFFIXES):
                command.extend(["--include", f"**/*{suffix}", "--include", f"*{suffix}"])
            converter = converter_directory(repo_root, execution.policy).relative_to(repo_root).as_posix()
            command.extend(["--include", f"{converter}/**"])
            for name in ("HF_TOKEN", "CW_KEY_ID", "CW_KEY_SECRET", "GITHUB_TOKEN"):
                if os.environ.get(name):
                    command.extend(["--secret", name, f"env:{name}"])
            subprocess.run([*command, "--", *execution.worker_argv], cwd=bundle, check=True)
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(bundle)], cwd=repo_root, check=True)
    if json.loads(path.read_bytes())["execution_sha256"] != canonical_sha256(execution.identity()):
        raise ValueError("Worker report belongs to another frozen execution")
    return ArtifactRef(uri=str(path), sha256=file_sha256(path))


def review_sample(prepared_report: ArtifactRef, execution: IngestExecution, repo_root: Path) -> ArtifactRef:
    """Use the maintained launcher from the host so QC has its own regional root."""
    require_review_serving_root(QC_CLUSTER)
    report = read_ref(prepared_report)
    request_ref = ArtifactRef.model_validate(report["sample_request"])
    request = QCRequest.model_validate(read_ref(request_ref))
    outcome_path = StoragePath(report["qc_outcome_uri"])
    if outcome_path.exists():
        outcome = QCOutcome.model_validate_json(outcome_path.read_bytes())
        if outcome.request_sha256 != request_ref.sha256:
            raise ValueError("Existing QC outcome belongs to another request")
        return ArtifactRef(uri=str(outcome_path), sha256=file_sha256(outcome_path))
    panels = read_ref(request.queue)["panels"]
    array_path = StoragePath(request.queue.uri).parent / "queue.json"
    payload = (json.dumps(panels, sort_keys=True, indent=2) + "\n").encode()
    if array_path.exists() and array_path.read_bytes() != payload:
        raise ValueError("SAMPLE launch queue changed after its request was sealed")
    array_path.write_bytes(payload, auto_mkdir=True)
    queue_sha = file_sha256(array_path)
    name = sample_job_name(execution)
    command = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "experiments.post_training.task_curation.launch_sample_qc",
        "--revision",
        execution.source_revision,
        "--queue-uri",
        str(array_path),
        "--queue-sha256",
        queue_sha,
        "--output-root",
        request.output_root,
        "--review-cache",
        prefix_join(request.output_root, "cache"),
        "--job-name",
        name,
        "--target-cluster",
        QC_CLUSTER,
    ]
    launched = subprocess.run(command, cwd=repo_root, check=False)
    job_id = f"/{JOB_OWNER}/{name}"
    code = launched.returncode
    if not code:
        completed = subprocess.run(
            ["uv", "run", "--no-sync", "iris", "--cluster=marin", "job", "wait", job_id],
            cwd=repo_root,
            check=False,
        )
        code = completed.returncode
    outcome = sample_outcome(request_ref, job_id, code)
    return seal_manifest(outcome.model_dump(mode="json"), report["qc_outcome_uri"])
