# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify an approved LOCAL ownership setup failure without rewriting its zero Result."""

import hashlib
import json
from pathlib import Path, PurePosixPath

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256, task_spec_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.oracle_filter import InfrastructureRetryApproval, OracleTrialReceipt
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.archive_evidence import verified_archive_members

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def require_bootstrap_failure(task: TaskSpec, parent: OracleTrialReceipt, approval: InfrastructureRetryApproval) -> None:
    """Check real archived logs, unchanged source bootstrap, reproduction and exact-parent decision."""
    proof = approval.bootstrap_failure
    if proof is None:
        return
    proof.require_parent(parent)
    if proof.source_task_sha256 != task_spec_sha256(task):
        raise ValueError("Bootstrap failure belongs to another frozen source task")
    archive_uri, separator, log_member = proof.oracle_log.uri.partition("#")
    if not separator or archive_uri != proof.archive.uri:
        raise ValueError("Bootstrap log must be an exact member of the original oracle archive")
    observed = verified_archive_members(proof.archive, {log_member, proof.result_member, proof.oracle_exit_member})
    log = observed[log_member]
    if (
        hashlib.sha256(log).hexdigest() != proof.oracle_log.sha256
        or log.decode() != parent.failure_log
        or canonical_sha256(json.loads(observed[proof.result_member])) != parent.result_sha256
        or observed[proof.oracle_exit_member].strip() != b"1"
    ):
        raise ValueError("Bootstrap evidence differs from the immutable production oracle attempt")
    root = PurePosixPath(proof.result_member).parent
    if (
        PurePosixPath(log_member) != root / "agent/oracle.txt"
        or PurePosixPath(proof.oracle_exit_member) != root / "agent/exit-code.txt"
    ):
        raise ValueError("Bootstrap log, exit code and Result must come from the same production attempt")
    resources = {
        resource.path: resource_bytes(resource)
        for group in (task.resources.all, task.resources.worker, task.resources.oracle)
        for resource in group
    }
    oracle = resources["solution/solve.sh"]
    setup = resources["setup_files/setup.sh"]
    commands = [
        line.strip() for line in oracle.decode().splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    if (
        hashlib.sha256(oracle).hexdigest() != proof.oracle_script_sha256
        or hashlib.sha256(setup).hexdigest() != proof.setup_script_sha256
        or commands[:2] != ["set -e", "bash /setup_files/setup.sh"]
        or "set -euo pipefail" not in setup.decode().splitlines()
        or f"chown -R {proof.target_uid}:{proof.target_gid} /task_file" not in setup.decode()
    ):
        raise ValueError("Original source must fail in its unchanged setup before the oracle payload")
    documents = []
    for reference in (proof.reproduction, proof.decision):
        raw = StoragePath(reference.uri).read_bytes()
        if hashlib.sha256(raw).hexdigest() != reference.sha256:
            raise ValueError("Bootstrap diagnosis or exact-parent decision bytes changed")
        documents.append(json.loads(raw))
    reproduction, decision = documents
    baseline, candidate = reproduction["results"]
    files = reproduction.get("inputs", {}).get("files", {})
    required = {"candidate/shellbox/backends/local/launch.py", "candidate/shellbox/backends/local/machine.py"}
    if not required <= files.keys():
        raise ValueError("Bootstrap reproduction lacks the executed LOCAL backend source hashes")
    for name, digest in files.items():
        if not name.startswith("candidate/shellbox/"):
            continue
        relative = PurePosixPath(name.removeprefix("candidate/"))
        if (
            ".." in relative.parts
            or hashlib.sha256((PROJECT_ROOT / "lib/shellbox/src" / relative).read_bytes()).hexdigest() != digest
        ):
            raise ValueError("LOCAL backend source differs from its clean ownership reproduction")
    if (
        reproduction.get("passed") is not True
        or baseline["variant"] != "baseline"
        or baseline["image_exit"] != 1
        or "Invalid argument" not in baseline["image_stderr"]
        or candidate["variant"] != "candidate"
        or candidate["image_exit"] != 0
        or candidate["image_stderr"]
        or decision.get("classification") != "our_infrastructure"
        or decision.get("approved_parents", {}).get(task.id) != parent.result_sha256
        or decision.get("source_contracts_changed") is not False
    ):
        raise ValueError("Bootstrap retry lacks a clean backend reproduction or exact-parent authority")
