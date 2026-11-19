# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify source-verifier exceptions against original parser and captured log bytes."""

import ast
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ScriptGrader, TaskSpec
from taskcompendium.pipeline.oracle_filter import OracleTrialReceipt, SourceVerifierError
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.archive_evidence import verified_archive_members
from experiments.post_training.task_curation.datasets.prime_intellect.multiswe.host_grade import source_test_output

UPSTREAM_TASKSET_SHA256 = "e9709befcbba78af2092cf489866b0dda183a28b02ad15e18da1477de67a8f1b"
UPSTREAM_REVISION = "b41d8759070715dc102000e06fba0d73ebce9955"
PARSER_FUNCTIONS = frozenset({"make_instance", "validate_report", "calculate_reward"})
OVERLAP_ERROR = "ValueError: Passed tests and failed tests should not have common items:"


def multiswe_parser_sha256(task: TaskSpec) -> str:
    """Fingerprint exact source parser resources and its declared dependency context."""
    if not isinstance(task.grader, ScriptGrader) or task.grader.environment.docker_build is None:
        raise ValueError("Multi-SWE source errors require the original declared parser build")
    files = [
        (resource.path, hashlib.sha256(resource_bytes(resource)).hexdigest())
        for resource in task.resources.verifier
        if resource.path.startswith("multiswe/")
    ]
    files.extend(
        ("parser-context/" + resource.path, hashlib.sha256(resource_bytes(resource)).hexdigest())
        for resource in task.grader.environment.docker_build.files
    )
    return canonical_sha256(sorted(files))


def collected_source_files(trial: OracleTrialReceipt, proof: SourceVerifierError) -> dict[str, bytes]:
    """Read SHA-verified files that the production trial copied from the actor."""
    directory = PurePosixPath(trial.path).parent.relative_to("/tmp") / "verifier/source-artifacts"
    index_member = str(directory / "index.json")
    index = json.loads(verified_archive_members(proof.archive, {index_member})[index_member])
    members = verified_archive_members(proof.archive, {str(directory / item["file"]) for item in index})
    files = {}
    for item in index:
        payload = members[str(directory / item["file"])]
        if len(payload) != item["bytes"] or hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise ValueError("Source-artifact archive differs from the production index")
        with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
            for member in archive.getmembers():
                if not member.isfile() or "/" in member.name or member.name in files:
                    raise ValueError("Source-artifact archive has ambiguous files")
                stream = archive.extractfile(member)
                assert stream is not None
                files[member.name] = stream.read()
    return files


def require_source_verifier_error(task: TaskSpec, trial: OracleTrialReceipt, proof: SourceVerifierError) -> None:
    """Recheck the approved upstream exception without changing its parser or score."""
    if (
        task.source.dataset != "PrimeIntellect/Multi-SWE-RL-Verified"
        or task.source.revision != "64b3ebf62750b1373e26106cfc658a1861809d0f"
        or proof.source.sha256 != UPSTREAM_TASKSET_SHA256
    ):
        raise ValueError("Multi-SWE source-verifier error belongs to another dataset")
    references = (proof.source, proof.traceback, proof.reproduction, proof.decision)
    evidence = {}
    for reference in references:
        payload = StoragePath(reference.uri).read_bytes()
        if hashlib.sha256(payload).hexdigest() != reference.sha256:
            raise ValueError("Source-verifier exception evidence bytes changed")
        evidence[reference.uri] = payload
    decision = json.loads(evidence[proof.decision.uri])
    reproduction = json.loads(evidence[proof.reproduction.uri])
    if (
        decision.get("kind") != proof.kind
        or decision.get("authority_message") != "7d38152f01f4c573c67e5f9a8f8a41e2"
        or reproduction.get("task_id") != task.id
        or reproduction.get("returncode") != 1
        or reproduction.get("reward_written") is not False
        or reproduction.get("source_archive_sha256") != proof.archive.sha256
        or reproduction.get("fixture") != proof.fixture.model_dump(mode="json")
        or reproduction.get("files", {}).get("parser.stderr", {}).get("sha256") != proof.traceback.sha256
        or OVERLAP_ERROR not in evidence[proof.traceback.uri].decode("utf-8")
    ):
        raise ValueError("Source-verifier reproduction does not prove the approved exception")
    assert isinstance(task.grader, ScriptGrader)
    assert task.grader.environment.docker_build is not None
    resources = {
        resource.path: resource_bytes(resource)
        for resource in task.resources.verifier
        if resource.path.startswith("multiswe/") or resource.path == "row.json"
    }
    resources.update(
        ("parser-context/" + resource.path, resource_bytes(resource))
        for resource in task.grader.environment.docker_build.files
    )
    provenance = json.loads(resources["parser-context/provenance.json"])
    if (
        provenance.get("upstream_revision") != UPSTREAM_REVISION
        or provenance.get("benchmark_version") != "1.1.2"
        or provenance.get("requirements_lock_sha256")
        != hashlib.sha256(resources["parser-context/requirements.lock"]).hexdigest()
    ):
        raise ValueError("Reproduced dependency context differs from the approved upstream parser")
    source_files = collected_source_files(trial, proof)
    names = set(resources) | {"source-artifacts/" + name for name in source_files} | {"fixture-provenance.json"}
    fixture = verified_archive_members(proof.fixture, names)
    if any(fixture[name] != data for name, data in resources.items()) or any(
        fixture["source-artifacts/" + name] != data for name, data in source_files.items()
    ):
        raise ValueError("Reproduced parser or logs differ from the exact source trial")
    if multiswe_parser_sha256(task) != proof.parser_sha256:
        raise ValueError("Reproduced parser fingerprint differs from the task")
    definitions = []
    for payload in (evidence[proof.source.uri], resources["multiswe/score.py"]):
        definitions.append(
            {
                node.name: ast.dump(node, include_attributes=False)
                for node in ast.parse(payload).body
                if isinstance(node, ast.FunctionDef) and node.name in PARSER_FUNCTIONS
            }
        )
    if definitions[0].keys() != PARSER_FUNCTIONS or definitions[0] != definitions[1]:
        raise ValueError("Extracted source parser differs from the pinned upstream functions")
    with TemporaryDirectory(prefix="source-verifier-proof-") as directory:
        output = Path(directory) / "test-output.txt"
        output.write_bytes(source_files["multiswe-test-output.txt"])
        text = source_test_output(
            output,
            int(source_files["multiswe-read-exit-code.txt"]),
            source_files["multiswe-harness-stdout.txt"].decode("utf-8"),
            source_files["multiswe-harness-stderr.txt"].decode("utf-8"),
        )
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != proof.log_sha256:
        raise ValueError("Reproduced parser input differs from source log selection")
