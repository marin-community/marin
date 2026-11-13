# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Version the archived CrossCode Python fragment oracle and four-tier verifier."""

import argparse
import hashlib
import io
import json
import tarfile
import tomllib
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from marin.execution.artifact import Artifact, ArtifactRecord, result_type_name, write_record
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    DockerBuildContext,
    EnvironmentRequirements,
    ResourceGroups,
    TaskSpec,
    VerifyitGrader,
)
from taskcompendium.pipeline.models import NormalizationChange, NormalizedTask
from taskcompendium.runtime.resources import inline_resource, resource_bytes
from verifyit.spec import ScriptSpec

from experiments.post_training.task_curation.datasets.tasktrove.built_images import build_context_sha256
from experiments.post_training.task_curation.full_qc_artifacts import FullQCInputs
from experiments.post_training.task_curation.full_qc_review import contract_batches, load_inputs
from experiments.post_training.task_curation.images.build import BASE_IMAGE, compile_lock
from experiments.post_training.task_curation.tasktrove.harbor_export import (
    archive_bytes,
    archive_file_mode,
    harbor_payload,
)

VERSION = "crosscode-python-four-tier-2026.11.13.1"
SOURCE = "laion__exp_rpt_crosscodeeval-python-v2"
NATIVE_GRADER_SHA256 = "dad8c034eb303e887c9923b379ac481b57b3caca0a37273a272de6b255882b2c"
ANSWER_PATH = "/app/solution.txt"
GRADER_PATH = "grade-crosscode.sh"
VERDICT_FILE = "crosscode-verdict.json"
ORACLE_ADAPTER = b"""#!/bin/bash
set -euo pipefail
mkdir -p /app
cp "$(dirname "$0")/solution.py" /app/solution.txt
"""
STRUCTURED_PREAMBLE = r"""#!/bin/bash
set -u
AGENT_OUT="${VERIFYIT_WORKSPACE:?}/solution.txt"
GOLD_OUT="${VERIFYIT_TESTS_DIR:?}/solution.txt"
emit_verdict() {
    python3 - "$1" "$2" "$3" "$4" "$5" <<'PY'
import json
import os
import sys
from pathlib import Path
status, reward, tier, reason, total = sys.argv[1:]
detail = {"tier": tier, "reason": reason, "total": int(total)}
path = Path(os.environ["VERIFYIT_LOGS_DIR"]) / "crosscode-verdict.json"
path.write_text(json.dumps({"status": status, "reward": float(reward), "detail": detail}))
PY
}
if [ ! -f "$GOLD_OUT" ]; then
    emit_verdict invalid_task 0 "" missing_reference 0
    exit $?
fi
if [ ! -f "$AGENT_OUT" ]; then
    emit_verdict scored 0 "" missing_submission 1
    exit $?
fi
"""
STRUCTURED_EPILOGUE = """
rm -f "$A" "$B"
emit_verdict scored "$reward" "$tier" compared 1
"""


class CrossCodeCandidate(Artifact):
    inputs: FullQCInputs
    remediation_version: str
    actor_image: str
    actor_context_sha256: str
    grader_image: str
    task_count: int
    cohort_path: str
    cohort_sha256: str
    native_archive_path: str
    native_archive_sha256: str
    changes_path: str
    qualification_status: Literal["pending"]
    grading_environment: Literal["separate"]


class CrossCodeGraderBuild(Artifact):
    base_image: str
    context_path: str
    context_sha256: str
    archive_sha256: str


def prepare_grader_build(output: str) -> CrossCodeGraderBuild:
    """Resolve a hashed runtime dependency lock for the separate Python 3.12 grader image."""
    destination = StoragePath(output) / "attempts" / uuid4().hex
    with TemporaryDirectory() as directory:
        lock = Path(directory) / "requirements.lock"
        compile_lock(("tomli-w==1.2.0",), lock)
        dockerfile = (
            f"FROM {BASE_IMAGE}\n"
            "COPY requirements.lock /opt/crosscode/requirements.lock\n"
            "RUN uv pip sync --system --require-hashes /opt/crosscode/requirements.lock\n"
        ).encode()
        files = {"Dockerfile": dockerfile, "requirements.lock": lock.read_bytes()}
    build = DockerBuildContext(files=tuple(inline_resource(name, data) for name, data in files.items()))
    archive_path = destination / "context.tar.gz"
    with archive_path.open("wb", auto_mkdir=True) as stream:
        stream.write(archive_bytes(files, {}))
    result = CrossCodeGraderBuild(
        path=str(destination),
        base_image=BASE_IMAGE,
        context_path=str(archive_path),
        context_sha256=build_context_sha256(build),
        archive_sha256=file_sha256(archive_path),
    )
    write_record(
        ArtifactRecord(
            output_path=result.path, result_type=result_type_name(type(result)), result=result.result_payload()
        )
    )
    return result


def structured_grader(native: bytes) -> bytes:
    """Preserve the archived comparison statements and replace their reporting channel."""
    if hashlib.sha256(native).hexdigest() != NATIVE_GRADER_SHA256:
        raise ValueError("CrossCode native verifier differs from the reviewed four-tier producer")
    script = native.decode()
    start = script.index("normalise() {")
    end = script.index('\necho "$reward" > /logs/verifier/reward.txt', start)
    return (STRUCTURED_PREAMBLE + script[start:end] + STRUCTURED_EPILOGUE).encode()


def repair_task(task: TaskSpec, *, actor_image: str, grader_image: str) -> NormalizedTask:
    """Make a new, unqualified candidate using pinned images and the original fragment."""
    if task.source.dataset != "open-thoughts/TaskTrove" or not task.source.row.startswith(SOURCE + "/"):
        raise ValueError("Expected the archived CrossCode Python source")
    private = {resource.path: resource for resource in task.resources.verifier}
    original = resource_bytes(private["tests/test.sh"])
    producer = structured_grader(original)
    fragment = next(resource for resource in task.resources.oracle if resource.path == "solution/solution.py")
    if any(resource.path == "solution/solve.sh" for resource in task.resources.oracle):
        raise ValueError("CrossCode fragment candidate already contains an executable oracle")
    native_build = task.environment_requirements.docker_build
    if native_build is None:
        raise ValueError("The source must retain its archived actor build context")
    grader_environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        docker_image=grader_image,
        working_directory="/app",
    )
    native_config = tomllib.loads(resource_bytes(private["task.toml"]).decode())
    if native_config["agent"]["timeout_sec"] != 900.0:
        raise ValueError("The native actor deadline differs from the supported Harbor export contract")
    spec = ScriptSpec(
        path=GRADER_PATH,
        workspace="/app",
        timeout=native_config["verifier"]["timeout_sec"],
        verdict_file=VERDICT_FILE,
    )
    grader = VerifyitGrader(
        mode="script",
        parameters=json.loads(json.dumps(asdict(spec))),
        environment=grader_environment,
    )
    provenance = {
        "version": VERSION,
        "original_task_sha256": canonical_sha256(task.model_dump(mode="json")),
        "original_grader_sha256": NATIVE_GRADER_SHA256,
        "actor_build_sha256": canonical_sha256(native_build.model_dump(mode="json")),
        "fragment_sha256": hashlib.sha256(resource_bytes(fragment)).hexdigest(),
        "oracle_adapter_sha256": hashlib.sha256(ORACLE_ADAPTER).hexdigest(),
        "actor_image": actor_image,
        "grader_image": grader_image,
        "scoring": "Preserve exact, startswith, leading-identifier and non-empty-partial tiers",
        "missing_submission": "scored reward zero; producer exits successfully",
        "missing_reference": "invalid_task, unscored",
        "qualification_status": "pending",
    }
    archived = tuple(
        resource.model_copy(update={"path": "source_archive/" + resource.path}) for resource in task.resources.verifier
    )
    repaired = task.model_copy(
        update={
            "environment_requirements": task.environment_requirements.model_copy(
                update={"docker_build": None, "docker_image": actor_image, "working_directory": "/app"}
            ),
            "answer_type": AnswerType.FILE,
            "output_paths": (ANSWER_PATH,),
            "grader": grader,
            "resources": ResourceGroups(
                all=task.resources.all,
                worker=task.resources.worker,
                verifier=(
                    inline_resource(GRADER_PATH, producer),
                    private["tests/solution.txt"].model_copy(update={"path": "solution.txt"}),
                ),
                oracle=(
                    *task.resources.oracle,
                    *archived,
                    inline_resource("solution/solve.sh", ORACLE_ADAPTER).model_copy(update={"mode": "0755"}),
                    inline_resource("taskcompendium/crosscode-remediation.json", json.dumps(provenance).encode()),
                ),
            ),
            "labels": {**task.labels, "remediation_version": VERSION},
        }
    )
    # model_copy does not validate replacement fields, including the actor's digest pin.
    repaired = TaskSpec.model_validate_json(repaired.model_dump_json())
    changes = (
        NormalizationChange(
            field="environment_requirements",
            reason="Freeze the archived actor build as an immutable image; declare a separate private grader",
            original=task.environment_requirements.model_dump_json(),
            replacement=repaired.environment_requirements.model_dump_json(),
        ),
        NormalizationChange(
            field="grader",
            reason="Preserve the four-tier comparator with an authoritative structured verdict",
            original=task.grader.model_dump_json(),
            replacement=grader.model_dump_json(),
        ),
        NormalizationChange(
            field="resources.oracle",
            reason="Copy the original completion fragment to the declared answer path without executing it",
            original="solution/solution.py",
            replacement="solution/solution.py plus solution/solve.sh",
        ),
    )
    return NormalizedTask(repaired, changes)


def stage_candidate(
    inputs: FullQCInputs,
    *,
    actor_image: str,
    actor_context_sha256: str,
    grader_image: str,
    output: str,
) -> CrossCodeCandidate:
    """Stage all retained new contracts and their faithful Harbor packages, without publication."""
    if inputs.source_id != "TaskTrove:" + SOURCE or inputs.retained_count != 500:
        raise ValueError("Expected the frozen 500-task CrossCode Python retained population")
    destination = StoragePath(output) / "attempts" / uuid4().hex
    native_archive = destination / "native-tasks.tar.gz"
    cohort = destination / "tasks.parquet"
    changes_path = destination / "normalization-changes.jsonl"
    rows = []
    with native_archive.open("wb", auto_mkdir=True) as native_stream:
        with tarfile.open(fileobj=native_stream, mode="w|gz", compresslevel=1) as archive:
            with changes_path.open("w", auto_mkdir=True) as changes_stream:
                for batch in contract_batches(inputs):
                    if any(contract["runtime_sha256"] != canonical_sha256({}) for contract in batch.contracts):
                        raise ValueError("The frozen source declares runtime settings this candidate would replace")
                    for original in batch.tasks:
                        build = original.environment_requirements.docker_build
                        if build is None or build_context_sha256(build) != actor_context_sha256:
                            raise ValueError(f"Actor image does not match the archived build: {original.id}")
                        normalized = repair_task(original, actor_image=actor_image, grader_image=grader_image)
                        task = normalized.task
                        row = {
                            "path": task.id,
                            "task_json": task.model_dump_json(),
                            "lowered_task_spec": json.dumps({"task": task.model_dump(mode="json"), "runtime": {}}),
                        }
                        payload = harbor_payload(
                            {**row, "source_row": original.source.row, "original_path": task.id},
                            grader_image=grader_image,
                            family="crosscode",
                        )
                        modes = {**payload.modes, **payload.solution_modes}
                        for name, content in sorted({**payload.files, **payload.solution}.items()):
                            member = tarfile.TarInfo(f"{task.id}/{name}")
                            member.size = len(content)
                            member.mode = archive_file_mode(name, modes)
                            archive.addfile(member, io.BytesIO(content))
                        rows.append(row)
                        changes_stream.write(
                            json.dumps(
                                {
                                    "task_id": task.id,
                                    "changes": [change.model_dump(mode="json") for change in normalized.changes],
                                }
                            )
                            + "\n"
                        )
    if len(rows) != inputs.retained_count or len({row["path"] for row in rows}) != inputs.retained_count:
        raise ValueError("The new candidate does not cover the retained population exactly once")
    with cohort.open("wb", auto_mkdir=True) as stream:
        pq.write_table(pa.Table.from_pylist(rows), stream, compression="zstd")
    result = CrossCodeCandidate(
        path=str(destination),
        inputs=inputs,
        remediation_version=VERSION,
        actor_image=actor_image,
        actor_context_sha256=actor_context_sha256,
        grader_image=grader_image,
        task_count=len(rows),
        cohort_path=str(cohort),
        cohort_sha256=file_sha256(cohort),
        native_archive_path=str(native_archive),
        native_archive_sha256=file_sha256(native_archive),
        changes_path=str(changes_path),
        qualification_status="pending",
        grading_environment="separate",
    )
    write_record(
        ArtifactRecord(
            output_path=result.path, result_type=result_type_name(type(result)), result=result.result_payload()
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare-grader")
    preparation.add_argument("--output", required=True)
    preparation.add_argument("--run", action="store_true")
    stage = commands.add_parser("stage")
    stage.add_argument("--inputs", required=True)
    stage.add_argument("--actor-image", required=True)
    stage.add_argument("--actor-context-sha256", required=True)
    stage.add_argument("--grader-image", required=True)
    stage.add_argument("--output", required=True)
    stage.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        print(
            json.dumps({"command": args.command, "output": args.output, "version": VERSION, "qualification": "pending"})
        )
        return
    configure_coreweave_s3()
    if args.command == "prepare-grader":
        prepared = prepare_grader_build(args.output)
        print(json.dumps({"grader_context": prepared.path, "context_sha256": prepared.context_sha256}))
        return
    result = stage_candidate(
        load_inputs(args.inputs),
        actor_image=args.actor_image,
        actor_context_sha256=args.actor_context_sha256,
        grader_image=args.grader_image,
        output=args.output,
    )
    print(json.dumps({"candidate": result.path, "task_count": result.task_count, "qualification": "pending"}))


if __name__ == "__main__":
    main()
