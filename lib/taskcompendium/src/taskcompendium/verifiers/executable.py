# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade captured submissions in a fresh sandbox with private TaskTrove tests."""

import asyncio
import io
import json
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.machine import Command, DockerImage, MachineFactory, MachineSpec, NetworkPolicy
from verifyit.spec import PytestSpec, ScriptSpec, StdioSpec, render_spec, spec_from_table

from taskcompendium.grading import GradeResult, GradingAttempt, Outcome, Verifier
from taskcompendium.models import TaskResource
from taskcompendium.runtime.models import RuntimeEvidence

SPEC_PATH = "/tests/verifier.toml"
VERDICT_PATH = "/logs/verifier/verdict.json"


class TaskTroveExecutableVerifier(Verifier):
    """A pinned grader and image; only named submission files cross the boundary.

    Grading happens in a new network-disabled machine. The solving environment
    never receives private tests, and its reward files are not grading evidence.
    Executed submissions still share the grading sandbox with the tests; this
    prototype does not prevent submitted programs from reading hidden tests.
    """

    grader_spec_json: str
    resources: tuple[TaskResource, ...]
    submission_paths: tuple[str, ...]
    image: str
    timeout: float
    memory_mb: int

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        if not isinstance(attempt.environment, RuntimeEvidence):
            return GradeResult(Outcome.INFRA_ERROR, None, "Missing captured submission files")
        return asyncio.run(grade_submission(self, attempt.environment.files, DockerMachineFactory()))


async def grade_submission(
    verifier: TaskTroveExecutableVerifier, files: dict[str, bytes], factory: MachineFactory
) -> GradeResult:
    """Run the shared grader independently of the agent's environment."""
    try:
        return await _sandbox_grade(verifier, files, factory)
    except (RuntimeError, OSError, json.JSONDecodeError) as error:
        return GradeResult(Outcome.INFRA_ERROR, None, str(error))


async def _sandbox_grade(
    verifier: TaskTroveExecutableVerifier, files: dict[str, bytes], factory: MachineFactory
) -> GradeResult:
    submissions = {path: files[path] for path in verifier.submission_paths if path in files}
    if not submissions:
        return GradeResult(Outcome.GRADED, 0.0, "Missing submission")
    spec = spec_from_table(json.loads(verifier.grader_spec_json))
    if not isinstance(spec, StdioSpec | PytestSpec | ScriptSpec):
        raise ValueError("Executable task requires stdio, pytest, or script grading")
    machine = await factory.create(
        MachineSpec(
            DockerImage(verifier.image), workdir=spec.workspace, network=NetworkPolicy.DENY, memory_mb=verifier.memory_mb
        )
    )
    try:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            # Stdio tasks can carry hundreds of tiny case files. One archive
            # crosses the machine boundary rather than one RPC per fixture.
            archive_path = root / "submission.tar"
            with tarfile.open(archive_path, "w") as archive:
                for path, data in [
                    *((resource.path, resource.data()) for resource in verifier.resources),
                    *submissions.items(),
                    (SPEC_PATH, render_spec(spec).encode()),
                ]:
                    member = tarfile.TarInfo(path.removeprefix("/"))
                    member.size = len(data)
                    member.mode = 0o644
                    archive.addfile(member, io.BytesIO(data))
            remote_archive = "/tmp/taskcompendium-submission.tar"
            await machine.upload(archive_path, remote_archive)
            unpacked = await machine.run(
                Command(("tar", "-xf", remote_archive, "-C", "/"), cwd="/", timeout=verifier.timeout)
            )
            if unpacked.exit_code != 0:
                return GradeResult(Outcome.INFRA_ERROR, None, "Could not unpack grading fixtures")
            initialized = await machine.run(Command(("mkdir", "-p", spec.workspace), cwd="/", timeout=verifier.timeout))
            if initialized.exit_code != 0:
                return GradeResult(Outcome.INFRA_ERROR, None, "Could not initialize grading workspace")
            result = await machine.run(
                Command(
                    (
                        "python3",
                        "-c",
                        "from verifyit.grade import main; raise SystemExit(main())",
                        SPEC_PATH,
                        "--workspace",
                        spec.workspace,
                    ),
                    timeout=verifier.timeout,
                    output_limit_bytes=16_384,
                )
            )
            if result.exit_code != 0:
                return GradeResult(Outcome.INFRA_ERROR, None, result.stderr.decode(errors="replace"))
            verdict_file = root / "verdict.json"
            await machine.download(VERDICT_PATH, verdict_file)
            verdict = json.loads(verdict_file.read_text())
            status = Outcome.GRADED if verdict["status"] == "scored" else Outcome.INFRA_ERROR
            return GradeResult(
                status, verdict["reward"] if status == Outcome.GRADED else None, json.dumps(verdict["detail"])
            )
    finally:
        await machine.close()
