# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade captured submissions in a fresh sandbox with private TaskTrove tests."""

import io
import json
import tarfile
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from shellbox.machine import Command, DockerImage, MachineFactory, MachineSpec, NetworkPolicy
from verifyit.spec import GotestSpec, JunitSpec, PytestSpec, ScriptSpec, StdioSpec, render_spec, spec_from_table

from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.resources import resource_bytes

GRADING_TIMEOUT = 600.0
GRADING_MEMORY_MB = 4096

SPEC_PATH = "/tests/verifier.toml"
VERDICT_PATH = "/logs/verifier/verdict.json"


async def grade_submission(
    task: TaskSpec,
    files: dict[str, bytes],
    factory: MachineFactory,
    *,
    timeout: float = GRADING_TIMEOUT,
    memory_mb: int = GRADING_MEMORY_MB,
) -> GradeResult:
    """Run the shared grader independently of the agent's environment."""
    try:
        return await _sandbox_grade(task, files, factory, timeout=timeout, memory_mb=memory_mb)
    except (RuntimeError, OSError, json.JSONDecodeError) as error:
        return GradeResult(Outcome.INFRA_ERROR, None, str(error))


async def _sandbox_grade(
    task: TaskSpec, files: dict[str, bytes], factory: MachineFactory, *, timeout: float, memory_mb: int
) -> GradeResult:
    spec = spec_from_table({"mode": task.verifier.kind, **json.loads(task.verifier.parameters_json)})
    paths = task.output_paths
    if not isinstance(spec, StdioSpec | PytestSpec | ScriptSpec | JunitSpec | GotestSpec):
        paths = (*paths, spec.output)
    elif isinstance(spec, ScriptSpec) and not paths:
        paths = ("/app/answer.txt",)
    for path in paths:
        candidate_path = PurePosixPath(path)
        if not candidate_path.is_absolute() or ".." in candidate_path.parts:
            raise ValueError(f"Invalid submission path: {path}")
        if candidate_path.is_relative_to("/tests") or candidate_path.is_relative_to("/logs/verifier"):
            raise ValueError(f"Submission overlaps private grading files: {path}")
    submissions = {path: files[path] for path in paths if path in files}
    if task.verifier.kind == "script" and "/app/state.json" in files:
        submissions["/app/state.json"] = files["/app/state.json"]
    if not submissions:
        return GradeResult(Outcome.GRADED, 0.0, "Missing submission")
    requirements = task.verifier.environment_requirements
    if (
        requirements.capabilities
        or requirements.setup_commands
        or requirements.environment_variables
        or requirements.tool_providers
    ):
        raise ValueError("Unsupported private grading environment requirements")
    workspace = (
        spec.workspace if isinstance(spec, StdioSpec | PytestSpec | ScriptSpec | JunitSpec | GotestSpec) else "/app"
    )
    if requirements.working_directory is not None and requirements.working_directory != workspace:
        raise ValueError("Private grading workspace disagrees with verifier specification")
    image = requirements.docker_image
    if image is None:
        raise ValueError("Isolated grading requires a pinned image")
    machine = await factory.create(
        MachineSpec(DockerImage(image), workdir=workspace, network=NetworkPolicy.DENY, memory_mb=memory_mb)
    )
    try:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            # Stdio tasks can carry hundreds of tiny case files. One archive
            # crosses the machine boundary rather than one RPC per fixture.
            archive_path = root / "submission.tar"
            metadata = {
                **{"tests/" + resource.path: resource for resource in task.resources.verifier},
                **{resource.path: resource for resource in task.resources.all + task.resources.worker},
            }
            with tarfile.open(archive_path, "w") as archive:
                for path, data in [
                    *(("tests/" + resource.path, resource_bytes(resource)) for resource in task.resources.verifier),
                    *(
                        (resource.path, resource_bytes(resource))
                        for resource in task.resources.all + task.resources.worker
                    ),
                    *submissions.items(),
                    (SPEC_PATH, render_spec(spec).encode()),
                ]:
                    member = tarfile.TarInfo(path.removeprefix("/"))
                    member.size = len(data)
                    resource = metadata.get(path.removeprefix("/"))
                    member.mode = int(resource.mode, 8) if resource is not None and resource.mode is not None else 0o644
                    if resource is not None and resource.mtime_ns is not None:
                        seconds, nanos = divmod(resource.mtime_ns, 1_000_000_000)
                        member.pax_headers = {"mtime": f"{seconds}.{nanos:09d}"}
                    archive.addfile(member, io.BytesIO(data))
            remote_archive = "/tmp/taskcompendium-submission.tar"
            await machine.upload(archive_path, remote_archive)
            unpacked = await machine.run(Command(("tar", "-xf", remote_archive, "-C", "/"), cwd="/", timeout=timeout))
            if unpacked.exit_code != 0:
                return GradeResult(Outcome.INFRA_ERROR, None, "Could not unpack grading fixtures")
            initialized = await machine.run(Command(("mkdir", "-p", workspace), cwd="/", timeout=timeout))
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
                        workspace,
                    ),
                    timeout=timeout,
                    output_limit_bytes=16_384,
                )
            )
            if result.exit_code != 0:
                return GradeResult(Outcome.INFRA_ERROR, None, result.stderr.decode(errors="replace"))
            verdict_file = root / "verdict.json"
            await machine.download(VERDICT_PATH, verdict_file)
            verdict = json.loads(verdict_file.read_text())
            status = {
                "scored": Outcome.GRADED,
                "invalid_task": Outcome.INVALID_TASK,
                "infra_error": Outcome.INFRA_ERROR,
            }[verdict["status"]]
            return GradeResult(
                status,
                verdict["reward"] if status == Outcome.GRADED else None,
                verdict["detail"].get("error"),
                detail=verdict["detail"],
            )
    finally:
        await machine.close()
