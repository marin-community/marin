# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade captured submissions in a fresh sandbox with private TaskTrove tests."""

import io
import json
import tarfile
from dataclasses import replace
from math import isfinite
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from pydantic import ValidationError
from shellbox.machine import Command, ExitReason, MachineFactory, MachineSpec
from verifyit.spec import GotestSpec, JunitSpec, PytestSpec, ScriptSpec, StdioSpec, render_spec, spec_from_table

from taskcompendium.grader import SOURCE_UNAVAILABLE_KIND
from taskcompendium.grading import parse_grade_result
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import TaskSpec, require_compatible_backend
from taskcompendium.native_grader import NATIVE_COMMAND_KIND, NativeCommandSpec
from taskcompendium.runtime.output_capture import selected_directory_files, validate_output_directories
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.runtime.shell import require_image

GRADING_TIMEOUT = 600.0

SPEC_PATH = "/tests/verifier.toml"
VERDICT_PATH = "/logs/verifier/verdict.json"


async def grade_submission(
    task: TaskSpec,
    files: dict[str, bytes],
    factory: MachineFactory,
    *,
    machine_spec: MachineSpec,
    timeout: float | None = None,
) -> GradeResult:
    """Run the shared grader independently of the agent's environment."""
    if task.verifier.kind == SOURCE_UNAVAILABLE_KIND:
        return GradeResult(Outcome.UNAVAILABLE, None, "Source evaluator is unavailable")
    require_compatible_backend(task.verifier.environment_requirements, factory.backend)
    if task.verifier.kind == NATIVE_COMMAND_KIND:
        try:
            spec = NativeCommandSpec.model_validate_json(task.verifier.parameters_json)
        except ValidationError as error:
            return GradeResult(Outcome.INVALID_TASK, None, f"Invalid native grader contract: {error}")
        try:
            return await _sandbox_grade(
                task,
                files,
                factory,
                machine_spec=machine_spec,
                timeout=timeout if timeout is not None else spec.timeout,
                native=spec,
            )
        except (RuntimeError, OSError) as error:
            return GradeResult(Outcome.INFRA_ERROR, None, str(error))
    try:
        return await _sandbox_grade(
            task,
            files,
            factory,
            machine_spec=machine_spec,
            timeout=timeout if timeout is not None else GRADING_TIMEOUT,
        )
    except (RuntimeError, OSError, json.JSONDecodeError) as error:
        return GradeResult(Outcome.INFRA_ERROR, None, str(error))


async def _sandbox_grade(
    task: TaskSpec,
    files: dict[str, bytes],
    factory: MachineFactory,
    *,
    machine_spec: MachineSpec,
    timeout: float,
    native: NativeCommandSpec | None = None,
) -> GradeResult:
    spec = (
        None
        if native is not None
        else spec_from_table({"mode": task.verifier.kind, **json.loads(task.verifier.parameters_json)})
    )
    workspace = (
        spec.workspace if isinstance(spec, StdioSpec | PytestSpec | ScriptSpec | JunitSpec | GotestSpec) else "/app"
    )
    validate_output_directories(task.output_directories, workspace)
    paths = task.output_paths
    if spec is not None and not isinstance(spec, StdioSpec | PytestSpec | ScriptSpec | JunitSpec | GotestSpec):
        paths = (*paths, spec.output)
    elif isinstance(spec, ScriptSpec) and not paths:
        paths = ("/app/answer.txt",)
    for path in paths:
        candidate_path = PurePosixPath(path)
        if not candidate_path.is_absolute() or ".." in candidate_path.parts:
            raise ValueError(f"Invalid submission path: {path}")
        if candidate_path.is_relative_to("/tests") or candidate_path.is_relative_to("/logs/verifier"):
            raise ValueError(f"Submission overlaps private grading files: {path}")
        if native is not None and candidate_path == PurePosixPath(native.result_path):
            raise ValueError(f"Submission overlaps native grader result: {path}")
    submissions = {path: files[path] for path in paths if path in files}
    for selection in task.output_directories:
        for path, data in selected_directory_files(selection, files).items():
            submissions.setdefault(path, data)
    if (task.verifier.kind == "script" or native is not None) and "/app/state.json" in files:
        submissions["/app/state.json"] = files["/app/state.json"]
    if native is not None and native.result_path in submissions:
        raise ValueError("Submission overlaps native grader result")
    if not submissions and native is None:
        return GradeResult(Outcome.GRADED, 0.0, "Missing submission")
    requirements = task.verifier.environment_requirements
    if (
        requirements.capabilities
        or requirements.setup_commands
        or requirements.environment_variables
        or requirements.tool_providers
    ):
        raise ValueError("Unsupported private grading environment requirements")
    if requirements.working_directory is not None and requirements.working_directory != workspace:
        raise ValueError("Private grading workspace disagrees with verifier specification")
    image = requirements.docker_image
    if image is None:
        raise ValueError("Isolated grading requires a pinned image")
    require_image(machine_spec, image)
    machine = await factory.create(replace(machine_spec, workdir=workspace))
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
                    *([(SPEC_PATH, render_spec(spec).encode())] if spec is not None else []),
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
            # The archive also contains private references; protecting /tests
            # does not protect a readable fixture copy left under /tmp.
            removed = await machine.run(Command(("rm", remote_archive), cwd="/", timeout=timeout))
            if removed.exit_code != 0:
                return GradeResult(Outcome.INFRA_ERROR, None, "Could not remove private grading fixture archive")
            initialized = await machine.run(Command(("mkdir", "-p", workspace), cwd="/", timeout=timeout))
            if initialized.exit_code != 0:
                return GradeResult(Outcome.INFRA_ERROR, None, "Could not initialize grading workspace")
            if native is not None:
                result_directory = str(PurePosixPath(native.result_path).parent)
                prepared = await machine.run(Command(("mkdir", "-p", result_directory), cwd="/", timeout=timeout))
                if prepared.exit_code != 0:
                    return GradeResult(Outcome.INFRA_ERROR, None, "Could not initialize native grader result directory")
                cleared = await machine.run(Command(("rm", "-f", native.result_path), cwd="/", timeout=timeout))
                if cleared.exit_code != 0:
                    return GradeResult(Outcome.INFRA_ERROR, None, "Could not clear native grader result")
                result = await machine.run(
                    Command(
                        native.argv,
                        cwd=native.cwd,
                        env=native.env,
                        timeout=min(timeout, native.timeout),
                        output_limit_bytes=16_384,
                    )
                )
                if result.reason != ExitReason.EXITED:
                    return GradeResult(
                        Outcome.INVALID_TASK,
                        None,
                        f"Source grader did not complete ({result.reason}, exit_code={result.exit_code}): "
                        f"{result.stderr.decode(errors='replace')}",
                    )
                exists = await machine.run(Command(("test", "-f", native.result_path), cwd="/", timeout=timeout))
                if exists.reason != ExitReason.EXITED or exists.exit_code != 0:
                    return GradeResult(
                        Outcome.INVALID_TASK,
                        None,
                        f"Source grader did not produce its declared result (exit_code={result.exit_code}): "
                        f"{result.stderr.decode(errors='replace')}",
                    )
                reward_file = root / "native-reward"
                await machine.download(native.result_path, reward_file)
                try:
                    detail = None
                    if native.result_format in ("reward_json", "score_json"):
                        document = json.loads(reward_file.read_bytes())
                        fields = {"reward"} if native.result_format == "reward_json" else {"reward", "detail"}
                        if not isinstance(document, dict) or set(document) != fields:
                            raise ValueError(f"JSON score must contain exactly {sorted(fields)}")
                        value = document["reward"]
                        if isinstance(value, bool) or not isinstance(value, int | float):
                            raise ValueError("JSON reward must be numeric")
                        if native.result_format == "score_json":
                            detail = document["detail"]
                            if not isinstance(detail, dict):
                                raise ValueError("JSON score detail must be an object")
                        reward = float(value)
                    else:
                        reward = float(reward_file.read_text().strip())
                except (UnicodeError, ValueError, OverflowError) as error:
                    return GradeResult(Outcome.INVALID_TASK, None, f"Invalid source reward: {error}")
                if not isfinite(reward):
                    return GradeResult(Outcome.INVALID_TASK, None, "Source reward must be finite")
                return GradeResult(Outcome.GRADED, reward, detail=detail)
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
                    # Bootstrap trusted imports outside submitted Python modules.
                    cwd="/",
                    timeout=timeout,
                    output_limit_bytes=16_384,
                )
            )
            if result.exit_code != 0:
                return GradeResult(
                    Outcome.INFRA_ERROR,
                    None,
                    f"Grader {result.reason} (exit_code={result.exit_code}): {result.stderr.decode(errors='replace')}",
                )
            verdict_file = root / "verdict.json"
            await machine.download(VERDICT_PATH, verdict_file)
            return parse_grade_result(spec, verdict_file.read_bytes())
    finally:
        await machine.close()
