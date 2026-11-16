# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trusted CPython execution with disposable project directories."""

import asyncio
import base64
import binascii
import io
import json
import mimetypes
import os
import signal
import stat
import sys
import sysconfig
import tempfile
import time
import zipfile
from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from marina.applets import AppletServices
from marina.apps import RegisteredApi, registered_api
from marina.mcp import OperationRisk, operation_extension
from pydantic import BaseModel, Field
from rigging.config_discovery import find_project_root
from rigging.filesystem.cluster_config import MARIN_CLUSTER_CONFIG_DIRS
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine
from starlette.concurrency import run_in_threadpool

MAX_FILES = 100
MAX_INPUT_BYTES = 5 * 1024 * 1024
MAX_OUTPUT_BYTES = 10 * 1024 * 1024
MAX_LOG_BYTES = 256 * 1024
RUN_TIMEOUT = 120
PAGE_SIZE = 25
MAX_ARCHIVE_BYTES = MAX_INPUT_BYTES + 1024 * 1024

# Explicit extra packages take precedence; then expose Marina's installed runtime and editable packages.
RUNNER_BOOTSTRAP = """import json
import runpy
import site
import sys
from pathlib import Path

site.addsitedir(sys.argv[1])
from rigging.filesystem import cluster_config

cluster_config.MARIN_CLUSTER_CONFIG_DIRS = tuple(json.loads(sys.argv[2]))
entrypoint = sys.argv[3]
sys.argv = sys.argv[3:]
sys.path.insert(0, str(Path(entrypoint).resolve().parent))
runpy.run_path(entrypoint, run_name="__main__")
"""


class ProjectFile(BaseModel):
    path: str
    content: str
    encoding: str = "utf-8"


class RunRequest(BaseModel):
    files: list[ProjectFile] = Field(max_length=MAX_FILES)
    entrypoint: str
    packages: list[str] = Field(default_factory=list)
    timeout: int = Field(default=RUN_TIMEOUT, ge=1, le=RUN_TIMEOUT)


class Artifact(BaseModel):
    path: str
    content: str
    media_type: str
    size: int


class RunResult(BaseModel):
    exit_code: int
    timed_out: bool
    duration: float
    logs: str
    artifacts: list[Artifact]
    warnings: list[str]


def project_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise HTTPException(422, f"Expected a relative project path: {value}")
    if path.parts[0] == "output":
        raise HTTPException(422, "output/ is reserved for generated files")
    return path


def decoded_files(files: list[ProjectFile]) -> list[tuple[PurePosixPath, bytes]]:
    total = 0
    decoded = []
    seen = set()
    for file in files:
        path = project_path(file.path)
        if path in seen:
            raise HTTPException(422, f"Duplicate file: {file.path}")
        seen.add(path)
        if file.encoding == "base64":
            try:
                data = base64.b64decode(file.content, validate=True)
            except (ValueError, binascii.Error) as error:
                raise HTTPException(422, f"Invalid base64: {file.path}") from error
        elif file.encoding == "utf-8":
            data = file.content.encode()
        else:
            raise HTTPException(422, f"Unsupported encoding: {file.encoding}")
        total += len(data)
        if total > MAX_INPUT_BYTES:
            raise HTTPException(413, "Project exceeds the 5 MiB input limit")
        decoded.append((path, data))
    paths = {path for path, _ in decoded}
    if any(parent in paths for path in paths for parent in path.parents):
        raise HTTPException(422, "A file cannot also be a parent directory")
    return decoded


def materialize_project(root: Path, request: RunRequest) -> None:
    for path, data in decoded_files(request.files):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    entrypoint = root / project_path(request.entrypoint)
    if not entrypoint.is_file() or entrypoint.suffix != ".py":
        raise HTTPException(422, "Select an existing Python entrypoint")


async def capture_logs(stream: asyncio.StreamReader, buffer: bytearray) -> bool:
    """Drain process output into the bounded buffer, returning whether it was truncated."""
    truncated = False
    while chunk := await stream.read(8192):
        available = MAX_LOG_BYTES - len(buffer)
        buffer.extend(chunk[:available])
        truncated |= len(chunk) > available
    return truncated


def collect_artifacts(output: Path) -> tuple[list[Artifact], list[str]]:
    artifacts = []
    warnings = []
    total = 0
    # Do not follow generated symlinks while collecting the output tree.
    for parent, directories, files in os.walk(output, followlinks=False):
        directories[:] = sorted(name for name in directories if not (Path(parent) / name).is_symlink())
        for name in sorted(files):
            path = Path(parent) / name
            if path.is_symlink() or not path.is_file():
                continue
            size = path.stat().st_size
            if total + size > MAX_OUTPUT_BYTES or len(artifacts) >= MAX_FILES:
                warnings.append(f"Skipped {path.relative_to(output)}: output limit (10 MiB / 100 files)")
                continue
            data = path.read_bytes()
            total += len(data)
            artifacts.append(
                Artifact(
                    path=path.relative_to(output).as_posix(),
                    content=base64.b64encode(data).decode(),
                    media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                    size=len(data),
                )
            )
    return artifacts, warnings


async def execute_project(request: RunRequest) -> RunResult:
    """Run trusted code and return logs and generated files before cleanup."""
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="marina-python-") as directory:
        root = Path(directory)
        materialize_project(root, request)
        output = root / "output"
        output.mkdir()
        command = ["uv", "run", "--no-project", "--no-config", "--python", sys.executable]
        for package in request.packages:
            command.extend(["--with", package])
        config_root = find_project_root() or Path.cwd()
        config_dirs = [Path(directory).expanduser() for directory in MARIN_CLUSTER_CONFIG_DIRS]
        config_dirs = [path if path.is_absolute() else config_root / path for path in config_dirs]
        command.extend(
            [
                "python",
                "-u",
                "-c",
                RUNNER_BOOTSTRAP,
                sysconfig.get_path("purelib"),
                json.dumps([str(path) for path in config_dirs]),
                request.entrypoint,
            ]
        )
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=root,
            env={**os.environ, "OUTPUT_DIR": str(output), "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        assert process.stdout is not None
        buffer = bytearray()
        reader = asyncio.create_task(capture_logs(process.stdout, buffer))
        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), timeout=request.timeout)
        except TimeoutError:
            timed_out = True
        finally:
            # Descendants may outlive the entrypoint and keep its pipes open.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # The entire process group already exited.
            await process.wait()
            truncated = await reader
        artifacts, warnings = collect_artifacts(output)
        if truncated:
            warnings.append("Logs truncated after 256 KiB")
        if timed_out:
            warnings.append(f"Run stopped after {request.timeout} seconds (including package installation)")
        return RunResult(
            exit_code=process.returncode if process.returncode is not None else -1,
            timed_out=timed_out,
            duration=round(time.monotonic() - started, 3),
            logs=buffer.decode(errors="replace"),
            artifacts=artifacts,
            warnings=warnings,
        )


class ProjectDocument(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    files: list[ProjectFile] = Field(max_length=MAX_FILES)
    entrypoint: str
    packages: str = Field(max_length=8192)


class SaveProject(BaseModel):
    revision: int
    document: ProjectDocument


class RunProject(BaseModel):
    revision: int
    timeout: int = Field(default=RUN_TIMEOUT, ge=1, le=RUN_TIMEOUT)


class ProjectRecord(BaseModel):
    id: UUID
    revision: int
    document: ProjectDocument
    result: RunResult | None
    result_revision: int | None
    created_at: datetime
    updated_at: datetime
    last_run_at: datetime | None


class RunStatus(StrEnum):
    NEVER = "never"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class ProjectSummary(BaseModel):
    id: UUID
    title: str
    entrypoint: str
    file_count: int
    revision: int
    created_at: datetime
    updated_at: datetime
    last_run_at: datetime | None
    status: RunStatus


class ProjectPage(BaseModel):
    items: list[ProjectSummary]
    total: int
    page_size: int


class ProjectSort(StrEnum):
    TITLE = "title"
    CREATED = "created"
    UPDATED = "updated"
    LAST_RUN = "last_run"


class SortDirection(StrEnum):
    ASC = "asc"
    DESC = "desc"


SORT_COLUMNS = {
    ProjectSort.TITLE: "lower(p.title)",
    ProjectSort.CREATED: "p.created_at",
    ProjectSort.UPDATED: "p.updated_at",
    ProjectSort.LAST_RUN: "r.completed_at",
}


def migrate(connection: Connection) -> None:
    """Create project metadata, binary blobs, input files and latest-run outputs."""
    for statement in (
        """CREATE TABLE IF NOT EXISTS projects (
            id UUID PRIMARY KEY,
            title TEXT NOT NULL,
            entrypoint TEXT NOT NULL,
            packages TEXT NOT NULL,
            revision BIGINT NOT NULL DEFAULT 1,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )""",
        """CREATE TABLE IF NOT EXISTS blobs (
            id UUID PRIMARY KEY,
            content BYTEA NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS input_files (
            project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            path TEXT NOT NULL,
            encoding TEXT NOT NULL,
            blob_id UUID NOT NULL UNIQUE REFERENCES blobs(id) ON DELETE CASCADE,
            PRIMARY KEY (project_id, path)
        )""",
        """CREATE TABLE IF NOT EXISTS runs (
            id UUID PRIMARY KEY,
            project_id UUID NOT NULL UNIQUE REFERENCES projects(id) ON DELETE CASCADE,
            source_revision BIGINT NOT NULL,
            exit_code INTEGER NOT NULL,
            timed_out BOOLEAN NOT NULL,
            duration DOUBLE PRECISION NOT NULL,
            logs BYTEA NOT NULL,
            warnings TEXT[] NOT NULL,
            completed_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )""",
        """CREATE TABLE IF NOT EXISTS output_files (
            run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            path TEXT NOT NULL,
            media_type TEXT NOT NULL,
            blob_id UUID NOT NULL UNIQUE REFERENCES blobs(id) ON DELETE CASCADE,
            PRIMARY KEY (run_id, path)
        )""",
    ):
        connection.execute(text(statement))


def write_blob(connection: Connection, content: bytes) -> UUID:
    blob_id = uuid4()
    connection.execute(
        text("INSERT INTO blobs (id, content) VALUES (:id, :content)"), {"id": blob_id, "content": content}
    )
    return blob_id


def replace_inputs(connection: Connection, project_id: UUID, files: list[ProjectFile]) -> None:
    decoded = decoded_files(files)
    existing = {
        row["path"]: row
        for row in connection.execute(
            text(
                """
            SELECT f.path, f.encoding, f.blob_id, b.content
            FROM input_files f JOIN blobs b ON b.id = f.blob_id WHERE f.project_id = :id
        """
            ),
            {"id": project_id},
        ).mappings()
    }
    requested = {str(path) for path, _ in decoded}
    for path, row in existing.items():
        if path not in requested:
            connection.execute(text("DELETE FROM blobs WHERE id = :id"), {"id": row["blob_id"]})
    for file, (path, data) in zip(files, decoded, strict=True):
        old = existing.get(str(path))
        # Autosaving a title or package edit must not rewrite every uploaded blob.
        if old is not None and bytes(old["content"]) == data:
            if old["encoding"] != file.encoding:
                connection.execute(
                    text("UPDATE input_files SET encoding = :encoding WHERE blob_id = :id"),
                    {"encoding": file.encoding, "id": old["blob_id"]},
                )
            continue
        if old is not None:
            connection.execute(text("DELETE FROM blobs WHERE id = :id"), {"id": old["blob_id"]})
        blob_id = write_blob(connection, data)
        connection.execute(
            text(
                """
            INSERT INTO input_files (project_id, path, encoding, blob_id)
            VALUES (:id, :path, :encoding, :blob_id)
        """
            ),
            {"id": project_id, "path": str(path), "encoding": file.encoding, "blob_id": blob_id},
        )


def read_project_record(connection: Connection, project_id: UUID) -> ProjectRecord:
    # Share the project row lock with writers so metadata, files and results form one snapshot.
    project = (
        connection.execute(text("SELECT * FROM projects WHERE id = :id FOR SHARE"), {"id": project_id})
        .mappings()
        .one_or_none()
    )
    if project is None:
        raise HTTPException(404, "Project not found")
    files = []
    for row in connection.execute(
        text(
            """
        SELECT f.path, f.encoding, b.content FROM input_files f JOIN blobs b ON b.id = f.blob_id
        WHERE f.project_id = :id ORDER BY f.path
    """
        ),
        {"id": project_id},
    ).mappings():
        data = bytes(row["content"])
        files.append(
            ProjectFile(
                path=row["path"],
                encoding=row["encoding"],
                content=data.decode() if row["encoding"] == "utf-8" else base64.b64encode(data).decode(),
            )
        )
    run = (
        connection.execute(text("SELECT * FROM runs WHERE project_id = :id"), {"id": project_id})
        .mappings()
        .one_or_none()
    )
    result = None
    if run is not None:
        artifacts = []
        for row in connection.execute(
            text(
                """
            SELECT f.path, f.media_type, b.content FROM output_files f JOIN blobs b ON b.id = f.blob_id
            WHERE f.run_id = :id ORDER BY f.path
        """
            ),
            {"id": run["id"]},
        ).mappings():
            data = bytes(row["content"])
            artifacts.append(
                Artifact(
                    path=row["path"],
                    media_type=row["media_type"],
                    content=base64.b64encode(data).decode(),
                    size=len(data),
                )
            )
        result = RunResult(
            exit_code=run["exit_code"],
            timed_out=run["timed_out"],
            duration=run["duration"],
            logs=bytes(run["logs"]).decode(),
            warnings=run["warnings"],
            artifacts=artifacts,
        )
    return ProjectRecord(
        id=project["id"],
        revision=project["revision"],
        document=ProjectDocument(
            title=project["title"], entrypoint=project["entrypoint"], packages=project["packages"], files=files
        ),
        result=result,
        result_revision=run["source_revision"] if run is not None else None,
        created_at=project["created_at"],
        updated_at=project["updated_at"],
        last_run_at=run["completed_at"] if run is not None else None,
    )


def project_record(engine: Engine, project_id: UUID) -> ProjectRecord:
    with engine.begin() as connection:
        return read_project_record(connection, project_id)


def store_result(engine: Engine, project_id: UUID, revision: int, result: RunResult) -> ProjectRecord:
    with engine.begin() as connection:
        project = connection.execute(
            text("SELECT revision FROM projects WHERE id = :id FOR UPDATE"), {"id": project_id}
        ).scalar_one_or_none()
        if project != revision:
            raise HTTPException(409, "Project changed or was deleted during execution. The result was not saved.")
        connection.execute(
            text(
                """
            DELETE FROM blobs WHERE id IN (
                SELECT f.blob_id FROM output_files f JOIN runs r ON r.id = f.run_id WHERE r.project_id = :id
            )
        """
            ),
            {"id": project_id},
        )
        connection.execute(text("DELETE FROM runs WHERE project_id = :id"), {"id": project_id})
        run_id = uuid4()
        connection.execute(
            text(
                """
            INSERT INTO runs (id, project_id, source_revision, exit_code, timed_out, duration, logs, warnings)
            VALUES (:id, :project_id, :revision, :exit_code, :timed_out, :duration, :logs, CAST(:warnings AS text[]))
        """
            ),
            {
                "id": run_id,
                "project_id": project_id,
                "revision": revision,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "duration": result.duration,
                "logs": result.logs.encode(),
                "warnings": result.warnings,
            },
        )
        for artifact in result.artifacts:
            blob_id = write_blob(connection, base64.b64decode(artifact.content))
            connection.execute(
                text(
                    """
                INSERT INTO output_files (run_id, path, media_type, blob_id)
                VALUES (:id, :path, :media_type, :blob_id)
            """
                ),
                {"id": run_id, "path": artifact.path, "media_type": artifact.media_type, "blob_id": blob_id},
            )
        return read_project_record(connection, project_id)


def project_from_zip(payload: bytes, title: str, entrypoint: str, packages: list[str] | None) -> ProjectDocument:
    """Decode a root-level project ZIP within the input file and byte budgets."""
    files = []
    total = 0
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            if len(members) > MAX_FILES:
                raise HTTPException(413, "ZIP exceeds the 100 input file limit")
            for member in members:
                project_path(member.filename)
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise HTTPException(422, "ZIP symlinks are not supported")
                total += member.file_size
                if total > MAX_INPUT_BYTES:
                    raise HTTPException(413, "ZIP exceeds the 5 MiB input limit")
                data = archive.read(member)
                try:
                    content = data.decode("utf-8")
                    encoding = "utf-8"
                except UnicodeDecodeError:
                    content = base64.b64encode(data).decode()
                    encoding = "base64"
                files.append(ProjectFile(path=member.filename, content=content, encoding=encoding))
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError) as error:
        raise HTTPException(422, f"Cannot read project ZIP: {error}") from error
    decoded_files(files)
    if entrypoint not in {file.path for file in files} or not entrypoint.endswith(".py"):
        raise HTTPException(422, "ZIP must include the selected Python entrypoint at its project-relative path")
    if packages is None:
        requirements = next((file for file in files if file.path == "requirements.txt"), None)
        if requirements is not None and requirements.encoding != "utf-8":
            raise HTTPException(422, "requirements.txt must be UTF-8 text")
        package_text = requirements.content if requirements is not None else ""
    else:
        package_text = "\n".join(packages)
    return ProjectDocument(title=title, entrypoint=entrypoint, packages=package_text, files=files)


class AgentConfiguration(BaseModel):
    enabled: bool
    origin: str | None
    profile: str = "marina"
    repository: str = "marin-community/marin"
    starters: tuple[str, ...] = (
        "Explain this project's code and latest results.",
        "What should I change to improve this analysis?",
    )


def create_api(services: AppletServices) -> RegisteredApi:
    """Create the shared project directory and trusted execution API."""
    api = FastAPI(title="Python playground")
    engine = services.engine()
    capacity = asyncio.Semaphore(2)
    origin = os.environ.get("MARINA_AGENT_ORIGIN")
    agent_configuration = AgentConfiguration(enabled=bool(origin), origin=origin)

    @api.get("/agent/config")
    def agent_config() -> AgentConfiguration:
        return agent_configuration

    @api.get(
        "/projects",
        operation_id="list_projects",
        description="Search and sort saved Python playground projects.",
        openapi_extra=operation_extension(OperationRisk.READ),
    )
    def projects(
        query: str = Query(default="", max_length=200),
        sort: ProjectSort = ProjectSort.UPDATED,
        direction: SortDirection = SortDirection.DESC,
        offset: int = Query(default=0, ge=0),
    ) -> ProjectPage:
        with engine.connect() as connection:
            count = connection.execute(
                text("SELECT count(*) FROM projects WHERE strpos(lower(title), lower(:query)) > 0"), {"query": query}
            ).scalar_one()
            rows = connection.execute(
                text(
                    f"""
                SELECT p.id, p.title, p.entrypoint,
                       (SELECT count(*) FROM input_files f WHERE f.project_id = p.id) AS file_count,
                       p.revision, p.created_at, p.updated_at, r.completed_at AS last_run_at,
                       CASE WHEN r.id IS NULL THEN 'never'
                            WHEN r.timed_out THEN 'timed_out'
                            WHEN r.exit_code = 0 THEN 'succeeded'
                            ELSE 'failed' END AS status
                FROM projects p LEFT JOIN runs r ON r.project_id = p.id
                WHERE strpos(lower(p.title), lower(:query)) > 0
                ORDER BY {SORT_COLUMNS[sort]} {direction.value} NULLS LAST, p.id
                LIMIT :limit OFFSET :offset
            """
                ),
                {"query": query, "limit": PAGE_SIZE, "offset": offset},
            ).mappings()
            return ProjectPage(
                items=[ProjectSummary.model_validate(row) for row in rows], total=count, page_size=PAGE_SIZE
            )

    @api.post(
        "/projects",
        status_code=201,
        operation_id="create_project",
        description="Create a saved project from Python source, binary inputs and package requirements.",
        openapi_extra=operation_extension(OperationRisk.WRITE),
    )
    def create_project(document: ProjectDocument) -> ProjectRecord:
        decoded_files(document.files)
        project_id = uuid4()
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                INSERT INTO projects (id, title, entrypoint, packages) VALUES (:id, :title, :entrypoint, :packages)
            """
                ),
                {
                    "id": project_id,
                    "title": document.title,
                    "entrypoint": document.entrypoint,
                    "packages": document.packages,
                },
            )
            replace_inputs(connection, project_id, document.files)
            return read_project_record(connection, project_id)

    @api.post(
        "/projects/import",
        status_code=201,
        description="Create a project from a raw ZIP body with files at the archive root.",
    )
    async def import_project(
        request: Request,
        title: str = Query(min_length=1, max_length=200),
        entrypoint: str = "main.py",
        packages: list[str] | None = Query(default=None),
    ) -> ProjectRecord:
        payload = bytearray()
        async for chunk in request.stream():
            payload.extend(chunk)
            if len(payload) > MAX_ARCHIVE_BYTES:
                raise HTTPException(413, "Project ZIP exceeds the 6 MiB upload limit")
        document = await run_in_threadpool(project_from_zip, bytes(payload), title, entrypoint, packages)
        return await run_in_threadpool(create_project, document)

    @api.get(
        "/projects/{project_id}",
        operation_id="get_project",
        description="Read project source, inputs and its latest saved report, plots and logs.",
        openapi_extra=operation_extension(OperationRisk.READ),
    )
    def read_project(project_id: UUID) -> ProjectRecord:
        return project_record(engine, project_id)

    @api.put(
        "/projects/{project_id}",
        operation_id="save_project",
        description="Replace project files and settings at the supplied expected revision.",
        openapi_extra=operation_extension(OperationRisk.WRITE),
    )
    def save_project(project_id: UUID, request: SaveProject) -> ProjectRecord:
        decoded_files(request.document.files)
        with engine.begin() as connection:
            updated = connection.execute(
                text(
                    """
                UPDATE projects SET title = :title, entrypoint = :entrypoint, packages = :packages,
                                    revision = revision + 1, updated_at = now()
                WHERE id = :id AND revision = :revision RETURNING id
            """
                ),
                {
                    "id": project_id,
                    "revision": request.revision,
                    "title": request.document.title,
                    "entrypoint": request.document.entrypoint,
                    "packages": request.document.packages,
                },
            ).scalar_one_or_none()
            if updated is None:
                raise HTTPException(409, "Project changed or was deleted. Export your edits before reopening it.")
            replace_inputs(connection, project_id, request.document.files)
            return read_project_record(connection, project_id)

    @api.delete(
        "/projects/{project_id}",
        status_code=204,
        operation_id="delete_project",
        description="Delete a project and all its saved input and output blobs at the expected revision.",
        openapi_extra=operation_extension(OperationRisk.DESTRUCTIVE),
    )
    def delete_project(project_id: UUID, revision: int) -> None:
        with engine.begin() as connection:
            found = connection.execute(
                text("SELECT revision FROM projects WHERE id = :id FOR UPDATE"), {"id": project_id}
            ).scalar_one_or_none()
            if found != revision:
                raise HTTPException(409, "Project changed or was deleted. Refresh the directory before deleting.")
            connection.execute(
                text(
                    """
                DELETE FROM blobs WHERE id IN (
                    SELECT blob_id FROM input_files WHERE project_id = :id
                    UNION ALL
                    SELECT f.blob_id FROM output_files f JOIN runs r ON r.id = f.run_id WHERE r.project_id = :id
                )
            """
                ),
                {"id": project_id},
            )
            connection.execute(text("DELETE FROM projects WHERE id = :id"), {"id": project_id})

    @api.post(
        "/projects/{project_id}/run",
        operation_id="run_project",
        description=(
            "Execute trusted Python at the expected source revision, install packages and save outputs. "
            "Code inherits host network and credentials."
        ),
        openapi_extra=operation_extension(OperationRisk.EXTERNAL_WRITE),
    )
    async def run(project_id: UUID, request: RunProject) -> ProjectRecord:
        async with capacity:
            project = await run_in_threadpool(project_record, engine, project_id)
            if project.revision != request.revision:
                raise HTTPException(409, "Project changed. Reopen it before running.")
            packages = [
                line.strip()
                for line in project.document.packages.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            result = await execute_project(
                RunRequest(
                    files=project.document.files,
                    entrypoint=project.document.entrypoint,
                    packages=packages,
                    timeout=request.timeout,
                )
            )
            return await run_in_threadpool(store_result, engine, project_id, project.revision, result)

    return registered_api(api)
