# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dynamic Marina applets: packages, persistence, data, and Python backends."""

import hashlib
import importlib
import io
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import tomllib
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import cast

from rigging.filesystem.storage_path import prefix_join
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine
from starlette.types import ASGIApp

from marina.apps import RegisteredApi
from marina.database_setup import (
    APPLET_READER_ROLE,
    APPLET_ROLE_PREFIX,
    MARINA_SCHEMA,
    PROVISION_APPLET_FUNCTION,
)
from marina.db import DatabaseSpec, engine_for, engine_for_role, grant_read_on_connection, schema_name
from marina.host_bindings import migrate_host_bindings, normalize_hostname
from marina.manifest import APP_NAME_PATTERN, KERNEL_PREFIXES, AgentPanelManifest, AppJob, agent_panel, app_jobs

APPLET_MANIFEST = "applet.toml"
DIST_PREFIX = "dist/"
INDEX_FILE = "dist/index.html"
MAX_PACKAGE_BYTES = 25 * 1024 * 1024
MAX_ARCHIVE_BYTES = 30 * 1024 * 1024
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_FILES = 2_000
QUERY_TIMEOUT_MS = 10_000
MAX_QUERY_ROWS = 10_000
MAX_QUERY_RESPONSE_BYTES = 5 * 1024 * 1024
MAX_REVISIONS = 5
BACKEND_VALIDATION_TIMEOUT = 15
SCHEMA_ADOPTION_LOCK_TIMEOUT = "5s"
KNOWN_MANIFEST_KEYS = frozenset(
    {
        "title",
        "description",
        "connect_src",
        "build_command",
        "python_entrypoint",
        "name",
        "data_url",
        "jobs",
        "agent",
        "applet_id",
        "source_paths",
    }
)
SOURCE_EXCLUSIONS = frozenset({"tests", "journeys", "web", "dist", "node_modules", "__pycache__"})
ENTRYPOINT_PATTERN = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*$")
BACKEND_CACHE_ROOT = Path(tempfile.gettempdir()) / "marina-applet-backends"
APPLET_LOCK_SQL = "SELECT pg_advisory_xact_lock(hashtextextended(:id, 0))"
EXECUTABLE_INLINE_SCRIPT_TYPES = frozenset(
    {
        "application/ecmascript",
        "application/javascript",
        "application/x-ecmascript",
        "application/x-javascript",
        "importmap",
        "module",
        "speculationrules",
        "text/ecmascript",
        "text/javascript",
        "text/javascript1.0",
        "text/javascript1.1",
        "text/javascript1.2",
        "text/javascript1.3",
        "text/javascript1.4",
        "text/javascript1.5",
        "text/jscript",
        "text/livescript",
        "text/x-ecmascript",
        "text/x-javascript",
    }
)


class AppletNotFound(Exception):
    pass


class AppletConflict(Exception):
    pass


class AppletForbidden(Exception):
    pass


class QueryLimitExceeded(Exception):
    pass


class InvalidQuery(Exception):
    pass


class AppletBackendUnavailable(Exception):
    pass


class AppletMode(StrEnum):
    PRIVATE = "private"
    PUBLIC = "public"


@dataclass(frozen=True)
class AppletManifest:
    title: str
    description: str
    connect_src: tuple[str, ...] = ()
    build_command: str | None = None
    python_entrypoint: str | None = None
    name: str | None = None
    data_url: str | None = None
    jobs: tuple[AppJob, ...] = ()
    agent: AgentPanelManifest | None = None
    applet_id: str | None = None
    source_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class AppletPackage:
    manifest: AppletManifest
    files: dict[str, bytes]
    digest: bytes
    byte_size: int


@dataclass(frozen=True)
class AppletSummary:
    id: uuid.UUID
    title: str
    description: str
    owner: str
    current_version: int
    mode: AppletMode
    name: str | None = None

    @property
    def path(self) -> str:
        return f"/{self.name}/" if self.name is not None else f"/a/{self.id}/"


@dataclass(frozen=True)
class AppletVersion:
    applet_id: uuid.UUID
    version: int
    manifest: AppletManifest
    digest: bytes


@dataclass(frozen=True)
class AppletSnapshot:
    applet_id: uuid.UUID
    version: int
    package: AppletPackage


@dataclass(frozen=True)
class AppletVersionSummary:
    version: int
    published_by: str
    published_at: datetime
    byte_size: int


@dataclass(frozen=True)
class StoredFile:
    body: bytes
    media_type: str
    digest: bytes
    content_encoding: str | None = None


@dataclass(frozen=True)
class PublishResult:
    applet_id: uuid.UUID
    version: int
    mode: AppletMode

    @property
    def path(self) -> str:
        return f"/a/{self.applet_id}/v/{self.version}/"


class _InlineScriptParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.inline_script_line: int | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.inline_script_line is not None or tag.casefold() != "script":
            return
        attributes = {name.casefold(): value for name, value in attrs}
        if attributes.get("src"):
            return
        script_type = (attributes.get("type") or "text/javascript").strip().casefold()
        if script_type in EXECUTABLE_INLINE_SCRIPT_TYPES:
            self.inline_script_line = self.getpos()[0]


def _validate_inline_scripts(files: dict[str, bytes]) -> None:
    for path, body in files.items():
        if not path.startswith(DIST_PREFIX) or not path.casefold().endswith((".htm", ".html")):
            continue
        parser = _InlineScriptParser()
        parser.feed(body.decode(errors="replace"))
        parser.close()
        if parser.inline_script_line is not None:
            raise ValueError(
                f"{path}:{parser.inline_script_line}: inline <script> conflicts with Marina's script-src 'self'; "
                "move the code to a packaged JavaScript file"
            )


def parse_applet_manifest(content: bytes) -> AppletManifest:
    """Parse an applet manifest and reject fields the publisher would otherwise ignore."""
    try:
        raw = tomllib.loads(content.decode())
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"invalid {APPLET_MANIFEST}: {error}") from error
    unknown = set(raw) - KNOWN_MANIFEST_KEYS
    if unknown:
        raise ValueError(f"{APPLET_MANIFEST}: unknown keys {sorted(unknown)}")
    for field in ("title", "description"):
        if not isinstance(raw.get(field), str) or not raw[field].strip():
            raise ValueError(f"{APPLET_MANIFEST}: {field!r} must be a non-empty string")
    connect_src = raw.get("connect_src", [])
    if not isinstance(connect_src, list) or not all(isinstance(item, str) for item in connect_src):
        raise ValueError(f"{APPLET_MANIFEST}: 'connect_src' must be a list of strings")
    if any(not item or any(character.isspace() or character == ";" for character in item) for item in connect_src):
        raise ValueError(f"{APPLET_MANIFEST}: connect_src entries must be single CSP source expressions")
    for optional in ("build_command", "python_entrypoint", "data_url", "applet_id"):
        if optional in raw and not isinstance(raw[optional], str):
            raise ValueError(f"{APPLET_MANIFEST}: {optional!r} must be a string")
    entrypoint = raw.get("python_entrypoint")
    if entrypoint is not None and not ENTRYPOINT_PATTERN.fullmatch(entrypoint):
        raise ValueError(f"{APPLET_MANIFEST}: invalid python_entrypoint {entrypoint!r}")
    name = raw.get("name")
    if name is not None and (
        not isinstance(name, str) or len(name) > 63 or not APP_NAME_PATTERN.fullmatch(name) or name in KERNEL_PREFIXES
    ):
        raise ValueError(f"{APPLET_MANIFEST}: invalid or reserved application name {name!r}")
    source_paths = raw.get("source_paths", [])
    if not isinstance(source_paths, list) or any(
        not isinstance(path, str) or Path(path).is_absolute() or ".." in Path(path).parts for path in source_paths
    ):
        raise ValueError("source_paths must be repository-relative paths")
    return AppletManifest(
        title=raw["title"].strip(),
        description=raw["description"].strip(),
        connect_src=tuple(connect_src),
        build_command=raw.get("build_command"),
        python_entrypoint=entrypoint,
        name=name,
        data_url=raw.get("data_url"),
        jobs=app_jobs(raw.get("jobs", [])),
        agent=agent_panel(raw["agent"]) if "agent" in raw else None,
        applet_id=str(uuid.UUID(raw["applet_id"])) if "applet_id" in raw else None,
        source_paths=tuple(source_paths),
    )


def _clean_package_path(name: str) -> str:
    stripped = name.removeprefix("./")
    path = PurePosixPath(stripped)
    if not stripped or path.is_absolute() or "\\" in stripped or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"unsafe package path {name!r}")
    return path.as_posix()


def _validate_package_files(files: dict[str, bytes]) -> AppletPackage:
    if APPLET_MANIFEST not in files:
        raise ValueError(f"package has no {APPLET_MANIFEST}")
    if INDEX_FILE not in files:
        raise ValueError(f"package has no {INDEX_FILE}")
    manifest = parse_applet_manifest(files[APPLET_MANIFEST])
    for path in files:
        if path == APPLET_MANIFEST or path.startswith(DIST_PREFIX):
            continue
        if path.endswith(".py") and not SOURCE_EXCLUSIONS.intersection(PurePosixPath(path).parts):
            continue
        raise ValueError(f"package entry {path!r} is outside applet.toml, dist/, or application Python source")
    _validate_inline_scripts(files)
    if manifest.python_entrypoint is not None:
        module, _factory = manifest.python_entrypoint.split(":", 1)
        module_file = module.replace(".", "/") + ".py"
        package_file = module.replace(".", "/") + "/__init__.py"
        if module_file not in files and package_file not in files:
            raise ValueError(f"python_entrypoint module {module!r} is absent from the package")
    for job in manifest.jobs:
        module = job.module.replace(".", "/")
        if f"{module}.py" not in files and f"{module}/__init__.py" not in files:
            raise ValueError(f"job module {job.module!r} is absent from the package")
    digest = hashlib.sha256()
    for path, body in sorted(files.items()):
        digest.update(path.encode())
        digest.update(b"\0")
        digest.update(body)
        digest.update(b"\0")
    return AppletPackage(manifest=manifest, files=files, digest=digest.digest(), byte_size=sum(map(len, files.values())))


def read_applet_package(payload: bytes) -> AppletPackage:
    """Validate and unpack an applet tarball without writing uploaded paths to disk."""
    files: dict[str, bytes] = {}
    total = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
            for member in archive:
                if member.isdir():
                    continue
                if not member.isreg():
                    raise ValueError(f"package entry {member.name!r} is not a regular file")
                if len(files) >= MAX_FILES:
                    raise ValueError(f"package contains more than {MAX_FILES} files")
                path = _clean_package_path(member.name)
                if path in files:
                    raise ValueError(f"package contains duplicate path {path!r}")
                if member.size > MAX_FILE_BYTES:
                    raise ValueError(f"package entry {path!r} exceeds {MAX_FILE_BYTES} bytes")
                total += member.size
                if total > MAX_PACKAGE_BYTES:
                    raise ValueError(f"package exceeds {MAX_PACKAGE_BYTES} bytes")
                extracted = archive.extractfile(member)
                assert extracted is not None
                body = extracted.read(MAX_FILE_BYTES + 1)
                if len(body) != member.size:
                    raise ValueError(f"package entry {path!r} has an invalid size")
                files[path] = body
    except tarfile.TarError as error:
        raise ValueError(f"invalid applet archive: {error}") from error
    return _validate_package_files(files)


def package_applet(app_dir: Path, *, extra_files: dict[str, bytes] | None = None) -> bytes:
    """Build a validated applet tarball, including optional generated files."""
    paths = [app_dir / APPLET_MANIFEST]
    for path in sorted(app_dir.rglob("*")):
        relative = path.relative_to(app_dir)
        if path.is_dir() or any(part.startswith(".") for part in relative.parts):
            continue
        if relative.parts[0] == "dist" and path.suffix not in {".pyc", ".pyo"}:
            paths.append(path)
        elif path.suffix == ".py" and not SOURCE_EXCLUSIONS.intersection(relative.parts):
            paths.append(path)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"package entry {path} is not a regular file")
            archive.add(path, arcname=path.relative_to(app_dir).as_posix(), recursive=False)
        for name, body in (extra_files or {}).items():
            if (app_dir / name).exists():
                raise ValueError(f"generated package entry {name!r} already exists in the source directory")
            member = tarfile.TarInfo(name)
            member.size = len(body)
            archive.addfile(member, io.BytesIO(body))
    payload = buffer.getvalue()
    read_applet_package(payload)
    return payload


def applet_schema(applet_id: uuid.UUID, name: str | None = None) -> str:
    return schema_name(name) if name is not None else applet_role(applet_id)


def applet_role(applet_id: uuid.UUID) -> str:
    return f"{APPLET_ROLE_PREFIX}{applet_id.hex}"


def _manifest_json(manifest: AppletManifest) -> str:
    return json.dumps(asdict(manifest))


def _manifest_from_json(value: dict[str, object]) -> AppletManifest:
    connect_src = value.get("connect_src", [])
    assert isinstance(connect_src, list)
    source_paths = value.get("source_paths", [])
    assert isinstance(source_paths, list)
    return AppletManifest(
        title=str(value["title"]),
        description=str(value["description"]),
        connect_src=tuple(str(item) for item in connect_src),
        build_command=str(value["build_command"]) if value.get("build_command") is not None else None,
        python_entrypoint=str(value["python_entrypoint"]) if value.get("python_entrypoint") is not None else None,
        name=str(value["name"]) if value.get("name") is not None else None,
        data_url=str(value["data_url"]) if value.get("data_url") is not None else None,
        jobs=app_jobs(value.get("jobs", [])),
        agent=agent_panel(value["agent"]) if value.get("agent") is not None else None,
        applet_id=str(value["applet_id"]) if value.get("applet_id") is not None else None,
        source_paths=tuple(str(item) for item in source_paths),
    )


class AppletStore:
    """Postgres-backed registry and inline blob store for dynamic applets."""

    def __init__(self, database: DatabaseSpec):
        self.database = database
        self.engine = engine_for(database, MARINA_SCHEMA)

    def migrate(self) -> None:
        statements = (
            """CREATE TABLE IF NOT EXISTS applets (
                id UUID PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                owner TEXT NOT NULL,
                mode TEXT NOT NULL DEFAULT 'private' CHECK (mode IN ('private', 'public')),
                current_version INTEGER,
                archived_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )""",
            """ALTER TABLE applets ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT 'private'
                CHECK (mode IN ('private', 'public'))""",
            "ALTER TABLE applets ADD COLUMN IF NOT EXISTS name TEXT UNIQUE",
            """CREATE TABLE IF NOT EXISTS applet_versions (
                applet_id UUID NOT NULL REFERENCES applets(id) ON DELETE CASCADE,
                version INTEGER NOT NULL,
                manifest JSONB NOT NULL,
                package_digest BYTEA NOT NULL,
                byte_size BIGINT NOT NULL,
                published_by TEXT NOT NULL,
                published_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (applet_id, version)
            )""",
            """CREATE TABLE IF NOT EXISTS blobs (
                id UUID PRIMARY KEY,
                digest BYTEA NOT NULL,
                byte_size BIGINT NOT NULL,
                media_type TEXT NOT NULL,
                inline_bytes BYTEA NOT NULL,
                UNIQUE (digest, media_type)
            )""",
            """CREATE TABLE IF NOT EXISTS applet_files (
                applet_id UUID NOT NULL,
                version INTEGER NOT NULL,
                path TEXT NOT NULL,
                blob_id UUID NOT NULL REFERENCES blobs(id),
                PRIMARY KEY (applet_id, version, path),
                FOREIGN KEY (applet_id, version) REFERENCES applet_versions(applet_id, version) ON DELETE CASCADE
            )""",
            f"""CREATE OR REPLACE VIEW applet_catalog AS
                SELECT a.id AS applet_id, a.title, a.owner,
                       COALESCE(replace(a.name, '-', '_'),
                                '{APPLET_ROLE_PREFIX}' || replace(a.id::TEXT, '-', '')) AS schema_name,
                       c.table_name, c.column_name, c.data_type, c.ordinal_position
                FROM applets a
                LEFT JOIN information_schema.columns c
                  ON c.table_schema = COALESCE(replace(a.name, '-', '_'),
                                               '{APPLET_ROLE_PREFIX}' || replace(a.id::TEXT, '-', ''))
                WHERE a.archived_at IS NULL""",
            f"GRANT USAGE ON SCHEMA {MARINA_SCHEMA} TO {APPLET_READER_ROLE}",
            f"GRANT SELECT ON applet_catalog TO {APPLET_READER_ROLE}",
        )
        with self.engine.begin() as connection:
            for statement in statements:
                connection.execute(text(statement))
            migrate_host_bindings(connection)

    def host_applet(self, hostname: str) -> uuid.UUID | None:
        """Resolve a hostname; disabled hosts raise to keep their origins isolated."""
        with self.engine.connect() as connection:
            row = connection.execute(
                text("SELECT applet_id FROM applet_hosts WHERE hostname = :hostname"),
                {"hostname": hostname},
            ).first()
        if row is None:
            return None
        if row.applet_id is None:
            raise AppletNotFound(hostname)
        return row.applet_id

    def host_bindings(self) -> dict[str, str | None]:
        with self.engine.connect() as connection:
            return {
                row.hostname: str(row.applet_id) if row.applet_id is not None else None
                for row in connection.execute(text("SELECT hostname, applet_id FROM applet_hosts ORDER BY hostname"))
            }

    def bind_host(self, hostname: str, applet_id: uuid.UUID) -> None:
        """Bind a hostname to an active applet; callers must authorize the operator."""
        hostname = normalize_hostname(hostname)
        with self.engine.begin() as connection:
            applet = connection.execute(
                text("SELECT id FROM applets WHERE id = :id AND archived_at IS NULL FOR UPDATE"),
                {"id": applet_id},
            ).scalar()
            if applet is None:
                raise AppletNotFound(str(applet_id))
            connection.execute(
                text(
                    "INSERT INTO applet_hosts (hostname, applet_id) VALUES (:hostname, :id) "
                    "ON CONFLICT (hostname) DO UPDATE SET applet_id = EXCLUDED.applet_id, updated_at = now()"
                ),
                {"hostname": hostname, "id": applet_id},
            )

    def unbind_host(self, hostname: str) -> None:
        """Disable routing while reserving the hostname's isolated applet origin."""
        with self.engine.begin() as connection:
            result = connection.execute(
                text("UPDATE applet_hosts SET applet_id = NULL, updated_at = now() WHERE hostname = :hostname"),
                {"hostname": normalize_hostname(hostname)},
            )
            if result.rowcount == 0:
                raise AppletNotFound(hostname)

    def publish(
        self,
        package: AppletPackage,
        owner: str,
        applet_id: uuid.UUID | None = None,
        base_version: int | None = None,
        operators: frozenset[str] = frozenset(),
        mode: AppletMode | None = None,
        *,
        reader_roles: tuple[str, ...] = (),
    ) -> PublishResult:
        applet_id = applet_id or uuid.uuid4()
        if package.manifest.applet_id is not None and package.manifest.applet_id != str(applet_id):
            raise ValueError(
                "Upload target differs from applet.toml's applet_id; supply the declared UUID and base version"
            )
        if package.manifest.name is None and (
            package.manifest.data_url is not None or package.manifest.jobs or package.manifest.agent is not None
        ):
            raise ValueError("Data URLs, scheduled jobs and agent panels require a named application")
        if package.manifest.name is not None and mode is AppletMode.PUBLIC:
            raise ValueError("Named applications are private; public applets use the isolated applet origin")
        with self.engine.begin() as connection:
            self._lock_applet(connection, applet_id)
            version, effective_mode = self._allocate_version(
                connection,
                applet_id,
                package.manifest,
                owner,
                base_version,
                operators,
                mode,
            )
            self._insert_version(connection, applet_id, version, package, owner)
            self._store_files(connection, applet_id, version, package.files)
            connection.execute(text(f'SET LOCAL ROLE "{applet_role(applet_id)}"'))
            schema = applet_schema(applet_id, package.manifest.name)
            connection.execute(text(f'SET LOCAL search_path TO "{schema}", public'))
            self._run_migration(connection, applet_id, version, package)
            grant_read_on_connection(connection, schema, APPLET_READER_ROLE)
            for reader in reader_roles:
                grant_read_on_connection(connection, schema, reader)
            connection.execute(text("RESET ROLE"))
            connection.execute(text("SET LOCAL search_path TO marina, public"))
            self._activate_version(connection, applet_id, version, package.manifest, effective_mode)
            self._delete_unreferenced_blobs(connection)
        return PublishResult(applet_id=applet_id, version=version, mode=effective_mode)

    @staticmethod
    def _lock_applet(connection: Connection, applet_id: uuid.UUID) -> None:
        connection.execute(text(APPLET_LOCK_SQL), {"id": str(applet_id)})

    @classmethod
    def _authorize_versioned_mutation(
        cls,
        connection: Connection,
        applet_id: uuid.UUID,
        actor: str,
        base_version: int,
        operators: frozenset[str],
    ) -> None:
        cls._lock_applet(connection, applet_id)
        applet = (
            connection.execute(
                text("SELECT owner, current_version FROM applets WHERE id = :id AND archived_at IS NULL FOR UPDATE"),
                {"id": applet_id},
            )
            .mappings()
            .first()
        )
        if applet is None:
            raise AppletNotFound(str(applet_id))
        if applet["owner"] != actor and actor not in operators:
            raise AppletForbidden(str(applet_id))
        if int(applet["current_version"]) != base_version:
            raise AppletConflict(str(applet_id))

    @staticmethod
    def _allocate_version(
        connection: Connection,
        applet_id: uuid.UUID,
        manifest: AppletManifest,
        owner: str,
        base_version: int | None,
        operators: frozenset[str],
        mode: AppletMode | None,
    ) -> tuple[int, AppletMode]:
        existing = (
            connection.execute(
                text("SELECT owner, mode, current_version, archived_at, name FROM applets WHERE id = :id FOR UPDATE"),
                {"id": applet_id},
            )
            .mappings()
            .first()
        )
        if existing is not None:
            if existing["archived_at"] is not None:
                raise AppletNotFound(str(applet_id))
            if existing["owner"] != owner and owner not in operators:
                raise AppletForbidden(str(applet_id))
            if existing["name"] != manifest.name:
                raise ValueError("An application's registered name cannot change")
            if base_version is None or existing["current_version"] != base_version:
                raise AppletConflict(str(applet_id))
            version = int(
                connection.execute(
                    text("SELECT COALESCE(MAX(version), 0) + 1 FROM applet_versions WHERE applet_id = :id"),
                    {"id": applet_id},
                ).scalar_one()
            )
            effective_mode = AppletMode(existing["mode"]) if mode is None else mode
            return version, effective_mode
        if base_version not in (None, 0):
            raise AppletNotFound(str(applet_id))
        if manifest.name is not None and owner not in operators:
            raise AppletForbidden("Only an operator may register a named application")
        if manifest.name is not None:
            connection.execute(text(APPLET_LOCK_SQL), {"id": f"marina:name:{manifest.name}"})
            if connection.execute(text("SELECT 1 FROM applets WHERE name = :name"), {"name": manifest.name}).scalar():
                raise AppletConflict(f"Application name {manifest.name!r} is already registered")
        effective_mode = mode or AppletMode.PRIVATE
        connection.execute(text(f"SELECT {PROVISION_APPLET_FUNCTION}(:id)"), {"id": applet_id})
        if manifest.name is not None:
            AppletStore._adopt_schema(connection, schema_name(manifest.name), applet_role(applet_id))
        connection.execute(
            text(
                "INSERT INTO applets (id, title, description, owner, mode, name) "
                "VALUES (:id, :title, :description, :owner, :mode, :name)"
            ),
            {
                "id": applet_id,
                "title": manifest.title,
                "description": manifest.description,
                "owner": owner,
                "mode": effective_mode.value,
                "name": manifest.name,
            },
        )
        return 1, effective_mode

    @staticmethod
    def _adopt_schema(connection: Connection, schema: str, role: str) -> None:
        """Keep existing application data while transferring its schema to the package role."""
        connection.execute(text(f"SET LOCAL lock_timeout = '{SCHEMA_ADOPTION_LOCK_TIMEOUT}'"))
        connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}" AUTHORIZATION "{role}"'))
        connection.execute(text(f'ALTER SCHEMA "{schema}" OWNER TO "{role}"'))
        relations = connection.execute(
            text(
                "SELECT c.relname, c.relkind FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = :schema AND c.relkind IN ('r', 'p', 'v', 'm', 'S', 'f') "
                "AND (c.relkind <> 'S' OR NOT EXISTS (SELECT 1 FROM pg_depend d "
                "WHERE d.classid = 'pg_class'::regclass AND d.objid = c.oid AND d.deptype IN ('a', 'i')))"
            ),
            {"schema": schema},
        ).all()
        kinds = {
            "r": "TABLE",
            "p": "TABLE",
            "v": "VIEW",
            "m": "MATERIALIZED VIEW",
            "S": "SEQUENCE",
            "f": "FOREIGN TABLE",
        }
        for name, kind in relations:
            quoted = connection.dialect.identifier_preparer.quote_identifier(name)
            connection.execute(text(f'ALTER {kinds[kind]} "{schema}".{quoted} OWNER TO "{role}"'))

    @staticmethod
    def _insert_version(
        connection: Connection,
        applet_id: uuid.UUID,
        version: int,
        package: AppletPackage,
        owner: str,
    ) -> None:
        connection.execute(
            text(
                "INSERT INTO applet_versions "
                "(applet_id, version, manifest, package_digest, byte_size, published_by) "
                "VALUES (:id, :version, CAST(:manifest AS jsonb), :digest, :byte_size, :owner)"
            ),
            {
                "id": applet_id,
                "version": version,
                "manifest": _manifest_json(package.manifest),
                "digest": package.digest,
                "byte_size": package.byte_size,
                "owner": owner,
            },
        )

    @staticmethod
    def _blob_id(connection: Connection, path: str, body: bytes) -> uuid.UUID:
        digest = hashlib.sha256(body).digest()
        media_type = mimetypes.guess_type(path.removesuffix(".gz"))[0] or "application/octet-stream"
        existing = connection.execute(
            text("SELECT id FROM blobs WHERE digest = :digest AND media_type = :media_type"),
            {"digest": digest, "media_type": media_type},
        ).scalar()
        if existing is not None:
            return existing
        blob_id = uuid.uuid4()
        inserted = connection.execute(
            text(
                "INSERT INTO blobs (id, digest, byte_size, media_type, inline_bytes) "
                "VALUES (:id, :digest, :byte_size, :media_type, :body) "
                "ON CONFLICT (digest, media_type) DO NOTHING RETURNING id"
            ),
            {
                "id": blob_id,
                "digest": digest,
                "byte_size": len(body),
                "media_type": media_type,
                "body": body,
            },
        ).scalar()
        if inserted is not None:
            return inserted
        return connection.execute(
            text("SELECT id FROM blobs WHERE digest = :digest AND media_type = :media_type"),
            {"digest": digest, "media_type": media_type},
        ).scalar_one()

    @classmethod
    def _store_files(
        cls,
        connection: Connection,
        applet_id: uuid.UUID,
        version: int,
        files: dict[str, bytes],
    ) -> None:
        for path, body in files.items():
            connection.execute(
                text(
                    "INSERT INTO applet_files (applet_id, version, path, blob_id) "
                    "VALUES (:id, :version, :path, :blob_id)"
                ),
                {
                    "id": applet_id,
                    "version": version,
                    "path": path,
                    "blob_id": cls._blob_id(connection, path, body),
                },
            )

    @staticmethod
    def _run_migration(
        connection: Connection,
        applet_id: uuid.UUID,
        version: int,
        package: AppletPackage,
    ) -> None:
        if package.manifest.python_entrypoint is None:
            return
        try:
            module, _factory = load_backend_entrypoint(applet_id, version, package)
            migration = getattr(module, "migrate", None)
            if migration is not None:
                migration(connection)
        finally:
            remove_backend_revision(applet_id, version, package.digest)

    @staticmethod
    def _activate_version(
        connection: Connection,
        applet_id: uuid.UUID,
        version: int,
        manifest: AppletManifest,
        mode: AppletMode,
    ) -> None:
        connection.execute(
            text(
                "UPDATE applets SET title = :title, description = :description, mode = :mode, "
                "current_version = :version, "
                "updated_at = now() WHERE id = :id"
            ),
            {
                "id": applet_id,
                "title": manifest.title,
                "description": manifest.description,
                "mode": mode.value,
                "version": version,
            },
        )
        connection.execute(
            text(
                "DELETE FROM applet_versions WHERE applet_id = :id AND version <> :current "
                "AND version NOT IN (SELECT version FROM applet_versions WHERE applet_id = :id "
                "AND version <> :current ORDER BY version DESC LIMIT :retained)"
            ),
            {"id": applet_id, "current": version, "retained": MAX_REVISIONS - 1},
        )

    @staticmethod
    def _delete_unreferenced_blobs(connection: Connection) -> None:
        connection.execute(
            text("DELETE FROM blobs WHERE NOT EXISTS (SELECT 1 FROM applet_files WHERE blob_id = blobs.id)")
        )

    def active_applets(self) -> list[AppletSummary]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT id, title, description, owner, current_version, mode, name FROM applets "
                    "WHERE archived_at IS NULL AND current_version IS NOT NULL ORDER BY title, id"
                )
            ).mappings()
            return [
                AppletSummary(
                    id=row["id"],
                    title=row["title"],
                    description=row["description"],
                    owner=row["owner"],
                    current_version=row["current_version"],
                    mode=AppletMode(row["mode"]),
                    name=row["name"],
                )
                for row in rows
            ]

    def named_applet(self, name: str) -> AppletSummary:
        for applet in self.active_applets():
            if applet.name == name:
                return applet
        raise AppletNotFound(name)

    def schema(self, applet_id: uuid.UUID) -> str:
        with self.engine.connect() as connection:
            row = connection.execute(text("SELECT name FROM applets WHERE id = :id"), {"id": applet_id}).first()
        if row is None:
            raise AppletNotFound(str(applet_id))
        return applet_schema(applet_id, row[0])

    def snapshots(self, runner: str) -> tuple[AppletSnapshot, ...]:
        """Read a runner's current packages atomically so pruning cannot race its snapshot."""
        with self.engine.connect() as connection:
            rows = (
                connection.execute(
                    text(
                        "SELECT a.id, a.current_version, f.path, b.inline_bytes FROM applets a "
                        "JOIN applet_versions v ON v.applet_id = a.id AND v.version = a.current_version "
                        "JOIN applet_files f ON f.applet_id = v.applet_id AND f.version = v.version "
                        "JOIN blobs b ON b.id = f.blob_id WHERE a.archived_at IS NULL "
                        "AND v.manifest->'jobs' @> CAST(:jobs AS jsonb) ORDER BY a.id, f.path"
                    ),
                    {"jobs": json.dumps([{"runner": runner}])},
                )
                .mappings()
                .all()
            )
        files: dict[tuple[uuid.UUID, int], dict[str, bytes]] = {}
        for row in rows:
            files.setdefault((row["id"], row["current_version"]), {})[row["path"]] = bytes(row["inline_bytes"])
        return tuple(AppletSnapshot(key[0], key[1], _validate_package_files(value)) for key, value in files.items())

    def access_mode(self, applet_id: uuid.UUID) -> AppletMode:
        with self.engine.connect() as connection:
            mode = connection.execute(
                text("SELECT mode FROM applets WHERE id = :id AND archived_at IS NULL"),
                {"id": applet_id},
            ).scalar()
        if mode is None:
            raise AppletNotFound(str(applet_id))
        return AppletMode(mode)

    def set_mode(
        self,
        applet_id: uuid.UUID,
        mode: AppletMode,
        actor: str,
        base_version: int,
        operators: frozenset[str] = frozenset(),
    ) -> None:
        with self.engine.begin() as connection:
            self._authorize_versioned_mutation(connection, applet_id, actor, base_version, operators)
            if (
                mode is AppletMode.PUBLIC
                and connection.execute(text("SELECT name FROM applets WHERE id = :id"), {"id": applet_id}).scalar_one()
                is not None
            ):
                raise ValueError("Named applications remain private")
            connection.execute(
                text("UPDATE applets SET mode = :mode, updated_at = now() WHERE id = :id"),
                {"id": applet_id, "mode": mode.value},
            )

    def current_version(self, applet_id: uuid.UUID) -> int:
        with self.engine.connect() as connection:
            version = connection.execute(
                text("SELECT current_version FROM applets WHERE id = :id AND archived_at IS NULL"),
                {"id": applet_id},
            ).scalar()
        if version is None:
            raise AppletNotFound(str(applet_id))
        return int(version)

    def version(self, applet_id: uuid.UUID, version: int) -> AppletVersion:
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    text(
                        "SELECT v.manifest, v.package_digest FROM applet_versions v "
                        "JOIN applets a ON a.id = v.applet_id "
                        "WHERE v.applet_id = :id AND v.version = :version AND a.archived_at IS NULL"
                    ),
                    {"id": applet_id, "version": version},
                )
                .mappings()
                .first()
            )
        if row is None:
            raise AppletNotFound(f"{applet_id}/v/{version}")
        return AppletVersion(
            applet_id=applet_id,
            version=version,
            manifest=_manifest_from_json(row["manifest"]),
            digest=bytes(row["package_digest"]),
        )

    def versions(self, applet_id: uuid.UUID) -> list[AppletVersionSummary]:
        self.current_version(applet_id)
        with self.engine.connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT version, published_by, published_at, byte_size FROM applet_versions "
                    "WHERE applet_id = :id ORDER BY version DESC"
                ),
                {"id": applet_id},
            ).mappings()
            return [
                AppletVersionSummary(
                    version=int(row["version"]),
                    published_by=row["published_by"],
                    published_at=row["published_at"],
                    byte_size=int(row["byte_size"]),
                )
                for row in rows
            ]

    def rollback(
        self,
        applet_id: uuid.UUID,
        version: int,
        actor: str,
        base_version: int,
        operators: frozenset[str] = frozenset(),
    ) -> None:
        with self.engine.begin() as connection:
            self._authorize_versioned_mutation(connection, applet_id, actor, base_version, operators)
            exists = connection.execute(
                text("SELECT 1 FROM applet_versions WHERE applet_id = :id AND version = :version"),
                {"id": applet_id, "version": version},
            ).scalar()
            if exists is None:
                raise AppletNotFound(f"{applet_id}/v/{version}")
            connection.execute(
                text("UPDATE applets SET current_version = :version, updated_at = now() WHERE id = :id"),
                {"id": applet_id, "version": version},
            )

    def archive(self, applet_id: uuid.UUID, actor: str, operators: frozenset[str] = frozenset()) -> None:
        with self.engine.begin() as connection:
            applet = (
                connection.execute(
                    text("SELECT owner FROM applets WHERE id = :id AND archived_at IS NULL FOR UPDATE"),
                    {"id": applet_id},
                )
                .mappings()
                .first()
            )
            if applet is None:
                raise AppletNotFound(str(applet_id))
            if applet["owner"] != actor and actor not in operators:
                raise AppletForbidden(str(applet_id))
            connection.execute(
                text("UPDATE applets SET archived_at = now(), updated_at = now() WHERE id = :id"), {"id": applet_id}
            )

    def package(self, applet_id: uuid.UUID, version: int) -> AppletPackage:
        record = self.version(applet_id, version)
        with self.engine.connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT f.path, b.inline_bytes FROM applet_files f "
                    "JOIN blobs b ON b.id = f.blob_id "
                    "WHERE f.applet_id = :id AND f.version = :version"
                ),
                {"id": applet_id, "version": version},
            ).mappings()
            files = {row["path"]: bytes(row["inline_bytes"]) for row in rows}
        package = _validate_package_files(files)
        if package.digest != record.digest:
            raise RuntimeError(f"stored package digest does not match {applet_id}/v/{version}")
        return package

    def file(
        self,
        applet_id: uuid.UUID,
        version: int,
        path: str,
        accept_gzip: bool,
        accept_html: bool,
    ) -> StoredFile:
        self.version(applet_id, version)
        requested = path.strip("/")
        candidates = [f"{DIST_PREFIX}{requested}" if requested else INDEX_FILE]
        if requested.endswith("/"):
            candidates.append(f"{DIST_PREFIX}{requested}index.html")
        elif requested and "." not in PurePosixPath(requested).name and accept_html:
            candidates.append(INDEX_FILE)
        for candidate in candidates:
            variants = [(candidate + ".gz", "gzip"), (candidate, None)] if accept_gzip else [(candidate, None)]
            for stored_path, encoding in variants:
                found = self._stored_file(applet_id, version, stored_path, encoding)
                if found is not None:
                    return found
        raise AppletNotFound(f"{applet_id}/v/{version}/{path}")

    def _stored_file(
        self, applet_id: uuid.UUID, version: int, path: str, content_encoding: str | None
    ) -> StoredFile | None:
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    text(
                        "SELECT b.inline_bytes, b.media_type, b.digest FROM applet_files f "
                        "JOIN blobs b ON b.id = f.blob_id "
                        "WHERE f.applet_id = :id AND f.version = :version AND f.path = :path"
                    ),
                    {"id": applet_id, "version": version, "path": path},
                )
                .mappings()
                .first()
            )
        if row is None:
            return None
        return StoredFile(
            body=bytes(row["inline_bytes"]),
            media_type=row["media_type"],
            digest=bytes(row["digest"]),
            content_encoding=content_encoding,
        )

    def query(self, applet_id: uuid.UUID, sql: str, parameters: dict[str, object]) -> dict[str, object]:
        self.current_version(applet_id)
        command = validate_query(sql)
        engine = engine_for_role(self.database, self.schema(applet_id), applet_role(applet_id))
        try:
            with engine.begin() as connection:
                connection.execute(text(f"SET LOCAL statement_timeout = '{QUERY_TIMEOUT_MS}ms'"))
                if command in {"SELECT", "WITH"}:
                    statement = sql.strip().removesuffix(";").rstrip()
                    bounded = text(
                        "SELECT CASE WHEN pg_column_size(payload) <= "
                        f"{MAX_QUERY_RESPONSE_BYTES} THEN payload END AS payload, "
                        "row_count, pg_column_size(payload) AS byte_size FROM ("
                        "SELECT COALESCE(json_agg(row_to_json(applet_query)), '[]'::json) AS payload, "
                        "count(*) AS row_count FROM (SELECT * FROM ("
                        f"{statement}"
                        f") AS applet_result LIMIT {MAX_QUERY_ROWS + 1}) AS applet_query"
                        ") AS applet_payload"
                    )
                    result = connection.execute(bounded, parameters).mappings().one()
                    row_count = int(result["row_count"])
                    if row_count > MAX_QUERY_ROWS:
                        raise QueryLimitExceeded(f"query returned more than {MAX_QUERY_ROWS} rows")
                    if int(result["byte_size"]) > MAX_QUERY_RESPONSE_BYTES:
                        raise QueryLimitExceeded(f"query response exceeds {MAX_QUERY_RESPONSE_BYTES} bytes")
                    rows = cast(list[dict[str, object]], result["payload"])
                    columns = list(rows[0]) if rows else []
                    response: dict[str, object] = {"columns": columns, "rows": rows, "row_count": row_count}
                    if len(json.dumps(response).encode()) > MAX_QUERY_RESPONSE_BYTES:
                        raise QueryLimitExceeded(f"query response exceeds {MAX_QUERY_RESPONSE_BYTES} bytes")
                    return response
                result = connection.execute(text(sql), parameters)
                if result.returns_rows:
                    raise InvalidQuery(f"{command} statements may not return rows")
                return {"columns": [], "rows": [], "row_count": result.rowcount}
        finally:
            engine.dispose()


@dataclass(frozen=True)
class AppletServices:
    """Services passed to a dynamic Python applet backend."""

    applet_id: uuid.UUID
    version: int
    database: DatabaseSpec
    name: str
    data_url: str
    schema: str

    def engine(self) -> Engine:
        return engine_for_role(self.database, self.schema, applet_role(self.applet_id))


def validate_query(sql: str) -> str:
    """Accept one data or schema statement while blocking transaction and role control."""
    statement = sql.strip()
    if not statement or ";" in statement.rstrip(";"):
        raise InvalidQuery("query must contain exactly one statement")
    command = re.match(r"[A-Za-z]+", statement)
    if command is None or command.group(0).upper() not in {
        "ALTER",
        "COMMENT",
        "CREATE",
        "DELETE",
        "DROP",
        "INSERT",
        "SELECT",
        "TRUNCATE",
        "UPDATE",
        "WITH",
    }:
        raise InvalidQuery("query command is not allowed")
    normalized = command.group(0).upper()
    if normalized == "WITH" and re.search(r"\b(INSERT|UPDATE|DELETE)\b", statement, re.IGNORECASE):
        raise InvalidQuery("data-modifying WITH statements are not allowed")
    if normalized not in {"SELECT", "WITH"} and re.search(r"\bRETURNING\b", statement, re.IGNORECASE):
        raise InvalidQuery(f"{normalized} statements may not return rows")
    return normalized


def _module_name(applet_id: uuid.UUID, version: int, digest: bytes) -> str:
    return f"marina_applet_{applet_id.hex}_v{version}_{digest.hex()[:12]}"


def materialize_package(module_root: Path, package: AppletPackage) -> None:
    module_root.mkdir(parents=True, exist_ok=True)
    (module_root / "__init__.py").touch()
    for path, body in package.files.items():
        if not path.endswith(".py") or path.startswith(DIST_PREFIX):
            continue
        target = module_root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)


def validate_backend_import(package: AppletPackage) -> None:
    """Import a Python backend with Marina's module layout and without environment credentials."""
    entrypoints = [job.module for job in package.manifest.jobs]
    if package.manifest.python_entrypoint is not None:
        entrypoints.insert(0, package.manifest.python_entrypoint)
    if not entrypoints:
        return
    with tempfile.TemporaryDirectory(prefix="marina-applet-validate-") as directory:
        module_root = Path(directory) / "candidate"
        materialize_package(module_root, package)
        environment = {name: os.environ[name] for name in ("PATH", "LANG", "LC_ALL") if name in os.environ}
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-m",
                    "marina.applet_validator",
                    str(module_root),
                    *entrypoints,
                ],
                capture_output=True,
                text=True,
                timeout=BACKEND_VALIDATION_TIMEOUT,
                env=environment,
            )
        except subprocess.TimeoutExpired as error:
            raise ValueError(f"Python backend validation exceeded {BACKEND_VALIDATION_TIMEOUT} seconds") from error
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "import failed"
        raise ValueError(f"Python backend validation failed: {detail[:500]}")


def load_backend_entrypoint(applet_id: uuid.UUID, version: int, package: AppletPackage) -> tuple[object, str]:
    """Return the imported backend module and factory name for one applet revision."""
    assert package.manifest.python_entrypoint is not None
    module_path, factory_name = package.manifest.python_entrypoint.split(":", 1)
    root_name = _module_name(applet_id, version, package.digest)
    module_root = BACKEND_CACHE_ROOT / root_name
    materialize_package(module_root, package)
    cache_parent = str(BACKEND_CACHE_ROOT)
    if cache_parent not in sys.path:
        sys.path.insert(0, cache_parent)
    module = importlib.import_module(f"{root_name}.{module_path}")
    return module, factory_name


def remove_backend_revision(applet_id: uuid.UUID, version: int, digest: bytes) -> None:
    """Remove imported modules and source files for one applet revision."""
    root_name = _module_name(applet_id, version, digest)
    for name in [name for name in sys.modules if name == root_name or name.startswith(root_name + ".")]:
        del sys.modules[name]
    shutil.rmtree(BACKEND_CACHE_ROOT / root_name, ignore_errors=True)


class AppletRuntime:
    """Load and retain revision-specific ASGI applications."""

    def __init__(self, store: AppletStore, data_root: str = ""):
        self.store = store
        self.data_root = data_root
        self._apis: dict[tuple[uuid.UUID, int], ASGIApp] = {}
        self._registered: dict[tuple[uuid.UUID, int], RegisteredApi] = {}
        self._digests: dict[tuple[uuid.UUID, int], bytes] = {}
        self._initializers: dict[tuple[uuid.UUID, int], threading.Lock] = {}
        self._lock = threading.Lock()

    def api(self, applet_id: uuid.UUID, version: int) -> ASGIApp:
        key = (applet_id, version)
        try:
            self.store.version(applet_id, version)
        except AppletNotFound:
            self.retain_versions(applet_id, self._retained_versions(applet_id))
            raise
        with self._lock:
            if key in self._apis:
                return self._apis[key]
            initializer = self._initializers.setdefault(key, threading.Lock())
        with initializer:
            with self._lock:
                if key in self._apis:
                    return self._apis[key]
            package = self.store.package(applet_id, version)
            if package.manifest.python_entrypoint is None:
                raise AppletNotFound(f"{applet_id}/v/{version}/api")
            try:
                module, factory_name = load_backend_entrypoint(applet_id, version, package)
                factory = getattr(module, factory_name)
                manifest = package.manifest
                name = manifest.name or str(applet_id)
                candidate = factory(
                    AppletServices(
                        applet_id=applet_id,
                        version=version,
                        database=self.store.database,
                        name=name,
                        data_url=manifest.data_url or prefix_join(self.data_root, name),
                        schema=self.store.schema(applet_id),
                    )
                )
                if isinstance(candidate, RegisteredApi):
                    with self._lock:
                        self._registered[key] = candidate
                    candidate = candidate.app
                if not callable(candidate):
                    raise TypeError("create_api did not return an ASGI application")
                api = cast(ASGIApp, candidate)
            except Exception as error:
                remove_backend_revision(applet_id, version, package.digest)
                message = f"{type(error).__name__}: {error}"
                raise AppletBackendUnavailable(message) from error
            with self._lock:
                self._apis[key] = api
                self._digests[key] = package.digest
            try:
                self.store.version(applet_id, version)
            except AppletNotFound:
                self.retain_versions(applet_id, self._retained_versions(applet_id))
                raise
            return api

    def registered_api(self, applet_id: uuid.UUID, version: int) -> RegisteredApi | None:
        self.api(applet_id, version)
        with self._lock:
            return self._registered.get((applet_id, version))

    def _retained_versions(self, applet_id: uuid.UUID) -> set[int]:
        try:
            return {item.version for item in self.store.versions(applet_id)}
        except AppletNotFound:
            return set()

    def retain_versions(self, applet_id: uuid.UUID, versions: set[int]) -> None:
        """Discard cached runtime state for revisions outside the retained set."""
        removed: list[tuple[int, bytes]] = []
        with self._lock:
            stale_apis = [key for key in self._apis if key[0] == applet_id and key[1] not in versions]
            for key in stale_apis:
                self._apis.pop(key)
                self._registered.pop(key, None)
                removed.append((key[1], self._digests.pop(key)))
            stale_initializers = [key for key in self._initializers if key[0] == applet_id and key[1] not in versions]
            for key in stale_initializers:
                self._initializers.pop(key)
        for version, digest in removed:
            remove_backend_revision(applet_id, version, digest)
