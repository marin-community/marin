# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Forward-only migration from the sealed Iris schema to resource schema v2."""

import hashlib
import json
import re
import sqlite3
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from iris.cluster.controller.persistence.schema import metadata
from iris.cluster.controller.persistence.schema.version import (
    MERGE_BASE_MIGRATION_NAMES,
    MERGE_BASE_SCHEMA_FINGERPRINT,
    RESOURCE_SCHEMA_EPOCH,
    RESOURCE_SCHEMA_NAME,
)
from rigging.timing import Timestamp
from sqlalchemy.dialects import sqlite as sqlalchemy_sqlite
from sqlalchemy.schema import CreateIndex, CreateTable

_RESOURCE_UID_NAMESPACE = uuid.UUID("2c72b7f4-a156-5d27-8b58-7de28d5ec4cc")
_RESOURCE_UID_PREFIX = "iris-resource-v2"
_SAMPLE_LIMIT = 20
_LOCAL_CLUSTER_SENTINEL = "local"
_FINAL_MIGRATION_NAMES = (RESOURCE_SCHEMA_NAME,)


@dataclass(frozen=True, slots=True)
class SchemaStatus:
    epoch: int | None
    schema_fingerprint: str
    migration_names: tuple[str, ...]
    accepted: bool
    problems: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MigrationContext:
    cluster_id: str
    backend_kinds: Mapping[str, str]
    scale_group_to_backend: Mapping[str, str]
    backend_namespaces: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class MigrationProblem:
    name: str
    count: int
    sample_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MigrationReport:
    schema: SchemaStatus
    problems: tuple[MigrationProblem, ...]

    @property
    def accepted(self) -> bool:
        return self.schema.accepted and not self.problems


class ResourceSchemaMigrationError(RuntimeError):
    """The database cannot be migrated without losing resource identity."""


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _normalize_sql(value: str) -> str:
    return " ".join(value.strip().split())


def _normalize_default(column_type: str, value: object) -> object:
    if not isinstance(value, str):
        return value
    normalized = _normalize_sql(value)
    if "INT" in column_type and re.fullmatch(r"'-?\d+'", normalized):
        return normalized[1:-1]
    return normalized


def _check_predicates(create_sql: str) -> tuple[str, ...]:
    predicates: list[str] = []
    upper = create_sql.upper()
    offset = 0
    while True:
        match = re.search(r"\bCHECK\s*\(", upper[offset:])
        if match is None:
            break
        opening = offset + match.end() - 1
        depth = 1
        index = opening + 1
        quote = ""
        while index < len(create_sql) and depth:
            char = create_sql[index]
            if quote:
                if char == quote:
                    if index + 1 < len(create_sql) and create_sql[index + 1] == quote:
                        index += 1
                    else:
                        quote = ""
            elif char in {"'", '"'}:
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            index += 1
        if depth:
            raise ResourceSchemaMigrationError("malformed CHECK predicate in SQLite schema")
        predicates.append(_normalize_sql(create_sql[opening + 1 : index - 1]))
        offset = index
    return tuple(sorted(predicates))


def _index_shape(connection: sqlite3.Connection, index_name: str) -> tuple[tuple[object, ...], ...]:
    quoted = _quote_identifier(index_name)
    return tuple(
        (row[0], row[2], row[3], row[4], row[5])
        for row in connection.execute(f"PRAGMA index_xinfo({quoted})").fetchall()
    )


def _table_shape(connection: sqlite3.Connection, table_name: str, create_sql: str) -> dict[str, object]:
    quoted = _quote_identifier(table_name)
    columns = sorted(
        (
            row[1],
            str(row[2]).upper(),
            int(row[3]),
            _normalize_default(str(row[2]).upper(), row[4]),
            int(row[5]),
        )
        for row in connection.execute(f"PRAGMA table_info({quoted})").fetchall()
    )
    foreign_keys = sorted(
        (row[2], row[3], row[4], row[5], row[6], row[7])
        for row in connection.execute(f"PRAGMA foreign_key_list({quoted})").fetchall()
    )
    automatic_indexes: list[tuple[object, ...]] = []
    for row in connection.execute(f"PRAGMA index_list({quoted})").fetchall():
        _, index_name, unique, origin, partial = row[:5]
        if origin != "c":
            automatic_indexes.append((origin, int(unique), int(partial), _index_shape(connection, index_name)))
    normalized_sql = _normalize_sql(create_sql)
    return {
        "columns": columns,
        "foreign_keys": foreign_keys,
        "checks": _check_predicates(create_sql),
        "automatic_indexes": sorted(automatic_indexes, key=repr),
        "autoincrement": " AUTOINCREMENT" in normalized_sql.upper(),
        "deferred_foreign_keys": normalized_sql.upper().count("DEFERRABLE INITIALLY DEFERRED"),
    }


def schema_fingerprint(connection: sqlite3.Connection) -> str:
    """Return a stable fingerprint of persisted SQLite schema semantics."""
    tables: dict[str, object] = {}
    indexes: dict[str, object] = {}
    other_objects: dict[str, object] = {}
    rows = connection.execute(
        """
        SELECT type, name, tbl_name, sql FROM sqlite_schema
        WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name
        """
    ).fetchall()
    for kind, name, table_name, sql in rows:
        if kind == "table":
            tables[name] = _table_shape(connection, name, str(sql))
            continue
        if kind != "index":
            other_objects[name] = {
                "type": kind,
                "table": table_name,
                "sql": _normalize_sql(str(sql)) if sql is not None else None,
            }
            continue
        if sql is None:
            continue
        normalized_sql = _normalize_sql(str(sql))
        where = re.search(r"\bWHERE\b(.+)$", normalized_sql, re.IGNORECASE)
        indexes[name] = {
            "table": table_name,
            "unique": normalized_sql.upper().startswith("CREATE UNIQUE INDEX"),
            "columns": _index_shape(connection, name),
            "predicate": _normalize_sql(where.group(1)) if where is not None else None,
        }
    payload = json.dumps(
        {"tables": tables, "indexes": indexes, "other_objects": other_objects},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _migration_names(connection: sqlite3.Connection) -> tuple[str, ...]:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'schema_migrations'"
    ).fetchone()
    if exists is None:
        return ()
    return tuple(row[0] for row in connection.execute("SELECT name FROM schema_migrations ORDER BY name"))


def _has_user_schema(connection: sqlite3.Connection) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
        ).fetchone()
        is not None
    )


def inspect_schema(connection: sqlite3.Connection) -> SchemaStatus:
    """Inspect the physical schema and migration ledger without writing."""
    fingerprint = schema_fingerprint(connection)
    migration_names = _migration_names(connection)
    if not _has_user_schema(connection):
        return SchemaStatus(None, fingerprint, migration_names, True, ())

    if fingerprint == final_schema_fingerprint() and migration_names == _FINAL_MIGRATION_NAMES:
        return SchemaStatus(RESOURCE_SCHEMA_EPOCH, fingerprint, migration_names, True, ())

    problems: list[str] = []
    if fingerprint != MERGE_BASE_SCHEMA_FINGERPRINT:
        problems.append("schema_fingerprint_mismatch")
    if migration_names != MERGE_BASE_MIGRATION_NAMES:
        problems.append("migration_ledger_mismatch")
    return SchemaStatus(None, fingerprint, migration_names, not problems, tuple(problems))


def _memory_connection_with_final_schema() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        _create_final_schema(connection)
        connection.commit()
    except Exception:
        connection.close()
        raise
    return connection


_FINAL_SCHEMA_FINGERPRINT: str | None = None


def final_schema_fingerprint() -> str:
    """Return the fingerprint emitted by the final declarative metadata."""
    global _FINAL_SCHEMA_FINGERPRINT
    if _FINAL_SCHEMA_FINGERPRINT is None:
        connection = _memory_connection_with_final_schema()
        try:
            _FINAL_SCHEMA_FINGERPRINT = schema_fingerprint(connection)
        finally:
            connection.close()
    return _FINAL_SCHEMA_FINGERPRINT


def _create_final_schema(connection: sqlite3.Connection) -> None:
    dialect = sqlalchemy_sqlite.dialect()
    tables = sorted(metadata.tables.values(), key=lambda table: table.name)
    for table in tables:
        connection.execute(str(CreateTable(table).compile(dialect=dialect)))
    for table in tables:
        for index in sorted(table.indexes, key=lambda item: item.name or ""):
            connection.execute(str(CreateIndex(index).compile(dialect=dialect)))


def _resource_uid(kind: str, *parts: object) -> str:
    name = "\0".join((_RESOURCE_UID_PREFIX, kind, *(str(part) for part in parts)))
    return str(uuid.uuid5(_RESOURCE_UID_NAMESPACE, name))


def _open_database(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def _problem(
    connection: sqlite3.Connection,
    *,
    name: str,
    select_sql: str,
    params: Sequence[object] = (),
) -> MigrationProblem | None:
    count = int(connection.execute(f"SELECT count(*) FROM ({select_sql})", params).fetchone()[0])
    if not count:
        return None
    samples = tuple(
        str(row[0])
        for row in connection.execute(
            f"SELECT sample_id FROM ({select_sql}) ORDER BY sample_id LIMIT {_SAMPLE_LIMIT}",
            params,
        ).fetchall()
    )
    return MigrationProblem(name=name, count=count, sample_ids=samples)


def _append_problem(
    problems: list[MigrationProblem],
    connection: sqlite3.Connection,
    *,
    name: str,
    select_sql: str,
    params: Sequence[object] = (),
) -> None:
    problem = _problem(connection, name=name, select_sql=select_sql, params=params)
    if problem is not None:
        problems.append(problem)


def _json_mapping(value: Mapping[str, str]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))


def _context_problems(context: MigrationContext) -> list[MigrationProblem]:
    problems: list[MigrationProblem] = []
    if not context.cluster_id.strip():
        problems.append(MigrationProblem("cluster_id_missing", 1, ("cluster_id",)))

    invalid_backends = sorted(
        backend_id
        for backend_id, kind in context.backend_kinds.items()
        if not backend_id.strip() or kind not in {"rpc", "kubernetes"}
    )
    if not context.backend_kinds:
        invalid_backends.append("no_backends")
    if invalid_backends:
        problems.append(
            MigrationProblem(
                "backend_configuration_invalid", len(invalid_backends), tuple(invalid_backends[:_SAMPLE_LIMIT])
            )
        )
    invalid_scale_groups = sorted(
        scale_group
        for scale_group, backend_id in context.scale_group_to_backend.items()
        if not scale_group.strip()
        or backend_id not in context.backend_kinds
        or context.backend_kinds.get(backend_id) != "rpc"
    )
    if invalid_scale_groups:
        problems.append(
            MigrationProblem(
                "scale_group_configuration_invalid",
                len(invalid_scale_groups),
                tuple(invalid_scale_groups[:_SAMPLE_LIMIT]),
            )
        )
    invalid_namespaces = sorted(
        backend_id
        for backend_id, namespace in context.backend_namespaces.items()
        if not namespace.strip() or context.backend_kinds.get(backend_id) != "kubernetes"
    )
    if invalid_namespaces:
        problems.append(
            MigrationProblem(
                "backend_namespace_configuration_invalid",
                len(invalid_namespaces),
                tuple(invalid_namespaces[:_SAMPLE_LIMIT]),
            )
        )
    return problems


def _source_preflight(connection: sqlite3.Connection, context: MigrationContext) -> tuple[MigrationProblem, ...]:
    problems = _context_problems(context)

    configured_backends_json = json.dumps(sorted(context.backend_kinds), separators=(",", ":"))
    scale_group_mapping_json = _json_mapping(context.scale_group_to_backend)
    backend_namespace_json = _json_mapping(context.backend_namespaces)

    _append_problem(
        problems,
        connection,
        name="source_json_malformed",
        select_sql="""
            SELECT 'job_config:' || job_id AS sample_id FROM job_config
            WHERE (res_device_json IS NOT NULL AND NOT json_valid(res_device_json))
               OR (constraints_json IS NOT NULL AND NOT json_valid(constraints_json))
               OR NOT json_valid(entrypoint_json)
               OR NOT json_valid(environment_json)
               OR NOT json_valid(ports_json)
               OR NOT json_valid(submit_argv_json)
            UNION ALL
            SELECT 'worker:' || worker_id FROM workers
            WHERE NOT json_valid(md_device_json) OR NOT json_valid(md_provenance_json)
            UNION ALL
            SELECT 'slice:' || slice_id FROM slices
            WHERE NOT json_valid(worker_ids) OR json_type(worker_ids) <> 'array'
            UNION ALL
            SELECT 'endpoint:' || endpoint_id FROM endpoints
            WHERE NOT json_valid(metadata_json) OR json_type(metadata_json) <> 'object'
        """,
    )
    _append_problem(
        problems,
        connection,
        name="source_relationship_orphaned",
        select_sql="""
            SELECT 'job_config:' || c.job_id AS sample_id
            FROM job_config c LEFT JOIN jobs j ON j.job_id = c.job_id WHERE j.job_id IS NULL
            UNION ALL
            SELECT 'job_without_config:' || j.job_id
            FROM jobs j LEFT JOIN job_config c ON c.job_id = j.job_id WHERE c.job_id IS NULL
            UNION ALL
            SELECT 'task:' || t.task_id
            FROM tasks t LEFT JOIN jobs j ON j.job_id = t.job_id WHERE j.job_id IS NULL
            UNION ALL
            SELECT 'attempt:' || a.attempt_uid
            FROM task_attempts a LEFT JOIN tasks t ON t.task_id = a.task_id WHERE t.task_id IS NULL
            UNION ALL
            SELECT 'endpoint_job:' || e.endpoint_id
            FROM endpoints e LEFT JOIN jobs j ON j.job_id = e.job_id WHERE j.job_id IS NULL
            UNION ALL
            SELECT 'endpoint_task:' || e.endpoint_id
            FROM endpoints e LEFT JOIN tasks t ON t.task_id = e.task_id
            WHERE e.task_id IS NOT NULL AND t.task_id IS NULL
        """,
    )
    _append_problem(
        problems,
        connection,
        name="task_index_invalid",
        select_sql="""
            SELECT task_id AS sample_id FROM tasks
            WHERE typeof(task_index) <> 'integer' OR task_index < 0
               OR task_id <> job_id || '/' || CAST(task_index AS TEXT)
        """,
    )
    _append_problem(
        problems,
        connection,
        name="attempt_identity_invalid",
        select_sql="""
            SELECT task_id || ':' || attempt_id AS sample_id FROM task_attempts
            WHERE typeof(attempt_id) <> 'integer' OR attempt_id < 0
               OR length(attempt_uid) <> 16 OR attempt_uid <> lower(attempt_uid)
               OR attempt_uid GLOB '*[^0-9a-f]*'
        """,
    )
    _append_problem(
        problems,
        connection,
        name="source_value_invalid",
        select_sql="""
            SELECT 'job:' || job_id AS sample_id FROM jobs
            WHERE depth < 0 OR num_tasks < 0 OR job_id = '' OR root_job_id = ''
            UNION ALL
            SELECT 'job_config:' || job_id FROM job_config
            WHERE max_task_failures < 0 OR max_retries_failure < 0
               OR max_retries_preemption < 0 OR fail_if_exists NOT IN (0, 1)
            UNION ALL
            SELECT 'task:' || task_id FROM tasks
            WHERE max_retries_failure < 0 OR max_retries_preemption < 0
            UNION ALL
            SELECT 'worker:' || worker_id FROM workers
            WHERE total_cpu_millicores < 0 OR total_memory_bytes < 0
               OR md_disk_bytes < 0 OR total_gpu_count < 0 OR total_tpu_count < 0
            UNION ALL
            SELECT 'slice:' || slice_id FROM slices
            WHERE lifecycle NOT IN ('requesting', 'booting', 'initializing', 'ready', 'failed')
            UNION ALL
            SELECT 'changelog:' || seq FROM federation_changelog
            WHERE tombstone NOT IN (0, 1)
        """,
    )
    _append_problem(
        problems,
        connection,
        name="resource_relationship_ambiguous",
        select_sql="""
            SELECT 'job_parent:' || j.job_id AS sample_id
            FROM jobs j LEFT JOIN jobs parent ON parent.job_id = j.parent_job_id
            WHERE (j.parent_job_id IS NULL AND j.depth <> 0)
               OR (j.parent_job_id IS NOT NULL AND (parent.job_id IS NULL OR parent.depth + 1 <> j.depth))
            UNION ALL
            SELECT 'job_root:' || j.job_id
            FROM jobs j LEFT JOIN jobs root ON root.job_id = j.root_job_id
            WHERE root.job_id IS NULL OR root.depth <> 0
            UNION ALL
            SELECT 'task_attempt:' || t.task_id
            FROM tasks t LEFT JOIN task_attempts a
              ON a.task_id = t.task_id AND a.attempt_id = t.current_attempt_id
            WHERE t.current_attempt_id <> -1 AND a.attempt_uid IS NULL
            UNION ALL
            SELECT 'endpoint_task:' || e.endpoint_id
            FROM endpoints e JOIN tasks t ON t.task_id = e.task_id
            WHERE e.task_id IS NOT NULL AND t.job_id <> e.job_id
        """,
    )
    _append_problem(
        problems,
        connection,
        name="federation_direction_ambiguous",
        select_sql="""
            SELECT job_id AS sample_id FROM federated_jobs
            WHERE direction NOT IN (0, 1)
               OR (direction = 0 AND (handoff_state IS NULL OR handoff_state NOT IN (0, 1, 2, 3)))
               OR (direction = 1 AND handoff_state IS NOT NULL)
        """,
    )
    _append_problem(
        problems,
        connection,
        name="endpoint_access_invalid",
        select_sql="""
            SELECT endpoint_id AS sample_id FROM endpoints
            WHERE access IS NOT NULL AND access NOT IN (0, 2)
        """,
    )
    _append_problem(
        problems,
        connection,
        name="backend_id_unknown",
        select_sql="""
            WITH configured(value) AS (SELECT value FROM json_each(?))
            SELECT 'job:' || job_id AS sample_id FROM jobs
            WHERE backend_id <> '' AND backend_id NOT IN configured
            UNION ALL
            SELECT 'task:' || task_id FROM tasks
            WHERE backend_id <> '' AND backend_id NOT IN configured
            UNION ALL
            SELECT 'attempt:' || attempt_uid FROM task_attempts
            WHERE backend_id <> '' AND backend_id NOT IN configured
        """,
        params=(configured_backends_json,),
    )
    _append_problem(
        problems,
        connection,
        name="backend_coordinates_conflict",
        select_sql="""
            SELECT j.job_id AS sample_id
            FROM jobs j
            WHERE (
                SELECT count(DISTINCT backend_id) FROM (
                    SELECT j.backend_id AS backend_id WHERE j.backend_id <> ''
                    UNION ALL SELECT t.backend_id FROM tasks t
                        WHERE t.job_id = j.job_id AND t.backend_id <> ''
                    UNION ALL SELECT a.backend_id FROM task_attempts a JOIN tasks t ON t.task_id = a.task_id
                        WHERE t.job_id = j.job_id AND a.backend_id <> ''
                )
            ) > 1
        """,
    )
    one_backend = next(iter(context.backend_kinds)) if len(context.backend_kinds) == 1 else ""
    _append_problem(
        problems,
        connection,
        name="backend_identity_missing",
        select_sql="""
            SELECT j.job_id AS sample_id
            FROM jobs j LEFT JOIN federated_jobs f ON f.job_id = j.job_id
            WHERE ? = ''
              AND j.backend_id = ''
              AND NOT EXISTS (SELECT 1 FROM tasks t WHERE t.job_id = j.job_id AND t.backend_id <> '')
              AND NOT EXISTS (
                  SELECT 1 FROM task_attempts a JOIN tasks t ON t.task_id = a.task_id
                  WHERE t.job_id = j.job_id AND a.backend_id <> ''
              )
              AND NOT (f.direction = 0 AND NOT EXISTS (
                  SELECT 1 FROM task_attempts a JOIN tasks t ON t.task_id = a.task_id
                  WHERE t.job_id = j.job_id
              ))
        """,
        params=(one_backend,),
    )
    _append_problem(
        problems,
        connection,
        name="scale_group_backend_unknown",
        select_sql="""
            WITH mapping AS (SELECT key, value FROM json_each(?))
            SELECT 'scaling_group:' || name AS sample_id FROM scaling_groups
            WHERE name NOT IN (SELECT key FROM mapping)
            UNION ALL
            SELECT 'slice:' || slice_id FROM slices
            WHERE scale_group NOT IN (SELECT key FROM mapping)
            UNION ALL
            SELECT 'worker:' || worker_id FROM workers
            WHERE scale_group <> '' AND scale_group NOT IN (SELECT key FROM mapping)
        """,
        params=(scale_group_mapping_json,),
    )
    _append_problem(
        problems,
        connection,
        name="worker_backend_ambiguous",
        select_sql="""
            WITH mapping AS (SELECT key, value FROM json_each(?)),
                 kinds AS (SELECT key, value FROM json_each(?))
            SELECT w.worker_id AS sample_id
            FROM workers w
            WHERE COALESCE((SELECT value FROM mapping WHERE key = w.scale_group), ?) = ''
               OR COALESCE((SELECT value FROM kinds WHERE key = COALESCE(
                    (SELECT value FROM mapping WHERE key = w.scale_group), ?
               )), '') <> 'rpc'
        """,
        params=(scale_group_mapping_json, _json_mapping(context.backend_kinds), one_backend, one_backend),
    )
    _append_problem(
        problems,
        connection,
        name="runtime_identity_ambiguous",
        select_sql="""
            WITH namespaces AS (SELECT key, value FROM json_each(?)),
                 kinds AS (SELECT key, value FROM json_each(?))
            SELECT a.attempt_uid AS sample_id
            FROM task_attempts a JOIN tasks t ON t.task_id = a.task_id
            JOIN jobs j ON j.job_id = t.job_id
            WHERE (a.pod_name <> '' OR a.pod_uid <> '' OR a.node_name <> '')
              AND (
                  a.pod_name = '' OR a.pod_uid = ''
                  OR COALESCE(NULLIF(a.backend_id, ''), NULLIF(t.backend_id, ''), NULLIF(j.backend_id, ''), ?) = ''
                  OR COALESCE((SELECT value FROM kinds WHERE key = COALESCE(
                      NULLIF(a.backend_id, ''), NULLIF(t.backend_id, ''), NULLIF(j.backend_id, ''), ?
                  )), '') <> 'kubernetes'
                  OR COALESCE((SELECT value FROM namespaces WHERE key = COALESCE(
                      NULLIF(a.backend_id, ''), NULLIF(t.backend_id, ''), NULLIF(j.backend_id, ''), ?
                  )), '') = ''
              )
        """,
        params=(backend_namespace_json, _json_mapping(context.backend_kinds), one_backend, one_backend, one_backend),
    )
    _append_problem(
        problems,
        connection,
        name="node_attribute_invalid",
        select_sql="""
            SELECT worker_id || ':' || key AS sample_id FROM worker_attributes
            WHERE (value_type = 'str' AND NOT (
                       str_value IS NOT NULL AND int_value IS NULL AND float_value IS NULL
                   ))
               OR (value_type = 'int' AND NOT (
                       str_value IS NULL AND int_value IS NOT NULL AND float_value IS NULL
                   ))
               OR (value_type = 'float' AND NOT (
                       str_value IS NULL AND int_value IS NULL AND float_value IS NOT NULL
                   ))
               OR value_type NOT IN ('str', 'int', 'float')
        """,
    )
    _append_problem(
        problems,
        connection,
        name="slice_membership_invalid",
        select_sql="""
            SELECT s.slice_id || ':' || CAST(member.key AS TEXT) AS sample_id
            FROM slices s, json_each(s.worker_ids) AS member
            WHERE member.type <> 'text'
        """,
    )
    return tuple(sorted(problems, key=lambda problem: problem.name))


def preflight_database(database_path: Path, *, context: MigrationContext) -> MigrationReport:
    """Run exact-schema and retained-row validation without source writes."""
    connection = _open_database(database_path)
    try:
        schema = inspect_schema(connection)
        if not schema.accepted:
            return MigrationReport(schema=schema, problems=())
        if schema.epoch == RESOURCE_SCHEMA_EPOCH:
            return MigrationReport(schema=schema, problems=tuple(_context_problems(context)))
        if not _has_user_schema(connection):
            return MigrationReport(schema=schema, problems=tuple(_context_problems(context)))
        return MigrationReport(schema=schema, problems=_source_preflight(connection, context))
    finally:
        connection.close()


def _raise_for_report(report: MigrationReport) -> None:
    if report.accepted:
        return
    details = [*report.schema.problems, *(f"{problem.name}={problem.count}" for problem in report.problems)]
    raise ResourceSchemaMigrationError("resource schema migration rejected: " + ", ".join(details))


def _job_backend_expression(prefix: str = "j") -> str:
    return f"""
        COALESCE(
            NULLIF({prefix}.backend_id, ''),
            (SELECT min(NULLIF(t.backend_id, '')) FROM tasks t WHERE t.job_id = {prefix}.job_id),
            (SELECT min(NULLIF(a.backend_id, '')) FROM task_attempts a
                JOIN tasks t ON t.task_id = a.task_id WHERE t.job_id = {prefix}.job_id),
            :one_backend
        )
    """


def _prepare_mapping_tables(connection: sqlite3.Connection, context: MigrationContext) -> None:
    one_backend = next(iter(context.backend_kinds)) if len(context.backend_kinds) == 1 else ""
    connection.execute(
        "CREATE TEMP TABLE resource_job_map ("
        "job_id TEXT PRIMARY KEY, job_uid TEXT NOT NULL, authority_cluster_id TEXT NOT NULL, "
        "execution_cluster_id TEXT NOT NULL, backend_id TEXT NOT NULL, placement_state TEXT NOT NULL)"
    )
    connection.execute(
        f"""
        INSERT INTO resource_job_map
        SELECT
            j.job_id,
            resource_uid('job',
                CASE WHEN f.direction = 1 THEN f.peer_id ELSE :cluster_id END,
                j.job_id
            ),
            CASE WHEN f.direction = 1 THEN f.peer_id ELSE :cluster_id END,
            CASE
                WHEN f.direction = 0 THEN j.cluster
                ELSE :cluster_id
            END,
            COALESCE({_job_backend_expression()}, ''),
            CASE WHEN COALESCE({_job_backend_expression()}, '') = '' THEN 'pending' ELSE 'known' END
        FROM jobs j LEFT JOIN federated_jobs f ON f.job_id = j.job_id
        """,
        {"cluster_id": context.cluster_id, "one_backend": one_backend},
    )
    connection.execute(
        "CREATE TEMP TABLE resource_task_map ("
        "task_id TEXT PRIMARY KEY, task_uid TEXT NOT NULL, job_uid TEXT NOT NULL, backend_id TEXT NOT NULL)"
    )
    connection.execute(
        """
        INSERT INTO resource_task_map
        SELECT t.task_id, resource_uid('task', j.job_uid, t.task_index), j.job_uid,
               COALESCE(NULLIF(t.backend_id, ''), j.backend_id)
        FROM tasks t JOIN resource_job_map j ON j.job_id = t.job_id
        """
    )
    connection.execute(
        "CREATE TEMP TABLE resource_node_map ("
        "worker_id TEXT PRIMARY KEY, node_uid TEXT NOT NULL, backend_id TEXT NOT NULL)"
    )
    connection.execute(
        """
        INSERT INTO resource_node_map
        SELECT w.worker_id,
               resource_uid('node', :cluster_id,
                   COALESCE((SELECT value FROM json_each(:scale_groups) WHERE key = w.scale_group), :one_backend),
                   w.worker_id),
               COALESCE((SELECT value FROM json_each(:scale_groups) WHERE key = w.scale_group), :one_backend)
        FROM workers w
        """,
        {
            "cluster_id": context.cluster_id,
            "scale_groups": _json_mapping(context.scale_group_to_backend),
            "one_backend": one_backend,
        },
    )


def _rename_source_tables(connection: sqlite3.Connection) -> list[str]:
    names = [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    ]
    for name in names:
        connection.execute(f"ALTER TABLE {_quote_identifier(name)} RENAME TO {_quote_identifier('__v1_' + name)}")
    return names


def _copy_source_rows(
    connection: sqlite3.Connection,
    *,
    context: MigrationContext,
    source_fingerprint: str,
) -> None:
    one_backend = next(iter(context.backend_kinds)) if len(context.backend_kinds) == 1 else ""
    params: dict[str, object] = {
        "cluster_id": context.cluster_id,
        "one_backend": one_backend,
        "backend_kinds": _json_mapping(context.backend_kinds),
        "backend_namespaces": _json_mapping(context.backend_namespaces),
        "scale_groups": _json_mapping(context.scale_group_to_backend),
    }
    connection.execute(
        """
        INSERT INTO jobs (
            job_uid, authority_cluster_id, job_id, execution_cluster_id, backend_id,
            placement_state, owner_id, submitting_principal, parent_job_uid, root_job_uid,
            depth, state, submitted_at_ms, root_submitted_at_ms, started_at_ms,
            finished_at_ms, scheduling_deadline_at_ms, error_message, exit_code,
            num_tasks, name
        )
        SELECT m.job_uid, m.authority_cluster_id, j.job_id, m.execution_cluster_id,
               m.backend_id, m.placement_state, j.user_id, j.submitting_user,
               parent.job_uid, root.job_uid, j.depth, j.state, j.submitted_at_ms,
               j.root_submitted_at_ms, j.started_at_ms, j.finished_at_ms,
               j.scheduling_deadline_epoch_ms, COALESCE(j.error, ''), j.exit_code,
               j.num_tasks, j.name
        FROM __v1_jobs j
        JOIN resource_job_map m ON m.job_id = j.job_id
        LEFT JOIN resource_job_map parent ON parent.job_id = j.parent_job_id
        JOIN resource_job_map root ON root.job_id = j.root_job_id
        ORDER BY j.depth, j.job_id
        """
    )
    connection.execute(
        """
        INSERT INTO job_specs (
            job_uid, spec_version, resources_json, entrypoint_json, environment_json,
            constraints_json, coscheduling_json, bundle_id, ports_json,
            scheduling_timeout_ms, max_task_failures, max_retries_failure,
            max_retries_preemption, replicas, timeout_ms, fail_if_exists,
            preemption_policy, existing_job_policy, priority_band, task_image,
            submit_argv_json, client_revision_date, container_profile
        )
        SELECT m.job_uid, 1,
               json_object(
                   'cpu_millicores', c.res_cpu_millicores,
                   'memory_bytes', c.res_memory_bytes,
                   'disk_bytes', c.res_disk_bytes,
                   'device', json(COALESCE(c.res_device_json, '{}'))
               ),
               c.entrypoint_json, c.environment_json, COALESCE(c.constraints_json, '[]'),
               json_object('enabled', json(CASE c.has_coscheduling WHEN 0 THEN 'false' ELSE 'true' END),
                           'group_by', c.coscheduling_group_by),
               c.bundle_id, c.ports_json, c.scheduling_timeout_ms, c.max_task_failures,
               c.max_retries_failure, c.max_retries_preemption, max(j.num_tasks, 1), c.timeout_ms,
               c.fail_if_exists, c.preemption_policy, c.existing_job_policy,
               c.priority_band, c.task_image, c.submit_argv_json, '', c.container_profile
        FROM __v1_job_config c
        JOIN __v1_jobs j ON j.job_id = c.job_id
        JOIN resource_job_map m ON m.job_id = c.job_id
        """
    )
    connection.execute(
        """
        INSERT INTO job_workdir_files (job_uid, filename, data)
        SELECT m.job_uid, f.filename, f.data
        FROM __v1_job_workdir_files f JOIN resource_job_map m ON m.job_id = f.job_id
        """
    )
    connection.execute(
        """
        INSERT INTO tasks (
            task_uid, authority_cluster_id, task_id, job_uid, task_index,
            execution_cluster_id, backend_id, placement_state, state, submitted_at_ms,
            started_at_ms, finished_at_ms, error_message, status_message, exit_code,
            max_retries_failure, max_retries_preemption, current_attempt_uid,
            current_node_uid, priority_band, priority_neg_depth,
            priority_root_submitted_ms, priority_insertion
        )
        SELECT tm.task_uid, jm.authority_cluster_id, t.task_id, tm.job_uid, t.task_index,
               jm.execution_cluster_id, tm.backend_id, jm.placement_state, t.state,
               t.submitted_at_ms, t.started_at_ms, t.finished_at_ms, COALESCE(t.error, ''),
               COALESCE(t.status_message, ''), t.exit_code, t.max_retries_failure,
               t.max_retries_preemption, current.attempt_uid, nm.node_uid,
               t.priority_band, t.priority_neg_depth, t.priority_root_submitted_ms,
               t.priority_insertion
        FROM __v1_tasks t
        JOIN resource_task_map tm ON tm.task_id = t.task_id
        JOIN resource_job_map jm ON jm.job_uid = tm.job_uid
        LEFT JOIN __v1_task_attempts current
            ON current.task_id = t.task_id AND current.attempt_id = t.current_attempt_id
        LEFT JOIN resource_node_map nm ON nm.worker_id = current.worker_id
        """
    )
    connection.execute(
        """
        INSERT INTO attempts (
            attempt_uid, task_uid, attempt_number, execution_cluster_id, backend_id,
            node_uid, state, created_at_ms, started_at_ms, finished_at_ms,
            exit_code, error_message, terminal_reason
        )
        SELECT a.attempt_uid, tm.task_uid, a.attempt_id, jm.execution_cluster_id,
               COALESCE(NULLIF(a.backend_id, ''), tm.backend_id), nm.node_uid,
               a.state, a.created_at_ms, a.started_at_ms, a.finished_at_ms,
               a.exit_code, COALESCE(a.error, ''), a.terminal_reason
        FROM __v1_task_attempts a
        JOIN resource_task_map tm ON tm.task_id = a.task_id
        JOIN resource_job_map jm ON jm.job_uid = tm.job_uid
        LEFT JOIN resource_node_map nm ON nm.worker_id = a.worker_id
        """
    )
    connection.execute(
        """
        INSERT INTO attempt_runtime_objects (
            attempt_uid, provider_kind, namespace, name, provider_uid,
            provider_node_id, provider_node_uid, container_id, observed_at_ms
        )
        SELECT a.attempt_uid, 'kubernetes',
               (SELECT value FROM json_each(:backend_namespaces)
                WHERE key = COALESCE(NULLIF(a.backend_id, ''), tm.backend_id)),
               a.pod_name, a.pod_uid, a.node_name, '', '',
               COALESCE(a.finished_at_ms, a.started_at_ms, a.created_at_ms)
        FROM __v1_task_attempts a
        JOIN resource_task_map tm ON tm.task_id = a.task_id
        WHERE a.pod_name <> '' OR a.pod_uid <> '' OR a.node_name <> ''
        """,
        params,
    )
    connection.execute(
        """
        INSERT INTO endpoints (
            endpoint_id, authority_cluster_id, execution_cluster_id, name, address,
            owner_job_id, owner_task_id, owner_job_uid, owner_task_uid, peer_id,
            metadata_json, access, registered_at_ms, lease_deadline_at_ms
        )
        SELECT e.endpoint_id, jm.authority_cluster_id, jm.execution_cluster_id,
               e.name, e.address, e.job_id, e.task_id, jm.job_uid, tm.task_uid,
               e.peer_id, e.metadata_json, CASE COALESCE(e.access, 0) WHEN 2 THEN 1 ELSE 0 END,
               e.registered_at_ms, e.lease_deadline_ms
        FROM __v1_endpoints e
        JOIN resource_job_map jm ON jm.job_id = e.job_id
        LEFT JOIN resource_task_map tm ON tm.task_id = e.task_id
        """
    )
    connection.execute(
        """
        INSERT INTO rpc_nodes (
            node_uid, node_id, execution_cluster_id, backend_id, scaling_group_id,
            registered_at_ms, last_seen_at_ms, retired_at_ms
        )
        SELECT nm.node_uid, w.worker_id, :cluster_id, nm.backend_id,
               NULLIF(w.scale_group, ''), 0, 0, NULL
        FROM __v1_workers w JOIN resource_node_map nm ON nm.worker_id = w.worker_id
        """,
        params,
    )
    connection.execute(
        """
        INSERT INTO rpc_node_details (
            node_uid, address, hostname, ip_address, provider_instance_id,
            provider_zone, provenance_json
        )
        SELECT nm.node_uid, w.address, w.md_hostname, w.md_ip_address,
               w.md_gce_instance_name, w.md_gce_zone, w.md_provenance_json
        FROM __v1_workers w JOIN resource_node_map nm ON nm.worker_id = w.worker_id
        """
    )
    connection.execute(
        """
        INSERT INTO node_capacity (
            node_uid, cpu_millicores, memory_bytes, disk_bytes, accelerator_kind,
            accelerator_variant, accelerator_count
        )
        SELECT nm.node_uid, w.total_cpu_millicores, w.total_memory_bytes,
               w.md_disk_bytes, w.device_type, w.device_variant,
               CASE WHEN w.total_gpu_count > 0 THEN w.total_gpu_count ELSE w.total_tpu_count END
        FROM __v1_workers w JOIN resource_node_map nm ON nm.worker_id = w.worker_id
        """
    )
    connection.execute(
        """
        INSERT INTO node_attributes (node_uid, key, value_type, str_value, int_value, float_value)
        SELECT nm.node_uid, a.key, a.value_type, a.str_value, a.int_value, a.float_value
        FROM __v1_worker_attributes a JOIN resource_node_map nm ON nm.worker_id = a.worker_id
        """
    )
    connection.execute(
        """
        INSERT INTO scaling_groups (
            execution_cluster_id, backend_id, scaling_group_id, consecutive_failures,
            backoff_until_ms, last_scale_up_at_ms, last_scale_down_at_ms,
            quota_exceeded_until_ms, quota_reason, updated_at_ms
        )
        SELECT :cluster_id, (SELECT value FROM json_each(:scale_groups) WHERE key = g.name), g.name,
               g.consecutive_failures, g.backoff_until_ms, g.last_scale_up_ms,
               g.last_scale_down_ms, g.quota_exceeded_until_ms, g.quota_reason,
               g.updated_at_ms
        FROM __v1_scaling_groups g
        """,
        params,
    )
    connection.execute(
        """
        INSERT INTO slices (
            slice_uid, slice_id, execution_cluster_id, backend_id, scaling_group_id,
            management_mode, lifecycle, membership_state, created_at_ms,
            observed_at_ms, error_message
        )
        SELECT resource_uid('slice', :cluster_id,
                   (SELECT value FROM json_each(:scale_groups) WHERE key = s.scale_group), s.slice_id),
               s.slice_id, :cluster_id,
               (SELECT value FROM json_each(:scale_groups) WHERE key = s.scale_group), s.scale_group,
               'autoscaled',
               CASE s.lifecycle
                   WHEN 'ready' THEN 'ready'
                   WHEN 'failed' THEN 'failed'
                   ELSE 'creating'
               END,
               'observed', s.created_at_ms,
               s.created_at_ms, s.error_message
        FROM __v1_slices s
        """,
        params,
    )
    connection.execute(
        """
        INSERT INTO slice_members (slice_uid, provider_node_id, observed_at_ms)
        SELECT resource_uid('slice', :cluster_id,
                   (SELECT value FROM json_each(:scale_groups) WHERE key = s.scale_group), s.slice_id),
               member.value, s.created_at_ms
        FROM __v1_slices s, json_each(s.worker_ids) member
        """,
        params,
    )
    connection.execute(
        """
        INSERT INTO federated_jobs (
            job_uid, direction, peer_id, owner_principal, handoff_state,
            cancel_intent_version, handoff_nonce
        )
        SELECT jm.job_uid, CASE f.direction WHEN 0 THEN 'sent' ELSE 'received' END,
               f.peer_id, f.owner_principal,
               CASE f.handoff_state WHEN 0 THEN 'pending' WHEN 1 THEN 'handed_off'
                   WHEN 2 THEN 'rejected' WHEN 3 THEN 'queued' ELSE NULL END,
               f.cancel_intent_version, f.handoff_nonce
        FROM __v1_federated_jobs f JOIN resource_job_map jm ON jm.job_id = f.job_id
        """
    )
    connection.execute(
        """
        INSERT INTO federation_sync_state (peer_id, cursor)
        SELECT peer_id, cursor FROM __v1_federation_sync_state
        """
    )
    connection.execute(
        """
        INSERT INTO federated_tasks (task_uid, peer_node_label)
        SELECT tm.task_uid, f.peer_worker_label
        FROM __v1_federated_tasks f JOIN resource_task_map tm ON tm.task_id = f.task_id
        """
    )
    connection.execute(
        """
        INSERT INTO federation_changelog (
            seq, authority_cluster_id, job_id, job_uid, task_uid,
            requester_id, tombstone, written_at_ms
        )
        SELECT c.seq, COALESCE(jm.authority_cluster_id, :cluster_id), c.job_id,
               COALESCE(jm.job_uid, resource_uid('job', :cluster_id, c.job_id)),
               CASE WHEN c.task_index IS NULL THEN NULL ELSE COALESCE(
                   tm.task_uid,
                   resource_uid(
                       'task',
                       COALESCE(jm.job_uid, resource_uid('job', :cluster_id, c.job_id)),
                       c.task_index
                   )
               ) END,
               c.requester_id, c.tombstone, c.written_ms
        FROM __v1_federation_changelog c
        LEFT JOIN resource_job_map jm ON jm.job_id = c.job_id
        LEFT JOIN __v1_tasks t ON t.job_id = c.job_id AND t.task_index = c.task_index
        LEFT JOIN resource_task_map tm ON tm.task_id = t.task_id
        """,
        params,
    )
    connection.execute(
        """
        INSERT INTO user_budgets (owner_id, budget_limit, max_band, updated_at_ms)
        SELECT user_id, budget_limit, max_band, updated_at_ms FROM __v1_user_budgets
        """
    )
    connection.execute("INSERT INTO meta (key, value) SELECT key, CAST(value AS TEXT) FROM __v1_meta")
    connection.execute(
        "INSERT INTO schema_migrations (name, source_fingerprint, applied_at_ms) VALUES (?, ?, ?)",
        (RESOURCE_SCHEMA_NAME, source_fingerprint, Timestamp.now().epoch_ms()),
    )


def _validate_copy(connection: sqlite3.Connection) -> None:
    count_pairs = {
        "jobs": "__v1_jobs",
        "job_specs": "__v1_job_config",
        "job_workdir_files": "__v1_job_workdir_files",
        "tasks": "__v1_tasks",
        "attempts": "__v1_task_attempts",
        "endpoints": "__v1_endpoints",
        "rpc_nodes": "__v1_workers",
        "node_attributes": "__v1_worker_attributes",
        "scaling_groups": "__v1_scaling_groups",
        "slices": "__v1_slices",
        "federated_jobs": "__v1_federated_jobs",
        "federated_tasks": "__v1_federated_tasks",
        "federation_changelog": "__v1_federation_changelog",
        "federation_sync_state": "__v1_federation_sync_state",
        "user_budgets": "__v1_user_budgets",
    }
    for target, source in count_pairs.items():
        target_count = connection.execute(f"SELECT count(*) FROM {_quote_identifier(target)}").fetchone()[0]
        source_count = connection.execute(f"SELECT count(*) FROM {_quote_identifier(source)}").fetchone()[0]
        if target_count != source_count:
            raise ResourceSchemaMigrationError(
                f"post-copy count mismatch for {target}: source={source_count}, target={target_count}"
            )
    foreign_key_problem = connection.execute("PRAGMA foreign_key_check").fetchone()
    if foreign_key_problem is not None:
        raise ResourceSchemaMigrationError(f"post-copy foreign key failure: {tuple(foreign_key_problem)}")


def _drop_source_tables(connection: sqlite3.Connection, source_names: Sequence[str]) -> None:
    preferred_order = (
        "federated_tasks",
        "federated_jobs",
        "job_workdir_files",
        "task_attempts",
        "tasks",
        "job_config",
        "endpoints",
        "worker_attributes",
        "workers",
        "slices",
        "scaling_groups",
        "federation_changelog",
        "federation_sync_state",
        "user_budgets",
        "jobs",
        "meta",
        "schema_migrations",
    )
    ordered = [name for name in preferred_order if name in source_names]
    ordered.extend(name for name in source_names if name not in ordered)
    for name in ordered:
        connection.execute(f"DROP TABLE {_quote_identifier('__v1_' + name)}")


def _upgrade_source(
    connection: sqlite3.Connection,
    *,
    context: MigrationContext,
    source_fingerprint: str,
) -> None:
    connection.create_function("resource_uid", -1, _resource_uid, deterministic=True)
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute("PRAGMA defer_foreign_keys = ON")
        _prepare_mapping_tables(connection, context)
        source_names = _rename_source_tables(connection)
        _create_final_schema(connection)
        _copy_source_rows(connection, context=context, source_fingerprint=source_fingerprint)
        _validate_copy(connection)
        _drop_source_tables(connection, source_names)
        if schema_fingerprint(connection) != final_schema_fingerprint():
            raise ResourceSchemaMigrationError("post-copy final schema fingerprint mismatch")
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def _initialize_fresh(connection: sqlite3.Connection, source_fingerprint: str) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        _create_final_schema(connection)
        connection.execute(
            "INSERT INTO schema_migrations (name, source_fingerprint, applied_at_ms) VALUES (?, ?, ?)",
            (RESOURCE_SCHEMA_NAME, source_fingerprint, Timestamp.now().epoch_ms()),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def initialize_or_upgrade_database(database_path: Path, *, context: MigrationContext) -> None:
    """Create resource schema v2 or atomically upgrade its one accepted source."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = _open_database(database_path)
    try:
        schema = inspect_schema(connection)
        if schema.epoch == RESOURCE_SCHEMA_EPOCH and schema.accepted:
            _raise_for_report(MigrationReport(schema=schema, problems=tuple(_context_problems(context))))
            return
        if not _has_user_schema(connection):
            _raise_for_report(MigrationReport(schema=schema, problems=tuple(_context_problems(context))))
            _initialize_fresh(connection, schema.schema_fingerprint)
            return

        report = MigrationReport(schema=schema, problems=())
        if schema.accepted:
            report = MigrationReport(schema=schema, problems=_source_preflight(connection, context))
        _raise_for_report(report)
        _upgrade_source(connection, context=context, source_fingerprint=schema.schema_fingerprint)
    finally:
        connection.close()
