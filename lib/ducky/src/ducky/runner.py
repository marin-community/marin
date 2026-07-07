# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The DuckDB query runner over a shared embedded instance.

The runner runs the user's SQL exactly once via ``COPY (<sql>) TO '<parquet>'`` and
then reads the written parquet back for the row count and a capped preview. Running
once keeps non-deterministic SQL (``random()``, ``now()``, unordered ``LIMIT``)
consistent between the spilled file and the preview.

``run_query`` is safe to call concurrently: each call uses its own cursor on the
shared DuckDB instance (which provides the loaded extensions, secrets, and the
host-sized thread/memory budget). A per-query watchdog interrupts a query that
exceeds ``query_timeout``.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import re
import shutil
import threading
import time
from collections.abc import Callable

import duckdb
from iris.env_resources import TaskResources
from rigging.filesystem import (
    MARIN_CROSS_REGION_OVERRIDE_ENV,
    StoragePath,
    cached_marin_region,
    get_bucket_location,
    is_cross_region_url,
)

from ducky.catalog import DATAKIT_SCHEMA, FINELOG_SCHEMA, View, build_catalog
from ducky.config import DuckyConfig

logger = logging.getLogger(__name__)

# query_id is interpolated into the result path, so it must be a bare uuid4 hex.
_QUERY_ID_RE = re.compile(r"^[0-9a-f]{32}$")

# Object-store URIs referenced in the SQL (read_parquet('gs://…'), etc.). Stops at the
# closing quote/paren/whitespace of the SQL literal.
_OBJECT_URI_RE = re.compile(r"""(?:gs|s3|r2)://[^\s'")]+""", re.IGNORECASE)

# The opt-in directive for reading cross-region buckets: a leading SQL line comment such as
# `-- cross-region: allow`. Accepts `cross-region`/`cross region` and `allow`/`allowed`.
_CROSS_REGION_DIRECTIVE_RE = re.compile(r"^\s*--\s*cross[- ]region\s*:?\s*allow(ed)?\b", re.IGNORECASE)


def _opts_in_cross_region(sql: str) -> bool:
    """True if the SQL's leading comment block opts in to cross-region reads.

    Only the header is scanned — the run of blank/``--`` comment lines before the first
    statement line — so the directive can't hide inside a string literal further down.
    """
    for line in sql.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("--"):
            return False  # first real statement line ends the header
        if _CROSS_REGION_DIRECTIVE_RE.match(line):
            return True
    return False


def _object_uris(sql: str) -> list[str]:
    """Distinct gs://, s3://, r2:// URIs referenced literally in the SQL (order-preserving)."""
    return list(dict.fromkeys(_OBJECT_URI_RE.findall(sql)))


def _is_gcs_uri(uri: str) -> bool:
    """True for a ``gs://``/``gcs://`` URI. Only GCS is subject to the cross-region guard —
    the S3 backends (R2, CoreWeave) are region-agnostic to us and always allowed."""
    return uri.lower().startswith(("gs://", "gcs://"))


# Buckets whose GCS location we've successfully resolved once — cached so a same-region read
# doesn't re-probe metadata every query. Only *successes* are cached: a failed lookup retries
# so a transient blip can't wedge a bucket into fail-closed for the whole process lifetime.
_region_confirmed_buckets: set[str] = set()


def _region_resolvable(bucket: str) -> bool:
    """Whether ``bucket``'s GCS location metadata is readable. Successes are cached; failures
    retry (not cached) so a transient error or permission blip self-heals."""
    if bucket in _region_confirmed_buckets:
        return True
    try:
        get_bucket_location(bucket)
    except Exception:
        return False
    _region_confirmed_buckets.add(bucket)
    return True


def needs_cross_region_optin(uri: str) -> bool:
    """Whether reading GCS ``uri`` requires the ``-- cross-region: allow`` opt-in.

    Fails closed on uncertainty. A GCS URI needs the opt-in if it's confirmed cross-region by
    :func:`rigging.filesystem.is_cross_region_url`, or — on a GCP VM — if we cannot resolve the
    bucket's region at all, so a missing ``storage.buckets.get`` permission or a metadata lookup
    failure can't silently bypass the gate (``is_cross_region_url`` returns ``False`` on such
    failures). Off-GCP (no VM region) region gating doesn't apply and the fee override disables
    it, so both return ``False``.
    """
    if is_cross_region_url(uri):
        return True
    if cached_marin_region() is None or os.environ.get(MARIN_CROSS_REGION_OVERRIDE_ENV):
        return False
    bucket = StoragePath.parse(uri).netloc
    if not bucket or _region_resolvable(bucket):
        return False
    logger.warning("cross-region guard: could not resolve region for gs://%s; requiring opt-in", bucket)
    return True


def _is_allowed(uri: str, allowed: tuple[str, ...]) -> bool:
    """True if ``uri`` starts with any allowlist entry.

    Entries are URI prefixes: ``gs://marin-`` allows every ``gs://marin-*`` bucket,
    ``r2://`` allows all of R2, and a trailing slash (``gs://marin-us-east5/``) bounds
    a match to one bucket's contents.
    """
    return any(uri.startswith(entry) for entry in allowed)


def disallowed_uris(sql: str, allowed: tuple[str, ...]) -> list[str]:
    """Object-store URIs in ``sql`` not covered by ``allowed``. Empty allowlist allows all."""
    if not allowed:
        return []
    return [uri for uri in _object_uris(sql) if not _is_allowed(uri, allowed)]


def check_query_access(
    sql: str,
    allowed: tuple[str, ...],
    exempt_prefixes: tuple[str, ...],
    needs_opt_in: Callable[[str], bool],
) -> None:
    """Raise if ``sql`` references object-store URIs it may not read.

    ``allowed`` is the outer bound — the object-store prefixes ducky may read at all (marin
    buckets on either backend). A URI outside it is hard-refused. Among the allowed URIs, a
    *GCS* URI for which ``needs_opt_in`` returns True is egress-costly (cross-region, or a
    region we couldn't confirm), so it's read only when the query opts in with a leading
    ``-- cross-region: allow`` comment. S3 URIs (R2/CoreWeave) are never cross-region-gated.
    ``exempt_prefixes`` (the configured catalog roots) are deliberate always-on egress and skip
    the opt-in, so a literal read of a root behaves like its pre-baked view. An empty
    ``allowed`` disables all enforcement (allow-all).

    ``needs_opt_in`` is injected (:func:`needs_cross_region_optin` in prod) so the region
    decision uses live GCS bucket-location metadata rather than a naming convention, and fails
    closed when the region can't be resolved.

    :raises BucketNotAllowedError: a URI is outside ``allowed``.
    :raises CrossRegionNotAllowedError: a cross-region (or region-unconfirmed) GCS URI wasn't opted in.
    """
    if not allowed:
        return
    forbidden = disallowed_uris(sql, allowed)
    if forbidden:
        raise BucketNotAllowedError(
            f"query references buckets outside the allowlist: {', '.join(forbidden)}; "
            f"allowed prefixes: {', '.join(allowed)}"
        )
    # Every referenced URI is allowed at this point; gate cross-region GCS reads.
    cross = [
        uri
        for uri in _object_uris(sql)
        if _is_gcs_uri(uri) and not _is_allowed(uri, exempt_prefixes) and needs_opt_in(uri)
    ]
    if cross and not _opts_in_cross_region(sql):
        raise CrossRegionNotAllowedError(
            f"query reads cross-region buckets: {', '.join(cross)}; "
            f"add a leading '-- cross-region: allow' comment to opt in to the egress"
        )


class DuckyError(Exception):
    """Base for ducky errors surfaced to the dashboard as a clean message."""


class QueryError(DuckyError):
    """DuckDB failed to plan or execute the SQL. Wraps the DuckDB message."""


class BucketNotAllowedError(DuckyError):
    """The query references an object-store URI outside every configured allowlist.

    This is ducky's egress guardrail: GCS HMAC keys are region-agnostic, so the only way to
    bound what a query can read (and avoid unexpected cross-region/cross-cloud egress) is to
    refuse URIs whose bucket isn't allowlisted. Raised before any execution.
    """


class CrossRegionNotAllowedError(DuckyError):
    """The query reads an egress-costly GCS bucket without opting in.

    The referenced bucket lives in a different region than this VM, or (failing closed) its
    region couldn't be confirmed — see :func:`needs_cross_region_optin` — and the query lacks
    the leading ``-- cross-region: allow`` comment. Raised before execution; the user opts in by
    adding the comment.
    """


@dataclasses.dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    preview_rows: list[list]
    total_rows: int
    truncated: bool
    result_path: str
    elapsed_ms: int  # server-side execution wall time (COPY + readback)
    result_bytes: int  # on-disk size of the spilled result parquet


def _sql_literal(value: str) -> str:
    """Quote a string as a SQL literal, escaping embedded single quotes."""
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _coerce_cell(value: object) -> object:
    """Coerce a DuckDB/Arrow cell to a JSON-serializable scalar.

    Native scalars pass through; everything else (timestamp, decimal, interval,
    blob, list, struct) becomes its string form for the preview. Non-finite floats
    (``NaN``/``inf``) also become strings — Starlette's ``JSONResponse`` rejects them.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return str(value)


class QueryRunner:
    """Owns the shared embedded DuckDB instance for the process lifetime.

    ``run_query`` is concurrency-safe — each call runs on its own cursor.
    """

    def __init__(self, config: DuckyConfig, resources: TaskResources | None = None) -> None:
        self._config = config
        self._con = duckdb.connect()
        host = resources or TaskResources.from_environment()
        # threads = host CPU count; memory_limit = memory_fraction of host RAM (headroom for
        # Python/Arrow/OS). 0 memory leaves DuckDB's default (host memory unknown).
        threads = max(1, int(host.cpu_cores))
        memory_limit_bytes = int(host.memory_bytes * config.memory_fraction) if host.memory_bytes > 0 else 0
        self._con.execute(f"SET threads = {threads}")
        if memory_limit_bytes > 0:
            self._con.execute(f"SET memory_limit = '{memory_limit_bytes}B'")
        # Spill to local disk when a query exceeds memory_limit, so big sorts/joins/aggregates
        # go out-of-core instead of OOM-failing; a query that still doesn't fit fails alone.
        # DuckDB clears a query's spill on normal end (success or failure), but a process
        # crash (OOM-kill) orphans temp files — so wipe the dir on startup. Safe: only one
        # server child runs at a time and it hasn't served yet.
        shutil.rmtree(config.spill_directory, ignore_errors=True)
        os.makedirs(config.spill_directory, exist_ok=True)
        self._con.execute(f"SET temp_directory = {_sql_literal(config.spill_directory)}")
        # Bound the spill so a runaway query can't fill the (small, ~100 GB) boot disk and
        # crash the container; over the cap the query fails cleanly (caught per-query).
        self._con.execute(f"SET max_temp_directory_size = {_sql_literal(config.spill_limit)}")
        logger.info("DuckDB configured: threads=%d memory_limit_bytes=%d", threads, memory_limit_bytes)
        self._install_secrets()
        # A local scratch dir (smoke deploy / tests) needs the ducky/ subdir to exist;
        # object stores create the prefix implicitly.
        scratch_is_remote = "://" in config.scratch_bucket
        if not scratch_is_remote:
            os.makedirs(f"{config.scratch_bucket.rstrip('/')}/ducky", exist_ok=True)
        # Harden: when results go to object storage, block user SQL from touching the local
        # filesystem (e.g. read_text('/proc/self/environ') to exfil the injected creds). Object
        # stores are separate DuckDB filesystems and spilling is internal, so both still work.
        # Skipped for a local scratch dir, which needs LocalFileSystem for the result write.
        if scratch_is_remote:
            self._con.execute("SET disabled_filesystems = 'LocalFileSystem'")
        # Lock the configuration last so a query can't SET any of these guards back off.
        self._con.execute("SET lock_configuration = true")
        # Register the pre-baked catalog views on the shared connection (cursors inherit
        # the catalog). Done after secrets/locking so the views can read object storage.
        self._created_view_names: set[str] = set()
        self._create_catalog_views()

    @property
    def created_view_names(self) -> frozenset[str]:
        """Qualified identifiers of the catalog views that were successfully created — the
        subset actually queryable, after skipping absent/credential-less datasets."""
        return frozenset(self._created_view_names)

    def _create_catalog_views(self) -> None:
        """Create the pre-baked catalog views (finelog.*, datakit.*) on the shared connection.

        DuckDB binds a view's schema at CREATE time, so each view reads its dataset's footer
        now — this both validates the view and warms the parquet-metadata cache. A view over
        an absent/unreachable dataset fails to create; that's best-effort, so we log and move
        on rather than fail startup. Views over a root whose backend has no credentials are
        skipped up front (a creds-free smoke deploy would otherwise attempt doomed reads).

        Configured roots are readable regardless of ``allowed_buckets`` — they're part of
        ``effective_allowed_buckets`` — so a pre-baked view and a literal ``read_parquet`` of
        the same prefix behave identically; there's no view/allowlist inconsistency to guard.
        """
        for view in build_catalog(self._config).views:
            root = self._root_for(view)
            if root is None or not self._backend_ready(root):
                continue
            try:
                self._con.execute(f"CREATE SCHEMA IF NOT EXISTS {view.schema}")
                self._con.execute(f"CREATE OR REPLACE VIEW {view.qualified_name} AS {view.definition_sql}")
            except duckdb.Error as e:
                logger.warning("skipping catalog view %s: %s", view.qualified_name, str(e).splitlines()[0])
            else:
                self._created_view_names.add(view.qualified_name)
                logger.info("registered catalog view %s", view.qualified_name)

    def _root_for(self, view: View) -> str | None:
        """The configured object-store root backing a catalog view (by its schema)."""
        if view.schema == FINELOG_SCHEMA:
            return self._config.finelog_root
        if view.schema == DATAKIT_SCHEMA:
            return self._config.datakit_root
        return None

    def _backend_ready(self, root: str) -> bool:
        """Whether the object-store backend for ``root``'s scheme has credentials loaded."""
        if root.startswith("gs://"):
            return self._config.gcs_enabled
        if root.startswith("s3://"):
            return self._config.r2_enabled or self._config.cw_enabled
        return True  # local path (smoke/tests)

    def _install_secrets(self) -> None:
        """Load httpfs and create a DuckDB SECRET for each configured object-store backend."""
        cfg = self._config
        self._con.execute("INSTALL httpfs")
        self._con.execute("LOAD httpfs")
        # Retry transient object-store failures (5xx, throttling, brief DNS/connection
        # blips — more likely on cross-region reads) with exponential backoff, so a
        # network hiccup doesn't fail an expensive query outright. Defaults are 3/100ms;
        # 10 retries at 200ms * 2^n backoff spans ~100s of transient unavailability.
        # SET GLOBAL because these are connection-local settings: run_query uses a per-query
        # cursor, which only inherits the global scope, not this connection's local SETs.
        self._con.execute("SET GLOBAL http_retries = 10")
        self._con.execute("SET GLOBAL http_retry_wait_ms = 200")
        self._con.execute("SET GLOBAL http_retry_backoff = 2")
        # Cache parquet footers (row-group/column metadata) so repeat queries over the same
        # object-store files skip re-reading and re-parsing the footer — a big win for the
        # pre-baked views, which point at stable file sets queried over and over. GLOBAL so the
        # per-query cursors inherit it. enable_external_file_cache is on by default (caches file
        # bytes in memory); enable_http_metadata_cache avoids re-issuing HEADs for file size.
        self._con.execute("SET GLOBAL parquet_metadata_cache = true")
        self._con.execute("SET GLOBAL enable_http_metadata_cache = true")
        if cfg.gcs_enabled:
            self._con.execute(
                f"CREATE OR REPLACE SECRET ducky_gcs "
                f"(TYPE GCS, KEY_ID {_sql_literal(cfg.gcs_hmac_key_id)}, SECRET {_sql_literal(cfg.gcs_hmac_secret)})"
            )
        # R2 and CoreWeave are both s3://, so each S3 secret is SCOPE-d to its bucket prefix;
        # DuckDB routes an s3:// URI to the secret with the longest matching scope. URL_STYLE is
        # a fixed property of each endpoint: R2's account endpoint is path-style; CoreWeave
        # rejects path-style and needs vhost.
        if cfg.r2_enabled:
            self._con.execute(
                self._s3_secret("ducky_r2", cfg.r2_endpoint, "path", cfg.r2_scope, cfg.r2_access_key, cfg.r2_secret_key)
            )
        if cfg.cw_enabled:
            self._con.execute(
                self._s3_secret("ducky_cw", cfg.cw_endpoint, "vhost", cfg.cw_scope, cfg.cw_access_key, cfg.cw_secret_key)
            )

    @staticmethod
    def _s3_secret(name: str, endpoint: str, url_style: str, scope: str, key_id: str, secret: str) -> str:
        # REGION 'auto' keeps DuckDB from signing with a real AWS region for these custom endpoints.
        return (
            f"CREATE OR REPLACE SECRET {name} (TYPE S3, "
            f"ENDPOINT {_sql_literal(endpoint)}, URL_STYLE {_sql_literal(url_style)}, REGION 'auto', "
            f"SCOPE {_sql_literal(scope)}, KEY_ID {_sql_literal(key_id)}, SECRET {_sql_literal(secret)})"
        )

    def run_query(self, sql: str, query_id: str) -> QueryResult:
        """Run ``sql`` once, spill the full result to parquet, and return a capped preview.

        ``query_id`` must be a bare uuid4 hex (the server supplies it); anything else
        raises ``ValueError`` to prevent path injection. DuckDB failures — including
        cross-region/auth errors from httpfs — raise :class:`QueryError`.
        """
        if not _QUERY_ID_RE.match(query_id):
            raise ValueError(f"query_id must be a uuid4 hex, got {query_id!r}")

        check_query_access(
            sql,
            self._config.effective_allowed_buckets,
            self._config.catalog_root_prefixes,
            needs_cross_region_optin,
        )

        result_path = f"{self._config.scratch_bucket.rstrip('/')}/ducky/{query_id}.parquet"
        path_literal = _sql_literal(result_path)
        # hive_partitioning=false: the scratch path embeds a `tmp/ttl=Nd/` segment, which
        # DuckDB would otherwise read back as a phantom `ttl` partition column.
        readback = f"read_parquet({path_literal}, hive_partitioning=false)"

        # A fresh cursor per query so multiple queries run concurrently on the shared
        # DuckDB instance (one connection can't run parallel statements); the cursor
        # inherits the instance's loaded extensions, secrets, and settings.
        cursor = self._con.cursor()
        timed_out = threading.Event()
        watchdog = threading.Timer(self._config.query_timeout, lambda: (timed_out.set(), cursor.interrupt()))
        watchdog.start()
        start = time.monotonic()
        try:
            # Run the user SQL as a relation and write it out, rather than wrapping it in a
            # COPY (...) string — DuckDB parses it as a complete statement, so a trailing
            # ';' or '-- comment' just works.
            cursor.sql(sql).write_parquet(result_path)
            count_row = cursor.execute(f"SELECT count(*) FROM {readback}").fetchone()
            assert count_row is not None  # count(*) always returns exactly one row
            total_rows = int(count_row[0])
            size_row = cursor.execute(
                f"SELECT sum(total_compressed_size) FROM parquet_metadata({path_literal})"
            ).fetchone()
            result_bytes = int(size_row[0]) if size_row and size_row[0] is not None else 0
            preview = cursor.execute(
                f"SELECT * FROM {readback} LIMIT {self._config.preview_row_cap}"
            ).fetch_arrow_table()
        except duckdb.Error as e:
            if timed_out.is_set():
                raise QueryError(f"query exceeded the {self._config.query_timeout}s timeout and was cancelled") from e
            raise QueryError(str(e)) from e
        finally:
            watchdog.cancel()
            cursor.close()
        elapsed_ms = int((time.monotonic() - start) * 1000)

        columns = list(preview.column_names)
        preview_rows = [[_coerce_cell(row[col]) for col in columns] for row in preview.to_pylist()]
        return QueryResult(
            columns=columns,
            preview_rows=preview_rows,
            total_rows=total_rows,
            truncated=total_rows > len(preview_rows),
            result_path=result_path,
            elapsed_ms=elapsed_ms,
            result_bytes=result_bytes,
        )

    def close(self) -> None:
        self._con.close()
