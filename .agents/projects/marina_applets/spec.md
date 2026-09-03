# Spec: Marina applets

Contracts for [design.md](design.md). Paths are under `infra/marina/`.

## Registry

`src/marina/registry.py`

```python
class Registry(Protocol):
    def apps(self) -> list[AppManifest]:
        """Every app this registry serves, ordered by name. Cheap; may be cached."""

    def app(self, name: str) -> AppManifest | None:
        """The current manifest for ``name``, or None."""

    def read(self, app: AppManifest, path: str) -> bytes | None:
        """The bytes of ``path`` (relative, already cleaned) in the app's bundle, or None."""


@dataclass(frozen=True)
class DirectoryRegistry:
    """Checked-in apps under ``apps_dir``; the current ``discover_apps`` behind the protocol."""
    apps_dir: Path


@dataclass
class AppletRegistry:
    """Applets published to the ``applets`` tables, read through ``engine``.

    ``apps()`` and ``app()`` re-read the ``applets`` table (one indexed SELECT, in the
    request path) when the last read is older than ``refresh`` seconds; between reads
    they answer from the last list, unknown names included. If the read fails the last
    good list stands and the failure is logged. ``read()`` caches file bytes per
    ``(name, version, path)`` in an LRU bounded by ``cache_bytes``.
    """
    engine: Engine
    refresh: float = 5.0
    cache_bytes: int = 256 * 1024 * 1024


@dataclass(frozen=True)
class ComposedRegistry:
    """Directory apps by name, then applets by name; an applet whose name a directory app
    owns is never listed. ``landing_page``, ``/healthz``, and ``/api/marina/apps`` all use
    this order."""
    registries: tuple[Registry, ...]
```

`AppManifest` becomes `(name, title, description, connect_src, build_command, source,
version, package)`: `source` is the directory path for a checked-in app or
`applet:<name>@<version>`; `version` is 0 for directory apps; `package` is the directory
holding `__init__.py` for a Python app and `None` otherwise (`is_python_app`, `_module`,
`cli.build`, and `cli.check` use it instead of `root`). `AppManifest.dist` is removed;
every file read goes through `Registry.read`. `load_manifest(app_dir)` splits into
`parse_manifest(name, text)` (used by publish) and the directory wrapper.

`MarinaConfig` gains nothing. `create_app` builds
`ComposedRegistry((DirectoryRegistry(apps_dir), AppletRegistry(engine)))` when a database
is configured, else the directory registry alone. Route order: the kernel's `/healthz`,
`/`, `/api/marina/*`, then each Python app's `/<name>/api` mount (still installed at
startup), then `/{app}/data/{path}`, then `/{app}/{path}` and `/{app}`. An unknown app is
a 404 JSON body `{"error": "no app"}`.

## File serving

`serve_app_file(registry, app, path)` cleans `path` with `clean_relative_path`, then tries
`read(app, path)`, `read(app, path + ".gz")` (served with `Content-Encoding: gzip`), and
finally `read(app, "index.html")`. Responses carry the app's CSP, an `ETag` of
`"<name>-<version>-<path>"`, and `Cache-Control: private, max-age=31536000, immutable`
for hashed assets under `static/` or `Cache-Control: no-cache` otherwise. Directory apps
keep `FileResponse` semantics through a `DirectoryRegistry.read` that reads the file; the
gzip and index fallbacks are the same code for both registries.

## Persisted shapes

Kernel migration `src/marina/migrations/m0001_applets.py`, run by `marina migrate` before
any app's own migration, in schema `marina`.

```sql
CREATE TABLE applets (
    name            TEXT PRIMARY KEY CHECK (name ~ '^[a-z][a-z0-9-]{1,39}$'),
    current_version INTEGER NOT NULL,
    title           TEXT NOT NULL,
    description     TEXT NOT NULL,
    connect_src     TEXT[] NOT NULL DEFAULT '{}',
    published_by    TEXT NOT NULL,
    published_at    TIMESTAMPTZ NOT NULL
);

CREATE TABLE applet_versions (
    name         TEXT NOT NULL REFERENCES applets(name) ON DELETE CASCADE,
    version      INTEGER NOT NULL,
    manifest     JSONB NOT NULL,          -- the parsed app.toml
    published_by TEXT NOT NULL,
    published_at TIMESTAMPTZ NOT NULL,
    byte_size    BIGINT NOT NULL,
    PRIMARY KEY (name, version)
);

CREATE TABLE applet_files (
    name    TEXT NOT NULL,
    version INTEGER NOT NULL,
    path    TEXT NOT NULL,                -- relative to dist/, forward slashes
    bytes   BYTEA NOT NULL,
    PRIMARY KEY (name, version, path),
    FOREIGN KEY (name, version) REFERENCES applet_versions(name, version) ON DELETE CASCADE
);

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'marina_reader') THEN
        CREATE ROLE marina_reader NOLOGIN;
    END IF;
END $$;
```

The `name` check matches `manifest.APP_NAME_PATTERN`, which is tightened to the same
expression. Limits: 25 MB per version (sum of `bytes`), 2,000 files per version, the last
5 versions kept per applet (older versions deleted on publish).

Roles. The migrating identity is the service account, which must hold `CREATEROLE`
(granted once in Cloud SQL by an operator; the `marin` stack notes it) and owns every app
schema, so `ALTER DEFAULT PRIVILEGES` issued by it covers future tables. After each app's
migration, `marina migrate` runs for that schema:

```sql
GRANT USAGE ON SCHEMA "<app>" TO marina_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA "<app>" TO marina_reader;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA "<app>" TO marina_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA "<app>" GRANT SELECT ON TABLES TO marina_reader;
```

The kernel's own `marina` schema and `public` are not granted, and
`REVOKE CREATE ON SCHEMA public FROM PUBLIC` runs once. First publish of an applet
creates role `applet_<name>` (`NOLOGIN`, `IN ROLE marina_reader`), grants it to the
service account so `SET ROLE` is allowed, and creates schema `applet_<name>` (hyphens to
underscores) with `AUTHORIZATION applet_<name>`. Delete drops the schema and the role.

## HTTP surface

All under the existing kernel authentication; the caller is the verified identity.

| Method and path | Body | Result |
| --- | --- | --- |
| `GET /api/marina/apps` | | unchanged shape, now includes applets with `"kind": "applet"`, `"published_by"`, `"version"` |
| `POST /api/marina/applets/{name}` | `application/x-tar` (gzip allowed) with `app.toml` and `dist/` at the top | `201 {"name", "version", "path"}`; `400` for a bad manifest, missing `dist/index.html`, over-limit bundle, or a name a directory app owns; `409` when `name` is being published concurrently |
| `GET /api/marina/applets/{name}` | | `{"name", "current_version", "versions": [{"version", "published_by", "published_at", "byte_size"}]}` |
| `PUT /api/marina/applets/{name}/current` | `{"version": int}` | `200`; `404` for an unknown version |
| `DELETE /api/marina/applets/{name}` | | `204`; drops rows and the applet schema |
| `POST /{app}/query` | `{"sql": str, "params": {..}}` | `200 {"columns": [..], "rows": [[..]], "truncated": bool}`; `400` for more than one statement, a parse error, or a top-level `SET`, `RESET`, `BEGIN`, `COMMIT`, `ROLLBACK`; `404` for an unknown app; `422` with `sqlstate` and message for a permission or statement error; `504` on `sqlstate 57014` (timeout) |

Tarball contract: entries `app.toml` and `dist/**`, with or without a leading `./`, or
all under one uniform top-level directory that is stripped. Regular files only; symlinks,
hard links, absolute paths, `..`, and devices are rejected with `400`. Caps apply to
uncompressed bytes (25 MB total, 8 MB per file). `build_command` in the manifest is
ignored. Publish takes `pg_advisory_xact_lock(hashtext(name))`; a concurrent publisher
gets `409`. The new version is `max(version) + 1`.

Query execution runs on a dedicated engine (pool of 2) as one transaction:
`BEGIN; SET LOCAL ROLE applet_<app>; SET LOCAL search_path TO applet_<app>, public;
SET LOCAL statement_timeout = 10000; <statement>; COMMIT`, with `marina_reader` and
`search_path = <app>, public` for a directory app. `SET LOCAL` ends with the transaction
whichever way it ends, so no connection returns to the pool carrying a role. Parameters
bind through SQLAlchemy `text()` over pg8000's extended protocol, which is what refuses a
second statement in the string; the kernel additionally rejects the listed top-level
commands by their first keyword. A `DO` block is one statement and is bounded by the
timeout only. Rows are capped at 10,000 with `truncated: true`. Every call is logged with
the caller, app, `sqlstate` if any, and duration.

## CLI

`src/marina/cli.py`

```
marina publish DIR [--name NAME] [--url URL] [--build/--no-build] [--dry-run] [--json]
    Build DIR (its build_command) unless dist/ exists or --no-build, tar app.toml and
    dist/, POST to URL (default MARINA_URL or https://marina.oa.dev) with the desktop
    OAuth token (Echo's bearer_token() moves into marina.client), print the applet URL
    (URL + path). --name defaults to DIR's basename. --dry-run validates and prints the
    tarball listing without posting; --json prints the server's response body. Exit 1
    with the server's error text on 4xx.
marina applets list [--url URL]
    name, current version, publisher, published at.
marina applets rollback NAME VERSION [--url URL]
marina applets delete NAME [--url URL]
```

`marina check`, `marina build`, `marina dev`, and `marina journey` accept an applet's
source directory in `apps/` unchanged.

## Shell

`web/Shell.vue` shows applets in the switcher after the checked-in apps, with the
publisher's email as the entry's title attribute. No other change.

## Errors

- `AppletTooLarge(name, byte_size)`: bundle over 25 MB or 2,000 files; `400`.
- `AppletNameReserved(name)`: a directory app owns the name; `400`.
- `AppletNotFound(name)`: `404` on rollback, delete, and info.
- `QueryRefused(reason)`: multiple statements or a refused top-level command; `400`.
- `AppletBundleInvalid(reason)`: a bad tarball entry or manifest; `400`.
- Postgres permission and statement errors pass through as `422` with `sqlstate` and message.

## Out of scope

- Running uploaded Python in the kernel or in a per-applet service.
- Vanity hosts for applets; they live at `https://marina.oa.dev/<name>/` only.
- Per-user or per-app authorization; every signed-in person can publish, roll back,
  delete, and query, as with every other Marina action.
- Data-root uploads for applets (use `gsutil` as for checked-in apps) and a bucket
  store for bundles.
- Build-on-server; the publisher builds.

## Files

| Path | Purpose |
| --- | --- |
| `src/marina/registry.py` | `Registry`, `DirectoryRegistry`, `AppletRegistry`, `ComposedRegistry` |
| `src/marina/applets.py` | bundle validation, publish/rollback/delete, the applets routes |
| `src/marina/query.py` | the query endpoint and role handling |
| `src/marina/migrations/m0001_applets.py` | tables and the role |
| `src/marina/cli.py` | `publish`, `applets` |
| `tests/test_registry.py`, `tests/test_applets.py`, `tests/test_query.py` | unit tests on a throwaway Postgres |
| `tests/journeys/test_applets.py` | the publish-then-visit journey; `journey_plugin` gains `--journey-app NAME` for specs outside `apps/<name>/journeys/`, and `Journey.app` is settable so the spec names the applet it just published; needs `MARINA_DATABASE_URL` |
