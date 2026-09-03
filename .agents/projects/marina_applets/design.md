# Marina applets: a dynamic app registry

_Why are we doing this? What's the benefit?_

Marina hosts a small web app once its code is in the repository: an `app.toml`,
a built `dist/`, a PR, an image build, a deploy. That is the right bar for
Echo and EvalDash. It is the wrong bar for the table of last week's failed
evals someone wants during an incident, or the little review page an agent
builds mid-session. Those should cost what an artifact costs: build the page,
publish it, share the link. Applets are Marina apps that are published to the
running service instead of committed, stored in the database, and served,
listed, and authenticated exactly like the checked-in ones. An agent gets a
`marina publish` that takes a directory and returns a URL.

## Challenges

_What's hard?_

Static and dynamic apps must feel the same, to people and to the code. Today
the kernel reads a directory once at startup
([`discover_apps`](https://github.com/marin-community/marin/blob/0780825da0fe6b5654717780b9a8349c1e8e6e89/infra/marina/src/marina/manifest.py#L68)), installs a route pair per
app ([`install_app_routes`](https://github.com/marin-community/marin/blob/0780825da0fe6b5654717780b9a8349c1e8e6e89/infra/marina/src/marina/server.py#L219)), and serves files
from a local `dist` ([`serve_app_file`](https://github.com/marin-community/marin/blob/0780825da0fe6b5654717780b9a8349c1e8e6e89/infra/marina/src/marina/server.py#L145)). An
applet published at 14:02 must be visible at 14:02 on every instance without a
restart, so app resolution moves from startup to request time.

Applets that only ship files are easy. Applets that want data are the point,
and the first thing they ask for is the database. Running uploaded Python
inside the kernel process means arbitrary code under the service account, so
data access for applets has to come through a surface the kernel owns.

## Costs / Risks

_What's bad about doing this?_

- A second registry and a request-time lookup make the kernel a little less
  obvious than a directory scan. The two registries share one interface to
  keep that cost small.
- Applets are code nobody reviewed, served on the same origin as reviewed apps.
  CSP per app and same-origin cookies already exist; the query surface is the
  new attack surface, bounded by Postgres roles and a statement timeout.
- The database grows a table of bundle bytes. Bundles are small (a Vue build is
  under a few megabytes) and capped; large data goes to the data root as it
  does for checked-in apps.
- Anyone can overwrite anyone's applet, and through it reach that applet's
  tables. That follows the earlier decision that users can do everything; the
  owner and publish time are recorded and shown.

## Design

_How are we doing this?_

**One app model, two sources.** An app is a manifest plus a bundle of files
plus, for checked-in apps only, a Python package. `AppManifest` gains
`source`, the fsspec URL or database key its files come from, and
`serve_app_file` reads through that source rather than a `Path`. A
`Registry` protocol has two implementations: `DirectoryRegistry` over
`apps/` (what `discover_apps` is now) and `AppletRegistry` over two tables,
`applets` and `applet_files`, in the kernel's own `marina` schema.
`create_app` composes them; checked-in names are reserved, so an applet can
never shadow Echo.

**Resolve at request time.** The per-app route pair becomes one route family,
`/{app}/...`, registered after the kernel's own routes and the Python apps'
API mounts, that asks the composed registry for the app and 404s when there
is none. The directory registry answers from memory; the applet registry
re-reads the small `applets` table at most every few seconds (unknown names
included, so a stray `/favicon.ico` is not a query per request) and keeps
current versions' files in a bounded per-process cache, so a publish is live
everywhere within one window and bytes are fetched once per version per
instance. File serving keeps what checked-in apps have: the `.gz` fallback,
the `index.html` fallback, and now a version-derived ETag. The Shell's
switcher already reads `/api/marina/apps` and needs no change; the directory
just grows.

**Publishing.** `POST /api/marina/applets/{name}` takes a tarball of a
directory shaped like a checked-in app: `app.toml` and `dist/`. The kernel
validates the manifest with the existing loader, writes one row per file, and
points `current_version` at the new version. Old versions stay for rollback
(`PUT .../current`) and are pruned beyond the last five. `marina publish DIR`
runs the app's `build_command` if `dist/` is missing, then posts; it prints
the URL. Because an applet's source directory is a valid `apps/<name>/`, the
same journeys run against it locally before publishing, and promotion to a
checked-in app is `git mv` plus a PR.

**Data for applets.** Two things cover most applets without any server code.
The data route already serves `<data root>/<name>/`, so an applet can publish
a parquet or JSON alongside itself. For live data, `POST /<app>/query` runs one
SQL statement for the app in the path. Each applet gets a Postgres role that
owns its own schema, and a shared `marina_reader` role with SELECT on every
app schema is granted to each of them. The kernel runs the statement in one
transaction with `SET LOCAL ROLE` and `SET LOCAL search_path`, so the role
never outlives the request and Postgres, not the kernel, decides what a
statement may touch: an applet reads Echo's wiki and EvalDash's runs, and
writes only its own tables. A checked-in app can call the same endpoint and
gets the reader role, so for it the surface is read-only. Uploaded Python is
not run in v1 (see Open Questions).

The isolation is against accidents, not adversaries: any signed-in person can
publish under any applet's name, so any signed-in person can reach any
applet's tables by publishing there. That is the same trust every other
Marina action already assumes.

## Testing

Unit tests cover both registries against a throwaway Postgres: publish, list,
serve a file, roll back, prune, and reject a name that a checked-in app owns.
The query surface gets tests that prove a SELECT across schemas works, a write
inside the applet's schema works, a write to another app's schema is refused
by Postgres, and a long statement hits the timeout. One journey publishes a
small applet through the API from the test, then opens it in the browser,
sees it in the switcher, and reads a row it stored through the query endpoint.
The image smoke from the Marina deploy covers the migration that creates the
tables and the role.

## Open Questions

- Server code for applets. The two candidates are a restricted query surface
  only (this proposal) and a generic runner that starts a Cloud Run service per
  applet from the kernel image with the bundle pulled at boot. The second gives
  applets real APIs and background work at the cost of a service per applet
  and a slower publish. Is the query surface enough for the first year?
- Query role scope. SELECT on every app schema is the simplest useful policy
  and matches "users can do everything". Should Echo's work log or EvalDash's
  review tables be excluded by default, with apps opting schemas in?
- Bundle storage. Rows in Postgres keep one dependency and make an applet a
  transaction; the bucket would allow larger bundles. Is a 25 MB cap per
  version acceptable?
