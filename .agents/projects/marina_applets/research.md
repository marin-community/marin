# Research: Marina applets

Findings behind [design.md](design.md). Kernel references are pinned to
`0780825da0fe6b5654717780b9a8349c1e8e6e89` on the `marina-kernel-tasktrove` branch (PR #8867).

## In-repo

- App discovery is a startup directory scan:
  [`discover_apps`](https://github.com/marin-community/marin/blob/0780825da0fe6b5654717780b9a8349c1e8e6e89/infra/marina/src/marina/manifest.py#L68) returns a list of
  [`AppManifest`](https://github.com/marin-community/marin/blob/0780825da0fe6b5654717780b9a8349c1e8e6e89/infra/marina/src/marina/manifest.py#L24) (name, title, description,
  root, connect_src, build_command). `dist` is `root / "dist"` and
  `path` is `/<name>/`. Unknown manifest keys are rejected, which applets
  inherit for free.
- Routes are installed per app at startup by
  [`install_app_routes`](https://github.com/marin-community/marin/blob/0780825da0fe6b5654717780b9a8349c1e8e6e89/infra/marina/src/marina/server.py#L219): a redirect for the
  bare prefix, `/<name>/data/{path}` from the data root, and
  `/<name>/{path}` from `dist` with an `index.html` fallback.
  [`serve_app_file`](https://github.com/marin-community/marin/blob/0780825da0fe6b5654717780b9a8349c1e8e6e89/infra/marina/src/marina/server.py#L145) takes a local `Path`;
  the data route already reads through `rigging.filesystem.factory.url_to_fs`,
  so a URL-backed file source has a precedent in the same file.
- Python apps are imported by package name from the apps directory
  ([`apps.py`](https://github.com/marin-community/marin/blob/0780825da0fe6b5654717780b9a8349c1e8e6e89/infra/marina/src/marina/apps.py#L46)) and mounted at `/<name>/api/`
  behind [`AuthenticatedMount`](https://github.com/marin-community/marin/blob/0780825da0fe6b5654717780b9a8349c1e8e6e89/infra/marina/src/marina/server.py#L193). Their engine
  comes from [`engine_for`](https://github.com/marin-community/marin/blob/0780825da0fe6b5654717780b9a8349c1e8e6e89/infra/marina/src/marina/db.py), which creates the app's
  schema and pins `search_path` per connection. The same call gives an
  applet its schema.
- The kernel has no schema of its own yet; `marina migrate` runs each app's
  `migrate(engine)`. The applet tables and the query role need a kernel
  migration, run first.
- The Shell reads `/api/marina/apps` ([`Shell.vue`](https://github.com/marin-community/marin/blob/0780825da0fe6b5654717780b9a8349c1e8e6e89/infra/marina/web/Shell.vue#L43))
  and renders whatever is listed; the directory is computed by
  [`app_directory`](https://github.com/marin-community/marin/blob/0780825da0fe6b5654717780b9a8349c1e8e6e89/infra/marina/src/marina/server.py#L124) from the manifest list.
- Journeys ([`journey_plugin.py`](https://github.com/marin-community/marin/blob/0780825da0fe6b5654717780b9a8349c1e8e6e89/infra/marina/src/marina/journey_plugin.py)) point a
  kernel at an apps directory. An applet's source directory is that shape, so
  the existing runner covers applets before publishing.
- The CLI ([`cli.py`](https://github.com/marin-community/marin/blob/0780825da0fe6b5654717780b9a8349c1e8e6e89/infra/marina/src/marina/cli.py)) already has `build` (runs
  `build_command`) and reads the apps directory; `publish` composes those.
- Echo's CLI ([`apps/echo/cli.py`](https://github.com/marin-community/marin/blob/0780825da0fe6b5654717780b9a8349c1e8e6e89/infra/marina/apps/echo/cli.py)) shows the
  agent-side pattern for calling Marina through IAP with a desktop OAuth
  token; `marina publish` reuses it.
- Cloud Run runs up to four instances (`infra/marina/__main__.py`), so any
  in-process cache must tolerate a publish landing on another instance; a
  short re-read window on the `applets` table is the coordination.

## Prior art

- Claude artifacts and Observable notebooks: publish is one action, the
  result is a URL, versions are kept, and there is no build step visible to
  the author. Applets copy the publish-and-link feel; the build stays on the
  author's side because Marina apps are compiled Vue.
- Val Town and Cloudflare Workers run uploaded server code in an isolated
  runtime per unit. Marina has no such runtime; the design defers server code
  and offers a query surface instead, which is what Metabase and Grafana do
  for dashboards over a shared database.
- Hyperbase (the inspiration for Marina) keeps every site in the repository;
  applets are the deliberate departure.

## Surprises

- Nothing in the kernel depends on apps being local except `serve_app_file`
  and the per-app route installation. Request-time resolution is a smaller
  change than expected.
- Per-app schemas plus `search_path` already give applets state isolation; the
  only new database concept is a role with cross-schema SELECT.

## Unclear

- Whether applets need server code at all in the first year, and if so whether
  a Cloud Run service per applet is acceptable operationally.
- Whether every app schema should be readable through the query surface by
  default (design assumes yes, per the earlier "users can do everything" call).
