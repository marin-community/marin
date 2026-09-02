# Marina

One Cloud Run service that hosts Marin's small internal web apps. An app is a
directory under `apps/` with an `app.toml` and a `web/` frontend; the kernel in
`src/marina` discovers every app at startup and serves each one under
`/<name>/` on a single IAP-gated origin. Everything an app would otherwise
rebuild for itself (login, deployment, a top bar, a place for an agent to plug
in) lives in the kernel and the shared kit in `web/`.

## Layout

```
infra/marina/
  src/marina/      the kernel: manifest loading, auth, the FastAPI process, the CLI
  web/             the shared Vue kit apps import as @marina (tokens, Shell, client)
  apps/<name>/     one app: app.toml, web/ (Vue + rsbuild), dist/ after a build
  Dockerfile       node stage builds every apps/*/web; python stage runs `marina serve`
  __main__.py      Pulumi: one CloudRunService for the whole thing
```

`web/README.md` describes the kit and the contract an app builds against.

## Adding an app

1. Create `apps/<name>/app.toml`:

   ```toml
   title = "TaskTrove"
   description = "Browse the TaskTrove task collection."
   class = "static"          # "python" once the app registers ops
   connect_src = []          # extra origins the page may fetch; 'self' is implied
   build_command = "cd web && npm ci && npm run build"
   ```

   Unknown keys are rejected. `schedules` (a list of `{op, cron}`) is reserved
   for Python apps and becomes a Cloud Scheduler job per entry.
2. Put a Vue + rsbuild app in `apps/<name>/web/`, aliasing `@marina` to
   `../../../web`, emitting to `../dist` with asset prefix `/<name>/`, and
   rendering inside `<Shell app="<name>">`. Copy `apps/tasktrove/web` as the
   starting point.
3. Run it locally:

   ```bash
   uv run marina check                 # parse manifests, report build state
   uv run marina build --only <name>   # run the app's build_command
   uv run marina dev                   # http://127.0.0.1:8080/<name>/
   ```

   Nothing in Pulumi or the Dockerfile changes for a new app.

## Serving

`marina serve` reads `MARINA_APPS_DIR` and `MARINA_IAP_AUDIENCE`. The auth
chain is IAP's signed assertion header when an audience is configured, then
loopback; on Cloud Run a missing audience is a startup error rather than an
open service. `/api/marina/apps` lists the apps and `/api/marina/me` reports
the caller; `/healthz` is public.

## Deploying

```bash
cd infra/marina && uv run pulumi up
```

The image builds from the repository root (`Dockerfile.dockerignore` allowlists
`lib/rigging` and `infra/marina`). IAM for the deploy account and IAP viewers is
declared in the `marin` stack by `infra/pulumi/src/iac/gcp/marina.py`; apply
that stack first when the service is new.

## Tests

```bash
uv run pytest infra/marina/tests
```
