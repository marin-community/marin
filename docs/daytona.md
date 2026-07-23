# Daytona operations

Marin uses Daytona as a supported container provider. The provider helpers are
available after installing the optional dependency:

~~~bash
uv sync --package marin-core --extra daytona
~~~

The commands take connection settings and the name of an environment variable
that holds the API key. They never read dotenv files or accept an API key in an
argument. For example:

~~~bash
export MARIN_DAYTONA_API_KEY=...
uv run --package marin-core python scripts/daytona/sandboxes.py \
  --api-key-env MARIN_DAYTONA_API_KEY --endpoint https://app.daytona.io/api --target us
~~~

## Safe resource reclamation

Sandbox and snapshot commands are audits by default. They print the selected
resources but make no provider mutation. To delete, pass `--delete` and answer
the exact-count prompt, or pass `--yes` in a non-interactive invocation.
Snapshot deletion also requires an explicit `--name-prefix`, so a shared base
image is never selected by an implicit organization-wide sweep. Pass `--json`
to save a machine-readable audit; JSON deletion requires `--yes`.

~~~bash
uv run --package marin-core python scripts/daytona/snapshots.py \
  --api-key-env MARIN_DAYTONA_API_KEY --endpoint https://app.daytona.io/api --target us \
  --name-prefix harbor__ --stale-after-days 14
~~~

Use the equivalent command with `--delete --yes` only after inspecting the
audit output. The library exposes the same policy through
`marin.daytona.sandboxes` and `marin.daytona.snapshots` for scheduled or
programmatic operations.

## Harbor task validation

Harbor validation treats Daytona and Iris as distinct backends. Daytona is the
backend for Dockerfile-backed tasks; Iris Harbor sandboxes run prebuilt task
images. A task validation report should record which backend ran each stage and
whether a failure came from the task or the provider.
